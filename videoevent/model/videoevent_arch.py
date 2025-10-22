#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2023 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import math
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLMHeadModel as BertLMHeadModelRaw

from .qformer import (
    BertConfig,
    MyExtractBlock,
    MySlidingBlock,
)
from .qformer import BertLMHeadModel as BertLMHeadModelQF

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from videoevent.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class VideoEventMetaModel:

    def __init__(self, config):
        super(VideoEventMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_processor = getattr(model_args, "image_processor", None)

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.max_token = max_token

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )

    def initialize_cross_attention(self, model_args, stage, vs=2):
        extract_config = {
            "num_attention_heads": 8,
            "hidden_size": 1408,
            "encoder_width": 1408,
            "attention_probs_dropout_prob": 0.1,
            "position_embedding_type": "absolute",
        }
        L1_config = {
            "num_attention_heads": 8,
            "hidden_size": 1408,
            "encoder_width": 1408,
            "attention_probs_dropout_prob": 0.1,
            "position_embedding_type": "absolute",
            "window_size": 10,
            "stride": 4,
            "k": 10,
            "select_config": {
                "num_attention_heads": 8,
                "hidden_size": 1408,
                "encoder_width": 768,
                "attention_probs_dropout_prob": 0.1,
                "position_embedding_type": "absolute",
            },
        }
        L2_config = {
            "num_attention_heads": 8,
            "hidden_size": 1408,
            "encoder_width": 1408,
            "attention_probs_dropout_prob": 0.1,
            "position_embedding_type": "absolute",
            "window_size": 5,
            "stride": 2,
            "k": 5,
            "select_config": {
                "num_attention_heads": 8,
                "hidden_size": 1408,
                "encoder_width": 768,
                "attention_probs_dropout_prob": 0.1,
                "position_embedding_type": "absolute",
            },
        }
        L3_config = {
            "num_attention_heads": 8,
            "hidden_size": 1408,
            "encoder_width": 1408,
            "attention_probs_dropout_prob": 0.1,
            "position_embedding_type": "absolute",
            "window_size": 2,
            "stride": 1,
            "k": 3,
            "select_config": {
                "num_attention_heads": 8,
                "hidden_size": 1408,
                "encoder_width": 768,
                "attention_probs_dropout_prob": 0.1,
                "position_embedding_type": "absolute",
            },
        }
        if stage == 1:
            self.extract = MyExtractBlock(extract_config, vs=vs)
            self.L1_block = MySlidingBlock(config=L1_config, vs=vs)
            self.L2_block = MySlidingBlock(config=L2_config, vs=vs)
            self.L3_block = MySlidingBlock(config=L3_config, vs=vs)
            self.time_emb = torch.nn.Embedding(1200, 1408)

            self.bert_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", truncation_side="right"
            )
            bert_config = BertConfig.from_pretrained(
                "bert-base-uncased"
            )
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased", config=bert_config
            )
            self.bert.requires_grad_(False)

        if stage == 2:
            self.extract = MyExtractBlock(extract_config, vs=vs)
            self.L1_block = MySlidingBlock(config=L1_config, vs=vs)
            self.L2_block = MySlidingBlock(config=L2_config, vs=vs)
            self.L3_block = MySlidingBlock(config=L3_config, vs=vs)
            self.time_emb = torch.nn.Embedding(1200, 1408)

            self.bert_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", truncation_side="right"
            )
            bert_config = BertConfig.from_pretrained(
                "bert-base-uncased"
            )
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased", config=bert_config
            )
            self.bert.requires_grad_(True)

            pretrain_mm_mlp_adapter = getattr(
                model_args, "pretrain_mm_mlp_adapter", None
            )
            if pretrain_mm_mlp_adapter is not None:
                print(f"pretrain_mm_mlp_adapter:{pretrain_mm_mlp_adapter}")
                att_projector_weights = torch.load(
                    pretrain_mm_mlp_adapter, map_location="cpu"
                )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.extract.load_state_dict(get_w(att_projector_weights, "extract"))
            self.L1_block.load_state_dict(get_w(att_projector_weights, "L1_block"))
            self.L2_block.load_state_dict(get_w(att_projector_weights, "L2_block"))
            self.L3_block.load_state_dict(get_w(att_projector_weights, "L3_block"))
            self.time_emb.load_state_dict(get_w(att_projector_weights, "time_emb"))

        if model_args.lora_resume and stage == 3:
            self.extract = MyExtractBlock(extract_config, vs=vs)
            self.L1_block = MySlidingBlock(config=L1_config, vs=vs)
            self.L2_block = MySlidingBlock(config=L2_config, vs=vs)
            self.L3_block = MySlidingBlock(config=L3_config, vs=vs)
            self.time_emb = torch.nn.Embedding(1200, 1408)
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", truncation_side="right"
            )
            bert_config = BertConfig.from_pretrained(
                "bert-base-uncased"
            )
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased", config=bert_config
            )

            pretrain_mm_mlp_adapter = getattr(
                model_args, "pretrain_mm_mlp_adapter", None
            )
            print(pretrain_mm_mlp_adapter)
            if pretrain_mm_mlp_adapter is not None:
                att_projector_weights = torch.load(
                    pretrain_mm_mlp_adapter, map_location="cpu"
                )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.extract.load_state_dict(get_w(att_projector_weights, "extract"))
            self.L1_block.load_state_dict(get_w(att_projector_weights, "L1_block"))
            self.L2_block.load_state_dict(get_w(att_projector_weights, "L2_block"))
            self.L3_block.load_state_dict(get_w(att_projector_weights, "L3_block"))
            self.time_emb.load_state_dict(get_w(att_projector_weights, "time_emb"))
            self.bert.load_state_dict(get_w(att_projector_weights, "bert"))

        if (not model_args.lora_resume) and stage == 3:
            self.extract = MyExtractBlock(extract_config, vs=vs)
            self.L1_block = MySlidingBlock(config=L1_config, vs=vs)
            self.L2_block = MySlidingBlock(config=L2_config, vs=vs)
            self.L3_block = MySlidingBlock(config=L3_config, vs=vs)
            self.time_emb = torch.nn.Embedding(1200, 1408)

            self.bert_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", truncation_side="right"
            )
            bert_config = BertConfig.from_pretrained(
                "bert-base-uncased"
            )
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased", config=bert_config
            )

            trainable_module = [
                "extract",
                "L1_block",
                "L2_block",
                "L3_block",
                "time_emb",
                "bert",
            ]
            if hasattr(model_args, "model_name_or_path"):
                model_save_path = model_args.model_name_or_path
            else:
                model_save_path = model_args.model_path
            model_idx_path = getattr(model_args, "model_path", model_save_path)
            weight_file = json.load(
                open(os.path.join(model_idx_path, "pytorch_model.bin.index.json"), "r")
            )["weight_map"]
            model_path = set(
                [
                    weight_file[_key]
                    for _key in weight_file
                    if any([_module in _key for _module in trainable_module])
                ]
            )
            att_projector_weights = {}
            for _model in model_path:
                att_projector_weights.update(
                    torch.load(os.path.join(model_idx_path, _model), map_location="cpu")
                )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.extract.load_state_dict(get_w(att_projector_weights, "extract"))
            self.L1_block.load_state_dict(get_w(att_projector_weights, "L1_block"))
            self.L2_block.load_state_dict(get_w(att_projector_weights, "L2_block"))
            self.L3_block.load_state_dict(get_w(att_projector_weights, "L3_block"))
            self.time_emb.load_state_dict(get_w(att_projector_weights, "time_emb"))
            self.bert.load_state_dict(get_w(att_projector_weights, "bert"))

        if stage == 3:
            weight_type = torch.bfloat16
            device_type = self.mm_projector[0].weight.device
            self.extract = self.extract.to(device=device_type, dtype=weight_type)
            self.L1_block = self.L1_block.to(device=device_type, dtype=weight_type)
            self.L2_block = self.L2_block.to(device=device_type, dtype=weight_type)
            self.L3_block = self.L3_block.to(device=device_type, dtype=weight_type)
            self.time_emb = self.time_emb.to(device=device_type, dtype=weight_type)
            self.bert = self.bert.to(device=device_type, dtype=weight_type)


class VideoEventMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def cal_cosine(self, frame_1, frame_2):
        cosine_similarity = torch.dot(frame_1.flatten(), frame_2.flatten()) / (
            torch.norm(frame_1) * torch.norm(frame_2)
        )
        return cosine_similarity

    def merge_images(self, image_features, batch_size=1, t=0.6):
        # features_device = image_features.device
        if image_features.shape[0] == batch_size:
            pass
        else:
            i = 0
            while i < (image_features.shape[0] - 1):
                cosine_score = self.cal_cosine(image_features[i], image_features[i + 1])
                if cosine_score > t:
                    image_features[i] = (image_features[i] + image_features[i + 1]) / 2
                    if i == (image_features.shape[0] - 2):
                        image_features = image_features[: i + 1]
                    else:
                        image_features = torch.cat(
                            (image_features[: i + 1], image_features[i + 2 :]), dim=0
                        )
                else:
                    i += 1
        return image_features

    def encode_events(self, images, prompts=None, image_counts=None):
        # image_features = self.get_model().get_vision_tower()(images)

        if image_counts is None:
            image_features = self.get_model().get_vision_tower()(images)
            position_ids = torch.zeros(
                (1, image_features.shape[0]),
                device=image_features.device,
                dtype=torch.long,
            )
            position_emb = (
                self.get_model()
                .time_emb(position_ids)
                .to(image_features.dtype)
                .squeeze(0)
                .unsqueeze(1)
            )
            image_features = image_features + position_emb
            all_output = self.get_events(image_features, prompts=prompts)
        else:
            total_count = 0
            new_count = []
            new_images = []
            for _idx in range(len(prompts)):
                current_images = images[total_count : total_count + image_counts[_idx]]
                current_images = self.get_model().get_vision_tower()(current_images)
                position_ids = (
                    torch.arange(current_images.shape[0])
                    .expand((1, -1))
                    .to(current_images.device)
                )
                position_emb = (
                    self.get_model()
                    .time_emb(position_ids)
                    .to(current_images.dtype)
                    .squeeze(0)
                    .unsqueeze(1)
                )
                current_images = current_images + position_emb
                current_images = self.merge_images(current_images)
                new_count.append(current_images.shape[0])
                new_images.append(current_images)
                total_count += image_counts[_idx]
            new_images = torch.cat(new_images, dim=0)
            all_output = self.get_events(
                new_images, prompts=prompts, image_counts=new_count
            )
        return all_output

    def get_events(self, image_features, prompts=None, image_counts=None):
        if image_counts is None:
            all_output = []
            for idx in range(image_features.shape[0]):
                current_images = image_features[idx].unsqueeze(0)

                input_token = (
                    self.get_model()
                    .bert_tokenizer(
                        prompts[idx],
                        padding="longest",
                        truncation=True,
                        max_length=256,
                        return_tensors="pt",
                    )
                    .to(image_features.device)
                )
                input_ids = input_token.input_ids
                attention_masks = input_token.attention_mask
                token_type_ids = input_token.token_type_ids
                prompt_feature = self.get_model().bert(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
                prompt_feature = prompt_feature.last_hidden_state

                image_num = current_images.shape[0]
                if image_num == 1:
                    window_size_1 = 1
                    stride_1 = 1
                    window_size_2 = 1
                    stride_2 = 1
                    window_size_3 = 1
                    stride_3 = 1

                L1_in = current_images.unsqueeze(0)
                L1_out, L1_sel = self.get_model().L1_block(
                    L1_in, prompt_feature, window_size_1, stride_1
                )
                L1_out = torch.cat(L1_out, dim=0)
                
                L2_in = L1_out.unsqueeze(0)
                L2_out, L2_sel = self.get_model().L2_block(
                    L2_in, prompt_feature, window_size_2, stride_2
                )
                L2_out = torch.cat(L2_out, dim=0)

                L3_in = L2_out.unsqueeze(0)
                L3_out, L3_sel = self.get_model().L3_block(
                    L3_in, prompt_feature, window_size_3, stride_3
                )
                L3_out = torch.cat(L3_out, dim=0)
                
                all_out = torch.cat((L1_sel, L2_sel, L3_sel), dim=1)
                all_out = torch.mean(all_out, dim=2)
                all_out = self.get_model().mm_projector(all_out)
                all_output.append(all_out)
        else:
            total_count = 0
            all_output = []
            for _idx in range(len(prompts)):
                current_images = image_features[
                    total_count : total_count + image_counts[_idx]
                ]
                current_images = self.get_model().extract(current_images)

                input_token = (
                    self.get_model()
                    .bert_tokenizer(
                        prompts[_idx],
                        padding="longest",
                        truncation=True,
                        max_length=256,
                        return_tensors="pt",
                    )
                    .to(image_features.device)
                )

                input_ids = input_token.input_ids
                attention_masks = input_token.attention_mask
                token_type_ids = input_token.token_type_ids

                prompt_feature = self.get_model().bert(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
                prompt_feature = prompt_feature.last_hidden_state

                image_num = current_images.shape[0]

                if image_num == 1:
                    window_size_1 = 1
                    stride_1 = 1
                elif image_num > 1:
                    window_size_1 = min(max(math.ceil(image_num / 10), 2), 10)
                    stride_1 = math.ceil(window_size_1 / 2)
                L1_in = current_images.unsqueeze(0)
                L1_out, L1_sel = self.get_model().L1_block(
                    L1_in, prompt_feature, window_size_1, stride_1
                )
                L1_out = torch.cat(L1_out, dim=0)
                
                L1_num = L1_out.shape[0]
                if L1_num == 1:
                    window_size_2 = 1
                    stride_2 = 1
                elif L1_num > 1:
                    # window_size_2 = max(math.ceil(L1_num / 5), 2)
                    window_size_2 = min(max(math.ceil(L1_num / 5), 2), 8)
                    stride_2 = math.ceil(window_size_2 / 2)
                L2_in = L1_out.unsqueeze(0)
                L2_out, L2_sel = self.get_model().L2_block(
                    L2_in, prompt_feature, window_size_2, stride_2
                )
                L2_out = torch.cat(L2_out, dim=0)
                
                L2_num = L2_out.shape[0]
                if L2_num == 1:
                    window_size_3 = 1
                    stride_3 = 1
                elif L2_num > 1:
                    # window_size_3 = max(math.ceil(L2_num / 3), 2)
                    # stride_3 = math.floor(window_size_3 / 2)
                    window_size_3 = min(max(math.ceil(L2_num / 3), 2), 5)
                    stride_3 = math.ceil(window_size_3 / 2)
                L3_in = L2_out.unsqueeze(0)
                L3_out, L3_sel = self.get_model().L3_block(
                    L3_in, prompt_feature, window_size_3, stride_3
                )
                L3_out = torch.cat(L3_out, dim=0)
                
                all_out = torch.cat((L1_sel, L2_sel, L3_sel), dim=1)
                all_out = torch.mean(all_out, dim=2)
                all_out = self.get_model().mm_projector(all_out)
                all_output.append(all_out)
                total_count += image_counts[_idx]
        return all_output

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, prompts=None
    ):
        if prompts is None and hasattr(self, "prompts"):
            prompts = self.prompts
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            images = [
                image if len(image.shape) == 4 else image.unsqueeze(0)
                for image in images
            ]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features = self.encode_events(concat_images, prompts, image_counts)
        else:
            image_features = self.encode_events(images, prompts)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids[:half_len]
                )
                cur_input_embeds_2 = self.get_model().embed_tokens(
                    cur_input_ids[half_len:]
                )
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2],
                    dim=0,
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if True:
                token_idx = 0
                while image_token_indices.numel() > 0:
                    if isinstance(image_features, list):
                        cur_image_features = image_features[cur_image_idx][token_idx]
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    image_token_start = image_token_indices[0]

                    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                        self.config, "mm_use_im_start_end", False
                    ):
                        cur_new_input_embeds.append(
                            self.get_model()
                            .embed_tokens(cur_input_ids[: image_token_start - 1])
                            .detach()
                        )
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(
                                cur_input_ids[image_token_start - 1 : image_token_start]
                            )
                        )
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(
                                cur_input_ids[
                                    image_token_start + 1 : image_token_start + 2
                                ]
                            )
                        )
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],),
                                    IGNORE_INDEX,
                                    device=labels.device,
                                    dtype=labels.dtype,
                                )
                            )
                            cur_new_labels.append(
                                cur_labels[image_token_start : image_token_start + 1]
                            )
                            cur_labels = cur_labels[image_token_start + 2 :]
                    else:
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(
                                cur_input_ids[:image_token_start]
                            )
                        )
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],),
                                    IGNORE_INDEX,
                                    device=labels.device,
                                    dtype=labels.dtype,
                                )
                            )
                            cur_labels = cur_labels[image_token_start + 1 :]
                    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                        self.config, "mm_use_im_start_end", False
                    ):
                        cur_input_ids = cur_input_ids[image_token_start + 2 :]
                    else:
                        cur_input_ids = cur_input_ids[image_token_start + 1 :]
                    image_token_indices = torch.where(
                        cur_input_ids == IMAGE_TOKEN_INDEX
                    )[0]
                    token_idx += 1

                # changle image idx after processing one sample
                cur_image_idx += 1
                if cur_input_ids.numel() > 0:
                    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                        self.config, "mm_use_im_start_end", False
                    ):
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids).detach()
                        )
                    else:
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids)
                        )
                    if labels is not None:
                        cur_new_labels.append(cur_labels)
                cur_new_input_embeds = [
                    x.to(device=self.device) for x in cur_new_input_embeds
                ]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    cur_new_labels = torch.cat(cur_new_labels, dim=0)
                    new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return (None, attention_mask, past_key_values, new_input_embeds, new_labels)

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
