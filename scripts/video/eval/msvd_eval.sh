#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=""
OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videoevent/eval/model_msvd_qa.py \
    --model-path ./work_dirs/$CKPT \
    --video_dir ./eval_data/MSVD-QA/video \
    --gt_file ./eval_data/MSVD-QA/test_qa.json \
    --output_dir ./work_dirs/eval_msvd/$CKPT \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &
done

wait


python videoevent/eval/eval_msvd_qa.py \
    --pred_path ./work_dirs/eval_msvd/$CKPT \
    --output_dir ./work_dirs/eval_msvd/$CKPT/results \
    --output_json ./work_dirs/eval_msvd/$CKPT/results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE
