#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=""
OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videoevent/eval/model_video_chatgpt_general.py \
        --model-path ./work_dirs/$CKPT \
        --video_dir ./eval_data/videochatgpt_gen/Test_Videos \
        --gt_file ./eval_data/videochatgpt_gen/generic_qa.json \
        --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/1 \
        --output_name pred \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &

done

wait

python videoevent/eval/evaluate_benchmark_1_correctness.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT/1 \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/correctness_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/correctness_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE

python videoevent/eval/evaluate_benchmark_2_detailed_orientation.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT/1 \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/detail_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/detail_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE

python videoevent/eval/evaluate_benchmark_3_context.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT/1 \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/context_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/context_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16\
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE
