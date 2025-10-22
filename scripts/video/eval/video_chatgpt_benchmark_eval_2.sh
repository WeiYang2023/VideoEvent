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
        --gt_file ./eval_data/videochatgpt_gen/temporal_qa.json \
        --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/2 \
        --output_name pred_temporal \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &

done

wait

python videoevent/eval/evaluate_benchmark_4_temporal.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT/2 \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/temporal_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/temporal_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE
