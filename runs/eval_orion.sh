#!/bin/bash

cd ../../
source DocMLLM_env/bin/activate

mkdir -p logs/eval_logs
mkdir -p heatmaps

export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

IFS=' ' read -r -a current_config <<< "$1"
EXP=${current_config[0]}              # e.g., baselineoptimized
NAME=${current_config[1]}             # e.g., baselineoptimized
WEIGHTS=${current_config[2]}          # e.g., checkpoint-12474
HEATMAPS=${current_config[3]}         # True or False

echo -e "\n🔍 Starting inference for $EXP (weights: $WEIGHTS, heatmaps: $HEATMAPS)"

ARGS="src/scripts/test.py \
  --model_config_path src/configs/$EXP.yaml \
  --data_config_path src/configs/$EXP.yaml \
  --weights checkpoints/$EXP/$WEIGHTS \
  --do_train False \
  --dataloader_num_workers 12 \
  --per_device_eval_batch_size 10 \
  --bf16 True \
  --generate_heatmaps $HEATMAPS \
  --heatmap_dir heatmaps/$NAME \
  --output_dir checkpoints/$NAME \
  --dataloader_pin_memory False \
  --remove_unused_columns False"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 nohup python $ARGS > "logs/eval_logs/$NAME.txt" &
