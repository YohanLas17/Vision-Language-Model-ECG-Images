#!/bin/bash
cd ../../
source DocMLLM_env/bin/activate

mkdir -p logs

export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
"baselineoptimized 2 5 2 1e-4 0"
)

for config in "${CONFIGS[@]}"; do
  IFS=' ' read -r -a current_config <<< "$config"
  EXP=${current_config[0]}
  BS=${current_config[1]}
  EPOCHS=${current_config[2]}
  GA=${current_config[3]}
  LR=${current_config[4]}
  GPU=${current_config[5]}

  echo "🚀 Starting training: $EXP | BS=$BS | EPOCHS=$EPOCHS | GA=$GA | LR=$LR | GPU=$GPU"

  ARGS="src/scripts/train.py \
    --model_config_path src/configs/$EXP.yaml \
    --data_config_path src/configs/$EXP.yaml \
    --do_train True \
    --do_eval True \
    --dataloader_num_workers 8 \
    --per_device_eval_batch_size $BS \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps $GA \
    --warmup_steps 500 \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --num_train_epochs $EPOCHS \
    --save_total_limit 2 \
    --bf16 True \
    --push_to_hub False \
    --save_strategy epoch \
    --eval_strategy epoch \
    --metric_for_best_model ExactMatch \
    --load_best_model_at_end True \
    --logging_strategy steps \
    --logging_steps 20 \
    --output_dir checkpoints/$EXP \
    --logging_dir checkpoints/$EXP/runs \
    --dataloader_pin_memory False \
    --remove_unused_columns False"

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU nohup python $ARGS > "logs/$EXP.txt" &
done
