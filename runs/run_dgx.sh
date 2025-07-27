#!/bin/bash

#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dgx_logs/%x_%j.out
#SBATCH --time=2:00:00

ENV_NAME="DocMLLM_env"
pwd
mkdir -p logs

MASTER_PORT=$((10000 + RANDOM % 50000))
echo "Using MASTER_PORT: $MASTER_PORT"

export CUDA_LAUNCH_BLOCKING=1
IFS=' ' read -r -a current_config <<< "$1"
EXP=${current_config[0]}
BS=${current_config[1]}
EVAL_BS=${current_config[2]}
EPOCHS=${current_config[3]}
GA=${current_config[4]}
LR=${current_config[5]}
NAME=${current_config[6]}
NUM_GPUS=${current_config[7]}

export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/"${NAME%%/*}"
echo "Parsed config: EXP=$EXP | BS=$BS | EVAL_BS=$EVAL_BS | EPOCHS=$EPOCHS | GA=$GA | LR=$LR | NAME=$NAME | NUM_GPUS=$NUM_GPUS"

CONTAINER_IMAGE="/rg/behar_prj/yohan.lascar/prj/nvidia+pytorch+24.06-py3-new.sqsh"
MOUNT_PATH="/rg/behar_prj/yohan.lascar/prj/repos/"
WORKSPACE_PATH="/workspace/repos/DocMLLM_repo"
ENV_PATH="$WORKSPACE_PATH/DocMLLM_env/bin/activate"

ARGS="--nnodes 1 --nproc_per_node $NUM_GPUS \
--master_port $MASTER_PORT \
scripts/train.py \
--model_config_path configs/$EXP.yaml \
--data_config_path configs/$EXP.yaml \
--output_dir /workspace/repos/DocMLLM_repo/weights/$NAME \
--do_train \
--dataloader_num_workers 4 \
--per_device_eval_batch_size $EVAL_BS \
--per_device_train_batch_size $BS \
--gradient_accumulation_steps $GA \
--warmup_steps 100 \
--learning_rate $LR \
--lr_scheduler_type cosine \
--weight_decay 0.05 \
--num_train_epochs $EPOCHS \
--save_total_limit 2 \
--dataloader_pin_memory false \
--push_to_hub false \
--max_steps 10 \
--logging_steps 10"

srun --container-image=$CONTAINER_IMAGE \
     --container-mounts=$MOUNT_PATH:/workspace/repos \
     --export=ALL,OMP_NUM_THREADS=2,MKL_NUM_THREADS=4 \
     bash -c "
     source $ENV_PATH && \
     cd $WORKSPACE_PATH && \
     export PYTHONPATH=$WORKSPACE_PATH:\$PYTHONPATH && \
     $WORKSPACE_PATH/DocMLLM_env/bin/torchrun $ARGS" \
     > "/rg/behar_prj/yohan.lascar/prj/repos/DocMLLM_repo/runs/logs/$NAME.txt" 2>&1

if [ $? -ne 0 ]; then
  echo '❌ Experiment $NAME failed. Check logs/$NAME.txt for details.'
else
  echo '✅ Experiment $NAME completed successfully.'
fi
