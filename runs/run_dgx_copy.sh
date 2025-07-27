#!/bin/bash

#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --output=logs/dgx_logs/%x_%j.out  # SLURM job output
#SBATCH --cpus-per-task=80  # Corrected option
#SBATCH --time=24:00:00           # Maximum runtime of


ENV_NAME="amdenv"
eval "$(conda shell.bash hook)"

pwd
# Ensure the required packages are installed inside the container
mkdir -p logs  # Use -p to prevent error if the directory already exists

# **Add dynamic MASTER_PORT assignment**
MASTER_PORT=$((10000 + RANDOM % 50000))
echo "Using MASTER_PORT: $MASTER_PORT"

export CUDA_LAUNCH_BLOCKING=1  # Set a value if needed
IFS=' ' read -r -a current_config <<< "$1"
EXP=${current_config[0]}
BS=${current_config[1]}
EVAL_BS=${current_config[2]}
EP=${current_config[3]}
GA=${current_config[4]}
LR=${current_config[5]}
NAME=${current_config[6]}
NUM_GPUS=${current_config[7]}  # Parse the number of GPUs

mkdir -p logs/"${NAME%%/*}"

echo "starting experience $EXP, batch size : $BS, total steps : $EP"
echo "$NAME.txt"

# Print current working directory
echo "Current working directory:"
pwd

# Print current Python working directory
echo "Current Python working directory:"
srun --container-image /home/benjaminc/athena/nvidia+pytorch+24.06-py3-new.sqsh --container-mounts /home/benjaminc/work/:/workspace/,/home/benjaminc/work:/rg/behar_prj/benjaminc/ bash -c "cd /workspace/AMDNet && python -c 'import os; print(os.getcwd())'"

#
#--max_steps $TOTAL_STEPS \
#--evaluation_strategy steps \
#--save_steps 200 \
#--eval_steps 200 \
#--master_port $MASTER_PORT \

ARGS="--nnodes 1 --nproc_per_node $NUM_GPUS \
--master_port $MASTER_PORT \
scripts/train.py \
--model_config_path configs/$EXP.yaml \
--data_config_path configs/$EXP.yaml \
--do_train \
--dataloader_num_workers 40 \
--per_device_eval_batch_size $EVAL_BS \
--per_device_train_batch_size $BS \
--gradient_accumulation_steps $GA \
--warmup_steps 100 \
--learning_rate $LR \
--lr_scheduler_type cosine \
--weight_decay 0.05 \
--num_train_epochs $EPOCHS \
--save_total_limit 2 \
--bf16 \
--push_to_hub False \
--save_strategy epoch \
--logging_steps 20 \
--metric_for_best_model loss \
--load_best_model_at_end \
--output_dir /workspace/weights/$NAME"
"

srun --container-image /home/benjaminc/athena/nvidia+pytorch+24.06-py3-new.sqsh \
    --container-mounts /home/benjaminc/work/:/workspace/,/home/benjaminc/work/:/rg/behar_prj/benjaminc/,/rg/behar_prj/benjaminc/amd_datasets/:/workspace/amd_datasets/ \
    --export=ALL,OMP_NUM_THREADS=2,MKL_NUM_THREADS=4 \
    bash -c "
source /workspace/amdenv/bin/activate && \
cd /workspace/AMDNet && \
export PYTHONPATH=/workspace/AMDNet:$PYTHONPATH && \
/workspace/amdenv/bin/torchrun $ARGS
" > "/home/benjaminc/work/logs/$NAME.txt" 2>&1

if [ $? -ne 0 ]; then
  echo "Experiment $NAME failed. Check logs for details."
else
  echo "Experiment $NAME completed successfully."
fi