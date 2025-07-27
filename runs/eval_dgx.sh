#!/bin/bash

#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --output=logs/eval/dgx_logs/%x_%j.out  # SLURM job output
#SBATCH --cpus-per-task=40 # Corrected option
#SBATCH --time=2:00:00           # Maximum runtime of 12 hours-


ENV_NAME="docmlenv"
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
NAME=${current_config[1]}
WEIGHTS=${current_config[2]}
HEATMAPS=${current_config[3]}

mkdir -p logs/eval/"${NAME%%/*}"

echo "starting inference for experience $EXP, named $NAME, with weights $WEIGHTS, generating heatmaps? $HEATMAPS"
echo "$NAME.txt"

# Print current working directory
echo "Current working directory:"
pwd
# Define paths
CONTAINER_IMAGE="/rg/behar_prj/yohan.lascar/prj/nvidia+pytorch+24.06-py3-new.sqsh"
MOUNT_PATH="/rg/behar_prj/yohan.lascar/prj/repos/"
WORKSPACE_PATH="/workspace/repos/DocMLLM_repo"
ENV_PATH="$WORKSPACE_PATH/docmllmenv/bin/activate"
# Print current Python working directory
echo "Current Python working directory:"
srun --container-image /home/benjaminc/athena/nvidia+pytorch+24.06-py3-new.sqsh --container-mounts /home/benjaminc/work/:/workspace/,/home/benjaminc/work:/rg/behar_prj/benjaminc/ bash -c "cd /workspace/AMDNet && python -c 'import os; print(os.getcwd())'"

#--nnodes 1 --nproc_per_node $NUM_GPUS \
#--max_steps $TOTAL_STEPS \
#--evaluation_strategy steps \
#--save_steps 200 \
#--eval_steps 200 \
#--master_port $MASTER_PORT \

ARGS="scripts/test.py \
--model_config_path configs/$EXP.yaml \
--data_config_path configs/$EXP.yaml \
--do_train True \
--dataloader_num_workers 120 \
--per_device_eval_batch_size 10 \
--per_device_train_batch_size 10 \
--bf16 True \
--push_to_hub=False \
--save_strategy epoch \
--logging_steps 20 \
--metric_for_best_model loss \
--load_best_model_at_end True \
--output_dir /workspace/eval/$NAME \
--generate_heatmaps $HEATMAPS \
--heatmap_dir /workspace/heatmaps/$NAME \
--weights /workspace/weights/$WEIGHTS \
"

srun --container-image /home/yohan.lascar/athena/nvidia+pytorch+24.06-py3-new.sqsh \
    --container-mounts /home/yohan.lascar/work/:/workspace/,/home/yohan.lascar/work/:/rg/behar_prj/yohan.lascar/,/rg/behar_prj/yohan.lascar/Databases/:/workspace/amd_datasets/ \
    --export=ALL,OMP_NUM_THREADS=2,MKL_NUM_THREADS=4 \
    bash -c "
source /rg/behar_prj/yohan.lascar/prj/repos/DocMLLM_repo/docmlenv && \
cd /workspace/repos/DocMLLM_repo && \
export PYTHONPATH=/workspace/repos/DocMLLM_repo:$PYTHONPATH && \
/DocMLLM_repo/docmlenv/bin/python $ARGS
" > "/home/yohan.lascar/work/logs/eval/$NAME.txt" 2>&1

if [ $? -ne 0 ]; then
  echo "Inference $NAME failed. Check logs for details."
else
  echo "Inference $NAME completed successfully."
fi