#!/bin/bash

cd /home/yohan.lascar/work/

# List of configurations
CONFIGS=(

  # "{name of config} {name of experiment} {path of weights} {generate heatmaps bool}"
)

# Loop through each configuration and submit a job
for config in "${CONFIGS[@]}"; do
  echo "$config"
  IFS=' ' read -r -a current_config <<< "$config"
  sbatch --gpus=1 AMDNet/runs/all/eval_dgx.sh "$config"
done