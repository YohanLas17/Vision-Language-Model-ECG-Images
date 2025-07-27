#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CONFIGS=(
  "baselineoptimized 2 2 5 2 1e-4 baselineoptimized_dgx 1"
)

chmod +x "$SCRIPT_DIR/run_dgx.sh"

for config in "${CONFIGS[@]}"; do
  echo "Submitting config: $config"
  IFS=' ' read -r -a current_config <<< "$config"
  NUM_GPUS=${current_config[7]}
  sbatch --gpus="$NUM_GPUS" "$SCRIPT_DIR/run_dgx.sh" "$config"
done
