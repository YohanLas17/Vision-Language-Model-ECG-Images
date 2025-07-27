#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CONFIGS=(
  "baselineoptimized baselineoptimized checkpoint-12474 False"
)

for cfg in "${CONFIGS[@]}"; do
  echo "➤ Running eval_orion.sh with config: $cfg"
  bash ./eval_orion.sh "$cfg"
done
