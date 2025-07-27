#!/bin/bash

cd /workspace/repos/DocMLLM_repo || exit 1
ENV_NAME="docmllmenv39"

echo "[INFO] Removing previous $ENV_NAME..."
rm -rf $ENV_NAME

echo "[INFO] Creating venv using Python: $(which python3.9)"
python3.9 -m venv $ENV_NAME

echo "[INFO] Activating environment..."
source $ENV_NAME/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

echo "[INFO] Installing requirements..."
pip install -r requirements.txt

echo "[SUCCESS] Venv $ENV_NAME ready. To activate later:"
echo "source $ENV_NAME/bin/activate"
