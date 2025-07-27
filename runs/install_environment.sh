#!/bin/bash

# Chemin absolu vers ton projet (dans le container Pyxis)
cd /workspace/repos/DocMLLM_repo || exit 1

# Crée un environnement virtuel nommé docmllmenv
python3 -m venv docmllmenv

# Active l'environnement
source docmllmenv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Installe les dépendances de requirements.txt
pip install -r src/requirements.txt

# Installe les outils Hugging Face CLI (optionnel mais utile)
pip install -U "huggingface_hub[cli]"

# Installe le repo en mode éditable si un setup.py existe
pip install -e .

# Jupyter support (si jamais tu fais du notebook)
pip install ipykernel notebook

# Install scikit-learn (provoquait l'erreur plus tôt)
pip install scikit-learn

echo "✅ Environnement docmllmenv installé avec succès !"

