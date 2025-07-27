cd ../../
python -m venv DocMLLM_env
source DocMLLM_env/bin/activate
pip install --upgrade pip
pip install -r src/requirements.txt
pip install -U "huggingface_hub[cli]"
pip install -e .
pip install ipykernel notebook
pip install nltk
pip install pandas
pip install pyarrow
pip install scikit-learn