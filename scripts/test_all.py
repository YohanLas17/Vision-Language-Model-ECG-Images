import os, sys
import torch
import transformers
import pandas as pd
from transformers import AutoProcessor
from transformers import AutoTokenizer, AutoImageProcessor
from helpers.args import ModelArguments, TrainingArguments, EvalArguments
from helpers.utils import load_yaml, get_object_from_path
from Datasets.DataCollatorForSupervisedDataset import DataCollatorForSupervisedDataset
from Evaluator.evaluator import run_all_tests

def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, EvalArguments))
    model_args, training_args, eval_args = parser.parse_args_into_dataclasses()
    config = load_yaml(eval_args.data_config_path)
    model_class = get_object_from_path(config['model_class'])



#    tokenizer = AutoTokenizer.from_pretrained("SurfaceData/llava-v1.6-vicuna-7b-processor", trust_remote_code=True)
 #   image_processor = AutoImageProcessor.from_pretrained("SurfaceData/llava-v1.6-vicuna-7b-processor",
#                                                        trust_remote_code=True)

    processor = AutoProcessor.from_pretrained("SurfaceData/llava-v1.6-vicuna-7b-processor", trust_remote_code=True)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = 16
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = "default"

    # Ajouter token spécial "<image>" s’il manque
    if "<image>" not in processor.tokenizer.additional_special_tokens:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})


    # Charger le modèle (le processor est géré dedans)
    model = model_class(model_args, processor=processor)
    model.model_pretrained.resize_token_embeddings(len(processor.tokenizer))
    model = model.to("cuda").bfloat16()
    print("✅ Modèle chargé depuis pretrained (avec processor)")

    # Collator avec processor du modèle
    collator = DataCollatorForSupervisedDataset(
        processor, max_length=config.get('max_length', 512), is_train=False
    )

    # Définir les datasets
    test_datasets = [
        {
            "name": "CODE15",
            "path": os.path.join(config['root_dir'], "PULSE_TEST/code15-test/test-00000-of-00001.parquet"),
            "metrics": ["F1", "AUC", "HL"],
            "label_space": ["NORM","ABNORMAL","1DAVB","RBBB","LBBB","SB","ST","AF"],
        },
        {
            "name": "ECG-QA",
            "path": os.path.join(config['root_dir'], "PULSE_TEST/ecgqa-test/test-00000-of-00001.parquet"),
            "metrics": ["Accuracy"],
            "label_space": ["A","B","C","D","E"],
        },
        {
            "name": "CPSC",
            "path": os.path.join(config['root_dir'], "PULSE_TEST/cpsc-test/test-00000-of-00001.parquet"),
            "metrics": ["F1", "AUC", "HL"],
            "label_space": ["NORM","AF","I-AVB","LBBB","RBBB","PAC","PVC","STD","STE"],
        },
        {
            "name": "PTB-XL",
            "path": os.path.join(config['root_dir'], "PULSE_TEST/ptb-test/test-00000-of-00001.parquet"),
            "metrics": ["F1", "AUC", "HL"],
            "label_space": ["NORM","MI","STTC","CD","HYP"],
        },
        {
            "name": "MMMU",
            "path": os.path.join(config['root_dir'], "PULSE_TEST/mmmu-ecg/test-00000-of-00001.parquet"),
            "metrics": ["Accuracy"],
            "label_space": ["A","B","C","D"],
        },
        {
            "name": "CSN",
            "path": os.path.join(config['root_dir'], "PULSE_TEST/csn-test-no-cot/test-00000-of-00001.parquet"),
            "metrics": ["Accuracy"],
            "label_space": ["A","B","C","D","E","F","G","H"],
        },
        {
            "name": "G12EC",
            "path": os.path.join(config['root_dir'], "PULSE_TEST/g12-test-no-cot/test-00000-of-00001.parquet"),
            "metrics": ["Accuracy"],
            "label_space": ["A","B","C","D","E","F","G","H"],
        },
    ]

    gen_kwargs = config.get('generation_config', {})

    # Lancement de l’évaluation
    df_results = run_all_tests(
        test_datasets=test_datasets,
        model=model,
        processor=processor,
        collator=collator,
        device="cuda",
        gen_kwargs={"max_new_tokens": 16},
        n_samples=None
    )


    # Affichage des résultats
    print("\n=== Final Scores ===")
    print(df_results.fillna('-').to_markdown())

if __name__ == "__main__":
    main()
