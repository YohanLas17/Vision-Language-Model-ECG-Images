import os, sys
import torch
import transformers
import pandas as pd
from transformers import AutoProcessor
from helpers.args import ModelArguments, TrainingArguments, EvalArguments
from helpers.utils import load_yaml, get_object_from_path
from Datasets.DataCollatorForSupervisedDataset import DataCollatorForSupervisedDataset
from Evaluator.evaluator_code15 import run_all_tests

def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, EvalArguments))
    model_args, training_args, eval_args = parser.parse_args_into_dataclasses()
    config = load_yaml(eval_args.data_config_path)

    torch.backends.cuda.matmul.allow_tf32 = True  # ✅ pour A100 perf

    # Load model class
    model_class = get_object_from_path(config['model_class'])

    # Load processor
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", trust_remote_code=True)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = 16
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = "default"
    if "<image>" not in processor.tokenizer.additional_special_tokens:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

    # Load model
    model = model_class(model_args, processor=processor)
    model.model_pretrained.resize_token_embeddings(len(processor.tokenizer))

    if eval_args.weights:
        safetensor_path = os.path.join(eval_args.weights, "model.safetensors")
        try:
            print(f"🔁 Loading weights from: {safetensor_path}")
            from safetensors.torch import load_file
            state_dict = load_file(safetensor_path)
            model.model_pretrained.load_state_dict(state_dict)
        except Exception as e:
            print(f"⚠️ Failed to load weights from .safetensors: {e}")

    model = model.to("cuda").bfloat16()
    print(f"✅ Model loaded on CUDA with dtype bfloat16")

    collator = DataCollatorForSupervisedDataset(
        processor, max_length=config.get('max_length', 512), is_train=False
    )

    # CODE15 dataset
    test_datasets = [
        {
            "name": "CODE15",
            "path": os.path.join(config['root_dir'], "PULSE_TEST/code15-test/test-00000-of-00001.parquet"),
            "metrics": ["F1", "AUC", "HL"],
            "label_space": ["NORM", "ABNORMAL", "1DAVB", "RBBB", "LBBB", "SB", "ST", "AF"],
        }
    ]

    gen_kwargs = {
        "max_new_tokens": 32,
        "do_sample": False,
        "num_beams": 4,
        "repetition_penalty": 1.2
    }

    output_path = os.path.join(eval_args.weights, "eval_code15_predictions.jsonl") if eval_args.weights else "code15_preds.jsonl"

    try:
        df_results = run_all_tests(
            test_datasets=test_datasets,
            model=model,
            processor=processor,
            collator=collator,
            device="cuda",
            gen_kwargs=gen_kwargs,
            n_samples=None,
            save_preds=True,
            save_path=output_path
        )
    except RuntimeError as e:
        print(f"❌ Evaluation failed: {e}")
        return

    print("\n=== Final Scores ===")
    print(df_results.fillna('-').to_markdown())
    print(f"\n📁 Results saved to {output_path} with {len(df_results)} dataset(s) evaluated.")

if __name__ == "__main__":
    main()
