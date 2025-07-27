import logging
import transformers
from transformers import PreTrainedModel
from helpers.args import TrainingArguments,DataArguments,ModelArguments
from helpers.utils import load_yaml, get_object_from_path
import torch
from Datasets.DataModule import DataModule
from Trainer.trainer import CustomTrainer
import numpy as np
import random
from Trainer.metrics import Metrics
def train():
    parser = transformers.HfArgumentParser((ModelArguments,DataArguments,TrainingArguments))
    model_args,data_args,training_args = parser.parse_args_into_dataclasses()
    training_args.evaluation_strategy = training_args.eval_strategy
    save_dir = training_args.output_dir
    print("Experience will be saved in ",save_dir)
    logger = logging.getLogger()
    logger.info(
        f"training args {training_args}, model_args {model_args}, data_args {data_args}"
    )

    data_module = DataModule(data_args)
    kwargs = data_module.to_dict()

    model_class = load_yaml(model_args.model_config_path).get('model_class',None)
    model: PreTrainedModel = get_object_from_path(model_class)(model_args, processor = data_module.processor)
    model = model.to(torch.bfloat16)
    model.to("cuda")

    trainer = CustomTrainer(model = model,
                            args = training_args,
                            compute_metrics = Metrics,
                            **kwargs

                            )

    trainer.train()
    # model.save_pretrained('save_dir/best_model')
    # trainer.evaluate()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    logging.basicConfig(level=logging.INFO)
    train()


