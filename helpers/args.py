from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class ModelArguments:
    model_config_path: str = field(default=None)

@dataclass
class DataArguments:
    data_config_path: str = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    eval_strategy: str = field(default="epoch")   # remplace evaluation_strategy
    save_strategy: str = field(default="epoch")
    num_train_epochs: float = field(default=100)
    bf16: bool = field(default=True)               # active automatiquement bfloat16
    fp16: bool = field(default=False)          # fp16 désactivé par défaut
    gradient_checkpointing: bool = False
@dataclass
class EvalArguments:
    data_config_path: str = field(
        default=None, metadata={"help": "Path to YAML config with data parameters."}
    )
    weights: Optional[str] = field(
        default=None, metadata={"help": "Path to model checkpoint weights to load."}
    )
    generate_heatmaps: Optional[bool] = field(
        default=False, metadata={"help": "Whether to generate attention heatmaps."}
    )
    heatmap_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory where to save heatmaps."}
    )
