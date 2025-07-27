from helpers.utils import load_yaml
from Datasets.LazySupervisedDataset import LazySupervisedDataset
import torch
from Datasets.DataCollatorForSupervisedDataset import DataCollatorForSupervisedDataset
from transformers import AutoProcessor

class DataModule:

    def __init__(self, data_args):
        self.data_args = data_args
        self._train_dataset = None
        self._eval_dataset = None
        self._test_dataset = None
        self._data_colator = None

        # Charge le config YAML
        self.data_config = load_yaml(data_args.data_config_path)

        # Initialise processor et max_length avant de les utiliser
        self.processor = AutoProcessor.from_pretrained(self.data_config['model_pretrained'], trust_remote_code=True)
        self.max_length = self.data_config.get('max_length', 128)

        # Maintenant tu peux créer les collators
        self.collator_train = DataCollatorForSupervisedDataset(
            processor=self.processor, max_length=self.max_length, is_train=True)
        self.collator_eval = DataCollatorForSupervisedDataset(
            processor=self.processor, max_length=self.max_length, is_train=False)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            datasets = []
            for i in range(len(self.data_config.get("train_datasets"))):
                datasets.append(LazySupervisedDataset(data_config = self.data_config, split = "training" ,index = i))
            print(f"Datasets used for train, {len(datasets)}, {datasets}")
            self._train_dataset = torch.utils.data.ConcatDataset(datasets)
            print(f"Datasets used for train, containing a total of {len(self._train_dataset)} samples")
        return self._train_dataset

    @property
    def eval_dataset(self):
        if self._eval_dataset is None:
            datasets = []
            for i in range(len(self.data_config.get("val_datasets"))):
                datasets.append(LazySupervisedDataset(data_config=self.data_config, split="val", index=i))
            print(f"Datasets used for val, {len(datasets)}, {datasets}")
            self._eval_dataset = torch.utils.data.ConcatDataset(datasets)
            print(f"Datasets used for val, containing a total of {len(self._eval_dataset)} samples")
        return self._eval_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            datasets = []
            for i in range(len(self.data_config.get("test_datasets"))):
                datasets.append(LazySupervisedDataset(data_config=self.data_config, split="test", index=i))
            print(f"Datasets used for test, {len(datasets)}, {datasets}")
            self._test_dataset = torch.utils.data.ConcatDataset(datasets)
            print(f"Datasets used for test, containing a total of {len(self._test_dataset)} samples")
        return self._test_dataset


    def get_data_collator(self):
        return DataCollatorForSupervisedDataset(self.processor, max_length = self.max_length)

    @property
    def data_collator(self):
        return self.get_data_collator()

    def to_dict(self, do_train=True):
        ret = dict(
            data_collator=self.collator_train,
            eval_dataset=self.eval_dataset,
            train_dataset=self.train_dataset
        )
        if hasattr(self, "sampler"):
            ret["sampler"] = self.sampler
        return ret





