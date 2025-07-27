# Datasets/LazySupervisedDataset.py
import os, json
from torch.utils.data import Dataset
from PIL import Image
from helpers.utils import pad_to_square

class LazySupervisedDataset(Dataset):
    """Dataset pour le fine-tuning supervisé."""

    def __init__(self, data_config, split, index):
        super().__init__()
        self.data_config   = data_config
        self.root_dir      = data_config.get("root_dir")
        self.split         = split
        self.val_limit     = data_config.get("val_dataset_limit")
        self.image_res     = int(data_config["resolution"])

        if split == "training":
            self.datasets_list = data_config["train_datasets"]
        elif split == "val":
            self.datasets_list = data_config["val_datasets"]
        else:
            self.datasets_list = data_config["test_datasets"]

        self.current_dataset   = self.datasets_list[index]
        json_path = f"{self.root_dir}/{self.current_dataset}/{data_config.get('json_filename','data.json')}"

        with open(json_path, "r") as f:
            self.json_dataset = json.load(f)

        self.update_json_dataset()

    # ----------- nouvelle version, sans vérif d’image coûteuse -------------
    def update_json_dataset(self):
        """
        Construit self.json_dataset en ne gardant que les exemples valides :
        - image présente ;
        - première paire de messages (user + assistant) complète,
          chaque message contenant une clé 'value' non vide.
        """
        valid = []
        for item in self.json_dataset:
            img_path = f"{self.root_dir}/{self.current_dataset}/{item['image']}"
            if not os.path.exists(img_path):
                continue

            conv = item.get("conversations", [])
            if conv and conv[0].get("from") == "system":
                conv = conv[1:]
            if len(conv) < 2:
                continue

            first, second = conv[0], conv[1]
            if (
                    "value" not in first or not first["value"].strip() or
                    "value" not in second or not second["value"].strip()
            ):
                continue

            valid.append(item)

        self.json_dataset = valid
        if self.split == "val" and self.val_limit:
            self.json_dataset = self.json_dataset[: self.val_limit]

        print(f"Dataset updated. Remaining {len(self.json_dataset)} valid entries.")

    def __len__(self):
        return len(self.json_dataset)

    def __getitem__(self, idx):
        item        = self.json_dataset[idx]
        image_path  = f"{self.root_dir}/{self.current_dataset}/{item['image']}"
        image       = Image.open(image_path).convert("RGB")
        image       = pad_to_square(image).resize((self.image_res, self.image_res))

        conv = item["conversations"]
        if conv and conv[0].get("from") == "system":
            conv = conv[1:]

        sample = dict(
            images     = image,
            image_name = image_path,
            question   = conv[0]["value"],
            answer     = conv[1]["value"],
            id         = item["id"],
            metadata   = item["metadata"],
            source     = item["source"],
            is_train   = self.split == "training",
        )
        return sample



        ### Maybe add data augmentation, look at this but not usual if performance is good already

        # # Apply random jittering
        # if random.random() > 0.2:
        #     jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        #     image = jitter(image)
        #
        # # Apply the same random rotation to both image and labels
        # if random.random() > 0.2:
        #     angle = random.randint(0, 360)
        #     rotation = transforms.RandomRotation([angle, angle], interpolation=transforms.InterpolationMode.BILINEAR)
        #     image = rotation(image)
        #     rotation2 = transforms.RandomRotation([angle, angle], interpolation=transforms.InterpolationMode.NEAREST)
        #     targets = [rotation2(target) for target in targets]
        #
        # # Apply the same random horizontal and vertical flip
        # if random.random() > 0.5:
        #     image = transforms.functional.hflip(image)
        #     targets = [transforms.functional.hflip(target) for target in targets]
        # if random.random() > 0.5:
        #     image = transforms.functional.vflip(image)
        #     targets = [transforms.functional.vflip(target) for target in targets]
