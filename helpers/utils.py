import yaml
import importlib
import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors.torch import load_file
import PIL
from PIL import Image

def load_yaml(file):
    with open(file,'rt') as f:
        out = yaml.safe_load(f)
    return out

def get_object_from_path(p):
    parts = p.split(".")
    path, name = ".".join(parts[:-1]), parts[-1]
    pkg = importlib.import_module(path)
    obj = getattr(pkg,name)
    return obj

def extract_labels_from_text(text, label_space):
    """
    Extrait les labels contenus dans un texte, à partir de la liste des labels disponibles.
    """
    text = text.lower()
    return [lbl for lbl in label_space if lbl.lower() in text]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def load_checkpoint(model, name, exception = None, replace = None):
    state_dict = load_file(name, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)
    print("missing keys: {}".format(missing))
    print("unexpected keys: {}".format(unexpected))
    return missing, unexpected


def pad_to_square(image: Image.Image) -> Image.Image:
    # Get the current dimensions of the image
    width, height = image.size

    # Determine the new size, which is the maximum of width and height
    new_size = max(width, height)

    # Calculate padding for each side
    padding_left = (new_size - width) // 2
    padding_top = (new_size - height) // 2
    padding_right = new_size - width - padding_left
    padding_bottom = new_size - height - padding_top

    # Add padding and return the padded image
    padded_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))  # white padding
    padded_image.paste(image, (padding_left, padding_top))

    return padded_image