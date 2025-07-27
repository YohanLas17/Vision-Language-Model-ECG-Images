import nltk as nltk
import torch
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
from anls import anls_score
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    hamming_loss
)

from transformers import AutoTokenizer
import anls


class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, eval_pred):
        pred, target = eval_pred

        target_strings = self.tokenizer.batch_decode(target, True)
        target_strings = [element.strip() for element in target_strings]

        pred_strings = self.tokenizer.batch_decode(pred, True)
        pred_strings = [element.strip() for element in pred_strings]

        unique_labels = sorted(list(set(target_strings + pred_strings)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        y_true = [label_to_id[t] for t in target_strings]
        y_pred = [label_to_id[p] for p in pred_strings]

        unique_labels = sorted(list(set(target_strings + pred_strings)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)


        exact_match = np.mean([p == t for p, t in zip(pred_strings, target_strings)])
        score_anls = [anls_score(pred, [target]) for pred, target in zip(pred_strings, target_strings)]
        anls_score_mean = np.mean(score_anls)

        f1_macro = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        acc = accuracy_score(y_true_np, y_pred_np)
        hamming = hamming_loss(y_true_np, y_pred_np)
        y_pred_onehot = np.eye(len(unique_labels))[y_pred_np]

        try:
            y_true_onehot = np.eye(len(unique_labels))[y_true_np]
            auc_macro = roc_auc_score(
                y_true_onehot,
                y_pred_onehot,
                average="macro",
                multi_class="ovo"
            )
        except ValueError:
            auc_macro = float('nan')

        return {
            "ExactMatch": exact_match,
            "ANLS": anls_score_mean,
            "F1_macro": f1_macro,
            "Accuracy": acc,
            "HammingLoss": hamming,
            "AUC_macro": auc_macro,
        }


if __name__== "__main__":
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = [
        'chat', 'chien', 'oiseau', 'chat',
        'chien', 'poisson', 'poisson', 'chat',
        'oiseau', 'chat'
    ]

    pred = [
        'chat', 'chat', 'oiseau', 'chien',
        'chien', 'poisson', 'chien', 'chat',
        'oiseau', 'oiseau'
    ]


    target_token = tokenizer(gt, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    pred_token = tokenizer(pred, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    eval_pred = (pred_token, target_token)##### fill me
    metric = Metrics(tokenizer=tokenizer)
    score = metric(eval_pred)
    print(score)

