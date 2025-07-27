import os
import io
import re
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from difflib import SequenceMatcher, get_close_matches
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from helpers.hf_chat_template import default_chat_template, render_chat_template

def build_chatml_prompt(prompt_text: str, is_multilabel: bool, label_space: list[str]) -> str:
    options = ', '.join([lbl.upper() for lbl in label_space])
    suffix = (
        f"Options: {options}. List your selections separated by commas." if is_multilabel
        else f"Options: {options}. Answer with the letter."
    )
    full = f"{prompt_text}\n{suffix}"
    messages = [{"role": "user", "content": "<image>\n" + full}]
    chat_template = default_chat_template()
    rendered = render_chat_template(messages, chat_template, add_generation_prompt=True)
    return rendered[0]

def safe_extract_label(conversation):
    value = conversation[-1]["value"]
    if isinstance(value, np.ndarray):
        value = value.tolist()
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else " ".join(value)
    return value

class Evaluator:
    def __init__(self, model, processor, collator, label_space=None, device=None, gen_kwargs=None):
        self.model = model
        self.processor = processor
        self.collator = collator
        self.label_space = label_space or []
        self.device = device or model.model_pretrained.device
        self.gen_kwargs = gen_kwargs or {}
        self.model.eval()

    def safe_conversations(self, conv):
        if isinstance(conv, str):
            try:
                conv = json.loads(conv)
            except:
                return None
        if isinstance(conv, np.ndarray):
            conv = conv.tolist()
        if isinstance(conv, list) and len(conv) >= 2 and isinstance(conv[0], dict):
            return conv
        return None

    def compute_anls(self, y_true, y_pred):
        total, count = 0.0, 0
        for t, p in zip(y_true, y_pred):
            t_str = str(t).strip().lower()
            p_str = str(p).strip().lower()
            if not t_str or not p_str:
                continue
            sim = SequenceMatcher(None, t_str, p_str).ratio()
            total += sim if sim >= 0.5 else 0
            count += 1
        return round(total / count * 100, 1) if count else 0.0

    def extract_single_label(self, raw):
        if isinstance(raw, (list, np.ndarray)):
            raw = ''.join(raw)
        txt = str(raw).strip().upper()
        txt = re.sub(r'[^A-Z]', '', txt)
        if txt in self.label_space:
            return txt
        for ch in txt:
            if ch in self.label_space:
                return ch
        return txt

    def evaluate_singlelabel(self, df, dataset_name="unnamed", n_samples=None):
        y_true, y_pred, outputs = [], [], []
        for i, row in tqdm(df.iterrows(), total=n_samples or len(df), desc=f"Eval {dataset_name}"):
            if n_samples and i >= n_samples:
                break
            convs = self.safe_conversations(row.get('conversations'))
            if not convs:
                continue
            raw_gt = safe_extract_label(convs)
            gt_lbl = self.extract_single_label(raw_gt)

            img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
            prompt = convs[0].get('value', '')
            full_prompt = build_chatml_prompt(prompt, is_multilabel=False, label_space=self.label_space)

            inputs = self.processor(images=img, text=full_prompt, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    attention_mask=inputs.get("attention_mask"),
                    **self.gen_kwargs
                )
            raw_pred = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            pred_lbl = self.extract_single_label(raw_pred)

            y_true.append(gt_lbl)
            y_pred.append(pred_lbl)
            outputs.append({'id': row.get('id', i), 'gt': gt_lbl, 'pred': pred_lbl, 'raw_pred': raw_pred})

        acc = accuracy_score(y_true, y_pred) * 100
        anls = self.compute_anls(y_true, y_pred)
        os.makedirs('results', exist_ok=True)
        with open(f"results/{dataset_name.lower()}_results.jsonl", 'w', encoding='utf-8') as f:
            for o in outputs:
                f.write(json.dumps(o, ensure_ascii=False) + '\n')
        scores = {'Accuracy': round(acc, 1), 'ANLS': anls}
        return scores

    def evaluate_multilabel(self, df, dataset_name="unnamed", n_samples=None):
        y_true, y_pred, outputs = [], [], []
        for i, row in tqdm(df.iterrows(), total=n_samples or len(df), desc=f"Eval {dataset_name}"):
            if n_samples and i >= n_samples:
                break
            convs = self.safe_conversations(row.get('conversations'))
            if not convs:
                continue
            gt_raw = safe_extract_label(convs)
            if isinstance(gt_raw, (list, np.ndarray)):
                gt_raw = ''.join(gt_raw)
            true_lbls = [lbl for lbl in self.label_space if lbl.upper() in str(gt_raw).upper()]

            img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
            prompt = convs[0].get('value', '')
            full_prompt = build_chatml_prompt(prompt, is_multilabel=True, label_space=self.label_space)

            inputs = self.processor(images=img, text=full_prompt, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    attention_mask=inputs.get("attention_mask"),
                    **self.gen_kwargs
                )
            raw_pred = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            pred_lbls = [lbl for lbl in self.label_space if lbl.upper() in raw_pred.upper()]
            if self.label_space and not pred_lbls:
                matches = get_close_matches(raw_pred.lower(), [l.lower() for l in self.label_space], cutoff=0.8)
                pred_lbls = [lbl for lbl in self.label_space if lbl.lower() in matches]

            y_true.append(true_lbls)
            y_pred.append(pred_lbls)
            outputs.append({'id': row.get('id', i), 'gt': true_lbls, 'pred': pred_lbls, 'raw_pred': raw_pred})

        mlb = MultiLabelBinarizer(classes=self.label_space)
        yt, yp = mlb.fit_transform(y_true), mlb.transform(y_pred)
        scores = {
            'F1': round(f1_score(yt, yp, average='macro', zero_division=0) * 100, 1),
            'AUC': round(roc_auc_score(yt, yp, average='macro') * 100, 1) if len(self.label_space) > 1 else 0.0,
            'HL': round(hamming_loss(yt, yp) * 100, 1),
            'ANLS': self.compute_anls(y_true, y_pred)
        }
        os.makedirs('results', exist_ok=True)
        with open(f"results/{dataset_name.lower()}_results.jsonl", 'w', encoding='utf-8') as f:
            for o in outputs:
                f.write(json.dumps(o, ensure_ascii=False) + '\n')

        return scores

def run_all_tests(test_datasets, model, processor, collator, device='cuda', gen_kwargs=None, n_samples=None):
    results = []
    for info in test_datasets:
        name = info['name']
        df = pd.read_parquet(info['path'])
        evaluator = Evaluator(
            model, processor, collator,
            label_space=[l.upper() for l in info.get('label_space', [])],
            device=device, gen_kwargs=gen_kwargs
        )
        if 'Accuracy' in info['metrics']:
            scores = evaluator.evaluate_singlelabel(df, name, n_samples)
        else:
            scores = evaluator.evaluate_multilabel(df, name, n_samples)
        print(f"{name} → {scores}")

        row = {'Dataset': name}
        row.update(scores)
        results.append(row)
    return pd.DataFrame(results).set_index('Dataset')
