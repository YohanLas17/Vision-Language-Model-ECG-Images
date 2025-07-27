import torch
from tqdm import tqdm
from PIL import Image
import io, os, json, re
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from difflib import SequenceMatcher, get_close_matches

class Evaluator:
    def __init__(self, model, processor, collator, label_space, device=None, gen_kwargs=None):
        self.model = model
        self.processor = processor
        self.collator = collator
        self.label_space = [l.upper() for l in label_space]
        self.device = device or model.model_pretrained.device
        self.gen_kwargs = gen_kwargs or {}
        self.model.eval()

    def safe_conversations(self, conv):
        if isinstance(conv, np.ndarray):
            conv = conv.tolist()
        if isinstance(conv, str):
            try:
                conv = json.loads(conv)
            except:
                return None
        return conv if isinstance(conv, list) and len(conv) >= 2 else None

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

    def evaluate_multilabel(self, df, dataset_name="CODE15", n_samples=None, save_preds=False, save_path=None):
        y_true, y_pred, outputs = [], [], []

        for i, row in tqdm(df.iterrows(), total=n_samples or len(df), desc=f"Eval {dataset_name}"):
            if n_samples and i >= n_samples:
                break

            convs = self.safe_conversations(row.get('conversations'))
            if not convs:
                continue

            gt_raw = convs[1].get('value', '')
            if isinstance(gt_raw, (list, np.ndarray)):
                gt_raw = ''.join(gt_raw.tolist() if hasattr(gt_raw, 'tolist') else gt_raw)
            true_lbls = [lbl for lbl in self.label_space if lbl.upper() in str(gt_raw).upper()]

            img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")

            # ✅ Prompt médical structuré
            full_prompt = (
                "You are a medical assistant. Based on the ECG image, identify the correct diagnoses.\n"
                "Select from the following options: NORM, ABNORMAL, 1DAVB, RBBB, LBBB, SB, ST, AF.\n"
                "Respond only with the selected labels, separated by commas."
            )

            message = self.processor.apply_chat_template(
                self.collator.build_message(f"<image>\n{full_prompt}"),
                add_generation_prompt=True
            ).strip()

            inputs = self.processor(images=img, text=message, return_tensors='pt').to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    attention_mask=inputs.get("attention_mask"),
                    **self.gen_kwargs
                )

            raw_pred = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

            # ✅ Nettoyage fin de la prédiction
            raw_pred_clean = re.sub(r'[^a-zA-Z0-9,]', '', raw_pred.upper())
            pred_lbls = [lbl for lbl in self.label_space if lbl in raw_pred_clean.split(',')]

            # 🔁 Fallback si aucune correspondance stricte
            if not pred_lbls:
                matches = get_close_matches(raw_pred_clean, self.label_space, n=3, cutoff=0.6)
                pred_lbls = [lbl for lbl in self.label_space if lbl in matches]

            # ⚠️ Cas de réponse vide
            if raw_pred.strip().lower() in ["", "unknown", "none"]:
                pred_lbls = []
                print(f"⚠️ Empty or unclear prediction at sample {i}: '{raw_pred}'")

            y_true.append(true_lbls)
            y_pred.append(pred_lbls)
            outputs.append({'id': row.get('id', i), 'gt': true_lbls, 'pred': pred_lbls, 'raw_pred': raw_pred})

        # ✅ Evaluation
        mlb = MultiLabelBinarizer(classes=self.label_space)
        yt, yp = mlb.fit_transform(y_true), mlb.transform(y_pred)

        scores = {
            'F1': round(f1_score(yt, yp, average='macro', zero_division=0) * 100, 1),
            'AUC': round(roc_auc_score(yt, yp, average='macro') * 100, 1),
            'HL': round(hamming_loss(yt, yp) * 100, 1),
            'ANLS': self.compute_anls(y_true, y_pred)
        }

        os.makedirs('results', exist_ok=True)
        path = f"results/{dataset_name.lower()}_results.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for o in outputs:
                f.write(json.dumps(o, ensure_ascii=False) + '\n')

        if save_preds and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                for o in outputs:
                    f.write(json.dumps(o, ensure_ascii=False) + '\n')
            print(f"📦 Predictions saved to {save_path}")

        return scores

def run_all_tests(test_datasets, model, processor, collator, device='cuda', gen_kwargs=None, n_samples=None, save_preds=False, save_path=None):
    results = []
    for info in test_datasets:
        name = info['name']
        df = pd.read_parquet(info['path'])
        evaluator = Evaluator(
            model, processor, collator,
            label_space=info['label_space'],
            device=device,
            gen_kwargs=gen_kwargs
        )
        scores = evaluator.evaluate_multilabel(
            df,
            dataset_name=name,
            n_samples=n_samples,
            save_preds=save_preds,
            save_path=save_path
        )
        print(f"{name} → {scores}")
        row = {'Dataset': name}
        row.update(scores)
        results.append(row)

    return pd.DataFrame(results).set_index('Dataset')
