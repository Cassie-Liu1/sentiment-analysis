"""Train and evaluate a binary IMDb sentiment classifier."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from preprocessing import clean_review
from tqdm.auto import tqdm


SEED = 42
MODEL_NAME = "distilbert-base-uncased"


class ReviewDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 256):
        self.reviews = frame["review"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.reviews[index],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def load_reviews(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"review", "sentiment"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame[["review", "sentiment"]].dropna().copy()
    labels = frame["sentiment"].astype(str).str.strip().str.lower().map(
        {"negative": 0, "positive": 1}
    )
    if labels.isna().any():
        raise ValueError("sentiment must contain only 'negative' and 'positive'")
    frame["label"] = labels.astype("int64")
    frame["review"] = frame["review"].map(clean_review)
    frame = frame[frame.review.str.len() > 0]
    if frame.groupby('review').label.nunique().gt(1).any():
        raise ValueError('Conflicting labels for identical cleaned reviews')
    frame = frame.drop_duplicates('review').reset_index(drop=True)
    if frame.empty:
        raise ValueError('No nonempty labelled reviews remain')
    return frame[["review", "label"]]


def make_loader(frame, tokenizer, batch_size: int, shuffle: bool, max_length=256):
    return DataLoader(
        ReviewDataset(frame, tokenizer, max_length), batch_size=batch_size, shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),
    ), frame["label"].to_numpy()


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in tqdm(loader, desc='Evaluating'):
            labels = batch["labels"]
            inputs = {key: value.to(device) for key, value in batch.items()}
            output = model(**inputs)
            logits = output.logits
            total_loss += output.loss.item() * len(labels)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            actuals.extend(labels.tolist())
    report = classification_report(actuals, predictions, labels=[0, 1], target_names=["negative", "positive"], zero_division=0, output_dict=True)
    return {'loss': total_loss / len(actuals), 'accuracy': float(np.mean(np.array(actuals) == predictions)), 'report': report,
            'confusion_matrix': confusion_matrix(actuals, predictions, labels=[0, 1]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--predict", help="Optional text to classify")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="weights")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-file", default="imdb_train.csv")
    parser.add_argument("--eval-file", default="imdb_test.csv")
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--validation-size', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()
    if not (args.train or args.evaluate or args.predict):
        parser.error("choose --train, --evaluate, or --predict")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not 0 < args.validation_size < 1 or not 0 <= args.warmup_ratio < 1 or args.patience < 1 or args.max_length < 2 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error('Invalid validation size, warmup, patience, length, or optimizer settings')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device_name)
    output_dir = Path(args.output_dir)
    if args.train:
        train_frame = load_reviews(str(Path(args.data_dir) / args.train_file))
        train_frame, val_frame = train_test_split(train_frame, test_size=args.validation_size, stratify=train_frame.label, random_state=SEED)
    if args.evaluate:
        test_frame = load_reviews(str(Path(args.data_dir) / args.eval_file))
        if args.train and test_frame.review.isin(pd.concat([train_frame, val_frame]).review).any():
            parser.error('Train/validation and test sets share cleaned review text')
    if not args.train and not (output_dir / "config.json").exists():
        parser.error("evaluation/prediction requires a trained checkpoint in --output-dir; run --train first")
    model_path = args.model_name if args.train else output_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)

    if args.train:
        output_dir.mkdir(parents=True, exist_ok=True)
        loader, _ = make_loader(train_frame, tokenizer, args.batch_size, True, args.max_length)
        val_loader, _ = make_loader(val_frame, tokenizer, args.batch_size, False, args.max_length)
        lengths = [len(tokenizer(text, truncation=False)['input_ids']) for text in train_frame.review.iloc[:1000]]
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        total_steps = len(loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
        history, best_loss, stale = [], float('inf'), 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch in tqdm(loader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * len(batch['labels'])
            validation = evaluate(model, val_loader, device)
            row = {'epoch': epoch + 1, 'train_loss': total_loss / len(train_frame), 'validation_loss': validation['loss'], 'validation_accuracy': validation['accuracy'], 'validation_macro_f1': validation['report']['macro avg']['f1-score']}
            history.append(row)
            print(json.dumps(row), flush=True)
            if validation['loss'] < best_loss:
                best_loss, stale = validation['loss'], 0
                model.config.id2label = {0: 'negative', 1: 'positive'}
                model.config.label2id = {'negative': 0, 'positive': 1}
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                stale += 1
            manifest = {'arguments': vars(args), 'seed': SEED, 'device': str(device), 'train_count': len(train_frame), 'validation_count': len(val_frame), 'selection': 'lowest validation loss', 'token_length_sample_count': len(lengths), 'token_length_percentiles': dict(zip(['p50', 'p90', 'p95'], np.percentile(lengths, [50, 90, 95]).tolist())), 'sample_truncation_fraction': float(np.mean(np.array(lengths) > args.max_length)), 'history': history, 'torch_version': torch.__version__}
            (output_dir / 'training_history.json').write_text(json.dumps(manifest, indent=2) + '\n')
            if stale >= args.patience:
                print('Early stopping: validation loss stopped improving', flush=True)
                break
        model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)

    if args.evaluate:
        loader, _ = make_loader(test_frame, tokenizer, args.batch_size, False, args.max_length)
        metrics = evaluate(model, loader, device)
        metrics['test_count'] = len(test_frame)
        (output_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2))

    if args.predict:
        encoded = tokenizer(clean_review(args.predict), truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model.eval()
        with torch.inference_mode():
            probs = model(**encoded).logits.softmax(dim=-1)[0]
        label = "positive" if probs.argmax().item() == 1 else "negative"
        print(f"{label} (confidence={probs.max().item():.3f})")


if __name__ == "__main__":
    main()
