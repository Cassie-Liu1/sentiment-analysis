"""Reproduce held-out predictions and keep review-level diagnostics local."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from run_imdb import load_reviews, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='weights-distilbert-amazon')
    parser.add_argument('--test-file', default='data/amazon_test.csv')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = Path(args.checkpoint)
    history = json.loads((checkpoint / 'training_history.json').read_text())
    max_length = history['arguments']['max_length']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(args.device).eval()
    frame = load_reviews(args.test_file)
    loader, _ = make_loader(frame, tokenizer, 16, False, max_length)
    probabilities = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop('labels')
            probabilities.extend(model(**{k: v.to(args.device) for k, v in batch.items()}).logits.softmax(-1).cpu().tolist())
    rows = []
    for index, ((_, row), probs) in enumerate(zip(frame.iterrows(), probabilities)):
        predicted = int(probs[1] > probs[0])
        token_count = len(tokenizer(row.review, truncation=False)['input_ids'])
        rows.append({'test_row': index, 'review_sha256': hashlib.sha256(row.review.encode()).hexdigest(), 'review': row.review, 'actual': 'positive' if row.label else 'negative', 'predicted': 'positive' if predicted else 'negative', 'score': max(probs), 'token_count': token_count, 'truncated': token_count > max_length, 'correct': predicted == int(row.label)})
    errors = [r for r in rows if not r['correct']]
    summary = {'test_count': len(rows), 'error_count': len(errors), 'accuracy': 1-len(errors)/len(rows), 'positive_predicted_negative': sum(r['actual']=='positive' for r in errors), 'negative_predicted_positive': sum(r['actual']=='negative' for r in errors), 'errors_with_score_at_least_0.9': sum(r['score'] >= .9 for r in errors), 'truncated_test_count': sum(r['truncated'] for r in rows), 'truncated_error_count': sum(r['truncated'] for r in errors), 'max_length': max_length, 'test_file_sha256': hashlib.sha256(Path(args.test_file).read_bytes()).hexdigest(), 'model_sha256': hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(), 'score_note': 'Uncalibrated maximum softmax score, not probability of correctness', 'row_note': 'Zero-based index after shared cleaning and deduplication'}
    expected = json.loads((checkpoint/'test_metrics.json').read_text())
    if abs(summary['accuracy'] - expected['accuracy']) > 1e-9:
        raise ValueError('Predictions do not reproduce saved accuracy')
    Path('artifacts').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('artifacts/test_predictions.json').write_text(json.dumps(rows, indent=2) + '\n')
    Path('results/error_analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
