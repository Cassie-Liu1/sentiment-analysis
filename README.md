# Transformer Sentiment Analysis

English | [简体中文](README.zh-CN.md)

A reproducible NLP portfolio project for binary sentiment classification of movie and product reviews.

## Verified model comparison

Real pretrained DistilBERT fine-tuning is complete. Both models used the same
16,000 training reviews, 4,000 validation reviews, and 2,000 held-out Amazon test
reviews (stratified split, seed 42; exact training/validation review order checked).

| Model | Test accuracy | Macro F1 |
| --- | --- | --- |
| TF-IDF + Logistic Regression | 88.70% | 0.8870 |
| Fine-tuned DistilBERT | 93.15% | 0.9315 |

The accuracy difference is 4.45 percentage points. Negative recall is 95.80%;
positive recall is 90.50%. This is a single-seed experiment on a balanced sample
from the first source-file chunks, not a full-dataset or significance claim.
Inference speed relative to BERT has not been measured.

Three epochs yielded validation accuracies of 93.85%, 94.33%, and 94.13%, while
validation losses rose from 0.1772 to 0.2008 and 0.2314. The predefined lowest
validation-loss rule selected epoch 1 for final test evaluation. Training loss
continued to fall, indicating overfitting in later epochs.

![Training and validation curves](results/distilbert_learning_curve.png)

Artifacts: [comparison](results/model_comparison.json),
[training history](results/distilbert_training_history.json),
[test metrics](results/distilbert_test_metrics.json).

### Error analysis

Re-inference reproduced all 137 test errors: 95 positive-to-negative and 42
negative-to-positive. Eight [representative cases](docs/error-analysis.md) cover
mixed sentiment, target confusion, positive idioms, possible label ambiguity,
and a truncated example. 71 errors had uncalibrated softmax scores ≥0.9; only one
error was truncated. These are diagnostic observations, not established causes.
See [reproducible counts](results/error_analysis_summary.json). No labels or
model parameters were changed during analysis.

To reproduce using the existing labelled Amazon splits:

```bash
python src/run_amazon_baseline.py --prepared-data-dir data --result-file results/amazon_baseline_matched_metrics.json
python src/run_amazon.py --train --evaluate --data-dir data --output-dir weights-distilbert-amazon --epochs 3 --batch-size 16 --max-length 256
python src/summarize_experiment.py --checkpoint weights-distilbert-amazon
```

Fine-tuning used the Apple MPS device. Device and package-version differences may
produce slightly different results. Checkpoints remain local and ignored by Git.

## Project highlights

- Fine-tunes DistilBERT/BERT-style transformer models with PyTorch and Hugging Face Transformers.
- Supports IMDb and Amazon review workflows.
- Includes stratified train/test splitting that preserves each review's label, training, evaluation, and single-text prediction.
- Keeps large datasets and model checkpoints outside Git; download instructions are provided below.
- Defaults to DistilBERT; uses dynamic padding, stratified validation, AdamW with
  linear warmup/decay, gradient clipping, early stopping, and best-checkpoint selection.
- Records validation history, token-length statistics, and final test metrics.

## Repository structure

```text
.
├── src/
│   ├── run_imdb.py            # IMDb training/evaluation/prediction entry point
│   ├── run_amazon.py          # Amazon training/evaluation/prediction entry point
│   ├── read_amazon.py         # Amazon raw-data preparation
│   └── split_data.py          # IMDb data splitting
├── data/                      # Local-only datasets (ignored by Git)
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/split_data.py /path/to/IMDB_Dataset.csv --output-dir data --test-size 0.2 --seed 42
```

Then train and evaluate:

```bash
python src/run_imdb.py --train --data-dir data --output-dir weights
python src/run_imdb.py --evaluate --data-dir data --output-dir weights
```

For Amazon polarity data in fastText `.bz2` format, prepare a balanced sample and train/evaluate it with the shared classifier:

```bash
python src/read_amazon.py /path/to/train.ft.txt.bz2 --output-dir data --sample-size 10000
python src/run_amazon.py --train --data-dir data --output-dir weights-amazon --epochs 3
python src/run_amazon.py --evaluate --data-dir data --output-dir weights-amazon
```

## Verified Amazon baseline run

Run the TF-IDF + logistic regression baseline from the original labelled Amazon
polarity CSVs (columns: label, title, review; labels 1=negative, 2=positive):

```bash
python src/run_amazon_baseline.py --source-dir /path/to/amazon_csv_directory
```

The 2026-10-13 local run trained on 20,000 reviews and evaluated on 2,000 reviews
from the separate original test file after shared text normalization: accuracy
89.60%, macro F1 0.8960 (the earlier unnormalized baseline was 89.50%).
This historical run used all 20,000 training reviews; the current baseline script
holds out 20% for validation and its matched comparison above is the main result.
Sampling is balanced from the first source-file chunks, with fixed seeds; this
is a sample baseline result, not a full-dataset or Transformer result.
Duplicate texts and train/test text overlap are removed before evaluation.
Prepared labelled splits are saved in `data/` and metrics in
`results/amazon_baseline_metrics.json`.

The legacy `imdb_train.csv` and `imdb_test.csv` contain review text only and
cannot be used for supervised evaluation. Do not reconstruct labels from row
positions. Transformer evaluation requires a trained checkpoint; the base
model's randomly initialized classifier does not provide valid sentiment metrics.

## Data and licensing

Datasets are intentionally excluded from this repository because of size and redistribution constraints. Use the original IMDb/Amazon sources and follow their terms. Model outputs and checkpoints should also remain local unless explicitly cleared for redistribution.

## Portfolio notes

The original experiments reported approximately 92% IMDb accuracy, but their fixed sample-label pairing was not reliable. The current scripts keep labels with their reviews and use stratified splits; report only metrics generated by rerunning this workflow.
