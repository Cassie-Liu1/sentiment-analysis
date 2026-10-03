# Transformer Sentiment Analysis

A reproducible NLP portfolio project for binary sentiment classification of movie and product reviews.

## Project highlights

- Fine-tunes DistilBERT/BERT-style transformer models with PyTorch and Hugging Face Transformers.
- Supports IMDb and Amazon review workflows.
- Includes deterministic train/test splitting, training, evaluation, and single-text prediction.
- Keeps large datasets and model checkpoints outside Git; download instructions are provided below.

## Repository structure

```text
.
├── notebooks/                 # Exploratory and DistilBERT training notebooks
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
python src/split_data.py /path/to/IMDB_Dataset.csv --output-dir data
```

Then train and evaluate:

```bash
python src/run_imdb.py --train --data-dir data --output-dir weights
python src/run_imdb.py --evaluate --data-dir data --output-dir weights
```

For Amazon reviews, prepare the source dataset and use the corresponding `run_amazon.py` workflow.

## Data and licensing

Datasets are intentionally excluded from this repository because of size and redistribution constraints. Use the original IMDb/Amazon sources and follow their terms. Model outputs and checkpoints should also remain local unless explicitly cleared for redistribution.

## Portfolio notes

The original experiments report approximately 92% IMDb accuracy. Re-run evaluation after changing the model, split, or preprocessing before quoting a metric in a résumé or interview.
