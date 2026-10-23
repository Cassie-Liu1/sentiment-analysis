"""Evaluate the portfolio's TF-IDF/logistic regression baseline on labelled Amazon CSVs."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from preprocessing import clean_review
from sklearn.model_selection import train_test_split


def sample_csv(path, per_class, seed):
    pools = {1: [], 2: []}
    # Shuffle candidates within each chunk; retain a bounded, balanced sample.
    for chunk in pd.read_csv(path, header=None, names=['label', 'title', 'review'], chunksize=50000):
        if not chunk.label.isin([1, 2]).all():
            raise ValueError('Expected Amazon polarity labels 1 and 2')
        for label in pools:
            remaining = per_class - sum(len(part) for part in pools[label])
            if remaining > 0:
                rows = chunk[chunk.label == label]
                pools[label].append(rows.sample(n=min(remaining, len(rows)), random_state=seed))
        if all(sum(len(part) for part in pools[label]) >= per_class for label in pools):
            break
    frame = pd.concat([part for parts in pools.values() for part in parts], ignore_index=True)
    frame['review'] = (frame.title.fillna('') + ' ' + frame.review.fillna('')).map(clean_review)
    frame['sentiment'] = frame.label.map({1: 'negative', 2: 'positive'})
    return frame.loc[frame.review.str.len() > 0, ['review', 'sentiment']].sample(frac=1, random_state=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir')
    parser.add_argument('--prepared-data-dir', help='Reuse prepared labelled CSVs without resampling')
    parser.add_argument('--validation-size', type=float, default=0.2)
    parser.add_argument('--result-file', default='results/amazon_baseline_metrics.json')
    parser.add_argument('--train-per-class', type=int, default=10000)
    parser.add_argument('--test-per-class', type=int, default=1000)
    args = parser.parse_args()
    if args.prepared_data_dir:
        from run_imdb import load_reviews
        source = Path(args.prepared_data_dir)
        train = load_reviews(str(source / 'amazon_train.csv')).rename(columns={'label': 'sentiment'})
        test = load_reviews(str(source / 'amazon_test.csv')).rename(columns={'label': 'sentiment'})
        for frame in [train, test]:
            frame['sentiment'] = frame.sentiment.map({0: 'negative', 1: 'positive'})
    elif args.source_dir:
        source = Path(args.source_dir)
        train = sample_csv(source / 'train.csv', args.train_per_class, 42)
        test = sample_csv(source / 'test.csv', args.test_per_class, 43)
    else:
        parser.error('Specify --source-dir or --prepared-data-dir')
    train = train.drop_duplicates('review')
    test = test.drop_duplicates('review')
    before = len(test)
    test = test[~test.review.isin(train.review)]
    if not args.prepared_data_dir:
        Path('data').mkdir(exist_ok=True)
        train.to_csv('data/amazon_train.csv', index=False)
        test.to_csv('data/amazon_test.csv', index=False)
    # Match the transformer's label encoding and seed exactly.
    labels = train.sentiment.map({'negative': 0, 'positive': 1})
    train, validation = train_test_split(train, test_size=args.validation_size, stratify=labels, random_state=42)
    model = make_pipeline(TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=500, random_state=42))
    print(f'Training baseline: {len(train)} reviews; evaluating: {len(test)} reviews', flush=True)
    model.fit(train.review, train.sentiment)
    predicted = model.predict(test.review)
    report = classification_report(test.sentiment, predicted, output_dict=True, zero_division=0)
    result = {'model': 'TF-IDF + LogisticRegression', 'seed': 42, 'sampling': 'balanced sample from first chunks of original train/test CSVs; not full-dataset evaluation', 'train_count': len(train), 'validation_count': len(validation), 'validation_size': args.validation_size, 'validation_report': classification_report(validation.sentiment, model.predict(validation.review), output_dict=True, zero_division=0), 'test_count': len(test), 'removed_test_overlap': before-len(test), 'report': report, 'confusion_matrix_labels': ['negative', 'positive'], 'confusion_matrix': confusion_matrix(test.sentiment, predicted, labels=['negative', 'positive']).tolist()}
    Path('results').mkdir(exist_ok=True)
    Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_file).write_text(json.dumps(result, indent=2) + '\n')
    print(classification_report(test.sentiment, predicted, digits=4, zero_division=0))


if __name__ == '__main__':
    main()
