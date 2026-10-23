"""Create reproducible, stratified IMDb train and test CSV files."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV with review and sentiment columns")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    frame = pd.read_csv(args.input)
    missing = {"review", "sentiment"}.difference(frame.columns)
    if missing:
        parser.error(f"input is missing required columns: {sorted(missing)}")
    frame = frame[["review", "sentiment"]].dropna().copy()
    frame["sentiment"] = frame["sentiment"].astype(str).str.strip().str.lower()
    if not frame["sentiment"].isin(["positive", "negative"]).all():
        parser.error("sentiment values must be 'positive' or 'negative'")
    train, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=frame["sentiment"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "imdb_train.csv", index=False)
    test.to_csv(output_dir / "imdb_test.csv", index=False)
    print(f"Wrote {len(train):,} train and {len(test):,} test reviews to {output_dir}")
    print("Train label proportions:")
    print(train["sentiment"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
