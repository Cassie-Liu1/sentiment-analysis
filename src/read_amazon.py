"""Prepare labelled CSV splits from the Amazon polarity fastText archive."""

import argparse
import bz2
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to train.ft.txt.bz2")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 2 or not 0 < args.test_size < 1:
        parser.error("sample-size must be >= 2 and test-size must be in (0, 1)")

    rng = random.Random(args.seed)
    per_class = max(1, args.sample_size // 2)
    samples = {"__label__1": [], "__label__2": []}
    counts = {"__label__1": 0, "__label__2": 0}
    with bz2.open(args.input, "rt", encoding="utf-8") as source:
        for line in source:
            label, sep, review = line.rstrip("\n").partition(" ")
            if not sep or label not in {"__label__1", "__label__2"}:
                continue
            counts[label] += 1
            sample = samples[label]
            row = {"sentiment": "positive" if label == "__label__2" else "negative", "review": review}
            if len(sample) < per_class:
                sample.append(row)
            else:
                position = rng.randrange(counts[label])
                if position < per_class:
                    sample[position] = row
    rows = samples["__label__1"] + samples["__label__2"]
    frame = pd.DataFrame(rows)
    if frame.empty or frame["sentiment"].nunique() != 2:
        raise ValueError("The input did not yield both Amazon polarity labels")
    train, test = train_test_split(
        frame, test_size=args.test_size, random_state=args.seed, stratify=frame["sentiment"]
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "amazon_train.csv", index=False)
    test.to_csv(out / "amazon_test.csv", index=False)
    print(f"Wrote {len(train):,} train and {len(test):,} test reviews to {out}")


if __name__ == "__main__":
    main()
