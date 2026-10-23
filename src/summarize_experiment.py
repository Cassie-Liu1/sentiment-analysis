"""Publish metrics and a learning curve from completed local training artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='weights-distilbert-amazon')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    out = Path('results')
    history = json.loads((checkpoint / 'training_history.json').read_text())
    test = json.loads((checkpoint / 'test_metrics.json').read_text())
    baseline = json.loads((out / 'amazon_baseline_matched_metrics.json').read_text())
    if (history['train_count'], history['validation_count'], test['test_count']) != (baseline['train_count'], baseline['validation_count'], baseline['test_count']):
        raise ValueError('Comparison requires matching split sizes')
    (out / 'distilbert_training_history.json').write_text(json.dumps(history, indent=2) + '\n')
    (out / 'distilbert_test_metrics.json').write_text(json.dumps(test, indent=2) + '\n')
    summary = {'dataset': 'Amazon balanced sample', 'train_count': history['train_count'], 'validation_count': history['validation_count'], 'test_count': test['test_count'], 'seed': history['seed'], 'baseline_accuracy': baseline['report']['accuracy'], 'baseline_macro_f1': baseline['report']['macro avg']['f1-score'], 'distilbert_accuracy': test['accuracy'], 'distilbert_macro_f1': test['report']['macro avg']['f1-score'], 'accuracy_difference_percentage_points': 100 * (test['accuracy'] - baseline['report']['accuracy']), 'selection': history['selection'], 'speed_comparison': 'not measured', 'scope': 'single seed, sampled data; no statistical significance claim'}
    (out / 'model_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    rows = history['history']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r['epoch'] for r in rows]
    axes[0].plot(epochs, [r['train_loss'] for r in rows], 'o-', label='Train')
    axes[0].plot(epochs, [r['validation_loss'] for r in rows], 'o-', label='Validation')
    axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axes[0].legend()
    axes[1].plot(epochs, [r['validation_accuracy'] for r in rows], 'o-', label='Accuracy')
    axes[1].plot(epochs, [r['validation_macro_f1'] for r in rows], 'o-', label='Macro F1')
    axes[1].set(title='Validation quality', xlabel='Epoch', ylabel='Score')
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(epochs)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'distilbert_learning_curve.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
