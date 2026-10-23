"""Offline checks use a tiny random model; they do not measure sentiment quality."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from preprocessing import clean_review
from run_imdb import load_reviews


class WorkflowTests(unittest.TestCase):
    def test_clean_preserves_sentiment_cues(self):
        self.assertEqual(clean_review('<br/>NOT bad &amp; nice 😊\n'), 'not bad & nice 😊')

    def test_train_evaluate_predict_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            vocab = base / 'vocab.txt'
            vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nmovie\nnot\n')
            tokenizer = BertTokenizer(vocab=str(vocab), do_lower_case=True)
            model_dir = base / 'tiny'
            tokenizer.save_pretrained(model_dir)
            model = DistilBertForSequenceClassification(DistilBertConfig(vocab_size=tokenizer.vocab_size, n_layers=1, dim=16, hidden_dim=32, n_heads=2, max_position_embeddings=64, num_labels=2))
            model.save_pretrained(model_dir)
            pd.DataFrame({'review': [f'{word} movie train {i}' for word in ['good', 'bad'] for i in range(6)], 'sentiment': [label for label in ['positive', 'negative'] for _ in range(6)]}).to_csv(base / 'imdb_train.csv', index=False)
            pd.DataFrame({'review': ['good held out review', 'bad held out review'], 'sentiment': ['positive', 'negative']}).to_csv(base / 'imdb_test.csv', index=False)
            output = base / 'output'
            cmd = [sys.executable, str(ROOT / 'src/run_imdb.py'), '--train', '--evaluate', '--predict', 'not bad', '--model-name', str(model_dir), '--data-dir', str(base), '--output-dir', str(output), '--epochs', '2', '--batch-size', '2', '--max-length', '32', '--device', 'cpu']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            history = json.loads((output / 'training_history.json').read_text())
            self.assertEqual(len(history['history']), 2)
            self.assertEqual(history['train_count'] + history['validation_count'], 12)
            metrics = json.loads((output / 'test_metrics.json').read_text())
            self.assertEqual(metrics['test_count'], 2)
            self.assertTrue((output / 'config.json').exists())
            self.assertIn('confidence=', result.stdout)

    def test_missing_and_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'reviews.csv'
            pd.DataFrame({'review': ['hello']}).to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, 'sentiment'):
                load_reviews(str(p))
            pd.DataFrame({'review': ['GOOD', '<br/>good'], 'sentiment': ['positive', 'negative']}).to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                load_reviews(str(p))


if __name__ == '__main__':
    unittest.main()
