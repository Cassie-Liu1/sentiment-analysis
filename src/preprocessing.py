"""Shared conservative text normalization for English review classifiers."""
import html
import re
import unicodedata


def clean_review(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unicodedata.normalize('NFKC', text)
    # Preserve punctuation, negation, emoji, and non-English text as sentiment cues.
    text = ''.join(c for c in text if not unicodedata.category(c).startswith('C') or c.isspace())
    return re.sub(r'\s+', ' ', text).strip().lower()
