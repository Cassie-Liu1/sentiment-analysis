# DistilBERT error analysis

English | [简体中文](error-analysis.zh-CN.md)

## Scope

Inference with the saved best checkpoint reproduced 93.15% accuracy on the fixed
2,000-review Amazon test sample. There were 137 errors: 95 positive reviews
predicted negative and 42 negative reviews predicted positive. Negative recall
was 95.80%; positive recall was 90.50%. This asymmetry is sample-specific and
does not by itself establish a general negative prediction bias.

No labels, parameters, or reported test metrics were changed during this analysis.

## Representative cases

Eight cases were manually selected from the errors; they do not quantify category
prevalence. Row IDs are zero-based positions after shared cleaning/deduplication.
Descriptions paraphrase the source text. Suggested causes are hypotheses, not
verified explanations of internal model behavior. Actual labels are dataset labels.

| Row | Review meaning | Actual → predicted | Score | Interpretation |
| --- | --- | --- | --- | --- |
| 1928 | An absorbing book series that the reader could not put down. | Positive → negative | 0.9876 | A positive idiom expressed through grammatical negation may be misunderstood. |
| 1717 | Disappointed by the choice of characters, but still calls it a good and absorbing book. | Positive → negative | 0.9916 | Mixed aspect-level sentiment makes a single polarity label difficult. |
| 1792 | Loves the album and song, while strongly disliking the associated movie. | Positive → negative | 0.9838 | The reviewed product and another evaluation target may be conflated. |
| 1039 | Criticizes another reviewer for revealing the ending, then gives the movie a mildly favorable, mixed assessment. | Positive → negative | 0.9960 | Attitude toward another reviewer differs from attitude toward the product. |
| 887 | Calls the work pathetic and questions what the authors were thinking. | Negative → positive | 0.9893 | A clearly negative example is missed with a high score. Short text, nonstandard spelling, and context-sensitive wording merit investigation; the mechanism is unconfirmed. |
| 756 | Strongly praises a singer and CD despite the negative dataset label. | Negative → positive | 0.9896 | Possible label ambiguity; original rating/record should be checked before any label correction. |
| 41 | Praises illustrations but repeatedly describes the guide as disappointing and suggests saving money. | Positive → negative | 0.9970 | Mixed aspects and possible rating-to-polarity ambiguity; the original record is needed to resolve it. |
| 1453 | Recommends a book while using critical language about viewpoints it disputes. | Positive → negative | 0.9597 | Multiple targets and quoted opinions; 264 tokens also exceed the 256-token limit. A controlled test is needed before attributing the error to truncation. |

## Findings and next experiments

71 of the 137 errors had maximum softmax scores of at least 0.9. This highlights
high-score errors, but is not an overall calibration measurement. Scores are
uncalibrated and must not be interpreted as probabilities of correctness.

Only four test reviews exceeded 256 tokens, and one was misclassified. Truncation
is therefore uncommon in this sample; these counts are too small to establish
long-text performance or causation.

Mixed sentiment, target separation, and negated positive idioms are useful
diagnostic themes, but all 137 errors have not been categorically annotated.
Potential label ambiguity must be investigated against original records rather
than removing inconvenient examples to improve the reported score.

Future changes should be developed on training/validation data. Since these test
errors have now been inspected, model changes informed by them should receive
final evaluation on a fresh, uninspected holdout.

## Reproduction

```bash
python src/analyze_errors.py --checkpoint weights-distilbert-amazon --test-file data/amazon_test.csv --device cpu
```

The script checks that accuracy matches the saved evaluation. Full review-level
diagnostics remain in Git-ignored `artifacts/test_predictions.json`. Counts,
model/test-file SHA-256 hashes, and score semantics are exported to
`results/error_analysis_summary.json`. Published examples use paraphrases rather
than redistributing the full test corpus. Local labelled data and a trained
checkpoint are required; no model download is needed.
