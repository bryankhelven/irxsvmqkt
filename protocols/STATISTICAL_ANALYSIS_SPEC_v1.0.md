# STATISTICAL ANALYSIS SPEC v1.0

Primary metric: Macro-F1.

Secondary:
- accuracy
- weighted-F1
- per-class precision/recall/F1
- confusion matrix
- irony->sarcasm and sarcasm->irony error counts

## Bootstrap
- stratified resampling over the frozen test rows
- 10,000 resamples
- RNG seed 20260902
- percentile 95% CI
- paired delta only when the prediction files have identical `_id` sequences

## Neural seeds
Frozen: 42, 2026, 7.

Report:
- every seed separately
- mean ± sample SD

Never select the best final-test seed.

A seed-42 paired bootstrap may be used for fixed-model comparison, but must not replace
the multi-seed summary.

## Multiple comparisons
The paper uses effect sizes and confidence intervals, not p-value/significance-star
claims. It therefore does not claim a classical multiple-testing correction framework.

## Interpretation
Classification performance is not evidence of linguistic “understanding”.
