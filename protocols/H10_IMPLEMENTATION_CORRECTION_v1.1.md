# H10 IMPLEMENTATION CORRECTION v1.1

Status: corrective implementation patch after demonstrated protocol mismatch.

The scientific protocol in `H10_EXTERNAL_TRANSFER_FREEZE_v1.0.md` is unchanged.

Only the frozen-marker matcher is corrected so that all exact marker forms are removed
even when immediately adjacent to a preceding word/hashtag. The correction was triggered
by consolidator inspection of the cleaned H10 v1.0 external inputs, not by model
performance.

H10 v1.0 numerical results are superseded.

Required rerun:
- external preparation;
- exact/high-similarity overlap audit;
- frozen S2 inference;
- frozen BERTweet inference for seeds 42/2026/7;
- frozen bootstrap/statistics.

No training, tuning, threshold change, or result-dependent preprocessing is authorized.
