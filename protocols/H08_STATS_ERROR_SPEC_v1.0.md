# H08 Statistical + Error-Queue Specification v1.0

Status: frozen after H07 completion and before H08 derived analyses.

H08 introduces **no new predictive model**. It derives uncertainty, effect sizes,
comparison tables, and deterministic qualitative-review queues from already frozen
predictions.

## A. Statistical analysis

Primary metric remains Macro-F1.

### Individual test-set uncertainty
For each selected frozen prediction file:
- class-stratified bootstrap;
- 10,000 resamples;
- RNG seed 20260902;
- percentile 95% CI.

Because Macro-F1 is a function only of the confusion matrix, H08 may implement the
stratified bootstrap through class-conditional multinomial resampling of prediction
counts. This is distributionally equivalent to row-level within-class resampling for
this metric and avoids materializing 10,000 full 5,947-row samples.

### Paired deltas
Paired bootstrap is allowed only after asserting identical ordered `_id` and `y_true`.
The paired implementation resamples the joint `(prediction_A, prediction_B)` outcome
within each gold class, preserving prediction dependence exactly.

Required paired deltas:
1. historical matched leaky SVC minus clean SVC;
2. BERTweet A-C seed 42 minus S2 A-C;
3. S2 G-C minus S2 A-C;
4. BERTweet G-C minus A-C separately for seeds 42, 2026, 7;
5. same A-L-trained BERTweet seed-42 model: A-L eval minus A-C eval.

No p-values or significance stars are produced.

### Neural variability
Report each frozen seed and mean ± sample SD for:
- BERTweet A-C;
- BERTweet G-C.

Also report per-class F1 mean ± sample SD across seeds.

## B. Error accounting

Derive deterministic counts for:
- irony -> sarcasm;
- sarcasm -> irony;
- irony/sarcasm -> regular;
- regular -> irony/sarcasm;
- S2 vs BERTweet seed-42 correctness overlap.

## C. Qualitative error pilot

The local canonical A-C test text is joined by `_id` only after prediction integrity
checks pass.

Generate a deterministic pilot sample using seed 20260902 from both:
- S2 A-C;
- BERTweet A-C seed 42.

Target strata per model:
- irony -> sarcasm: 8;
- sarcasm -> irony: 8;
- irony/sarcasm -> regular: up to 8;
- regular -> irony/sarcasm: all if <=8, otherwise 8.

Deduplicate identical `_id` across strata/models after selection while retaining the
selection key separately.

Produce:
- `ERROR_PILOT_BLIND.tsv`: `case_id`, `gold_label`, `text_clean` only;
- `ERROR_PILOT_KEY.tsv`: model, prediction, source stratum, `_id` and case mapping;
- `ERROR_PILOT_ANNOTATION_TEMPLATE.tsv`: blank fields for qualitative categories.

The blind file intentionally hides model identity and prediction. H08 does **not**
freeze the qualitative taxonomy. The consolidator reviews the pilot first; only then
may a full 25-per-main-direction annotation guideline be frozen and optionally routed
to Claude Haiku.

## D. Exit criterion for H08A

- all bootstrap outputs reproducible;
- all required paired comparisons traceable to frozen prediction IDs;
- neural seed summaries produced;
- error accounting produced;
- blinded pilot queue produced;
- no raw/model training rerun.
