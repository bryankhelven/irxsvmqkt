# H10 EXTERNAL TRANSFER FREEZE v1.0

Status: **FROZEN BEFORE EXTERNAL TEST RESULTS**
Date: 2026-09-03

## Goal

Test whether the already-frozen A-C classifiers retain useful predictive ability outside
their development benchmark.

No hyperparameter tuning, fine-tuning, threshold tuning, model selection, or label
remapping may use the external gold labels.

## External benchmark

TweetEval `irony` test split, pinned to repository commit:

`cardiffnlp/tweeteval@4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`

Expected files:
- `datasets/irony/test_text.txt`
- `datasets/irony/test_labels.txt`
- `datasets/irony/mapping.txt`

Expected size: 784 test tweets.

TweetEval label mapping:
- 0 = non_irony
- 1 = irony

This benchmark is derived from SemEval-2018 Task 3.

## Important marker-control decision

The pinned TweetEval test snapshot visibly contains irony-related hashtags in some test
items. Therefore the primary external transfer condition removes, case-insensitively:

- `#irony`
- `#ironic`
- `#sarcasm`
- `#sarcastic`
- `#not`

This choice is frozen **before model predictions are inspected**.

The same canonical external normalization also:
- maps user mentions to `<USER>`;
- maps HTTP(S) URLs to `<URL>`;
- collapses whitespace.

No other lexical cleanup is allowed.

## Label mapping from internal three-class model to external binary task

Frozen before results:

- internal `irony` -> external `irony` (1)
- internal `sarcasm` -> external `irony` (1)
- internal `regular` -> external `non_irony` (0)

Rationale: SemEval/TweetEval Task A asks for irony vs. non-irony rather than the finer
irony-vs.-sarcasm distinction.

This is an **external binary transfer**, not an external validation of the internal
three-way label ontology.

## Models

No new model family.

### S2
Reconstruct the frozen A-C primary sparse model exactly from:
`data/processed/h03_tcc_reproduction/train_final.csv`

Configuration:
- word TF-IDF
- ngram_range=(1,2)
- min_df=2
- sublinear_tf=True
- max_features=200000
- lowercase=True
- LinearSVC C=0.5

### BERTweet
Reuse the three already-trained A-C checkpoints:
- seed 42
- seed 2026
- seed 7

Reuse the already-frozen local `vinai/bertweet-base` snapshot.
No training occurs in H10.

## Primary external populations

### EXT-C-ALL
All 784 TweetEval test rows after frozen external marker removal/normalization.

### EXT-C-NOVEL — PRIMARY
Remove from EXT-C-ALL any external test item whose normalized clean text is an exact
match to any A-C internal train **or internal test** text.

This guards against exact cross-dataset item reuse.

If zero rows are removed, EXT-C-NOVEL equals EXT-C-ALL.

## Near-duplicate diagnostic sensitivity

Using the same broad H04 family-screen logic:
- token unigram+bigram MinHash
- 64 permutations
- LSH threshold 0.72
- exact candidate scores:
  - token Jaccard
  - character-5 Jaccard
  - RapidFuzz ratio
- high-similarity screen:
  `char5 >= 0.80 OR (RapidFuzz >= 0.92 AND token >= 0.70)`

The screen compares external clean test items against the internal A-C training set.

A conservative sensitivity, `EXT-C-HISIM-PURGED`, excludes external items with at least
one high-similarity training candidate. This is an **algorithmic sensitivity**, not
manual duplicate gold.

It may be reported if informative but is not allowed to replace EXT-C-NOVEL as the
primary external result after seeing outcomes.

## Metrics

Primary:
- binary Macro-F1 on EXT-C-NOVEL.

Secondary:
- F1 for external irony class;
- F1 for external non-irony class;
- accuracy;
- confusion matrix.

BERTweet:
- each seed separately;
- mean ± sample SD across seeds 42/2026/7.

Uncertainty:
- 10,000 stratified bootstrap resamples;
- seed 20260903;
- percentile 95% CI.

Paired BERTweet seed42 minus S2 delta is allowed because predictions share the identical
external item IDs.

## Interpretation constraints

Allowed:
- external performance is lower/similar/higher than internal under dataset shift;
- clean internal gains do/do not survive binary external transfer;
- cross-dataset exact/high-similarity overlap is absent/present at measured rates.

Forbidden:
- calling the external result a three-way irony/sarcasm validation;
- fine-tuning on external gold labels;
- choosing a preprocessing variant because it scores better;
- hiding a weak external result;
- claiming corpus-independent generalization from one external benchmark;
- claiming the TweetEval snapshot is marker-free.

## Stop rule

H10 is the only new empirical extension authorized before ERCEMAPI submission.
After H10, no additional dataset/model experiment is added before the 2026-09-06
submission deadline.
