# ERCEMAPI P1 -- METHOD FREEZE v1.0

Status: **FROZEN BEFORE H06/H07 RESULTS**
Freeze date: 2026-09-02
Consolidator: ChatGPT GPT-5.6 Sol
Independent red-team: Claude Code Sonnet, H05

No model/condition may be added, removed, retuned, or reframed because of final-test
performance after this freeze. Demonstrated implementation bugs may be fixed with an
auditable rerun.

## 1. Research questions

**RQ1.** How much do explicit label markers and broader dataset artifacts inflate
apparent performance in irony/sarcasm classification?

**RQ2.** After explicit marker leakage, exact/normalized duplication, train-test overlap,
and residual cross-split same-message/template families are controlled, how much
discriminative performance remains?

**RQ3.** Does a contextual model pretrained for Twitter improve robust clean
classification over a strong sparse baseline, especially at the irony-sarcasm boundary?

## 2. Frozen data conditions

### A-L -- Audited Matched Leaky
Exactly the final A-C row IDs, after the H03 structural audit, using `text_leaky`:
URL/user normalization is retained, explicit class markers remain.

**Purpose:** paired estimate of direct marker leakage against A-C.

### A-C -- Audited Clean -- PRIMARY
H03 canonical clean split:
- train: 58,784
- test: 5,947

Uses `text_clean`, with the four explicit class markers removed after conservative
normalization and after structural cleaning.

### G-C -- Group-Purged Clean -- MANDATORY SENSITIVITY
Same 5,947 test IDs, in the same order, as A-C.
Remove from the A-C training population every high-similarity train member connected
to one of the 85 manually confirmed affected test items.

Frozen inventory:
- A-C train: 58,784
- purged train IDs: 227
- G-C train: 58,557
- G-C test: 5,947
- confirmed affected A-C test fraction: 1.429292%
- predeclared trigger: 0.5%

The 85 representative best pairs were manually reviewed. The purge then extends to
all 227 train nodes connected to those confirmed test-family components. This
component-level extrapolation is an explicit methodological assumption.

A-C remains the primary estimate. G-C is a mandatory sensitivity analysis.

### U-L / U-C -- archival / illustrative only
The original unaudited split may be retained as a descriptive reproduction artifact,
but it is not required for the paper's core matrix.

**Headline marker-inflation claims MUST use A-L vs A-C**, not U-L vs U-C, because the
unaudited split mixes marker leakage with duplication/overlap contamination.

## 3. Diagnostics and sparse baselines

### D0 -- sanity baselines
Run on A-C and G-C:
- majority-class predictor;
- stratified-random predictor using training-label priors, RNG seed 42.

### D1 -- direct marker rule
Run on A-L only.
Search exact markers case-insensitively:
`#irony`, `#ironic`, `#sarcasm`, `#sarcastic`.

Prediction:
- if one or more class markers occur, use the class of the earliest occurring marker;
- if none occurs, predict `regular`.

Diagnostic only.

### D2 -- marker-only Logistic Regression
Run on A-L only.
Four binary features, one for presence of each exact class marker.
Logistic Regression:
- L2
- C=1.0
- solver=`lbfgs`
- max_iter=3000
- no class weighting.

Diagnostic only.

### D3 -- surface-only Logistic Regression
Run on A-C and G-C.
No lexical identities.

Exactly these nine deterministic features:
1. character length;
2. whitespace-token count;
3. `<URL>` count;
4. `<USER>` count;
5. hashtag count;
6. RT-prefix indicator;
7. exclamation-mark count;
8. question-mark count;
9. digit count.

Pipeline:
- StandardScaler;
- Logistic Regression, L2, C=1.0, solver=`lbfgs`, max_iter=3000.

Purpose: quantify nonlexical dataset-specific shortcuts that survive explicit-marker removal.

### S1 -- word TF-IDF + Logistic Regression -- SUPPLEMENTARY
Run on A-C and G-C if computationally trivial.

Vectorizer:
- word analyzer;
- ngram_range=(1,2);
- min_df=2;
- sublinear_tf=True;
- max_features=200000;
- lowercase=True.

Classifier:
- Logistic Regression, L2, C=1.0, solver=`lbfgs`, max_iter=3000.

### S2 -- word TF-IDF + LinearSVC -- REQUIRED PRIMARY SPARSE
Run on A-C and G-C.

Reuse the configuration selected in H03 validation:
- word analyzer;
- ngram_range=(1,2);
- min_df=2;
- sublinear_tf=True;
- max_features=200000;
- lowercase=True;
- LinearSVC C=0.5.

No retuning on G-C or final test.

### S3 -- char TF-IDF + LinearSVC -- OPTIONAL SUPPLEMENTARY
Run on A-C and G-C if resources permit.

Predeclared fixed configuration, with no test-based tuning:
- analyzer=`char_wb`;
- ngram_range=(3,5);
- min_df=2;
- sublinear_tf=True;
- max_features=200000;
- lowercase=True;
- LinearSVC C=0.5.

## 4. Neural model freeze

### N1 -- `vinai/bertweet-base` -- REQUIRED PRIMARY NEURAL

#### Text adapter
Use A-L/A-C/G-C frozen texts exactly as stored, except model-input-only normalization:
- `<USER>` -> `@USER`
- `<URL>` -> `HTTPURL`

No other rewriting, stemming, hashtag splitting, emoji normalization, or lexical cleanup.

#### Validation membership
Create one stratified 80/20 split of A-C train IDs with `random_state=42`.
That membership is canonical for all N1 runs:
- A-C uses it directly;
- A-L uses the same IDs with leaky text;
- G-C inherits the same membership and removes any of the 227 purged IDs from whichever
  side they originally belonged to.

The final 5,947 test IDs are never used for epoch selection or hyperparameter selection.

#### Frozen training hyperparameters
No hyperparameter search will be performed.

- model: `vinai/bertweet-base`
- max_length: 128
- optimizer: AdamW
- learning_rate: 2e-5
- weight_decay: 0.01
- train batch size: 16
- eval batch size: 32
- gradient accumulation: 2
- effective train batch size: 32
- warmup_ratio: 0.10
- max_epochs: 5
- evaluation: once per epoch
- model-selection metric: validation Macro-F1
- early-stopping patience: 1 completed epoch without improvement
- restore best validation checkpoint
- no mixed-precision requirement is part of the scientific protocol; fp16/bf16 may be
  used only as a numerically standard runtime optimization if supported.
- label order: irony, sarcasm, regular.

#### Seeds
Primary A-C:
- 42
- 2026
- 7

G-C:
- same three seeds if completed before the hard experimental freeze;
- minimum non-cuttable fallback: seed 42, declared before inspecting G-C test performance.

Leaky diagnostic:
- train A-L, seed 42;
- evaluate that same frozen model on A-L test and A-C test.

The A-L -> A-C evaluation directly measures shortcut sensitivity when explicit markers
disappear at test time.

#### Reporting
A-C 3-seed result:
- individual seed metrics;
- mean ± sample SD.

Never headline the best seed.

G-C:
- same policy if three seeds complete;
- if only seed 42 is run under the predeclared fallback, label it explicitly as a
  single-seed sensitivity.

### N2 -- `roberta-base` -- OPTIONAL
First neural model to cut. It is not required for validity.

## 5. Metrics

Primary:
- Macro-F1.

Secondary:
- accuracy;
- weighted-F1;
- per-class precision/recall/F1;
- confusion matrix;
- direct irony -> sarcasm and sarcasm -> irony error counts.

## 6. Statistical analysis

Test uncertainty:
- stratified bootstrap over the frozen test set;
- 10,000 resamples;
- seed 20260902;
- percentile 95% CI.

Paired delta bootstrap is permitted only when predictions share the identical frozen
test IDs, including:
- A-L vs A-C on matched IDs when the compared model/predictions are logically paired;
- A-C vs G-C;
- S2 vs N1 seed 42 on A-C, when comparing the same test IDs.

Neural uncertainty:
- individual frozen seed results;
- mean ± sample SD across seeds 42/2026/7.

Seed-42 paired bootstrap and multi-seed dispersion answer different uncertainty
questions and must be labeled separately.

No p-value/significance-star framework is planned. Claims are effect-size/CI based,
so no classical multiple-comparison correction is asserted.

## 7. Near-duplicate limitation

The H04 MinHash-LSH candidate threshold 0.72 and the high-similarity screen were fixed
before H06/H07 results. Near-duplicate contamination below the detection boundary is
unmeasured, not assumed absent.

## 8. Claims allowed if supported

- explicit class markers inflate Macro-F1 by X under A-L vs A-C;
- direct-marker / marker-only diagnostics achieve Y;
- surface-only features contain Z predictive signal;
- A-C robust performance is X;
- the G-C train-family purge changes A-C performance by Δ;
- BERTweet clean gain/loss over S2 is Δ;
- remaining error does or does not concentrate at the irony-sarcasm boundary.

## 9. Claims forbidden

- “the model understands irony/sarcasm”;
- general causal claims beyond the controlled interventions;
- equating marker removal with removal of all dataset bias;
- calling all algorithmically retrieved candidates gold duplicates;
- selecting models/conditions because final-test numbers look better;
- using U-L/U-C as the headline causal estimate of marker inflation;
- suppressing G-C because its observed effect is small.

## 10. Hard stop

Experimental freeze deadline: 2026-09-05 18:00 BRT.

After that time:
- no new model or condition;
- only demonstrated implementation-bug fixes may be rerun and must be documented.
