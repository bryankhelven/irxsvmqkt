# Double-Anonymous Review Artifact

This repository accompanies a double-anonymous submission on **dataset leakage,
shortcut dependence, robust in-benchmark evaluation, and cross-benchmark transfer in
English Twitter irony/sarcasm classification**.

The public source repository deliberately does **not** contain the manuscript title,
author names, affiliations, or raw tweet text. Reviewers should access it through the
anonymized review URL supplied in the paper.

## What the artifact lets you verify

The study asks four empirical questions:

1. **How much can explicit distant-supervision markers inflate apparent performance?**
2. **What signal remains after direct marker leakage, duplicates, conflicts, and
   train-test overlap are controlled?**
3. **Does tweet-specific contextual pretraining improve the audited clean benchmark
   without relying on the explicit marker shortcut?**
4. **Do the clean internal classifiers transfer to a separately constructed external
   irony benchmark without fine-tuning?**

The committed prediction files support a complete CPU-only check of the headline
metrics; the source code and frozen protocols support deeper reconstruction.

## Headline results

| Finding | Frozen result |
|---|---:|
| Direct marker rule on audited-leaky condition | **1.000 Macro-F1** |
| Marker-only logistic regression | **1.000 Macro-F1** |
| Surface-only diagnostic after marker removal | **0.594 Macro-F1** |
| Primary sparse TF-IDF + LinearSVC on audited clean data | **0.826 Macro-F1** |
| BERTweet on audited clean data, 3 seeds | **0.884 ± 0.002 Macro-F1** |
| Same leaky-trained BERTweet, evaluated after marker removal | **0.201 Macro-F1** |
| External TweetEval transfer: primary sparse model | **0.297 Macro-F1** |
| External TweetEval transfer: BERTweet, 3 seeds | **0.298 ± 0.005 Macro-F1** |

The external result is intentionally reported even though it is weak: cleaning and
auditing improve **benchmark validity**, but do not by themselves guarantee
cross-dataset generalization.

For condition definitions, confidence intervals, paired deltas, and interpretation
constraints, see [`RESULTS.md`](RESULTS.md).

## Quick reviewer check

CPU only; raw tweets are not required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r environment/requirements-core.txt
python scripts/artifact_doctor.py
python scripts/check_typography.py .
python scripts/verify_predictions.py
```

Expected final line:

```text
OVERALL: PASS
```

## Repository structure

```text
.
├── README.md
├── RESULTS.md
├── REPRODUCE.md
├── ARTIFACT_MAP.md
├── environment/          # dependency and hardware record
├── manifests/            # dataset/model hashes + expected numerical anchors
├── protocols/            # frozen pre-result methodological decisions
├── results/
│   ├── predictions/      # prediction-only evidence; no raw tweets
│   └── summaries/        # compact result tables
├── scripts/              # quick verification + reproduction helpers
└── src/                  # frozen research implementation
```

See [`ARTIFACT_MAP.md`](ARTIFACT_MAP.md) for a reviewer-oriented description of every
component.

## Reproduction levels

- **Level 0 -- verify committed results:** seconds/minutes, CPU only.
- **Level 1 -- rebuild audited sparse pipeline:** the artifact automatically downloads
  the pinned public Kaggle dataset, verifies the frozen hashes, then runs H03-H06.
- **Level 2 -- retrain BERTweet:** CUDA GPU required; frozen seeds and validation
  membership.
- **Level 3 -- recompute uncertainty and external transfer:** no external tuning;
  TweetEval is pinned to a fixed repository revision.

Exact commands are in [`REPRODUCE.md`](REPRODUCE.md).

## Data provenance and redistribution

Raw tweet text is intentionally not committed. Reproduction does **not** require the
reviewer to hunt for files manually. After installing the core requirements:

```bash
python scripts/fetch_internal_data.py
```

downloads the public Kaggle source dataset through KaggleHub, verifies the exact frozen
`train.csv` / `test.csv` SHA-256 values, and installs them into the expected layout.
`bash scripts/run.sh sparse` calls this downloader automatically.

`manifests/SOURCES.md` records the internal dataset identifier, exact source hashes, the
pinned TweetEval revision/file hashes, and the exact BERTweet model revision.

## Experimental safeguards preserved here

The artifact includes the method/statistical freeze records used before the main final
results. In particular:

- A-L and A-C use matched final test IDs;
- S2 was designated the primary sparse baseline before final comparison;
- G-C is a sensitivity analysis and does not replace A-C post hoc;
- BERTweet headline reporting uses all three frozen seeds;
- external transfer uses no fine-tuning or threshold selection;
- the weak external result is retained rather than suppressed.

## Start here

- **Understand the findings:** [`RESULTS.md`](RESULTS.md)
- **Verify them quickly:** `python scripts/verify_predictions.py`
- **Reproduce from source data:** [`REPRODUCE.md`](REPRODUCE.md)
- **Inspect methodology freezes:** [`protocols/`](protocols/)
- **Check provenance:** [`manifests/SOURCES.md`](manifests/SOURCES.md)
