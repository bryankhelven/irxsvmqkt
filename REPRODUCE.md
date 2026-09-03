# Reproduction guide

The artifact supports three distinct levels of verification. Reviewers do not need to
retrain BERTweet to check the reported numbers.

## Level 0 -- integrity + headline metric verification

Expected time: seconds to a few minutes. CPU only. No source tweet text required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r environment/requirements-core.txt
python scripts/artifact_doctor.py
python scripts/verify_predictions.py
```

Or:

```bash
bash scripts/run.sh quick
```

The verifier recomputes Macro-F1/accuracy directly from committed prediction files,
including the leaky-marker diagnostics, sparse baseline, BERTweet seed runs, shortcut
collapse, and external transfer.

## Level 1 -- reconstruct the audited sparse pipeline

Expected resources: CPU + Internet access for the public dataset download.

After activating the environment created in Level 0, the whole sparse path is one
command:

```bash
bash scripts/run.sh sparse
```

The runner first executes:

```bash
python scripts/fetch_internal_data.py
```

That helper downloads the public Kaggle dataset
`nikhiljohnk/tweets-with-sarcasm-and-irony` through KaggleHub, verifies the exact frozen
SHA-256 values for `train.csv` and `test.csv`, and installs them into the expected raw
data directory. Public Kaggle resources normally download without authentication; if
Kaggle requires consent/authentication, the helper stops explicitly rather than silently
using different data.

After the hashes pass, the runner executes:

- H03: audited dataset reconstruction and matched clean/leaky baseline;
- H04: forensic marker/duplicate/near-duplicate audit;
- H05: deterministic G-C construction from the frozen confirmed-family ID table;
- H06: diagnostics and sparse model matrix.

Expected A-C sizes:
- train: 58,784
- test: 5,947

Expected G-C sizes:
- train: 58,557
- test: 5,947

The acquisition step can also be run by itself:

```bash
python scripts/fetch_internal_data.py
```

No manual placement of `train.csv` or `test.csv` is required.

## Level 2 -- neural reproduction

GPU required.

After Level 1:

```bash
pip install -r environment/requirements-neural.txt
bash scripts/run.sh neural
```

The BERTweet runner:
- resolves `vinai/bertweet-base`;
- freezes validation membership;
- trains A-C seeds 42/2026/7;
- trains G-C seeds 42/2026/7;
- trains A-L seed 42 once and evaluates the same checkpoint on A-L and A-C;
- writes predictions and training metadata.

The frozen model/training protocol is in `protocols/`.

## Level 3 -- uncertainty + external transfer

After Levels 1-2:

```bash
bash scripts/run.sh stats
bash scripts/run.sh external
```

The external stage:
- downloads the pinned TweetEval irony test revision;
- removes the frozen marker set before prediction;
- checks exact and high-similarity cross-dataset overlap;
- evaluates the already-trained S2/BERTweet models with no external tuning;
- computes the frozen bootstrap statistics.

## Why raw tweets are absent

Tweet text is not redistributed in this review artifact. The source datasets remain
independently obtainable and are pinned by identifiers and SHA-256 hashes. Prediction
files contain IDs, gold labels, model predictions, and audit flags only.

## Expected-result reference

See:
- `RESULTS.md`
- `manifests/EXPECTED_RESULTS.json`
- `results/summaries/`
