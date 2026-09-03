# Data and model provenance

No raw tweet text is committed to this repository.

## Internal source dataset

Public dataset identifier:

`nikhiljohnk/tweets-with-sarcasm-and-irony`

Frozen source hashes:

- archive: `3b278d7767d62cbe50d077c5527db75ff4e54acb1e0b6825048f64c55436fb43`
- `train.csv`: `f1624804f995f6ca7003a7a023159e0326c283b3b104dfe6e66a55f7fe6c8d77`
- `test.csv`: `b4318a94673cf28f74a97410f5880b4c6bfd1cfb09090565e13587afbc108020`

Automatic acquisition:

```bash
python scripts/fetch_internal_data.py
```

The helper uses KaggleHub to download the public dataset, verifies the two frozen
SHA-256 values, and installs the files into the exact layout expected by H03. No manual
download or file placement is required.

## External benchmark

TweetEval repository revision:

`cardiffnlp/tweeteval@4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`

Frozen external files:

- `test_text.txt`: `53103da934a7308eee82f05f2a9781a8ea3e88604fdc1e02d3101108505c64be`
- `test_labels.txt`: `08e2095e1725e74907a380614c220204e356bb46e3e8c93deb74e83e5b15ab38`
- `mapping.txt`: `70e09046e181179b9870a2f59a3263bbb28c787831779e4ce4a18373d054296b`

`src/h10_prepare_external.py` downloads those exact files from the pinned revision.

## Contextual model

Model identifier:

`vinai/bertweet-base`

Frozen resolved revision:

`b349c1243407b0dcffeabb2337497477286e27ab`

The training script resolves/downloads the model and records the local snapshot metadata.
