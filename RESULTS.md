# Results guide

This page maps the paper-level claims to concrete committed artifacts. The repository
does **not** redistribute raw tweet text.

## Condition glossary

| Code | Meaning | Test population |
|---|---|---|
| **A-L** | Audited matched leaky condition; final audited IDs with explicit class markers retained | 5,947 internal test items |
| **A-C** | Audited clean condition; same IDs after marker removal and structural cleaning | 5,947 internal test items |
| **G-C** | Group-purged clean sensitivity; 227 connected training rows removed, A-C test unchanged | 5,947 internal test items |
| **EXT-C-NOVEL** | External TweetEval irony test after frozen marker removal, excluding exact internal overlap | 783 external test items |

## Internal benchmark: central findings

| Model / diagnostic | Condition | Macro-F1 | Interpretation |
|---|---:|---:|---|
| Direct marker rule | A-L | **1.000** | Tweet content is unnecessary when collection markers remain |
| Marker-only logistic regression | A-L | **1.000** | Four binary marker indicators perfectly recover the matched labels |
| Surface-only logistic regression | A-C | **0.594** | Nonlexical dataset-specific signal remains after direct-marker removal |
| TF-IDF + LinearSVC (primary sparse) | A-C | **0.826** | Strong cleaned sparse baseline |
| BERTweet, 3 seeds | A-C | **0.884 ± 0.002** | Contextual pretraining improves the audited clean benchmark |
| TF-IDF + LinearSVC | G-C | **0.824** | Near-duplicate-family purge changes little |
| BERTweet, 3 seeds | G-C | **0.886 ± 0.002** | Main contextual conclusion is stable to the purge sensitivity |

### Controlled shortcut-dependence intervention

The same BERTweet checkpoint trained on A-L scores:

- **1.000 Macro-F1** when evaluated on A-L;
- **0.201 Macro-F1** on the same test IDs after the class markers are removed.

The paired drop is approximately **0.799 Macro-F1**. This is a dataset/model-specific
shortcut-dependence result, not a claim about all Transformers.

## External transfer

The external evaluation uses the TweetEval irony test split with no fine-tuning and maps
internal `irony|sarcasm -> irony`, `regular -> non_irony`.

| Model | EXT-C-NOVEL Macro-F1 | Notes |
|---|---:|---|
| TF-IDF + LinearSVC | **0.297** | 95% CI [0.285, 0.312] |
| BERTweet, 3 seeds | **0.298 ± 0.005** | clean internal gain does not transfer |

The external result shows that stronger in-benchmark evaluation validity does not imply
cross-dataset generalization.

## Where to verify each claim

- fixed predictions: `results/predictions/`
- compact model tables: `results/summaries/H06_RESULTS.tsv`, `H07_RESULTS.tsv`
- bootstrap/delta summary: `results/summaries/H08_SUMMARY.md`
- external transfer: `results/summaries/H10_RESULTS.tsv`
- automated metric check: `python scripts/verify_predictions.py`
- methodological freeze documents: `protocols/`
