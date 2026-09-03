# Artifact map

| Path | Purpose | Needed for quick verification? |
|---|---|---:|
| `README.md` | Reviewer entry point | yes |
| `RESULTS.md` | Human-readable map of claims to results | yes |
| `REPRODUCE.md` | Exact verification/reproduction instructions | yes |
| `scripts/artifact_doctor.py` | Hash + anonymity/codename scan | yes |
| `scripts/verify_predictions.py` | Recompute headline metrics | yes |
| `scripts/fetch_internal_data.py` | Automatically download + hash-verify the public internal source dataset | no |
| `scripts/run.sh` | Stage-oriented reproduction runner | no |
| `src/` | Frozen research implementation | no |
| `protocols/` | Pre-result method/statistical freezes and H10 correction note | recommended |
| `results/predictions/` | Prediction-only evidence, no tweet text | yes |
| `results/summaries/` | Compact frozen result tables | recommended |
| `manifests/SOURCES.md` | Dataset/model provenance and hashes | recommended |
| `manifests/EXPECTED_RESULTS.json` | Numeric anchors used by quick verifier | yes |
| `ARTIFACT_MANIFEST.sha256` | Repository-file integrity | yes |
| `environment/` | Dependency versions and original compute environment | no |

The Hxx prefixes preserve execution provenance from the frozen research pipeline. They
are mapped to human-readable stages in `REPRODUCE.md`; reviewers do not need to know the
development history to use the artifact.
