#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ARTIFACT_ROOT="$ROOT"

MODE="${1:-quick}"

echo "============================================================"
echo " DOUBLE-ANONYMOUS ARTIFACT — REPRODUCTION RUNNER"
echo "============================================================"
echo "root=$ROOT"
echo "mode=$MODE"
echo

if [ "$MODE" = "quick" ]; then
  python "$ROOT/scripts/artifact_doctor.py" || true
  python "$ROOT/scripts/verify_predictions.py" || true

elif [ "$MODE" = "sparse" ]; then
  echo "Fetching/verifying the frozen internal source dataset..."
  python "$ROOT/scripts/fetch_internal_data.py" || true

  TRAIN="$ROOT/data/raw/kaggle_tweets_sarcasm_irony/3b278d7767d6/train.csv"
  TEST="$ROOT/data/raw/kaggle_tweets_sarcasm_irony/3b278d7767d6/test.csv"
  if [ ! -f "$TRAIN" ] || [ ! -f "$TEST" ]; then
    echo "BLOCKER: frozen internal source data is unavailable; sparse reproduction stopped."
  else
    python "$ROOT/src/h03_reproduce_tcc.py" --project "$ROOT" || true
    python "$ROOT/src/h04_forensic_audit.py" --project "$ROOT" || true
    python "$ROOT/src/h05_build_grouped_clean.py" || true
    python "$ROOT/src/h06_sparse_baselines.py" || true
  fi

elif [ "$MODE" = "neural" ]; then
  python "$ROOT/src/h07_bertweet.py" || true

elif [ "$MODE" = "stats" ]; then
  python "$ROOT/src/h08_stats_error.py" || true

elif [ "$MODE" = "external" ]; then
  python "$ROOT/src/h10_prepare_external.py" || true
  python "$ROOT/src/h10_eval_sparse.py" || true
  python "$ROOT/src/h10_eval_bertweet.py" || true
  python "$ROOT/src/h10_analyze.py" || true

elif [ "$MODE" = "all" ]; then
  bash "$0" sparse
  bash "$0" neural
  bash "$0" stats
  bash "$0" external

else
  echo "Unknown mode: $MODE"
  echo "Use: quick | sparse | neural | stats | external | all"
fi

true
