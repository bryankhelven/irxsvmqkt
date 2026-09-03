#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results/predictions"
EXPECTED = json.loads((ROOT / "manifests/EXPECTED_RESULTS.json").read_text())

def mf1(path, sep=",", mask=None):
    df = pd.read_csv(path, sep=sep)
    if mask is not None:
        df = df.loc[mask(df)].copy()
    return len(df), float(accuracy_score(df.y_true, df.y_pred)), float(f1_score(df.y_true, df.y_pred, average="macro", zero_division=0))

def close(a, b, tol=5e-6):
    return abs(float(a) - float(b)) <= tol

def main():
    checks = []
    rows = []

    mapping = [
        ("D1_direct_marker_AL", "D1_direct_marker_AL.csv", "D1_direct_marker_AL_macro_f1"),
        ("D2_marker_only_AL", "D2_marker_only_AL.csv", "D2_marker_only_AL_macro_f1"),
        ("D3_surface_AC", "D3_surface_AC.csv", "D3_surface_AC_macro_f1"),
        ("S2_AC", "S2_AC.csv", "S2_AC_macro_f1"),
        ("S2_GC", "S2_GC.csv", "S2_GC_macro_f1"),
        ("BERT_AC_seed42", "BERT_AC_seed42.csv", "BERT_AC_seed42_macro_f1"),
        ("BERT_AC_seed2026", "BERT_AC_seed2026.csv", "BERT_AC_seed2026_macro_f1"),
        ("BERT_AC_seed7", "BERT_AC_seed7.csv", "BERT_AC_seed7_macro_f1"),
        ("BERT_AL_eval_AL", "BERT_AL_seed42_eval_AL.csv", "BERT_AL_eval_AL_macro_f1"),
        ("BERT_AL_eval_AC", "BERT_AL_seed42_eval_AC.csv", "BERT_AL_eval_AC_macro_f1"),
    ]
    for label, fn, key in mapping:
        n, acc, score = mf1(PRED / fn)
        exp = EXPECTED["internal"][key]
        ok = close(score, exp)
        checks.append(ok)
        rows.append((label, n, acc, score, exp, ok))

    ext_scores = []
    for label, fn in [
        ("EXT_S2", "EXT_S2.tsv"),
        ("EXT_BERT_seed42", "EXT_BERT_seed42.tsv"),
        ("EXT_BERT_seed2026", "EXT_BERT_seed2026.tsv"),
        ("EXT_BERT_seed7", "EXT_BERT_seed7.tsv"),
    ]:
        n, acc, score = mf1(
            PRED / fn, sep="\t",
            mask=lambda d: ~d["exact_internal_overlap"].astype(bool)
        )
        ext_scores.append(score)
        rows.append((label, n, acc, score, None, n == EXPECTED["external"]["n_exact_novel"]))
        checks.append(n == EXPECTED["external"]["n_exact_novel"])

    s2 = ext_scores[0]
    bert_mean = float(np.mean(ext_scores[1:]))
    bert_sd = float(np.std(ext_scores[1:], ddof=1))
    checks += [
        close(s2, EXPECTED["external"]["S2_EXT_C_NOVEL_macro_f1"]),
        close(bert_mean, EXPECTED["external"]["BERT_EXT_C_NOVEL_mean_macro_f1"]),
        close(bert_sd, EXPECTED["external"]["BERT_EXT_C_NOVEL_sample_sd"]),
    ]

    print("=== Prediction verification ===")
    for label, n, acc, score, exp, ok in rows:
        target = "" if exp is None else f" expected={exp:.6f}"
        print(f"{'PASS' if ok else 'FAIL'}  {label:24s} n={n:5d} accuracy={acc:.6f} macro_f1={score:.6f}{target}")
    print()
    print(f"External BERTweet 3-seed mean Macro-F1: {bert_mean:.6f}")
    print(f"External BERTweet sample SD:          {bert_sd:.6f}")
    print("OVERALL:", "PASS" if all(checks) else "FAIL")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
