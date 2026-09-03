#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABELS = ["irony", "sarcasm", "regular"]
MARKER_PATTERNS = {
    "#irony": re.compile(r"(?i)(?<!\w)#irony\b"),
    "#ironic": re.compile(r"(?i)(?<!\w)#ironic\b"),
    "#sarcasm": re.compile(r"(?i)(?<!\w)#sarcasm\b"),
    "#sarcastic": re.compile(r"(?i)(?<!\w)#sarcastic\b"),
}
MARKER_CLASS = {
    "#irony": "irony",
    "#ironic": "irony",
    "#sarcasm": "sarcasm",
    "#sarcastic": "sarcasm",
}
HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
RT_RE = re.compile(r"(?i)^\s*RT\b")


def aggregate_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        ),
    }


def id_digest(df: pd.DataFrame) -> str:
    import hashlib
    joined = "\n".join(df["_id"].astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def save_run(out_root: Path, name: str, test_df: pd.DataFrame, pred, meta: dict):
    run = out_root / name
    run.mkdir(parents=True, exist_ok=True)
    y_true = test_df["class"].astype(str).to_numpy()
    pred = np.asarray(pred, dtype=object)

    metrics = aggregate_metrics(y_true, pred)
    report = classification_report(
        y_true, pred, labels=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, pred, labels=LABELS)

    pd.DataFrame({
        "_id": test_df["_id"].astype(str).values,
        "y_true": y_true,
        "y_pred": pred,
    }).to_csv(run / "predictions.csv", index=False)

    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(run / "confusion.csv")

    with (run / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    payload = {
        "run": name,
        "metrics": metrics,
        "test_n": int(len(test_df)),
        "test_id_sha256": id_digest(test_df),
        "irony_to_sarcasm": int(cm[0, 1]),
        "sarcasm_to_irony": int(cm[1, 0]),
        **meta,
    }
    with (run / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def direct_marker_predict(texts):
    preds = []
    for text in texts.astype(str):
        hits = []
        for marker, pat in MARKER_PATTERNS.items():
            m = pat.search(text)
            if m:
                hits.append((m.start(), marker))
        if not hits:
            preds.append("regular")
        else:
            hits.sort(key=lambda x: (x[0], x[1]))
            preds.append(MARKER_CLASS[hits[0][1]])
    return np.array(preds, dtype=object)


def marker_features(texts):
    arr = np.zeros((len(texts), 4), dtype=np.float64)
    names = list(MARKER_PATTERNS)
    for i, text in enumerate(texts.astype(str)):
        for j, name in enumerate(names):
            arr[i, j] = 1.0 if MARKER_PATTERNS[name].search(text) else 0.0
    return arr


def surface_features(texts):
    rows = []
    for text in texts.astype(str):
        rows.append([
            len(text),
            len(text.split()),
            text.count("<URL>"),
            text.count("<USER>"),
            len(HASHTAG_RE.findall(text)),
            1 if RT_RE.search(text) else 0,
            text.count("!"),
            text.count("?"),
            sum(ch.isdigit() for ch in text),
        ])
    return np.asarray(rows, dtype=np.float64)


def majority_predict(train_df, test_df):
    majority = train_df["class"].value_counts().idxmax()
    return np.array([majority] * len(test_df), dtype=object)


def stratified_random_predict(train_df, test_df, seed=42):
    counts = train_df["class"].value_counts()
    probs = np.array([counts.get(lbl, 0) for lbl in LABELS], dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(np.array(LABELS, dtype=object), size=len(test_df), p=probs)


def make_word_tfidf():
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=200000,
        lowercase=True,
    )


def run_all(project: Path):
    ac_train_p = project / "data/processed/h03_tcc_reproduction/train_final.csv"
    ac_test_p = project / "data/processed/h03_tcc_reproduction/test_final.csv"
    gc_train_p = project / "data/derived/h05_grouped_clean/train_grouped_clean.csv"
    gc_test_p = project / "data/derived/h05_grouped_clean/test_grouped_clean.csv"

    required = [ac_train_p, ac_test_p, gc_train_p, gc_test_p]
    missing = [str(p) for p in required if not p.exists()]

    report_root = project / "reports/h06_sparse_baselines"
    run_root = project / "runs/h06_sparse_baselines"
    report_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    if missing:
        status = {"status": "BLOCKED", "reason": "missing_inputs", "missing": missing}
        (report_root / "H06_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return status

    ac_train = pd.read_csv(ac_train_p)
    ac_test = pd.read_csv(ac_test_p)
    gc_train = pd.read_csv(gc_train_p)
    gc_test = pd.read_csv(gc_test_p)

    for name, df in [("ac_train", ac_train), ("ac_test", ac_test), ("gc_train", gc_train), ("gc_test", gc_test)]:
        needed = {"_id", "class", "text_clean", "text_leaky"}
        absent = sorted(needed - set(df.columns))
        if absent:
            status = {"status": "BLOCKED", "reason": "missing_columns", "dataset": name, "columns": absent}
            (report_root / "H06_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps(status, indent=2))
            return status

    invariant = {
        "ac_train_n": len(ac_train),
        "ac_test_n": len(ac_test),
        "gc_train_n": len(gc_train),
        "gc_test_n": len(gc_test),
        "same_test_ids_and_order": ac_test["_id"].astype(str).tolist() == gc_test["_id"].astype(str).tolist(),
        "gc_ids_subset_ac": set(gc_train["_id"].astype(str)).issubset(set(ac_train["_id"].astype(str))),
        "purged_train_n": len(ac_train) - len(gc_train),
    }

    invariant_pass = (
        invariant["ac_train_n"] == 58784
        and invariant["ac_test_n"] == 5947
        and invariant["gc_train_n"] == 58557
        and invariant["gc_test_n"] == 5947
        and invariant["same_test_ids_and_order"]
        and invariant["gc_ids_subset_ac"]
        and invariant["purged_train_n"] == 227
    )

    (report_root / "condition_invariants.json").write_text(
        json.dumps(invariant, indent=2), encoding="utf-8"
    )

    if not invariant_pass:
        status = {"status": "BLOCKED", "reason": "condition_invariants_failed", "invariants": invariant}
        (report_root / "H06_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return status

    inventory = []
    for cond, tr, te in [("A-C", ac_train, ac_test), ("G-C", gc_train, gc_test)]:
        inventory.append({
            "condition": cond,
            "train_n": len(tr),
            "test_n": len(te),
            "train_id_sha256": id_digest(tr),
            "test_id_sha256": id_digest(te),
        })
    pd.DataFrame(inventory).to_csv(report_root / "condition_inventory.tsv", sep="\t", index=False)

    results = []

    # D0 A-C/G-C
    for cond, tr, te in [("AC", ac_train, ac_test), ("GC", gc_train, gc_test)]:
        pred = majority_predict(tr, te)
        results.append(save_run(run_root, f"D0_majority_{cond}", te, pred, {
            "family": "D0", "condition": cond, "model": "majority"
        }))
        pred = stratified_random_predict(tr, te, seed=42)
        results.append(save_run(run_root, f"D0_stratified_random_{cond}", te, pred, {
            "family": "D0", "condition": cond, "model": "stratified_random", "seed": 42
        }))

    # D1 direct marker on A-L
    pred = direct_marker_predict(ac_test["text_leaky"])
    results.append(save_run(run_root, "D1_direct_marker_AL", ac_test, pred, {
        "family": "D1", "condition": "A-L", "model": "direct_marker_earliest"
    }))

    # D2 marker-only LR on A-L
    xtr = marker_features(ac_train["text_leaky"])
    xte = marker_features(ac_test["text_leaky"])
    d2 = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000)
    d2.fit(xtr, ac_train["class"])
    pred = d2.predict(xte)
    results.append(save_run(run_root, "D2_marker_only_LR_AL", ac_test, pred, {
        "family": "D2", "condition": "A-L", "model": "marker_only_logreg",
        "feature_order": list(MARKER_PATTERNS)
    }))

    # D3 surface-only LR on A-C/G-C
    for cond, tr, te in [("AC", ac_train, ac_test), ("GC", gc_train, gc_test)]:
        d3 = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000)),
        ])
        d3.fit(surface_features(tr["text_clean"]), tr["class"])
        pred = d3.predict(surface_features(te["text_clean"]))
        results.append(save_run(run_root, f"D3_surface_only_LR_{cond}", te, pred, {
            "family": "D3", "condition": cond, "model": "surface_only_logreg",
            "features": [
                "char_len","whitespace_token_count","url_count","user_count",
                "hashtag_count","rt_prefix","exclamation_count","question_count","digit_count"
            ],
        }))

    # S1 supplementary
    for cond, tr, te in [("AC", ac_train, ac_test), ("GC", gc_train, gc_test)]:
        s1 = Pipeline([
            ("tfidf", make_word_tfidf()),
            ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000)),
        ])
        s1.fit(tr["text_clean"], tr["class"])
        pred = s1.predict(te["text_clean"])
        results.append(save_run(run_root, f"S1_word_tfidf_LR_{cond}", te, pred, {
            "family": "S1", "condition": cond, "model": "word_tfidf_logreg"
        }))

    # S2 required
    for cond, tr, te in [("AC", ac_train, ac_test), ("GC", gc_train, gc_test)]:
        s2 = Pipeline([
            ("tfidf", make_word_tfidf()),
            ("clf", LinearSVC(C=0.5)),
        ])
        s2.fit(tr["text_clean"], tr["class"])
        pred = s2.predict(te["text_clean"])
        results.append(save_run(run_root, f"S2_word_tfidf_SVC_{cond}", te, pred, {
            "family": "S2", "condition": cond, "model": "word_tfidf_linearsvc", "C": 0.5
        }))

    # S3 optional, attempted because CPU cost is modest; a failure does not invalidate S2.
    s3_status = "NOT_ATTEMPTED"
    try:
        for cond, tr, te in [("AC", ac_train, ac_test), ("GC", gc_train, gc_test)]:
            s3 = Pipeline([
                ("tfidf", TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=200000,
                    lowercase=True,
                )),
                ("clf", LinearSVC(C=0.5)),
            ])
            s3.fit(tr["text_clean"], tr["class"])
            pred = s3.predict(te["text_clean"])
            results.append(save_run(run_root, f"S3_char_tfidf_SVC_{cond}", te, pred, {
                "family": "S3", "condition": cond, "model": "char_wb_tfidf_linearsvc",
                "C": 0.5, "ngram_range": [3, 5]
            }))
        s3_status = "PASS"
    except Exception as exc:
        s3_status = f"OPTIONAL_FAILED: {type(exc).__name__}: {exc}"
        (report_root / "S3_OPTIONAL_FAILURE.txt").write_text(traceback.format_exc(), encoding="utf-8")

    rows = []
    for item in results:
        row = {
            "run": item["run"],
            "family": item.get("family"),
            "condition": item.get("condition"),
            **item["metrics"],
            "irony_to_sarcasm": item["irony_to_sarcasm"],
            "sarcasm_to_irony": item["sarcasm_to_irony"],
            "test_n": item["test_n"],
            "test_id_sha256": item["test_id_sha256"],
        }
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(report_root / "H06_RESULTS.tsv", sep="\t", index=False)

    required_runs = {
        "D0_majority_AC",
        "D0_stratified_random_AC",
        "D0_majority_GC",
        "D0_stratified_random_GC",
        "D1_direct_marker_AL",
        "D2_marker_only_LR_AL",
        "D3_surface_only_LR_AC",
        "D3_surface_only_LR_GC",
        "S2_word_tfidf_SVC_AC",
        "S2_word_tfidf_SVC_GC",
    }
    observed = set(summary_df["run"].tolist())
    required_ok = required_runs.issubset(observed)

    status = {
        "status": "PASS" if required_ok else "INCOMPLETE",
        "required_runs_complete": bool(required_ok),
        "s3_status": s3_status,
        "n_runs": int(len(summary_df)),
        "condition_invariants": invariant,
    }
    (report_root / "H06_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    md = ["# H06 Sparse/Diagnostic Baselines", "", f"- status: **{status['status']}**",
          f"- runs produced: {len(summary_df)}", f"- S3: {s3_status}", "",
          "## Results", "", summary_df.to_markdown(index=False)]
    (report_root / "H06_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print((report_root / "H06_SUMMARY.md").read_text(encoding="utf-8"))
    return status


def main():
    project = Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))
    try:
        run_all(project)
    except Exception:
        report_root = project / "reports/h06_sparse_baselines"
        report_root.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (report_root / "H06_FAILURE.txt").write_text(tb, encoding="utf-8")
        (report_root / "H06_STATUS.json").write_text(
            json.dumps({"status": "BLOCKED", "reason": "unhandled_exception"}, indent=2),
            encoding="utf-8",
        )
        print("H06 encountered an operational blocker. Handoff will preserve the traceback.")
        print(tb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())