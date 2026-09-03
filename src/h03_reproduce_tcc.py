#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABELS = ["irony", "sarcasm", "regular"]
SOURCE_LABELS = ["figurative", "irony", "sarcasm", "regular"]
MARKERS = ["#irony", "#ironic", "#sarcasm", "#sarcastic"]
URL_RE = re.compile(r"(?i)\b(?:https?://\S+|www\.\S+)")
USER_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
# Marker alternatives are sorted by length only for clarity. Word boundary prevents partial matches.
MARKER_RE = re.compile(r"(?i)(?<!\w)#(?:sarcastic|sarcasm|ironic|irony)\b")
WS_RE = re.compile(r"\s+")

EXPECTED_FUNNEL = {
    "raw": (81408, 8128),
    "missing_removed": (81408, 8119),
    "figurative_removed": (60170, 6075),
    "initial_exact_duplicates_removed": (60124, 6075),
    "initial_cross_split_overlap_removed": (60124, 6072),
    "text_cleaning": (60118, 6072),
    "conflicts_removed": (60104, 6072),
    "normalized_deduplication": (58784, 6009),
    "final_cross_split_overlap_removed": (58784, 5947),
}
EXPECTED_FINAL_CLASSES = {
    "train": {"irony": 20805, "sarcasm": 20646, "regular": 17333},
    "test": {"irony": 2096, "sarcasm": 2102, "regular": 1749},
}
EXPECTED_MARKER_PREVALENCE = {"irony": 0.9996, "sarcasm": 0.9998, "regular": 0.0}
EXPECTED_VALIDATION = {
    ((1, 1), 0.5): (0.8005, 0.8078),
    ((1, 1), 1.0): (0.7948, 0.8027),
    ((1, 1), 2.0): (0.7859, 0.7943),
    ((1, 2), 0.5): (0.8153, 0.8222),
    ((1, 2), 1.0): (0.8100, 0.8173),
    ((1, 2), 2.0): (0.8027, 0.8105),
}
EXPECTED_FINAL = {
    "accuracy": 0.8189,
    "macro_precision": 0.8232,
    "macro_recall": 0.8287,
    "macro_f1": 0.8256,
}
EXPECTED_CLASS_F1 = {"irony": 0.7557, "sarcasm": 0.7530, "regular": 0.9680}
EXPECTED_LEAKY_ACCURACY = 0.9997
EXPECTED_BINARY = {"accuracy": 0.7592, "macro_f1": 0.7591}
EXPECTED_CM = np.array([[1584, 456, 56], [506, 1547, 49], [6, 4, 1739]])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_base(text: str) -> str:
    text = str(text)
    text = URL_RE.sub("<URL>", text)
    text = USER_RE.sub("<USER>", text)
    return WS_RE.sub(" ", text).strip()


def remove_class_markers(text: str) -> str:
    text = MARKER_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def marker_present(text: str) -> bool:
    return bool(MARKER_RE.search(str(text)))


def valid_text(x) -> bool:
    if pd.isna(x):
        return False
    return bool(str(x).strip())


def canonical_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def locate_csvs(project: Path) -> Tuple[Path, Path, Path]:
    ptr = project / "data/raw/kaggle_tweets_sarcasm_irony/CURRENT_PATH.txt"
    if ptr.exists():
        root = Path(ptr.read_text(encoding="utf-8").strip())
    else:
        root = project / "data/raw/kaggle_tweets_sarcasm_irony"
    trains = list(root.rglob("train.csv"))
    tests = list(root.rglob("test.csv"))
    if not trains or not tests:
        raise FileNotFoundError(f"train.csv/test.csv not found under {root}")
    return root, trains[0], tests[0]


def add_source_ids(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["_source_split"] = split
    df["_source_row"] = np.arange(len(df), dtype=np.int64)
    df["_id"] = [f"{split}:{i:08d}" for i in range(len(df))]
    return df


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["class"].value_counts(dropna=False).sort_index().items()}


def drop_conflicting_normalized(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = df.groupby("text_clean", dropna=False)["class"].nunique()
    bad = set(g[g > 1].index.tolist())
    removed = df[df["text_clean"].isin(bad)].copy()
    kept = df[~df["text_clean"].isin(bad)].copy()
    return kept, removed


def build_clean_dataset(raw_train: pd.DataFrame, raw_test: pd.DataFrame, out_dir: Path):
    funnel = []
    details = {}

    train = add_source_ids(raw_train, "train")
    test = add_source_ids(raw_test, "test")

    for df in (train, test):
        df["class"] = df["class"].map(canonical_label)

    funnel.append(("raw", len(train), len(test)))
    details["raw_train_class_counts"] = class_counts(train)
    details["raw_test_class_counts"] = class_counts(test)

    # Marker prevalence before marker removal, for target labels only.
    marker_rows = []
    for split, df in [("train", train), ("test", test), ("combined", pd.concat([train, test], ignore_index=True))]:
        for label in LABELS:
            sub = df[df["class"] == label]
            denom = len(sub)
            num = int(sub["tweets"].map(marker_present).sum()) if denom else 0
            marker_rows.append({"split": split, "class": label, "n": denom, "marker_n": num, "prevalence": num / denom if denom else np.nan})
    marker_df = pd.DataFrame(marker_rows)

    # 1) remove missing/invalid text or class. TCC's 9 test items had no valid class.
    def filter_valid(df):
        mask = df["tweets"].map(valid_text) & df["class"].isin(SOURCE_LABELS)
        return df[mask].copy(), df[~mask].copy()
    train, invalid_train = filter_valid(train)
    test, invalid_test = filter_valid(test)
    funnel.append(("missing_removed", len(train), len(test)))

    # 2) primary task excludes figurative.
    train = train[train["class"].isin(LABELS)].copy()
    test = test[test["class"].isin(LABELS)].copy()
    funnel.append(("figurative_removed", len(train), len(test)))

    # 3) initial exact duplicates: exact row content (tweet + label) within split.
    dup_train_mask = train.duplicated(subset=["tweets", "class"], keep="first")
    dup_test_mask = test.duplicated(subset=["tweets", "class"], keep="first")
    initial_dup_train = train[dup_train_mask].copy()
    initial_dup_test = test[dup_test_mask].copy()
    train = train[~dup_train_mask].copy()
    test = test[~dup_test_mask].copy()
    funnel.append(("initial_exact_duplicates_removed", len(train), len(test)))

    # 4) exact raw text overlap, remove from test only.
    raw_train_text = set(train["tweets"].astype(str).tolist())
    overlap0_mask = test["tweets"].astype(str).isin(raw_train_text)
    initial_overlap = test[overlap0_mask].copy()
    test = test[~overlap0_mask].copy()
    funnel.append(("initial_cross_split_overlap_removed", len(train), len(test)))

    # 5) conservative normalization and marker removal, with matched leaky view.
    for df in (train, test):
        df["text_leaky"] = df["tweets"].map(normalize_base)
        df["text_clean"] = df["text_leaky"].map(remove_class_markers)
    empty_train = train[train["text_clean"].eq("")].copy()
    empty_test = test[test["text_clean"].eq("")].copy()
    train = train[~train["text_clean"].eq("")].copy()
    test = test[~test["text_clean"].eq("")].copy()
    funnel.append(("text_cleaning", len(train), len(test)))

    # 6) conflicts after normalization, within split.
    train, conflict_train = drop_conflicting_normalized(train)
    test, conflict_test = drop_conflicting_normalized(test)
    funnel.append(("conflicts_removed", len(train), len(test)))

    # 7) normalized deduplication, within split.
    norm_dup_train_mask = train.duplicated(subset=["text_clean"], keep="first")
    norm_dup_test_mask = test.duplicated(subset=["text_clean"], keep="first")
    norm_dup_train = train[norm_dup_train_mask].copy()
    norm_dup_test = test[norm_dup_test_mask].copy()
    train = train[~norm_dup_train_mask].copy()
    test = test[~norm_dup_test_mask].copy()
    funnel.append(("normalized_deduplication", len(train), len(test)))

    # 8) final overlap on normalized clean text, remove from test.
    clean_train_text = set(train["text_clean"].tolist())
    overlap1_mask = test["text_clean"].isin(clean_train_text)
    final_overlap = test[overlap1_mask].copy()
    test = test[~overlap1_mask].copy()
    funnel.append(("final_cross_split_overlap_removed", len(train), len(test)))

    # Preserve only useful columns in final processed files.
    final_cols = ["_id", "_source_split", "_source_row", "class", "tweets", "text_leaky", "text_clean"]
    train = train[final_cols].reset_index(drop=True)
    test = test[final_cols].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train_final.csv", index=False)
    test.to_csv(out_dir / "test_final.csv", index=False)
    marker_df.to_csv(out_dir / "marker_prevalence.tsv", sep="\t", index=False)

    removed = {
        "invalid_train": invalid_train,
        "invalid_test": invalid_test,
        "initial_duplicate_train": initial_dup_train,
        "initial_duplicate_test": initial_dup_test,
        "initial_overlap_test": initial_overlap,
        "empty_after_clean_train": empty_train,
        "empty_after_clean_test": empty_test,
        "conflict_train": conflict_train,
        "conflict_test": conflict_test,
        "normalized_duplicate_train": norm_dup_train,
        "normalized_duplicate_test": norm_dup_test,
        "final_overlap_test": final_overlap,
    }
    removed_counts = {k: int(len(v)) for k, v in removed.items()}
    pd.DataFrame([{"category": k, "n": v} for k, v in removed_counts.items()]).to_csv(out_dir / "removed_counts.tsv", sep="\t", index=False)

    funnel_df = pd.DataFrame(funnel, columns=["stage", "train_n", "test_n"])
    funnel_df.to_csv(out_dir / "funnel.tsv", sep="\t", index=False)
    return train, test, funnel_df, marker_df, removed_counts, details


def make_pipeline(ngram_range=(1, 2), C=0.5):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=tuple(ngram_range),
            min_df=2,
            sublinear_tf=True,
            max_features=200000,
            lowercase=True,
        )),
        ("clf", LinearSVC(C=float(C))),
    ])


def aggregate_metrics(y_true, y_pred, labels=LABELS):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def evaluate_model(train_df, test_df, text_col, ngram_range, C, run_dir: Path, name: str, labels=LABELS):
    pipe = make_pipeline(ngram_range, C)
    pipe.fit(train_df[text_col], train_df["class"])
    pred = pipe.predict(test_df[text_col])
    metrics = aggregate_metrics(test_df["class"], pred, labels=labels)
    report = classification_report(test_df["class"], pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_df["class"], pred, labels=labels)
    pred_df = pd.DataFrame({"_id": test_df["_id"].values, "y_true": test_df["class"].values, "y_pred": pred})
    pred_df.to_csv(run_dir / f"predictions_{name}.csv", index=False)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(run_dir / f"confusion_{name}.csv")
    with (run_dir / f"classification_report_{name}.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return metrics, report, cm, pred_df


def validation_grid(train: pd.DataFrame, run_dir: Path):
    tr, dev = train_test_split(train, test_size=0.20, random_state=42, stratify=train["class"])
    rows = []
    for ngram_range in [(1, 1), (1, 2)]:
        for C in [0.5, 1.0, 2.0]:
            pipe = make_pipeline(ngram_range, C)
            pipe.fit(tr["text_clean"], tr["class"])
            pred = pipe.predict(dev["text_clean"])
            rows.append({
                "ngram_min": ngram_range[0],
                "ngram_max": ngram_range[1],
                "C": C,
                "accuracy": float(accuracy_score(dev["class"], pred)),
                "macro_f1": float(f1_score(dev["class"], pred, average="macro", zero_division=0)),
            })
    df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    df.to_csv(run_dir / "validation_grid.tsv", sep="\t", index=False)
    best = df.iloc[0]
    return df, (int(best["ngram_min"]), int(best["ngram_max"])), float(best["C"])


def divergence_rows(funnel_df, train, test, marker_df, val_df, clean_metrics, clean_report, clean_cm, leaky_metrics, binary_metrics):
    rows = []
    def add(kind, item, observed, expected, tolerance=0.0):
        try:
            delta = float(observed) - float(expected)
            ok = abs(delta) <= tolerance
        except Exception:
            delta = ""
            ok = observed == expected
        rows.append({"kind": kind, "item": item, "observed": observed, "expected": expected, "delta": delta, "within_tolerance": ok})

    for _, r in funnel_df.iterrows():
        exp = EXPECTED_FUNNEL.get(r["stage"])
        if exp:
            add("funnel", f"{r['stage']}:train", int(r["train_n"]), exp[0])
            add("funnel", f"{r['stage']}:test", int(r["test_n"]), exp[1])
    for split, df in [("train", train), ("test", test)]:
        obs = df["class"].value_counts().to_dict()
        for label, exp in EXPECTED_FINAL_CLASSES[split].items():
            add("class_count", f"{split}:{label}", int(obs.get(label, 0)), exp)

    combined = marker_df[marker_df["split"] == "combined"].set_index("class")
    for label, exp in EXPECTED_MARKER_PREVALENCE.items():
        if label in combined.index:
            add("marker_prevalence", label, float(combined.loc[label, "prevalence"]), exp, tolerance=0.0001)

    for key, (exp_acc, exp_f1) in EXPECTED_VALIDATION.items():
        ng, C = key
        row = val_df[(val_df.ngram_min == ng[0]) & (val_df.ngram_max == ng[1]) & (np.isclose(val_df.C, C))]
        if len(row):
            rr = row.iloc[0]
            add("validation", f"ngram={ng},C={C}:accuracy", float(rr.accuracy), exp_acc, tolerance=0.0015)
            add("validation", f"ngram={ng},C={C}:macro_f1", float(rr.macro_f1), exp_f1, tolerance=0.0015)

    for k, exp in EXPECTED_FINAL.items():
        add("final_clean", k, clean_metrics[k], exp, tolerance=0.0015)
    for label, exp in EXPECTED_CLASS_F1.items():
        add("final_clean_class", f"{label}:f1", float(clean_report[label]["f1-score"]), exp, tolerance=0.0015)
    add("final_clean", "confusion_matrix_exact", json.dumps(clean_cm.tolist()), json.dumps(EXPECTED_CM.tolist()))
    add("leaky", "accuracy", leaky_metrics["accuracy"], EXPECTED_LEAKY_ACCURACY, tolerance=0.0010)
    for k, exp in EXPECTED_BINARY.items():
        add("binary", k, binary_metrics[k], exp, tolerance=0.0015)
    return pd.DataFrame(rows)


def render_summary(path: Path, root: Path, train_csv: Path, test_csv: Path, funnel_df, marker_df, val_df, best_ngram, best_C, clean_metrics, clean_report, clean_cm, leaky_metrics, binary_metrics, divergences, removed_counts):
    lines = []
    lines.append("# H03 TCC reproduction summary")
    lines.append("")
    lines.append(f"- Raw root: `{root}`")
    lines.append(f"- train SHA-256: `{sha256_file(train_csv)}`")
    lines.append(f"- test SHA-256: `{sha256_file(test_csv)}`")
    lines.append(f"- Selected clean validation configuration: ngram={best_ngram}, C={best_C}")
    lines.append("")
    lines.append("## Cleaning funnel")
    lines.append("")
    lines.append(funnel_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Marker prevalence")
    lines.append("")
    lines.append(marker_df[marker_df.split == "combined"].to_markdown(index=False))
    lines.append("")
    lines.append("## Validation grid")
    lines.append("")
    lines.append(val_df.sort_values(["ngram_min","ngram_max","C"]).to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Final clean test")
    lines.append("")
    for k,v in clean_metrics.items(): lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("Per-class F1:")
    for label in LABELS: lines.append(f"- {label}: {clean_report[label]['f1-score']:.6f}")
    lines.append("")
    lines.append("Confusion matrix (rows=true; columns=predicted; irony/sarcasm/regular):")
    lines.append("")
    lines.append("```text")
    lines.extend(" ".join(map(str,row)) for row in clean_cm.tolist())
    lines.append("```")
    lines.append("")
    lines.append("## Matched leaky test")
    for k,v in leaky_metrics.items(): lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Binary irony vs sarcasm")
    for k,v in binary_metrics.items(): lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Removed-item counts")
    for k,v in removed_counts.items(): lines.append(f"- {k}: {v}")
    lines.append("")
    nbad = int((~divergences["within_tolerance"].astype(bool)).sum())
    lines.append("## Reproduction verdict")
    lines.append("")
    lines.append(f"- Checks outside declared tolerance: **{nbad} / {len(divergences)}**")
    lines.append("- A nonzero count is an audit result, not a failure to be hidden.")
    lines.append("")
    lines.append("See `divergences.tsv` for every target comparison and `predictions_*.csv` for instance-level outputs.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))))
    args = ap.parse_args()
    project = Path(args.project).expanduser().resolve()
    root, train_csv, test_csv = locate_csvs(project)
    run_dir = project / "runs/h03_tcc_reproduction"
    proc_dir = project / "data/processed/h03_tcc_reproduction"
    report_dir = project / "reports/h03_tcc_reproduction"
    for d in (run_dir, proc_dir, report_dir): d.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_csv(train_csv)
    raw_test = pd.read_csv(test_csv)
    for col in ["tweets", "class"]:
        if col not in raw_train.columns or col not in raw_test.columns:
            raise ValueError(f"Required column {col!r} missing. train={raw_train.columns.tolist()} test={raw_test.columns.tolist()}")

    train, test, funnel_df, marker_df, removed_counts, details = build_clean_dataset(raw_train, raw_test, proc_dir)
    val_df, best_ngram, best_C = validation_grid(train, run_dir)

    clean_metrics, clean_report, clean_cm, clean_pred = evaluate_model(train, test, "text_clean", best_ngram, best_C, run_dir, "clean")
    leaky_metrics, leaky_report, leaky_cm, leaky_pred = evaluate_model(train, test, "text_leaky", best_ngram, best_C, run_dir, "leaky_matched")

    btrain = train[train["class"].isin(["irony", "sarcasm"])].copy()
    btest = test[test["class"].isin(["irony", "sarcasm"])].copy()
    binary_metrics, binary_report, binary_cm, binary_pred = evaluate_model(btrain, btest, "text_clean", best_ngram, best_C, run_dir, "binary_clean", labels=["irony","sarcasm"])

    divergences = divergence_rows(funnel_df, train, test, marker_df, val_df, clean_metrics, clean_report, clean_cm, leaky_metrics, binary_metrics)
    divergences.to_csv(report_dir / "divergences.tsv", sep="\t", index=False)

    metrics = {
        "best_validation": {"ngram_range": list(best_ngram), "C": best_C},
        "clean": clean_metrics,
        "leaky_matched": leaky_metrics,
        "binary_clean": binary_metrics,
        "final_train_n": int(len(train)),
        "final_test_n": int(len(test)),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "raw_class_counts.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")

    render_summary(report_dir / "H03_SUMMARY.md", root, train_csv, test_csv, funnel_df, marker_df, val_df, best_ngram, best_C, clean_metrics, clean_report, clean_cm, leaky_metrics, binary_metrics, divergences, removed_counts)

    print((report_dir / "H03_SUMMARY.md").read_text(encoding="utf-8"))
    print("Artifacts:")
    print(" ", report_dir)
    print(" ", run_dir)
    print(" ", proc_dir)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        # Preserve the traceback while keeping the research runner recoverable.
        import traceback
        traceback.print_exc()
        print(f"\nH03_OPERATIONAL_ERROR: {type(e).__name__}: {e}")
        raise SystemExit(0)