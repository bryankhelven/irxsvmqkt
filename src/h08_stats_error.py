#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["irony", "sarcasm", "regular"]
L2I = {x: i for i, x in enumerate(LABELS)}
BOOT_N = 10000
BOOT_SEED = 20260902
PILOT_SEED = 20260902


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"_id": str})
    needed = ["_id", "y_true", "y_pred"]
    if list(df.columns) != needed:
        raise RuntimeError(f"Unexpected prediction columns for {path}: {df.columns.tolist()}")
    if len(df) != 5947:
        raise RuntimeError(f"Unexpected prediction row count for {path}: {len(df)}")
    if df["_id"].duplicated().any():
        raise RuntimeError(f"Duplicate test IDs in {path}")
    bad_gold = sorted(set(df["y_true"]) - set(LABELS))
    bad_pred = sorted(set(df["y_pred"]) - set(LABELS))
    if bad_gold or bad_pred:
        raise RuntimeError(f"Unknown labels in {path}: gold={bad_gold} pred={bad_pred}")
    return df


def confusion(df: pd.DataFrame) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for y, p in zip(df["y_true"], df["y_pred"]):
        cm[L2I[y], L2I[p]] += 1
    return cm


def macro_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    pred = cm.sum(axis=-2)
    true = cm.sum(axis=-1)
    precision = np.divide(tp, pred, out=np.zeros_like(tp, dtype=float), where=pred != 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp, dtype=float), where=true != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=float),
        where=(precision + recall) != 0,
    )
    return f1.mean(axis=-1)


def accuracy_from_cm(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum())


def class_f1_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    true = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros(3), where=pred != 0)
    recall = np.divide(tp, true, out=np.zeros(3), where=true != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) != 0)
    return {LABELS[i]: float(f1[i]) for i in range(3)}


def single_bootstrap(df: pd.DataFrame, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    cm = confusion(df)
    draws = np.empty((n, 3, 3), dtype=np.int64)
    for i in range(3):
        probs = cm[i].astype(float) / cm[i].sum()
        draws[:, i, :] = rng.multinomial(int(cm[i].sum()), probs, size=n)
    vals = macro_f1_from_cm(draws)
    return {
        "point": float(macro_f1_from_cm(cm)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def assert_pairable(a: pd.DataFrame, b: pd.DataFrame):
    if a["_id"].tolist() != b["_id"].tolist():
        raise RuntimeError("Paired bootstrap refused: ordered IDs differ")
    if a["y_true"].tolist() != b["y_true"].tolist():
        raise RuntimeError("Paired bootstrap refused: y_true differs")


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n=BOOT_N, seed=BOOT_SEED):
    assert_pairable(a, b)
    rng = np.random.default_rng(seed)
    cms_a = np.zeros((n, 3, 3), dtype=np.int64)
    cms_b = np.zeros((n, 3, 3), dtype=np.int64)

    for ti, true_label in enumerate(LABELS):
        mask = a["y_true"].eq(true_label).to_numpy()
        pa = a.loc[mask, "y_pred"].tolist()
        pb = b.loc[mask, "y_pred"].tolist()
        joint = np.zeros((3, 3), dtype=np.int64)
        for xa, xb in zip(pa, pb):
            joint[L2I[xa], L2I[xb]] += 1
        flat = joint.ravel()
        probs = flat.astype(float) / flat.sum()
        draws = rng.multinomial(int(flat.sum()), probs, size=n).reshape(n, 3, 3)
        cms_a[:, ti, :] = draws.sum(axis=2)
        cms_b[:, ti, :] = draws.sum(axis=1)

    va = macro_f1_from_cm(cms_a)
    vb = macro_f1_from_cm(cms_b)
    delta = va - vb
    point = float(macro_f1_from_cm(confusion(a)) - macro_f1_from_cm(confusion(b)))
    return {
        "delta_point": point,
        "ci_low": float(np.quantile(delta, 0.025)),
        "ci_high": float(np.quantile(delta, 0.975)),
    }


def id_sha(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(df["_id"].astype(str)).encode("utf-8")).hexdigest()


def run_paths(project: Path):
    return {
        "D3_AC": project / "runs/h06_sparse_baselines/D3_surface_only_LR_AC/predictions.csv",
        "D3_GC": project / "runs/h06_sparse_baselines/D3_surface_only_LR_GC/predictions.csv",
        "S1_AC": project / "runs/h06_sparse_baselines/S1_word_tfidf_LR_AC/predictions.csv",
        "S1_GC": project / "runs/h06_sparse_baselines/S1_word_tfidf_LR_GC/predictions.csv",
        "S2_AC": project / "runs/h06_sparse_baselines/S2_word_tfidf_SVC_AC/predictions.csv",
        "S2_GC": project / "runs/h06_sparse_baselines/S2_word_tfidf_SVC_GC/predictions.csv",
        "S3_AC": project / "runs/h06_sparse_baselines/S3_char_tfidf_SVC_AC/predictions.csv",
        "S3_GC": project / "runs/h06_sparse_baselines/S3_char_tfidf_SVC_GC/predictions.csv",
        "H03_LEAKY": project / "runs/h03_tcc_reproduction/predictions_leaky_matched.csv",
        "H03_CLEAN": project / "runs/h03_tcc_reproduction/predictions_clean.csv",
        "BERT_AC42": project / "runs/h07_bertweet/N1_BERTweet_AC_seed42/predictions_AC.csv",
        "BERT_AC2026": project / "runs/h07_bertweet/N1_BERTweet_AC_seed2026/predictions_AC.csv",
        "BERT_AC7": project / "runs/h07_bertweet/N1_BERTweet_AC_seed7/predictions_AC.csv",
        "BERT_GC42": project / "runs/h07_bertweet/N1_BERTweet_GC_seed42/predictions_GC.csv",
        "BERT_GC2026": project / "runs/h07_bertweet/N1_BERTweet_GC_seed2026/predictions_GC.csv",
        "BERT_GC7": project / "runs/h07_bertweet/N1_BERTweet_GC_seed7/predictions_GC.csv",
        "BERT_AL_AL": project / "runs/h07_bertweet/N1_BERTweet_AL_seed42/predictions_AL.csv",
        "BERT_AL_AC": project / "runs/h07_bertweet/N1_BERTweet_AL_seed42/predictions_AC.csv",
    }


def deterministic_sample(df: pd.DataFrame, n: int, rng: np.random.Generator):
    if len(df) <= n:
        return df.copy()
    idx = np.sort(rng.choice(np.arange(len(df)), size=n, replace=False))
    return df.iloc[idx].copy()


def build_error_pilot(project: Path, preds: dict, outdir: Path):
    test_path = project / "data/processed/h03_tcc_reproduction/test_final.csv"
    if not test_path.exists():
        raise RuntimeError(f"Canonical A-C test text missing: {test_path}")
    test = pd.read_csv(test_path, dtype={"_id": str})
    needed = {"_id", "class", "text_clean"}
    if not needed.issubset(test.columns):
        raise RuntimeError(f"A-C test text missing columns: {sorted(needed - set(test.columns))}")
    if len(test) != 5947 or test["_id"].duplicated().any():
        raise RuntimeError("Canonical A-C test text integrity failure")

    canonical_ids = preds["S2_AC"]["_id"].tolist()
    if test["_id"].tolist() != canonical_ids:
        # preserve canonical prediction order after strict set check
        if set(test["_id"]) != set(canonical_ids):
            raise RuntimeError("A-C test text IDs differ from frozen predictions")
        test = test.set_index("_id").loc[canonical_ids].reset_index()

    text_map = dict(zip(test["_id"], test["text_clean"].astype(str)))
    rng = np.random.default_rng(PILOT_SEED)
    selected_rows = []

    model_specs = [("S2_AC", "S2"), ("BERT_AC42", "BERTweet_seed42")]
    for pred_key, model_name in model_specs:
        d = preds[pred_key].copy()
        strata = {
            "irony_to_sarcasm": d[(d.y_true == "irony") & (d.y_pred == "sarcasm")],
            "sarcasm_to_irony": d[(d.y_true == "sarcasm") & (d.y_pred == "irony")],
            "figurative_to_regular_like": d[(d.y_true.isin(["irony", "sarcasm"])) & (d.y_pred == "regular")],
            "regular_false_positive": d[(d.y_true == "regular") & (d.y_pred.isin(["irony", "sarcasm"]))],
        }
        for stratum, sub in strata.items():
            take = min(8, len(sub))
            sample = deterministic_sample(sub, take, rng)
            for _, row in sample.iterrows():
                selected_rows.append({
                    "_id": row["_id"],
                    "gold_label": row["y_true"],
                    "predicted_label": row["y_pred"],
                    "source_model": model_name,
                    "stratum": stratum,
                })

    key = pd.DataFrame(selected_rows)
    # stable case ID by first appearance of unique _id
    unique_ids = []
    seen = set()
    for rid in key["_id"]:
        if rid not in seen:
            seen.add(rid)
            unique_ids.append(rid)
    case_map = {rid: f"E{idx:03d}" for idx, rid in enumerate(unique_ids, start=1)}
    key.insert(0, "case_id", key["_id"].map(case_map))
    key["text_sha256"] = key["_id"].map(lambda rid: hashlib.sha256(text_map[rid].encode("utf-8")).hexdigest())

    blind_rows = []
    for rid in unique_ids:
        # gold label is invariant across key rows
        gold = key.loc[key["_id"] == rid, "gold_label"].iloc[0]
        blind_rows.append({"case_id": case_map[rid], "gold_label": gold, "text_clean": text_map[rid]})
    blind = pd.DataFrame(blind_rows)

    annotation = blind[["case_id", "gold_label", "text_clean"]].copy()
    annotation["primary_category"] = ""
    annotation["secondary_category"] = ""
    annotation["confidence"] = ""
    annotation["notes"] = ""

    blind.to_csv(outdir / "ERROR_PILOT_BLIND.tsv", sep="\t", index=False)
    key.to_csv(outdir / "ERROR_PILOT_KEY.tsv", sep="\t", index=False)
    annotation.to_csv(outdir / "ERROR_PILOT_ANNOTATION_TEMPLATE.tsv", sep="\t", index=False)

    return {
        "selected_key_rows": int(len(key)),
        "unique_blind_cases": int(len(blind)),
        "source_counts": key.groupby(["source_model", "stratum"]).size().reset_index(name="n").to_dict(orient="records"),
    }


def main():
    project = Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))
    outdir = project / "reports/h08_stats_error"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        paths = run_paths(project)
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise RuntimeError("Missing frozen prediction files:\n" + "\n".join(missing))

        preds = {k: load_predictions(p) for k, p in paths.items()}
        canonical = preds["S2_AC"]
        for k, d in preds.items():
            assert_pairable(canonical, d)

        inventory = []
        for k, p in paths.items():
            d = preds[k]
            inventory.append({
                "prediction_key": k,
                "path": str(p),
                "sha256": file_sha256(p),
                "n": len(d),
                "test_id_sha256": id_sha(d),
            })
        pd.DataFrame(inventory).to_csv(outdir / "PREDICTION_INVENTORY.tsv", sep="\t", index=False)

        # Individual CIs.
        single_keys = [
            "D3_AC", "D3_GC", "S1_AC", "S1_GC", "S2_AC", "S2_GC", "S3_AC", "S3_GC",
            "H03_LEAKY", "H03_CLEAN",
            "BERT_AC42", "BERT_AC2026", "BERT_AC7",
            "BERT_GC42", "BERT_GC2026", "BERT_GC7",
            "BERT_AL_AL", "BERT_AL_AC",
        ]
        singles = []
        for k in single_keys:
            b = single_bootstrap(preds[k])
            cm = confusion(preds[k])
            singles.append({
                "prediction_key": k,
                "macro_f1": b["point"],
                "ci_low": b["ci_low"],
                "ci_high": b["ci_high"],
                "accuracy": accuracy_from_cm(cm),
                "irony_to_sarcasm": int(cm[0, 1]),
                "sarcasm_to_irony": int(cm[1, 0]),
                "irony_sarcasm_to_regular": int(cm[0, 2] + cm[1, 2]),
                "regular_false_positive": int(cm[2, 0] + cm[2, 1]),
            })
        singles_df = pd.DataFrame(singles)
        singles_df.to_csv(outdir / "BOOTSTRAP_SINGLE_CI.tsv", sep="\t", index=False)

        pairs = [
            ("H03_LEAKY", "H03_CLEAN", "sparse_matched_leaky_minus_clean"),
            ("BERT_AC42", "S2_AC", "bert_ac42_minus_s2_ac"),
            ("S2_GC", "S2_AC", "s2_gc_minus_ac"),
            ("S3_GC", "S3_AC", "s3_gc_minus_ac"),
            ("D3_GC", "D3_AC", "surface_gc_minus_ac"),
            ("BERT_GC42", "BERT_AC42", "bert_gc_minus_ac_seed42"),
            ("BERT_GC2026", "BERT_AC2026", "bert_gc_minus_ac_seed2026"),
            ("BERT_GC7", "BERT_AC7", "bert_gc_minus_ac_seed7"),
            ("BERT_AL_AL", "BERT_AL_AC", "same_leaky_trained_model_al_minus_ac_eval"),
        ]
        deltas = []
        for a, b, name in pairs:
            x = paired_bootstrap(preds[a], preds[b])
            deltas.append({"comparison": name, "A": a, "B": b, **x})
        delta_df = pd.DataFrame(deltas)
        delta_df.to_csv(outdir / "PAIRED_DELTA_BOOTSTRAP.tsv", sep="\t", index=False)

        # Neural seed + per-class summaries.
        seed_rows = []
        class_rows = []
        for cond, keys in {
            "AC": ["BERT_AC42", "BERT_AC2026", "BERT_AC7"],
            "GC": ["BERT_GC42", "BERT_GC2026", "BERT_GC7"],
        }.items():
            mf = [float(macro_f1_from_cm(confusion(preds[k]))) for k in keys]
            ac = [accuracy_from_cm(confusion(preds[k])) for k in keys]
            seed_rows.append({
                "condition": cond,
                "n_seeds": 3,
                "macro_f1_mean": float(np.mean(mf)),
                "macro_f1_sample_sd": float(np.std(mf, ddof=1)),
                "accuracy_mean": float(np.mean(ac)),
                "accuracy_sample_sd": float(np.std(ac, ddof=1)),
                "seeds": "42;2026;7",
            })
            class_by_seed = [class_f1_from_cm(confusion(preds[k])) for k in keys]
            for cls in LABELS:
                vals = [x[cls] for x in class_by_seed]
                class_rows.append({
                    "condition": cond,
                    "class": cls,
                    "f1_mean": float(np.mean(vals)),
                    "f1_sample_sd": float(np.std(vals, ddof=1)),
                })
        seed_df = pd.DataFrame(seed_rows)
        class_df = pd.DataFrame(class_rows)
        seed_df.to_csv(outdir / "NEURAL_SEED_SUMMARY.tsv", sep="\t", index=False)
        class_df.to_csv(outdir / "NEURAL_PER_CLASS_SUMMARY.tsv", sep="\t", index=False)

        # S2 vs BERT seed42 correctness overlap.
        s2 = preds["S2_AC"]
        bert = preds["BERT_AC42"]
        s2_ok = s2["y_true"].eq(s2["y_pred"]).to_numpy()
        b_ok = bert["y_true"].eq(bert["y_pred"]).to_numpy()
        overlap = {
            "both_correct": int((s2_ok & b_ok).sum()),
            "bert_only_correct": int((~s2_ok & b_ok).sum()),
            "s2_only_correct": int((s2_ok & ~b_ok).sum()),
            "both_wrong": int((~s2_ok & ~b_ok).sum()),
        }
        pd.DataFrame([overlap]).to_csv(outdir / "S2_BERT_CORRECTNESS_OVERLAP.tsv", sep="\t", index=False)

        # Main result table.
        lookup = singles_df.set_index("prediction_key")
        main_rows = []
        for k, model, cond, role in [
            ("D3_AC", "Surface-only LR", "A-C", "diagnostic"),
            ("S2_AC", "Word TF-IDF + LinearSVC", "A-C", "primary sparse"),
            ("S2_GC", "Word TF-IDF + LinearSVC", "G-C", "group sensitivity"),
            ("S3_AC", "Char TF-IDF + LinearSVC", "A-C", "supplementary sparse"),
            ("BERT_AC42", "BERTweet seed 42", "A-C", "paired neural reference seed"),
            ("BERT_GC42", "BERTweet seed 42", "G-C", "paired neural group sensitivity"),
            ("BERT_AL_AL", "BERTweet trained A-L seed 42", "A-L eval", "shortcut diagnostic"),
            ("BERT_AL_AC", "BERTweet trained A-L seed 42", "A-C eval", "shortcut removal diagnostic"),
        ]:
            r = lookup.loc[k]
            main_rows.append({
                "model": model, "condition": cond, "role": role,
                "macro_f1": r.macro_f1, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "accuracy": r.accuracy,
            })
        pd.DataFrame(main_rows).to_csv(outdir / "TABLE2_MAIN_RESULTS.tsv", sep="\t", index=False)

        pilot = build_error_pilot(project, preds, outdir)

        # Headline numbers.
        d = delta_df.set_index("comparison")
        ac_mean = float(seed_df.set_index("condition").loc["AC", "macro_f1_mean"])
        ac_sd = float(seed_df.set_index("condition").loc["AC", "macro_f1_sample_sd"])
        gc_mean = float(seed_df.set_index("condition").loc["GC", "macro_f1_mean"])
        gc_sd = float(seed_df.set_index("condition").loc["GC", "macro_f1_sample_sd"])
        s2_ac = float(lookup.loc["S2_AC", "macro_f1"])
        direct_counts = [
            int(lookup.loc[k, "irony_to_sarcasm"] + lookup.loc[k, "sarcasm_to_irony"])
            for k in ["BERT_AC42", "BERT_AC2026", "BERT_AC7"]
        ]
        s2_direct = int(lookup.loc["S2_AC", "irony_to_sarcasm"] + lookup.loc["S2_AC", "sarcasm_to_irony"])
        headline = {
            "bootstrap_n": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
            "bert_ac_macro_f1_mean": ac_mean,
            "bert_ac_macro_f1_sample_sd": ac_sd,
            "bert_gc_macro_f1_mean": gc_mean,
            "bert_gc_macro_f1_sample_sd": gc_sd,
            "bert_gc_minus_ac_mean_descriptive": gc_mean - ac_mean,
            "s2_ac_macro_f1": s2_ac,
            "bert_ac_mean_minus_s2_ac_descriptive": ac_mean - s2_ac,
            "bert_ac42_minus_s2_ac_paired": d.loc["bert_ac42_minus_s2_ac"].to_dict(),
            "sparse_matched_leaky_minus_clean": d.loc["sparse_matched_leaky_minus_clean"].to_dict(),
            "bert_same_leaky_trained_al_minus_ac_eval": d.loc["same_leaky_trained_model_al_minus_ac_eval"].to_dict(),
            "bert_ac_direct_irony_sarcasm_confusions_by_seed": direct_counts,
            "bert_ac_direct_confusion_mean": float(np.mean(direct_counts)),
            "s2_ac_direct_irony_sarcasm_confusions": s2_direct,
            "bert_direct_confusion_reduction_vs_s2_fraction": float((s2_direct - np.mean(direct_counts)) / s2_direct),
            "s2_bert_correctness_overlap": overlap,
            "error_pilot": pilot,
        }
        (outdir / "HEADLINE_NUMBERS.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")

        # Human-readable summary without optional tabulate dependency.
        def fmt_ci(key):
            r = lookup.loc[key]
            return f"{r.macro_f1:.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}]"

        summary = f'''# H08A Statistical Analysis + Error Pilot\n\n- status: **PASS**\n- bootstrap: {BOOT_N:,} stratified resamples, seed {BOOT_SEED}\n- canonical test ID SHA-256: `{id_sha(canonical)}`\n\n## Main uncertainty anchors\n\n- Surface-only A-C Macro-F1: {fmt_ci("D3_AC")}\n- S2 A-C Macro-F1: {fmt_ci("S2_AC")}\n- BERTweet A-C seed42 Macro-F1: {fmt_ci("BERT_AC42")}\n- BERTweet G-C seed42 Macro-F1: {fmt_ci("BERT_GC42")}\n- A-L-trained BERTweet evaluated on A-L: {fmt_ci("BERT_AL_AL")}\n- same A-L-trained BERTweet evaluated on A-C: {fmt_ci("BERT_AL_AC")}\n\n## Neural seed summary\n\n- BERTweet A-C: **{ac_mean:.6f} ± {ac_sd:.6f}** Macro-F1\n- BERTweet G-C: **{gc_mean:.6f} ± {gc_sd:.6f}** Macro-F1\n\n## Required paired deltas\n\n'''
        for _, r in delta_df.iterrows():
            summary += f'- {r["comparison"]}: {r["delta_point"]:+.6f} [{r["ci_low"]:+.6f}, {r["ci_high"]:+.6f}]\n'
        summary += f'''\n## Error accounting\n\n- S2 A-C direct irony<->sarcasm confusions: {s2_direct}\n- BERTweet A-C direct cross-confusions by seed: {direct_counts}\n- BERTweet mean reduction vs S2: {headline["bert_direct_confusion_reduction_vs_s2_fraction"]*100:.2f}%\n- S2/BERT seed42 both correct: {overlap["both_correct"]}\n- BERT-only correct: {overlap["bert_only_correct"]}\n- S2-only correct: {overlap["s2_only_correct"]}\n- both wrong: {overlap["both_wrong"]}\n\n## Error pilot\n\n- blind unique cases: {pilot["unique_blind_cases"]}\n- key selection rows: {pilot["selected_key_rows"]}\n\nThe qualitative taxonomy is intentionally **not frozen** in H08A.\nThe consolidator must review `ERROR_PILOT_BLIND.tsv` and `ERROR_PILOT_KEY.tsv` first.\n'''
        (outdir / "H08_SUMMARY.md").write_text(summary, encoding="utf-8")

        status = {
            "status": "PASS",
            "stage": "H08A_STATS_ERROR_QUEUE",
            "bootstrap_n": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
            "prediction_files_validated": len(preds),
            "paired_comparisons": len(delta_df),
            "error_pilot_unique_cases": pilot["unique_blind_cases"],
        }
        (outdir / "H08_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(summary)
        return 0

    except Exception:
        tb = traceback.format_exc()
        (outdir / "H08_FAILURE.txt").write_text(tb, encoding="utf-8")
        (outdir / "H08_STATUS.json").write_text(
            json.dumps({"status": "BLOCKED", "stage": "H08A_STATS_ERROR_QUEUE"}, indent=2),
            encoding="utf-8",
        )
        print("H08 BLOCKER")
        print(tb)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())