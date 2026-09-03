#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

LABELS = ["irony", "sarcasm", "regular"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for i, x in enumerate(LABELS)}
MODEL_ID = "vinai/bertweet-base"
MAX_LENGTH = 128
TRAIN_BATCH = 16
EVAL_BATCH = 32
GRAD_ACCUM = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_EPOCHS = 5
PATIENCE = 1
MAX_GRAD_NORM = 1.0
VAL_RANDOM_STATE = 42
SEEDS = [42, 2026, 7]


def adapt_text(s: str) -> str:
    return str(s).replace("<USER>", "@USER").replace("<URL>", "HTTPURL")


def id_digest(ids) -> str:
    payload = "\n".join(map(str, ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# H07_RESUME_PATCH_v1
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def metrics_dict(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        ),
        "irony_to_sarcasm": int(cm[0, 1]),
        "sarcasm_to_irony": int(cm[1, 0]),
    }, cm


def freeze_validation_membership(project: Path, ac_train: pd.DataFrame):
    outdir = project / "provenance/h07_bertweet"
    outdir.mkdir(parents=True, exist_ok=True)
    membership_path = outdir / "validation_membership.tsv"
    meta_path = outdir / "validation_membership.json"

    if membership_path.exists() and meta_path.exists():
        mem = pd.read_csv(membership_path, sep="\t", dtype={"_id": str})
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        current_ids = set(ac_train["_id"].astype(str))
        if set(mem["_id"]) != current_ids:
            raise RuntimeError("Frozen validation membership does not match current A-C train IDs")
        return mem, meta

    ids = ac_train["_id"].astype(str).to_numpy()
    labels = ac_train["class"].astype(str).to_numpy()
    train_ids, val_ids = train_test_split(
        ids,
        test_size=0.20,
        random_state=VAL_RANDOM_STATE,
        stratify=labels,
    )
    train_set, val_set = set(map(str, train_ids)), set(map(str, val_ids))
    rows = []
    for rid in ac_train["_id"].astype(str):
        rows.append({"_id": rid, "split": "train" if rid in train_set else "val"})
    mem = pd.DataFrame(rows)
    mem.to_csv(membership_path, sep="\t", index=False)

    meta = {
        "random_state": VAL_RANDOM_STATE,
        "method": "sklearn train_test_split stratified 80/20",
        "ac_train_n": int(len(ac_train)),
        "train_n": int((mem["split"] == "train").sum()),
        "val_n": int((mem["split"] == "val").sum()),
        "membership_sha256": id_digest(mem["_id"].astype(str) + ":" + mem["split"].astype(str)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return mem, meta


def apply_membership(df: pd.DataFrame, mem: pd.DataFrame):
    split_map = dict(zip(mem["_id"].astype(str), mem["split"].astype(str)))
    work = df.copy()
    work["_id"] = work["_id"].astype(str)
    work["_h07_split"] = work["_id"].map(split_map)
    if work["_h07_split"].isna().any():
        bad = work.loc[work["_h07_split"].isna(), "_id"].head().tolist()
        raise RuntimeError(f"Condition contains IDs absent from frozen membership: {bad}")
    tr = work[work["_h07_split"] == "train"].copy()
    va = work[work["_h07_split"] == "val"].copy()
    return tr, va


class EncodedDataset:
    def __init__(self, tokenizer, texts, labels=None):
        import torch
        enc = tokenizer(
            [adapt_text(x) for x in texts],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.enc = {k: v for k, v in enc.items()}
        self.labels = None if labels is None else torch.tensor(
            [LABEL2ID[str(x)] for x in labels], dtype=torch.long
        )

    def __len__(self):
        return next(iter(self.enc.values())).shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def build_optimizer(model):
    import torch
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if name.endswith(".bias") or "layernorm" in low or "layer_norm" in low:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LR,
    )


def evaluate(model, loader, device):
    import torch
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                out = model(**batch)
            p = out.logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            preds.extend(p)
            gold.extend(labels.numpy().tolist())
    return [ID2LABEL[x] for x in gold], [ID2LABEL[x] for x in preds]


def save_eval(run_dir: Path, eval_name: str, test_df: pd.DataFrame, y_true, y_pred, extra: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics, cm = metrics_dict(y_true, y_pred)
    pred_path = run_dir / f"predictions_{eval_name}.csv"
    pd.DataFrame({
        "_id": test_df["_id"].astype(str).values,
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(pred_path, index=False)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        run_dir / f"confusion_{eval_name}.csv"
    )
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    (run_dir / f"classification_report_{eval_name}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    payload = {
        "eval_name": eval_name,
        "metrics": metrics,
        "test_n": int(len(test_df)),
        "test_id_sha256": id_digest(test_df["_id"].astype(str)),
        **extra,
    }
    (run_dir / f"metrics_{eval_name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def load_model_snapshot(project: Path):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_root = project / "models/h07_bertweet"
    cache = model_root / "hf_cache"
    model_root.mkdir(parents=True, exist_ok=True)
    meta_path = project / "provenance/h07_bertweet/MODEL_SNAPSHOT.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        p = Path(meta["snapshot_path"])
        if p.exists():
            tokenizer = AutoTokenizer.from_pretrained(
                str(p), use_fast=False, local_files_only=True
            )
            return p, tokenizer, meta

    snapshot_path = Path(snapshot_download(repo_id=MODEL_ID, cache_dir=str(cache)))
    commit = snapshot_path.name
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path), use_fast=False, local_files_only=True
    )
    meta = {
        "repo_id": MODEL_ID,
        "snapshot_path": str(snapshot_path),
        "resolved_snapshot": commit,
        "use_fast": False,
        "normalization_enabled_by_runner": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return snapshot_path, tokenizer, meta


def smoke_test(project: Path, snapshot_path: Path, tokenizer):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    smoke_dir = project / "reports/h07_bertweet/smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    marker = smoke_dir / "SMOKE_PASS.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: full H07 is blocked")

    texts = [
        "plain tweet",
        "RT @USER this is ironic",
        "wow HTTPURL #topic",
        "sure, wonderful!",
    ] * 8
    labels = ["regular", "irony", "sarcasm", "irony"] * 8

    ds = EncodedDataset(tokenizer, texts, labels)
    dl = DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot_path),
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        local_files_only=True,
    ).cuda()
    opt = build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    model.train()
    opt.zero_grad(set_to_none=True)
    batch = next(iter(dl))
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

    info = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": TRAIN_BATCH,
        "max_length": MAX_LENGTH,
        "loss": float(loss.detach().cpu()) * GRAD_ACCUM,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    marker.write_text(json.dumps(info, indent=2), encoding="utf-8")
    del model, opt, scaler, batch, out
    gc.collect()
    torch.cuda.empty_cache()
    return info


def train_one(project: Path, snapshot_path: Path, tokenizer, condition: str, seed: int,
              train_df: pd.DataFrame, val_df: pd.DataFrame, train_text_col: str):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    run_name = f"N1_BERTweet_{condition}_seed{seed}"
    run_dir = project / "runs/h07_bertweet" / run_name
    report_dir = project / "reports/h07_bertweet" / run_name
    ckpt_dir = project / "models/h07_bertweet/checkpoints" / run_name
    for d in [run_dir, report_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    done = run_dir / "TRAINING_DONE.json"
    best_path = ckpt_dir / "best_state.pt"

    if done.exists() and best_path.exists():
        meta = json.loads(done.read_text(encoding="utf-8"))
        if "checkpoint_sha256" not in meta:
            meta["checkpoint_sha256"] = file_sha256(best_path)
            done.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{run_name} RESUME: completed training preserved; checkpoint reused", flush=True)
        return meta

    set_seed(seed)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_ds = EncodedDataset(
        tokenizer,
        train_df[train_text_col].astype(str).tolist(),
        train_df["class"].astype(str).tolist(),
    )
    val_ds = EncodedDataset(
        tokenizer,
        val_df[train_text_col].astype(str).tolist(),
        val_df["class"].astype(str).tolist(),
    )

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        generator=gen,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot_path),
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        local_files_only=True,
    ).to(device)

    opt = build_optimizer(model)
    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_updates = updates_per_epoch * MAX_EPOCHS
    warmup_steps = int(round(total_updates * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    history = []
    best_f1 = -1.0
    non_improving = 0
    global_updates = 0
    started = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen_batches = 0

        for bi, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                out = model(**batch)
                raw_loss = out.loss
                loss = raw_loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            loss_sum += float(raw_loss.detach().cpu())
            seen_batches += 1

            do_step = (bi % GRAD_ACCUM == 0) or (bi == len(train_loader))
            if do_step:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                global_updates += 1

            if bi % 250 == 0 or bi == len(train_loader):
                elapsed = time.time() - started
                print(
                    f"{run_name} epoch={epoch} batch={bi}/{len(train_loader)} "
                    f"loss={raw_loss.item():.5f} updates={global_updates} "
                    f"elapsed_min={elapsed/60:.1f}",
                    flush=True,
                )

        val_true, val_pred = evaluate(model, val_loader, device)
        val_metrics, _ = metrics_dict(val_true, val_pred)
        row = {
            "epoch": epoch,
            "train_loss_mean": loss_sum / max(1, seen_batches),
            **val_metrics,
            "optimizer_updates": global_updates,
            "lr_now": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(report_dir / "validation_history.tsv", sep="\t", index=False)

        print(
            f"{run_name} epoch={epoch} val_macro_f1={val_metrics['macro_f1']:.6f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            non_improving = 0
            torch.save(model.state_dict(), best_path)
        else:
            non_improving += 1
            if non_improving >= PATIENCE:
                print(f"{run_name} early_stop after epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    summary = {
        "run_name": run_name,
        "condition": condition,
        "seed": seed,
        "train_text_col": train_text_col,
        "train_n": int(len(train_df)),
        "val_n": int(len(val_df)),
        "train_id_sha256": id_digest(train_df["_id"].astype(str)),
        "val_id_sha256": id_digest(val_df["_id"].astype(str)),
        "best_validation_macro_f1": float(best_f1),
        "epochs_completed": int(len(history)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_seconds": float(time.time() - started),
        "checkpoint": str(best_path),
        "checkpoint_sha256": file_sha256(best_path),
    }
    done.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del train_ds, val_ds, train_loader, val_loader, opt, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def load_trained_model(snapshot_path: Path, training_meta: dict):
    import torch
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot_path),
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        local_files_only=True,
    )
    state = torch.load(training_meta["checkpoint"], map_location="cpu")
    model.load_state_dict(state)
    return model


def eval_trained(project: Path, snapshot_path: Path, tokenizer, training_meta: dict,
                 eval_name: str, test_df: pd.DataFrame, text_col: str):
    import torch
    from torch.utils.data import DataLoader

    run_dir = project / "runs/h07_bertweet" / training_meta["run_name"]
    metric_path = run_dir / f"metrics_{eval_name}.json"
    pred_path = run_dir / f"predictions_{eval_name}.csv"
    if metric_path.exists() and pred_path.exists():
        return json.loads(metric_path.read_text(encoding="utf-8"))

    model = load_trained_model(snapshot_path, training_meta).cuda()
    ds = EncodedDataset(
        tokenizer,
        test_df[text_col].astype(str).tolist(),
        test_df["class"].astype(str).tolist(),
    )
    dl = DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, pin_memory=True, num_workers=0)
    y_true, y_pred = evaluate(model, dl, torch.device("cuda"))
    payload = save_eval(
        run_dir,
        eval_name,
        test_df,
        y_true,
        y_pred,
        {
            "training_run": training_meta["run_name"],
            "training_condition": training_meta["condition"],
            "training_seed": training_meta["seed"],
            "eval_text_col": text_col,
            "best_validation_macro_f1": training_meta["best_validation_macro_f1"],
        },
    )
    del model, ds, dl
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def environment_snapshot():
    import torch, transformers
    import sklearn
    try:
        import huggingface_hub
        hub = huggingface_hub.__version__
    except Exception:
        hub = "unknown"
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None,
        "huggingface_hub": hub,
    }


def aggregate(project: Path):
    run_root = project / "runs/h07_bertweet"
    rows = []
    for mp in sorted(run_root.glob("N1_BERTweet_*/metrics_*.json")):
        x = json.loads(mp.read_text(encoding="utf-8"))
        rows.append({
            "training_run": x["training_run"],
            "training_condition": x["training_condition"],
            "seed": x["training_seed"],
            "eval_name": x["eval_name"],
            **x["metrics"],
            "test_n": x["test_n"],
            "test_id_sha256": x["test_id_sha256"],
            "best_validation_macro_f1": x["best_validation_macro_f1"],
        })
    out = pd.DataFrame(rows)
    report_root = project / "reports/h07_bertweet"
    report_root.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(report_root / "H07_RESULTS.tsv", sep="\t", index=False)

    required = {
        ("AC", 42, "AC"),
        ("AC", 2026, "AC"),
        ("AC", 7, "AC"),
        ("GC", 42, "GC"),
        ("GC", 2026, "GC"),
        ("GC", 7, "GC"),
        ("AL", 42, "AL"),
        ("AL", 42, "AC"),
    }
    observed = set()
    for _, r in out.iterrows():
        observed.add((str(r["training_condition"]), int(r["seed"]), str(r["eval_name"])))

    complete = required.issubset(observed)
    status = {
        "status": "PASS" if complete else "PARTIAL",
        "required_evaluations": len(required),
        "observed_evaluations": int(len(observed)),
        "missing": [list(x) for x in sorted(required - observed)],
    }

    if len(out):
        ac = out[(out.training_condition == "AC") & (out.eval_name == "AC")]
        gcdf = out[(out.training_condition == "GC") & (out.eval_name == "GC")]
        def seed_summary(df):
            vals = df["macro_f1"].astype(float).tolist()
            return {
                "n": len(vals),
                "macro_f1_mean": float(np.mean(vals)) if vals else None,
                "macro_f1_sample_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "seeds": {str(int(r.seed)): float(r.macro_f1) for _, r in df.iterrows()},
            }
        status["AC_seed_summary"] = seed_summary(ac)
        status["GC_seed_summary"] = seed_summary(gcdf)

    (report_root / "H07_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if len(out):
        try:
            result_table = out.to_markdown(index=False)
            result_table_format = "markdown"
        except ImportError:
            result_table = "```text\n" + out.to_string(index=False) + "\n```"
            result_table_format = "plain_text_fallback"
        status["summary_table_format"] = result_table_format
        (report_root / "H07_STATUS.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        md = ["# H07 BERTweet Results", "", f"- status: **{status['status']}**", "",
              result_table, "", "## Status", "",
              "```json", json.dumps(status, indent=2), "```"]
        (report_root / "H07_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print((report_root / "H07_SUMMARY.md").read_text(encoding="utf-8"))
    else:
        print(json.dumps(status, indent=2))
    return status


def self_test():
    assert adapt_text("<USER> x <URL>") == "@USER x HTTPURL"
    df = pd.DataFrame({
        "_id": [str(i) for i in range(30)],
        "class": LABELS * 10,
        "text_clean": ["x"] * 30,
        "text_leaky": ["x"] * 30,
    })
    with __import__("tempfile").TemporaryDirectory() as td:
        project = Path(td)
        mem, meta = freeze_validation_membership(project, df)
        tr, va = apply_membership(df, mem)
        assert len(tr) + len(va) == len(df)
        assert set(tr["_id"]).isdisjoint(set(va["_id"]))
        assert set(tr["_id"]) | set(va["_id"]) == set(df["_id"])
    y = ["irony", "sarcasm", "regular"]
    m, cm = metrics_dict(y, y)
    assert abs(m["macro_f1"] - 1.0) < 1e-12
    print("H07_SELF_TEST PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    project = Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))
    report_root = project / "reports/h07_bertweet"
    report_root.mkdir(parents=True, exist_ok=True)

    try:
        env = environment_snapshot()
        (report_root / "ENVIRONMENT.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        if not env["cuda_available"]:
            raise RuntimeError("CUDA unavailable")

        ac_train = pd.read_csv(project / "data/processed/h03_tcc_reproduction/train_final.csv")
        ac_test = pd.read_csv(project / "data/processed/h03_tcc_reproduction/test_final.csv")
        gc_train = pd.read_csv(project / "data/derived/h05_grouped_clean/train_grouped_clean.csv")
        gc_test = pd.read_csv(project / "data/derived/h05_grouped_clean/test_grouped_clean.csv")

        for df in [ac_train, ac_test, gc_train, gc_test]:
            df["_id"] = df["_id"].astype(str)

        if len(ac_train) != 58784 or len(ac_test) != 5947 or len(gc_train) != 58557 or len(gc_test) != 5947:
            raise RuntimeError("Frozen condition counts changed")
        if ac_test["_id"].tolist() != gc_test["_id"].tolist():
            raise RuntimeError("A-C and G-C test IDs/order differ")

        mem, mem_meta = freeze_validation_membership(project, ac_train)
        ac_tr, ac_va = apply_membership(ac_train, mem)
        gc_tr, gc_va = apply_membership(gc_train, mem)

        split_summary = {
            "AC_train_fit_n": len(ac_tr), "AC_val_n": len(ac_va),
            "GC_train_fit_n": len(gc_tr), "GC_val_n": len(gc_va),
            "AC_train_fit_id_sha256": id_digest(ac_tr["_id"]),
            "AC_val_id_sha256": id_digest(ac_va["_id"]),
            "GC_train_fit_id_sha256": id_digest(gc_tr["_id"]),
            "GC_val_id_sha256": id_digest(gc_va["_id"]),
            **mem_meta,
        }
        (report_root / "SPLIT_SUMMARY.json").write_text(
            json.dumps(split_summary, indent=2), encoding="utf-8"
        )

        snapshot_path, tokenizer, snapshot_meta = load_model_snapshot(project)
        smoke = smoke_test(project, snapshot_path, tokenizer)
        print("SMOKE:", json.dumps(smoke, indent=2), flush=True)
        if args.smoke_only:
            return 0

        plan = [
            ("AC", 42, ac_tr, ac_va, "text_clean"),
            ("AL", 42, ac_tr, ac_va, "text_leaky"),
            ("GC", 42, gc_tr, gc_va, "text_clean"),
            ("AC", 2026, ac_tr, ac_va, "text_clean"),
            ("AC", 7, ac_tr, ac_va, "text_clean"),
            ("GC", 2026, gc_tr, gc_va, "text_clean"),
            ("GC", 7, gc_tr, gc_va, "text_clean"),
        ]

        for condition, seed, tr, va, col in plan:
            print(f"\n=== TRAIN {condition} seed={seed} ===", flush=True)
            meta = train_one(project, snapshot_path, tokenizer, condition, seed, tr, va, col)
            if condition == "AC":
                eval_trained(project, snapshot_path, tokenizer, meta, "AC", ac_test, "text_clean")
            elif condition == "GC":
                eval_trained(project, snapshot_path, tokenizer, meta, "GC", gc_test, "text_clean")
            elif condition == "AL":
                eval_trained(project, snapshot_path, tokenizer, meta, "AL", ac_test, "text_leaky")
                eval_trained(project, snapshot_path, tokenizer, meta, "AC", ac_test, "text_clean")
            aggregate(project)

        aggregate(project)
    except Exception:
        tb = traceback.format_exc()
        (report_root / "H07_FAILURE.txt").write_text(tb, encoding="utf-8")
        print("H07 BLOCKER:")
        print(tb)
        try:
            aggregate(project)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())