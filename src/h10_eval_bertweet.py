#!/usr/bin/env python3
from pathlib import Path
import json
import gc
import hashlib
import sys
import pandas as pd
import numpy as np

LABELS=["irony","sarcasm","regular"]
ID2LABEL={i:x for i,x in enumerate(LABELS)}
LABEL2ID={x:i for i,x in enumerate(LABELS)}
SEEDS=[42,2026,7]
MAX_LENGTH=128
EVAL_BATCH=32

def adapt_text(s):
    return str(s).replace("<USER>","@USER").replace("<URL>","HTTPURL")

class EncodedDataset:
    def __init__(self,tokenizer,texts):
        enc=tokenizer(
            [adapt_text(x) for x in texts],
            truncation=True,padding="max_length",max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.enc=enc
    def __len__(self):
        return next(iter(self.enc.values())).shape[0]
    def __getitem__(self,idx):
        return {k:v[idx] for k,v in self.enc.items()}

def main():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    project=Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))
    ext_p=project/"runs/h10_external_transfer/EXTERNAL_POPULATIONS.tsv"
    model_meta_p=project/"provenance/h07_bertweet/MODEL_SNAPSHOT.json"
    out=project/"runs/h10_external_transfer"
    rep=project/"reports/h10_external_transfer"
    out.mkdir(parents=True,exist_ok=True); rep.mkdir(parents=True,exist_ok=True)

    if not torch.cuda.is_available():
        (rep/"BERT_STATUS.json").write_text(json.dumps({"status":"BLOCKED","reason":"CUDA unavailable"},indent=2),encoding="utf-8")
        return 0

    if not ext_p.exists() or not model_meta_p.exists():
        (rep/"BERT_STATUS.json").write_text(json.dumps({"status":"BLOCKED","reason":"missing inputs"},indent=2),encoding="utf-8")
        return 0

    ext=pd.read_csv(ext_p,sep="\t")
    meta=json.loads(model_meta_p.read_text(encoding="utf-8"))
    snapshot=Path(meta["snapshot_path"])
    if not snapshot.exists():
        (rep/"BERT_STATUS.json").write_text(json.dumps({"status":"BLOCKED","reason":"snapshot missing","snapshot":str(snapshot)},indent=2),encoding="utf-8")
        return 0

    tok=AutoTokenizer.from_pretrained(str(snapshot),use_fast=False,local_files_only=True)
    ds=EncodedDataset(tok,ext.text_clean.astype(str).tolist())
    dl=DataLoader(ds,batch_size=EVAL_BATCH,shuffle=False,pin_memory=True,num_workers=0)
    device=torch.device("cuda")

    completed=[]
    for seed in SEEDS:
        ckpt=project/f"models/h07_bertweet/checkpoints/N1_BERTweet_AC_seed{seed}/best_state.pt"
        if not ckpt.exists():
            print("BLOCKER missing checkpoint",ckpt)
            continue

        model=AutoModelForSequenceClassification.from_pretrained(
            str(snapshot),num_labels=3,label2id=LABEL2ID,id2label=ID2LABEL,
            local_files_only=True,
        )
        state=torch.load(ckpt,map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        pred=[]
        with torch.no_grad():
            for batch in dl:
                batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
                with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
                    logits=model(**batch).logits
                pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        pred3=[ID2LABEL[i] for i in pred]
        pred2=[1 if x in {"irony","sarcasm"} else 0 for x in pred3]

        pdf=pd.DataFrame({
            "_ext_id":ext._ext_id.astype(str),
            "y_true":ext.gold_binary.astype(int),
            "pred_internal_3class":pred3,
            "y_pred":pred2,
            "exact_internal_overlap":ext.exact_internal_overlap.astype(bool),
            "high_similarity_train":ext.high_similarity_train.astype(bool),
        })
        pdf.to_csv(out/f"BERTWEET_EXTERNAL_seed{seed}.tsv",sep="\t",index=False)
        completed.append(seed)
        print("completed seed",seed)

        del model,state
        gc.collect()
        torch.cuda.empty_cache()

    status={"status":"PASS" if completed==SEEDS else "PARTIAL","completed_seeds":completed}
    (rep/"BERT_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())