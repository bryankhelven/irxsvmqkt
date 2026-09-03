#!/usr/bin/env python3
from pathlib import Path
import json
import hashlib
import re
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def token_features(text):
    toks=str(text).lower().split()
    feats=set(toks)
    feats.update(f"{a}␟{b}" for a,b in zip(toks,toks[1:]))
    return feats

def char5(text):
    s=" ".join(str(text).lower().split())
    if len(s)<5:
        return {s} if s else set()
    return {s[i:i+5] for i in range(len(s)-4)}

def jacc(a,b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)

def minhash(features, num_perm=64):
    m=MinHash(num_perm=num_perm, seed=20260903)
    for f in sorted(features):
        m.update(f.encode("utf-8"))
    return m

def main():
    project=Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))
    train_p=project/"data/processed/h03_tcc_reproduction/train_final.csv"
    test_p=project/"data/processed/h03_tcc_reproduction/test_final.csv"
    ext_p=project/"data/derived/h10_external_transfer/tweeteval_irony_test_clean.tsv"
    out=project/"runs/h10_external_transfer"
    rep=project/"reports/h10_external_transfer"
    out.mkdir(parents=True,exist_ok=True); rep.mkdir(parents=True,exist_ok=True)

    missing=[str(p) for p in [train_p,test_p,ext_p] if not p.exists()]
    if missing:
        (rep/"SPARSE_STATUS.json").write_text(json.dumps({"status":"BLOCKED","missing":missing},indent=2),encoding="utf-8")
        return 0

    tr=pd.read_csv(train_p)
    itest=pd.read_csv(test_p)
    ext=pd.read_csv(ext_p,sep="\t")

    # Exact cross-dataset overlap against internal train OR test.
    internal_all=set(tr.text_clean.astype(str)) | set(itest.text_clean.astype(str))
    ext["exact_internal_overlap"]=ext.text_clean.astype(str).isin(internal_all)
    ext["novel_exact"]=~ext.exact_internal_overlap

    # H04-style high-sim screen against internal training set.
    num_perm=64
    lsh=MinHashLSH(threshold=0.72,num_perm=num_perm)
    train_text={}
    train_tok={}
    train_char={}
    for idx,text in enumerate(tr.text_clean.astype(str)):
        key=f"I{idx}"
        tf=token_features(text)
        train_text[key]=text
        train_tok[key]=tf
        train_char[key]=char5(text)
        lsh.insert(key,minhash(tf,num_perm=num_perm))

    hisim_flags=[]
    best_rows=[]
    for _,r in ext.iterrows():
        txt=str(r.text_clean)
        tf=token_features(txt); cf=char5(txt)
        candidates=lsh.query(minhash(tf,num_perm=num_perm))
        best=None
        high=False
        for key in candidates:
            tj=jacc(tf,train_tok[key])
            cj=jacc(cf,train_char[key])
            rr=fuzz.ratio(txt,train_text[key])/100.0
            is_high=(cj>=0.80) or (rr>=0.92 and tj>=0.70)
            score=max(cj, rr, tj)
            row=(score,key,tj,cj,rr,is_high)
            if best is None or row[0]>best[0]:
                best=row
            if is_high:
                high=True
        hisim_flags.append(high)
        if best:
            best_rows.append({
                "_ext_id":r._ext_id,
                "candidate_train_key":best[1],
                "token_jaccard":best[2],
                "char5_jaccard":best[3],
                "rapidfuzz_ratio":best[4],
                "high_similarity":best[5],
            })
        else:
            best_rows.append({
                "_ext_id":r._ext_id,
                "candidate_train_key":"",
                "token_jaccard":0.0,
                "char5_jaccard":0.0,
                "rapidfuzz_ratio":0.0,
                "high_similarity":False,
            })
    ext["high_similarity_train"]=hisim_flags
    ext["novel_hisim"]=~ext.high_similarity_train

    ext.to_csv(out/"EXTERNAL_POPULATIONS.tsv",sep="\t",index=False)
    pd.DataFrame(best_rows).to_csv(out/"EXTERNAL_HISIM_BEST_CANDIDATE.tsv",sep="\t",index=False)

    # Frozen S2 reconstruction.
    vec=TfidfVectorizer(
        analyzer="word",ngram_range=(1,2),min_df=2,sublinear_tf=True,
        max_features=200000,lowercase=True
    )
    xtr=vec.fit_transform(tr.text_clean.astype(str))
    clf=LinearSVC(C=0.5)
    clf.fit(xtr,tr["class"].astype(str))
    xext=vec.transform(ext.text_clean.astype(str))
    pred3=clf.predict(xext)

    pred_binary=np.array([1 if p in {"irony","sarcasm"} else 0 for p in pred3],dtype=int)
    pred_df=pd.DataFrame({
        "_ext_id":ext._ext_id.astype(str),
        "y_true":ext.gold_binary.astype(int),
        "pred_internal_3class":pred3,
        "y_pred":pred_binary,
        "exact_internal_overlap":ext.exact_internal_overlap.astype(bool),
        "high_similarity_train":ext.high_similarity_train.astype(bool),
    })
    pred_df.to_csv(out/"S2_EXTERNAL_PREDICTIONS.tsv",sep="\t",index=False)

    status={
        "status":"PASS",
        "test_n":int(len(ext)),
        "exact_internal_overlap_n":int(ext.exact_internal_overlap.sum()),
        "ext_novel_n":int(ext.novel_exact.sum()),
        "high_similarity_train_n":int(ext.high_similarity_train.sum()),
        "ext_hisim_purged_n":int((~ext.high_similarity_train).sum()),
        "vocab_size":int(len(vec.vocabulary_)),
    }
    (rep/"SPARSE_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())