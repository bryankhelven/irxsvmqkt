#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz

LABELS = ["irony", "sarcasm", "regular"]
SOURCE_LABELS = ["figurative", "irony", "sarcasm", "regular"]
MARKERS = ["#irony", "#ironic", "#sarcasm", "#sarcastic"]
URL_RE = re.compile(r"(?i)\b(?:https?://\S+|www\.\S+)")
USER_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
MARKER_RE = re.compile(r"(?i)(?<!\w)#(?:sarcastic|sarcasm|ironic|irony)\b")
HASHTAG_RE = re.compile(r"(?i)(?<!\w)#[A-Za-z0-9_]+")
TOKEN_RE = re.compile(r"(?u)<URL>|<USER>|\w+")
WS_RE = re.compile(r"\s+")
RT_RE = re.compile(r"(?i)^\s*RT\b")

SEED = 20260902
NUM_PERM = 64
LSH_THRESHOLD = 0.72
HIGH_CHAR5 = 0.80
HIGH_FUZZ = 0.92
HIGH_TOKEN = 0.70
GC_TRIGGER = 0.005
MAX_REVIEW_ROWS = 500


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def valid_text(x) -> bool:
    return (not pd.isna(x)) and bool(str(x).strip())


def normalize_base(text: str) -> str:
    text = str(text)
    text = URL_RE.sub("<URL>", text)
    text = USER_RE.sub("<USER>", text)
    return WS_RE.sub(" ", text).strip()


def remove_class_markers(text: str) -> str:
    return WS_RE.sub(" ", MARKER_RE.sub(" ", str(text))).strip()


def normalize_clean(text: str) -> str:
    return remove_class_markers(normalize_base(text))


def normalize_aggressive(text: str) -> str:
    t = unicodedata.normalize("NFKC", normalize_clean(text)).casefold()
    t = re.sub(r"[^\w<>]+", " ", t, flags=re.UNICODE)
    return WS_RE.sub(" ", t).strip()


def add_ids(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    out["_source_split"] = split
    out["_source_row"] = np.arange(len(out), dtype=np.int64)
    out["_id"] = [f"{split}:{i:08d}" for i in range(len(out))]
    out["class"] = out["class"].map(canonical_label)
    return out


def marker_list(text: str) -> list[str]:
    vals = [m.casefold() for m in MARKER_RE.findall(str(text))]
    # regex returns whole match under current pattern; defensive normalization follows.
    if not vals:
        vals = [m.casefold() for m in HASHTAG_RE.findall(str(text)) if m.casefold() in MARKERS]
    return vals


def exact_marker_counts(text: str) -> Counter:
    tags = [m.casefold() for m in HASHTAG_RE.findall(str(text))]
    return Counter(t for t in tags if t in MARKERS)


def marker_like_variants(text: str) -> list[str]:
    out=[]
    for tag in HASHTAG_RE.findall(str(text)):
        low=tag.casefold()
        if low not in MARKERS and ("iron" in low or "sarcas" in low):
            out.append(low)
    return out


def qstats(values: pd.Series) -> dict:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {k: np.nan for k in ["mean","std","min","p25","median","p75","p95","max"]}
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
        "min": float(vals.min()),
        "p25": float(vals.quantile(.25)),
        "median": float(vals.quantile(.50)),
        "p75": float(vals.quantile(.75)),
        "p95": float(vals.quantile(.95)),
        "max": float(vals.max()),
    }


def family_table(df: pd.DataFrame, key: str, split: str, view: str) -> pd.DataFrame:
    g = df.groupby(key, dropna=False)
    rows=[]
    for val, sub in g:
        if len(sub) < 2:
            continue
        labels=sorted(set(sub["class"].astype(str)))
        rows.append({
            "view": view, "split": split, "family_key": str(val),
            "n_rows": int(len(sub)), "n_labels": int(len(labels)),
            "labels": "|".join(labels),
            "source_ids": "|".join(sub["_id"].astype(str).tolist()),
        })
    return pd.DataFrame(rows)


def cross_overlap(train: pd.DataFrame, test: pd.DataFrame, key: str, view: str) -> tuple[pd.DataFrame, dict]:
    tr = defaultdict(list)
    for _,r in train.iterrows():
        tr[str(r[key])].append((r["_id"], r["class"]))
    rows=[]
    test_ids=set(); keys=set()
    for _,r in test.iterrows():
        k=str(r[key])
        if k in tr:
            keys.add(k); test_ids.add(str(r["_id"]))
            for tid,tlabel in tr[k]:
                rows.append({"view":view,"key":k,"train_id":tid,"train_label":tlabel,"test_id":r["_id"],"test_label":r["class"]})
    tab=pd.DataFrame(rows)
    summary={"view":view,"overlap_keys":len(keys),"test_rows_involved":len(test_ids),"pair_rows":len(rows)}
    return tab, summary


def word_tokens(text: str) -> list[str]:
    return [t.casefold() for t in TOKEN_RE.findall(str(text))]


def minhash_shingles(text: str) -> tuple[MinHash, set[str]]:
    toks=word_tokens(text)
    shingles=set(toks)
    shingles.update(f"{a}\u241f{b}" for a,b in zip(toks,toks[1:]))
    if not shingles:
        shingles={"<EMPTY>"}
    mh=MinHash(num_perm=NUM_PERM, seed=SEED)
    for sh in sorted(shingles):
        mh.update(sh.encode("utf-8", errors="ignore"))
    return mh, shingles


def ngram_set(text: str, n: int) -> set[str]:
    s=WS_RE.sub(" ", str(text).casefold()).strip()
    if not s: return {""}
    if len(s)<n: return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}


def jacc(a:set[str], b:set[str]) -> float:
    u=a|b
    return len(a&b)/len(u) if u else 1.0


def safe_snip(text: str, n=240) -> str:
    return WS_RE.sub(" ", str(text)).strip()[:n]


def near_duplicate_audit(train: pd.DataFrame, test: pd.DataFrame, report_dir: Path) -> dict:
    print(f"near-dup: building LSH on {len(train)} train rows", flush=True)
    lsh=MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    train_meta={}
    for i,r in train.reset_index(drop=True).iterrows():
        mh, toks=minhash_shingles(r["text_clean"])
        key=str(r["_id"])
        lsh.insert(key,mh)
        train_meta[key]={"text":r["text_clean"],"label":r["class"],"source_row":int(r["_source_row"]),"token_set":set(word_tokens(r["text_clean"]))}
        if (i+1)%5000==0:
            print(f"near-dup: indexed {i+1}/{len(train)}", flush=True)

    rows=[]
    candidate_test_ids=set(); high_test_ids=set(); high_train_ids=set()
    query_return_counts=[]
    for i,r in test.reset_index(drop=True).iterrows():
        mh,_=minhash_shingles(r["text_clean"])
        hits=lsh.query(mh)
        query_return_counts.append(len(hits))
        if hits:
            candidate_test_ids.add(str(r["_id"]))
        test_token=set(word_tokens(r["text_clean"]))
        test_c5=ngram_set(r["text_clean"],5)
        for tid in hits:
            tm=train_meta[tid]
            tj=jacc(tm["token_set"],test_token)
            cj=jacc(ngram_set(tm["text"],5),test_c5)
            fr=fuzz.ratio(tm["text"],r["text_clean"])/100.0
            high=(cj>=HIGH_CHAR5) or (fr>=HIGH_FUZZ and tj>=HIGH_TOKEN)
            if high:
                high_test_ids.add(str(r["_id"])); high_train_ids.add(tid)
            rows.append({
                "train_id":tid,"train_source_row":tm["source_row"],"train_label":tm["label"],
                "test_id":r["_id"],"test_source_row":int(r["_source_row"]),"test_label":r["class"],
                "token_jaccard":tj,"char5_jaccard":cj,"rapidfuzz_ratio":fr,"high_similarity_screen":bool(high),
                "train_text":safe_snip(tm["text"]),"test_text":safe_snip(r["text_clean"]),
            })
        if (i+1)%500==0:
            print(f"near-dup: queried {i+1}/{len(test)}", flush=True)

    cand=pd.DataFrame(rows)
    if len(cand):
        cand=cand.sort_values(["high_similarity_screen","char5_jaccard","rapidfuzz_ratio","token_jaccard"], ascending=[False,False,False,False])
    cand.to_csv(report_dir/"near_duplicate_candidates.tsv",sep="\t",index=False)
    high=cand[cand["high_similarity_screen"]==True].copy() if len(cand) else cand.copy()
    high.to_csv(report_dir/"near_duplicate_high_similarity_screen.tsv",sep="\t",index=False)

    # Consolidator-review sample: all high-sim up to MAX_REVIEW_ROWS, then top remaining if space.
    if len(cand):
        review=cand.head(MAX_REVIEW_ROWS).copy()
        review.insert(0,"review_rank",np.arange(1,len(review)+1))
        review["consolidator_decision"]=""
        review["consolidator_note"]=""
    else:
        review=pd.DataFrame(columns=["review_rank","consolidator_decision","consolidator_note"])
    review.to_csv(report_dir/"near_duplicate_review_sample.tsv",sep="\t",index=False)

    high_fraction=(len(high_test_ids)/len(test)) if len(test) else 0.0
    summary={
        "method":"token_unigram_bigram_minhash_lsh",
        "num_perm":NUM_PERM,"lsh_threshold":LSH_THRESHOLD,"seed":SEED,
        "candidate_pairs":int(len(cand)),"candidate_test_ids":int(len(candidate_test_ids)),
        "high_similarity_pairs":int(len(high)),"high_similarity_test_ids":int(len(high_test_ids)),
        "high_similarity_train_ids":int(len(high_train_ids)),"test_n":int(len(test)),
        "algorithmic_high_similarity_test_fraction":high_fraction,
        "roadmap_confirmed_family_trigger_fraction":GC_TRIGGER,
        "algorithmic_trigger_is_not_final":True,
        "max_query_hits":int(max(query_return_counts) if query_return_counts else 0),
        "mean_query_hits":float(np.mean(query_return_counts) if query_return_counts else 0.0),
    }
    (report_dir/"near_duplicate_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=str(Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))))
    args=ap.parse_args()
    project=Path(args.project).expanduser().resolve()
    ptr=project/"data/raw/kaggle_tweets_sarcasm_irony/CURRENT_PATH.txt"
    if not ptr.exists(): raise FileNotFoundError(ptr)
    raw_root=Path(ptr.read_text(encoding="utf-8").strip())
    train_csv=raw_root/"train.csv"; test_csv=raw_root/"test.csv"
    h03_train=project/"data/processed/h03_tcc_reproduction/train_final.csv"
    h03_test=project/"data/processed/h03_tcc_reproduction/test_final.csv"
    for p in [train_csv,test_csv,h03_train,h03_test]:
        if not p.exists(): raise FileNotFoundError(p)

    report_dir=project/"reports/h04_data_audit_v2"
    artifact_dir=project/"artifacts/h04_data_audit_v2"
    report_dir.mkdir(parents=True,exist_ok=True); artifact_dir.mkdir(parents=True,exist_ok=True)

    train=add_ids(pd.read_csv(train_csv),"train")
    test=add_ids(pd.read_csv(test_csv),"test")
    combined=pd.concat([train,test],ignore_index=True)

    # Raw/source distribution and validity.
    label_rows=[]
    for split,df in [("train",train),("test",test),("combined",combined)]:
        vc=df["class"].value_counts(dropna=False)
        for label,n in vc.items(): label_rows.append({"split":split,"class":str(label),"n":int(n)})
    pd.DataFrame(label_rows).to_csv(report_dir/"raw_label_distribution.tsv",sep="\t",index=False)

    valid_mask=combined["tweets"].map(valid_text)&combined["class"].isin(SOURCE_LABELS)
    invalid=combined[~valid_mask].copy()
    invalid[["_id","_source_split","_source_row","class"]].to_csv(report_dir/"invalid_rows.tsv",sep="\t",index=False)

    # Primary 3-class pre-audit population.
    target=combined[valid_mask & combined["class"].isin(LABELS)].copy()
    target["text_base"]=target["tweets"].map(normalize_base)
    target["text_clean"]=target["tweets"].map(normalize_clean)
    target["text_aggressive"]=target["tweets"].map(normalize_aggressive)

    # Marker exact prevalence and combinations.
    marker_rows=[]; combo_rows=[]; variant_counter=Counter()
    for _,r in target.iterrows():
        cnt=exact_marker_counts(r["tweets"])
        present=sorted([m for m,c in cnt.items() if c>0])
        for m in MARKERS:
            marker_rows.append({"split":r["_source_split"],"class":r["class"],"marker":m,"present":int(cnt[m]>0),"count":int(cnt[m])})
        combo_rows.append({"split":r["_source_split"],"class":r["class"],"marker_combo":"+".join(present) if present else "<NONE>"})
        for v in marker_like_variants(r["tweets"]): variant_counter[(r["class"],v)]+=1
    mdf=pd.DataFrame(marker_rows)
    magg=mdf.groupby(["split","class","marker"],as_index=False).agg(n_rows=("present","size"),marker_rows=("present","sum"),marker_occurrences=("count","sum"))
    magg["prevalence"]=magg["marker_rows"]/magg["n_rows"]
    magg.to_csv(report_dir/"marker_exact_prevalence.tsv",sep="\t",index=False)
    cdf=pd.DataFrame(combo_rows).groupby(["split","class","marker_combo"],as_index=False).size().rename(columns={"size":"n"})
    cdf.to_csv(report_dir/"marker_cooccurrence.tsv",sep="\t",index=False)
    vrows=[{"class":k[0],"variant":k[1],"n":v} for k,v in sorted(variant_counter.items(),key=lambda kv:(-kv[1],kv[0]))]
    pd.DataFrame(vrows,columns=["class","variant","n"]).to_csv(report_dir/"marker_like_variants.tsv",sep="\t",index=False)

    # Shortcut rule diagnostic (not a trained model).
    def direct_rule(text):
        cnt=exact_marker_counts(text)
        iron=cnt["#irony"]+cnt["#ironic"]
        sarc=cnt["#sarcasm"]+cnt["#sarcastic"]
        if iron>0 and sarc==0:return "irony"
        if sarc>0 and iron==0:return "sarcasm"
        if iron==0 and sarc==0:return "regular"
        return "<AMBIGUOUS>"
    target["marker_rule_pred"]=target["tweets"].map(direct_rule)
    rule=target.groupby(["class","marker_rule_pred"],as_index=False).size().rename(columns={"size":"n"})
    rule.to_csv(report_dir/"marker_direct_rule_diagnostic.tsv",sep="\t",index=False)

    # Duplicate families by view/split.
    fams=[]
    for split,df in [("train",target[target._source_split=="train"]),("test",target[target._source_split=="test"])]:
        for key,view in [("tweets","exact_text"),("text_base","base_normalized"),("text_clean","marker_clean"),("text_aggressive","aggressive_candidate_only")]:
            tab=family_table(df,key,split,view)
            if len(tab): fams.append(tab)
    fam=pd.concat(fams,ignore_index=True) if fams else pd.DataFrame()
    fam.to_csv(report_dir/"within_split_duplicate_families.tsv",sep="\t",index=False)
    if len(fam):
        fsum=fam.groupby(["view","split"],as_index=False).agg(families=("family_key","size"),rows_in_families=("n_rows","sum"),conflicting_families=("n_labels",lambda s:int((s>1).sum())))
    else: fsum=pd.DataFrame(columns=["view","split","families","rows_in_families","conflicting_families"])
    fsum.to_csv(report_dir/"within_split_duplicate_summary.tsv",sep="\t",index=False)

    # Cross-split overlap across multiple normalization views.
    overlap_summaries=[]; overlap_tabs=[]
    trt=target[target._source_split=="train"].copy(); tst=target[target._source_split=="test"].copy()
    for key,view in [("tweets","exact_text"),("text_base","base_normalized"),("text_clean","marker_clean"),("text_aggressive","aggressive_candidate_only")]:
        tab,summ=cross_overlap(trt,tst,key,view); overlap_summaries.append(summ)
        if len(tab): overlap_tabs.append(tab)
    pd.DataFrame(overlap_summaries).to_csv(report_dir/"cross_split_overlap_summary.tsv",sep="\t",index=False)
    (pd.concat(overlap_tabs,ignore_index=True) if overlap_tabs else pd.DataFrame()).to_csv(report_dir/"cross_split_overlap_pairs.tsv",sep="\t",index=False)

    # Structural/provenance clues and length stats.
    struct_rows=[]; length_rows=[]
    for split,df in [("train",target[target._source_split=="train"]),("test",target[target._source_split=="test"]),("combined",target)]:
        for label,sub in [(lab,df[df["class"]==lab]) for lab in LABELS]+[("<ALL>",df)]:
            if not len(sub): continue
            texts=sub["tweets"].astype(str)
            struct_rows.append({
                "split":split,"class":label,"n":len(sub),
                "url_present_n":int(texts.map(lambda x:bool(URL_RE.search(x))).sum()),
                "user_present_n":int(texts.map(lambda x:bool(USER_RE.search(x))).sum()),
                "rt_prefix_n":int(texts.map(lambda x:bool(RT_RE.search(x))).sum()),
                "any_hashtag_n":int(texts.map(lambda x:bool(HASHTAG_RE.search(x))).sum()),
                "mean_hashtag_count":float(texts.map(lambda x:len(HASHTAG_RE.findall(x))).mean()),
            })
            chars=texts.str.len(); tokens=texts.map(lambda x:len(str(x).split()))
            for metric,vals in [("characters",chars),("whitespace_tokens",tokens)]:
                row={"split":split,"class":label,"metric":metric,"n":len(vals)}; row.update(qstats(vals)); length_rows.append(row)
    pd.DataFrame(struct_rows).to_csv(report_dir/"structure_clues.tsv",sep="\t",index=False)
    pd.DataFrame(length_rows).to_csv(report_dir/"text_length_stats.tsv",sep="\t",index=False)

    # H03 final A-C integrity and near duplicates.
    ac_train=pd.read_csv(h03_train); ac_test=pd.read_csv(h03_test)
    final_hash={"train_final_sha256":sha256_file(h03_train),"test_final_sha256":sha256_file(h03_test),"train_n":len(ac_train),"test_n":len(ac_test)}
    (report_dir/"h03_final_input_integrity.json").write_text(json.dumps(final_hash,indent=2),encoding="utf-8")
    near=near_duplicate_audit(ac_train,ac_test,report_dir)

    # Summary.
    summary=[]
    summary.append("# H04 Forensic Data Audit v2")
    summary.append("")
    summary.append(f"- Raw dataset root: `{raw_root}`")
    summary.append(f"- Raw train SHA-256: `{sha256_file(train_csv)}`")
    summary.append(f"- Raw test SHA-256: `{sha256_file(test_csv)}`")
    summary.append(f"- Primary 3-class pre-audit rows: {len(target):,}")
    summary.append(f"- H03 final A-C train/test: {len(ac_train):,} / {len(ac_test):,}")
    summary.append("")
    summary.append("## Near-duplicate screen")
    summary.append("")
    for k,v in near.items(): summary.append(f"- {k}: {v}")
    summary.append("")
    summary.append("**No near-duplicate candidate is removed by H04.** `GROUPED_CLEAN` can only be activated after consolidator review confirms families and materiality.")
    summary.append("")
    summary.append("## Key output files")
    for name in [
        "raw_label_distribution.tsv","marker_exact_prevalence.tsv","marker_cooccurrence.tsv","marker_like_variants.tsv",
        "marker_direct_rule_diagnostic.tsv","within_split_duplicate_summary.tsv","within_split_duplicate_families.tsv",
        "cross_split_overlap_summary.tsv","cross_split_overlap_pairs.tsv","structure_clues.tsv","text_length_stats.tsv",
        "near_duplicate_summary.json","near_duplicate_candidates.tsv","near_duplicate_high_similarity_screen.tsv","near_duplicate_review_sample.tsv"]:
        summary.append(f"- `{name}`")
    (report_dir/"H04_SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")

    decision={
        "milestone":"H04_DATA_AUDIT_V2",
        "algorithmic_high_similarity_test_fraction":near["algorithmic_high_similarity_test_fraction"],
        "grouped_clean_trigger_fraction_for_confirmed_families":GC_TRIGGER,
        "grouped_clean_decision":"PENDING_CONSOLIDATOR_REVIEW",
        "automatic_removals":0,
    }
    (report_dir/"H04_DECISION_INPUT.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")

    print((report_dir/"H04_SUMMARY.md").read_text(encoding="utf-8"))
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nH04_OPERATIONAL_ERROR: {type(e).__name__}: {e}")
        raise SystemExit(0)