#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd, json, hashlib, sys

project=Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))
audit=project/"reports/h04_data_audit_v2"
adj_path=project/"protocols/CONFIRMED_NEAR_DUPLICATE_FAMILIES.tsv"
proc=project/"data/processed/h03_tcc_reproduction"
out=project/"data/derived/h05_grouped_clean"
out.mkdir(parents=True, exist_ok=True)

train_path=proc/"train_final.csv"
test_path=proc/"test_final.csv"
screen_path=audit/"near_duplicate_high_similarity_screen.tsv"

missing=[str(p) for p in [adj_path,train_path,test_path,screen_path] if not p.exists()]
if missing:
    print("BLOCKER missing files:")
    for x in missing: print(" -",x)
    sys.exit(0)

adj=pd.read_csv(adj_path, sep="\t")
screen=pd.read_csv(screen_path, sep="\t")
train=pd.read_csv(train_path)
test=pd.read_csv(test_path)

confirmed=set(adj.loc[adj["adjudication"].eq("CONFIRMED_FAMILY"),"test_id"].astype(str))
edges=screen[screen["test_id"].astype(str).isin(confirmed)].copy()
purge_ids=set(edges["train_id"].astype(str))
test_ids=set(edges["test_id"].astype(str))

if "_id" not in train.columns or "_id" not in test.columns:
    print("BLOCKER expected _id column missing from H03 final CSVs")
    sys.exit(0)

gc_train=train[~train["_id"].astype(str).isin(purge_ids)].copy()
gc_test=test.copy()

# invariants
same_test=list(gc_test["_id"].astype(str))==list(test["_id"].astype(str))
purged_present=int(gc_train["_id"].astype(str).isin(purge_ids).sum())
confirmed_test_present=int(gc_test["_id"].astype(str).isin(test_ids).sum())

summary={
 "method":"group-purged training sensitivity; canonical A-C test preserved",
 "ac_train_n":int(len(train)),
 "ac_test_n":int(len(test)),
 "confirmed_test_ids":int(len(confirmed)),
 "connected_screen_test_ids":int(len(test_ids)),
 "purged_train_ids":int(len(purge_ids)),
 "gc_train_n":int(len(gc_train)),
 "gc_test_n":int(len(gc_test)),
 "same_test_order_and_ids":bool(same_test),
 "purged_train_ids_remaining":purged_present,
 "confirmed_test_ids_present_in_gc_test":confirmed_test_present,
 "confirmed_test_fraction":float(len(confirmed)/len(test)),
}

gc_train.to_csv(out/"train_grouped_clean.csv", index=False)
gc_test.to_csv(out/"test_grouped_clean.csv", index=False)
edges.to_csv(out/"purged_cross_split_edges.tsv", sep="\t", index=False)
pd.DataFrame({"train_id":sorted(purge_ids)}).to_csv(out/"purged_train_ids.tsv",sep="\t",index=False)
(out/"G_C_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

checks = (
    len(train)==58784 and len(test)==5947 and len(confirmed)==85 and
    len(purge_ids)==227 and len(gc_train)==58557 and len(gc_test)==5947 and
    same_test and purged_present==0
)
print(json.dumps(summary,indent=2))
print("G_C_INVARIANTS", "PASS" if checks else "FAIL_REVIEW_REQUIRED")
sys.exit(0)