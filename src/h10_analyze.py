#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

BOOT=10000
SEED=20260903

def metrics(y,p):
    return {
        "accuracy":float(accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "f1_non_irony":float(f1_score(y,p,pos_label=0,average="binary",zero_division=0)),
        "f1_irony":float(f1_score(y,p,pos_label=1,average="binary",zero_division=0)),
        "confusion":confusion_matrix(y,p,labels=[0,1]).tolist(),
    }

def stratified_boot(y,p,seed=SEED,n=BOOT):
    y=np.asarray(y); p=np.asarray(p)
    rng=np.random.default_rng(seed)
    idx0=np.where(y==0)[0]; idx1=np.where(y==1)[0]
    vals=[]
    for _ in range(n):
        ii=np.concatenate([
            rng.choice(idx0,len(idx0),replace=True),
            rng.choice(idx1,len(idx1),replace=True),
        ])
        vals.append(f1_score(y[ii],p[ii],average="macro",zero_division=0))
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]

def paired_delta_boot(y,p1,p0,seed=SEED,n=BOOT):
    y=np.asarray(y); p1=np.asarray(p1); p0=np.asarray(p0)
    rng=np.random.default_rng(seed)
    idx0=np.where(y==0)[0]; idx1=np.where(y==1)[0]
    vals=[]
    for _ in range(n):
        ii=np.concatenate([
            rng.choice(idx0,len(idx0),replace=True),
            rng.choice(idx1,len(idx1),replace=True),
        ])
        d=f1_score(y[ii],p1[ii],average="macro",zero_division=0)-f1_score(y[ii],p0[ii],average="macro",zero_division=0)
        vals.append(d)
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]

def eval_pop(df,mask):
    sub=df.loc[mask].copy()
    m=metrics(sub.y_true,sub.y_pred)
    m["n"]=int(len(sub))
    m["macro_f1_ci95"]=stratified_boot(sub.y_true.to_numpy(),sub.y_pred.to_numpy())
    return m

def main():
    project=Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))
    run=project/"runs/h10_external_transfer"
    rep=project/"reports/h10_external_transfer"
    prep=json.loads((rep/"PREP_STATUS.json").read_text())
    sparse_status=json.loads((rep/"SPARSE_STATUS.json").read_text())
    bert_status=json.loads((rep/"BERT_STATUS.json").read_text())

    if sparse_status.get("status")!="PASS" or bert_status.get("status")!="PASS":
        status={"status":"PARTIAL","prep":prep,"sparse":sparse_status,"bert":bert_status}
        (rep/"H10_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(json.dumps(status,indent=2))
        return 0

    s2=pd.read_csv(run/"S2_EXTERNAL_PREDICTIONS.tsv",sep="\t")
    berts={seed:pd.read_csv(run/f"BERTWEET_EXTERNAL_seed{seed}.tsv",sep="\t") for seed in [42,2026,7]}

    masks={
        "EXT-C-ALL": np.ones(len(s2),dtype=bool),
        "EXT-C-NOVEL": ~s2.exact_internal_overlap.astype(bool).to_numpy(),
        "EXT-C-HISIM-PURGED": ~s2.high_similarity_train.astype(bool).to_numpy(),
    }

    results={"S2":{},"BERTweet":{}}
    rows=[]
    for pop,mask in masks.items():
        sm=eval_pop(s2,mask)
        results["S2"][pop]=sm
        rows.append({"model":"S2","seed":"","population":pop,**{k:v for k,v in sm.items() if k!="confusion" and k!="macro_f1_ci95"},
                     "ci_low":sm["macro_f1_ci95"][0],"ci_high":sm["macro_f1_ci95"][1]})
        results["BERTweet"][pop]={}
        vals=[]
        for seed,df in berts.items():
            bm=eval_pop(df,mask)
            results["BERTweet"][pop][str(seed)]=bm
            vals.append(bm["macro_f1"])
            rows.append({"model":"BERTweet","seed":seed,"population":pop,**{k:v for k,v in bm.items() if k!="confusion" and k!="macro_f1_ci95"},
                         "ci_low":bm["macro_f1_ci95"][0],"ci_high":bm["macro_f1_ci95"][1]})
        results["BERTweet"][pop]["seed_summary"]={
            "macro_f1_mean":float(np.mean(vals)),
            "macro_f1_sample_sd":float(np.std(vals,ddof=1)),
            "seeds":[42,2026,7],
        }

    # Primary paired external delta on EXT-C-NOVEL, seed42 BERTweet vs S2.
    mask=masks["EXT-C-NOVEL"]
    y=s2.loc[mask,"y_true"].to_numpy()
    pbert=berts[42].loc[mask,"y_pred"].to_numpy()
    ps2=s2.loc[mask,"y_pred"].to_numpy()
    delta=float(f1_score(y,pbert,average="macro",zero_division=0)-f1_score(y,ps2,average="macro",zero_division=0))
    dci=paired_delta_boot(y,pbert,ps2)
    results["primary_paired_delta_bert42_minus_s2"]={"delta_macro_f1":delta,"ci95":dci}

    pd.DataFrame(rows).to_csv(rep/"H10_RESULTS.tsv",sep="\t",index=False)
    (rep/"H10_RESULTS.json").write_text(json.dumps(results,indent=2),encoding="utf-8")

    primary=results["BERTweet"]["EXT-C-NOVEL"]["seed_summary"]
    s2p=results["S2"]["EXT-C-NOVEL"]
    status={
        "status":"PASS",
        "external_test_n":int(len(s2)),
        "external_novel_n":int(masks["EXT-C-NOVEL"].sum()),
        "external_hisim_purged_n":int(masks["EXT-C-HISIM-PURGED"].sum()),
        "tweets_with_frozen_markers_raw":prep["tweets_with_any_frozen_marker"],
        "S2_EXT_C_NOVEL_macro_f1":s2p["macro_f1"],
        "S2_EXT_C_NOVEL_ci95":s2p["macro_f1_ci95"],
        "BERT_EXT_C_NOVEL_mean_macro_f1":primary["macro_f1_mean"],
        "BERT_EXT_C_NOVEL_sample_sd":primary["macro_f1_sample_sd"],
        "BERT42_minus_S2_delta":delta,
        "BERT42_minus_S2_ci95":dci,
    }
    (rep/"H10_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")

    md=[
        "# H10 External Binary Transfer",
        "",
        f"- status: **PASS**",
        f"- TweetEval irony test: {len(s2)} rows",
        f"- raw test rows containing at least one frozen collection/class marker: {prep['tweets_with_any_frozen_marker']}",
        f"- exact-novel primary rows: {int(masks['EXT-C-NOVEL'].sum())}",
        f"- high-similarity-purged sensitivity rows: {int(masks['EXT-C-HISIM-PURGED'].sum())}",
        "",
        "## Primary EXT-C-NOVEL",
        "",
        f"- S2 Macro-F1: **{s2p['macro_f1']:.6f}** [{s2p['macro_f1_ci95'][0]:.6f}, {s2p['macro_f1_ci95'][1]:.6f}]",
        f"- BERTweet 3-seed Macro-F1: **{primary['macro_f1_mean']:.6f} ± {primary['macro_f1_sample_sd']:.6f}**",
        f"- seed42 BERTweet minus S2 paired delta: **{delta:+.6f}** [{dci[0]:+.6f}, {dci[1]:+.6f}]",
        "",
        "## Full result table",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
    ]
    (rep/"H10_SUMMARY.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print((rep/"H10_SUMMARY.md").read_text())
    return 0

if __name__=="__main__":
    raise SystemExit(main())