#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import re
import sys
import urllib.request
import pandas as pd

COMMIT = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
BASE = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{COMMIT}/datasets/irony"
FILES = ["test_text.txt", "test_labels.txt", "mapping.txt"]

MARKERS = ["#irony", "#ironic", "#sarcasm", "#sarcastic", "#not"]
MARKER_RE = re.compile(r"(?i)#(?:irony|ironic|sarcasm|sarcastic|not)(?![A-Za-z0-9_])")
URL_RE = re.compile(r"https?://\S+", flags=re.I)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
WS_RE = re.compile(r"\s+")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_external(text):
    s = str(text)
    s = URL_RE.sub("<URL>", s)
    s = MENTION_RE.sub("<USER>", s)
    s = MARKER_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def main():
    project = Path(os.environ.get("ARTIFACT_ROOT", Path.cwd()))
    raw = project / "data/external/tweeteval_irony" / COMMIT
    out = project / "data/derived/h10_external_transfer"
    report = project / "reports/h10_external_transfer"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for name in FILES:
        path = raw / name
        if not path.exists():
            url = f"{BASE}/{name}"
            print("Downloading", url)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                print("BLOCKER download failed", name, repr(exc))
                status = {"status":"BLOCKED","reason":"download_failed","file":name,"error":repr(exc)}
                (report/"PREP_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
                return 0
        downloaded[name] = {"sha256":sha256_file(path), "bytes":path.stat().st_size}

    texts = (raw/"test_text.txt").read_text(encoding="utf-8").splitlines()
    labels_raw = (raw/"test_labels.txt").read_text(encoding="utf-8").splitlines()
    mapping = (raw/"mapping.txt").read_text(encoding="utf-8").strip()

    if len(texts) != 784 or len(labels_raw) != 784:
        status = {
            "status":"BLOCKED","reason":"unexpected_test_size",
            "text_n":len(texts),"label_n":len(labels_raw)
        }
        (report/"PREP_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(json.dumps(status,indent=2))
        return 0

    labels = [int(x.strip()) for x in labels_raw]
    if set(labels) - {0,1}:
        print("BLOCKER unexpected labels", sorted(set(labels)))
        return 0

    marker_counts = {m:0 for m in MARKERS}
    any_marker = 0
    rows=[]
    for i,(text,y) in enumerate(zip(texts,labels), start=1):
        low=text.lower()
        has_any=False
        for m in MARKERS:
            if re.search(r"(?i)"+re.escape(m)+r"(?![A-Za-z0-9_])", text):
                marker_counts[m]+=1
                has_any=True
        any_marker += int(has_any)
        clean=normalize_external(text)
        rows.append({
            "_ext_id": f"TEI_{i:04d}",
            "gold_binary": y,
            "gold_name": "irony" if y==1 else "non_irony",
            "text_raw": text,
            "text_clean": clean,
            "had_frozen_marker": has_any,
        })

    df=pd.DataFrame(rows)

    # HARD postcondition: no frozen marker may remain in cleaned external text.
    remaining_marker_rows = int(
        df["text_clean"].astype(str).apply(lambda x: bool(MARKER_RE.search(x))).sum()
    )
    if remaining_marker_rows != 0:
        status = {
            "status":"BLOCKED",
            "reason":"frozen_marker_remaining_after_clean",
            "remaining_marker_rows":remaining_marker_rows,
        }
        (report/"PREP_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(json.dumps(status,indent=2))
        return 0

    df.to_csv(out/"tweeteval_irony_test_clean.tsv",sep="\t",index=False)

    status={
        "status":"PASS",
        "commit":COMMIT,
        "mapping_text":mapping,
        "test_n":len(df),
        "label_counts":df.gold_name.value_counts().to_dict(),
        "tweets_with_any_frozen_marker":int(any_marker),
        "marker_counts":marker_counts,
        "remaining_frozen_marker_rows_after_clean":remaining_marker_rows,
        "downloads":downloaded,
    }
    (report/"PREP_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())