#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import shutil
import sys

TRAIN_SHA = "f1624804f995f6ca7003a7a023159e0326c283b3b104dfe6e66a55f7fe6c8d77"
TEST_SHA  = "b4318a94673cf28f74a97410f5880b4c6bfd1cfb09090565e13587afbc108020"
RAW_KEY = "3b278d7767d6"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description="Optional fallback: install already-downloaded source CSVs into the frozen artifact layout.")
    ap.add_argument("--train", required=True, help="Path to source train.csv")
    ap.add_argument("--test", required=True, help="Path to source test.csv")
    ap.add_argument("--root", default=".", help="Artifact repository root")
    args = ap.parse_args()

    train = Path(args.train).expanduser().resolve()
    test = Path(args.test).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()

    problems = []
    if not train.exists(): problems.append(f"missing train file: {train}")
    if not test.exists(): problems.append(f"missing test file: {test}")
    if problems:
        print("\n".join("BLOCKER: " + x for x in problems))
        return 0

    observed_train = sha256(train)
    observed_test = sha256(test)
    print("train sha256:", observed_train)
    print("test  sha256:", observed_test)
    if observed_train != TRAIN_SHA or observed_test != TEST_SHA:
        print("BLOCKER: source-file hashes do not match the frozen snapshot.")
        return 0

    dest = root / "data/raw/kaggle_tweets_sarcasm_irony" / RAW_KEY
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train, dest / "train.csv")
    shutil.copy2(test, dest / "test.csv")

    ptr = root / "data/raw/kaggle_tweets_sarcasm_irony/CURRENT_PATH.txt"
    ptr.parent.mkdir(parents=True, exist_ok=True)
    ptr.write_text(str(dest) + "\n", encoding="utf-8")

    print("PASS: source data installed at", dest)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
