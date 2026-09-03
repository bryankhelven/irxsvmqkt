#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import os
import shutil

HANDLE = "nikhiljohnk/tweets-with-sarcasm-and-irony"
TRAIN_SHA = "f1624804f995f6ca7003a7a023159e0326c283b3b104dfe6e66a55f7fe6c8d77"
TEST_SHA  = "b4318a94673cf28f74a97410f5880b4c6bfd1cfb09090565e13587afbc108020"
RAW_KEY = "3b278d7767d6"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def locate(base: Path, name: str):
    hits = [p for p in base.rglob(name) if p.is_file()]
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]

def main():
    root = Path(os.environ.get("ARTIFACT_ROOT", Path.cwd())).expanduser().resolve()
    dest = root / "data/raw/kaggle_tweets_sarcasm_irony" / RAW_KEY
    train_dest = dest / "train.csv"
    test_dest = dest / "test.csv"

    if train_dest.exists() and test_dest.exists():
        if sha256(train_dest) == TRAIN_SHA and sha256(test_dest) == TEST_SHA:
            print("PASS: frozen internal source dataset already installed.")
            print("  train:", train_dest)
            print("  test: ", test_dest)
            return 0
        print("Existing source files do not match the frozen hashes; reacquiring them.")

    try:
        import kagglehub
    except Exception as exc:
        print("BLOCKER: kagglehub is unavailable.")
        print("Install the core environment first:")
        print("  pip install -r environment/requirements-core.txt")
        print("error:", repr(exc))
        return 0

    download_root = root / "data/raw/_kagglehub_download"
    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True, exist_ok=True)

    print("Downloading public Kaggle dataset automatically:")
    print(" ", HANDLE)
    try:
        resolved = kagglehub.dataset_download(
            HANDLE,
            output_dir=str(download_root),
            force_download=True,
        )
    except Exception as exc:
        print("BLOCKER: automatic KaggleHub download failed.")
        print("The dataset is public and normally does not require authentication.")
        print("If Kaggle asks for consent/authentication, configure KaggleHub and rerun.")
        print("error:", repr(exc))
        return 0

    resolved_path = Path(resolved)
    search_root = resolved_path if resolved_path.exists() else download_root
    train_src = locate(search_root, "train.csv") or locate(download_root, "train.csv")
    test_src = locate(search_root, "test.csv") or locate(download_root, "test.csv")
    if train_src is None or test_src is None:
        print("BLOCKER: downloaded dataset does not expose train.csv and test.csv.")
        print("resolved path:", resolved)
        return 0

    observed_train = sha256(train_src)
    observed_test = sha256(test_src)
    print("Downloaded hashes:")
    print("  train.csv", observed_train)
    print("  test.csv ", observed_test)
    if observed_train != TRAIN_SHA or observed_test != TEST_SHA:
        print("BLOCKER: Kaggle source snapshot changed or unexpected files were resolved.")
        print("The frozen experiment requires the exact hashes in manifests/SOURCES.md.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_src, train_dest)
    shutil.copy2(test_src, test_dest)
    pointer = root / "data/raw/kaggle_tweets_sarcasm_irony/CURRENT_PATH.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(dest) + "\n", encoding="utf-8")
    status = {
        "status": "PASS",
        "dataset_handle": HANDLE,
        "train_sha256": observed_train,
        "test_sha256": observed_test,
        "installed_path": str(dest),
    }
    (dest / "DOWNLOAD_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("PASS: frozen internal source dataset downloaded, verified, and installed.")
    print("  path:", dest)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
