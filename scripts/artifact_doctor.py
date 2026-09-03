#!/usr/bin/env python3
from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ARTIFACT_MANIFEST.sha256"
TEXT_SUFFIXES = {".md", ".py", ".sh", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini"}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    failures = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        p = ROOT / rel
        if not p.exists():
            failures.append(f"missing {rel}")
        elif sha(p) != expected:
            failures.append(f"hash mismatch {rel}")

    home_path = re.compile(r"/home/[A-Za-z0-9._-]+/")
    windows_user = re.compile(r"(?i)[A-Z]:\\Users\\[^\\]+\\")
    plausible_email = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    allow_emails = {"artifact@invalid.local"}

    for p in ROOT.rglob("*"):
        if not p.is_file() or p.name == "ARTIFACT_MANIFEST.sha256" or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if home_path.search(text):
            failures.append(f"absolute home path in {p.relative_to(ROOT)}")
        if windows_user.search(text):
            failures.append(f"Windows user path in {p.relative_to(ROOT)}")
        for email in plausible_email.findall(text):
            if email.lower() not in allow_emails:
                failures.append(f"email-like identifier in {p.relative_to(ROOT)}")

    print("ARTIFACT INTEGRITY:", "PASS" if not failures else "FAIL")
    for x in failures:
        print("-", x)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
