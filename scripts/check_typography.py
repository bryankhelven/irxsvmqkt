#!/usr/bin/env python3
from pathlib import Path
import re
import sys
import unicodedata

PROSE_EXTS = {".md", ".txt", ".rst", ".tex", ".bib"}
LONG_ASCII = re.compile(r"-{3,}")

def is_markdown_structure(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Markdown/YAML horizontal separator or front matter.
    if re.fullmatch(r"-{3,}", s):
        return True
    # Markdown table separator, e.g. | --- | :---: |
    if s.startswith("|") and s.endswith("|"):
        residue = s.replace("|", "").replace(":", "").replace("-", "").replace(" ", "").replace("\t", "")
        if residue == "":
            return True
    return False

def scan_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    failures = []
    in_fence = False
    fence_token = None

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()

        if path.suffix.lower() in {".md", ".rst"}:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                token = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_token = token
                elif token == fence_token:
                    in_fence = False
                    fence_token = None
                continue

        for col, ch in enumerate(line, start=1):
            if unicodedata.category(ch) == "Pd" and ch != "-":
                failures.append(
                    f"{path}:{lineno}:{col}: forbidden Unicode dash U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN')}"
                )

        if path.suffix.lower() in PROSE_EXTS and not in_fence and not is_markdown_structure(line):
            m = LONG_ASCII.search(line)
            if m:
                failures.append(
                    f"{path}:{lineno}:{m.start()+1}: inline ASCII dash run longer than '--': {m.group(0)!r}"
                )

    return failures

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    failures = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        failures.extend(scan_file(path))

    if failures:
        print("TYPOGRAPHY AUDIT: FAIL")
        for item in failures:
            print(item)
    else:
        print("TYPOGRAPHY AUDIT: PASS")
        print("No Unicode dash punctuation remains.")
        print("No inline prose dash run longer than '--' remains.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
