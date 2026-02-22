from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


BLOCK_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("Possible API key literal", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    (
        "Hardcoded OPENAI_API_KEY assignment",
        re.compile(r"OPENAI_API_KEY\s*[:=]\s*['\"](?!YOUR_KEY_HERE\b)[^'\"]+['\"]"),
    ),
    ("Internal note marker", re.compile(r"(?i)\b(private note|internal note|do not share)\b")),
    ("Project-internal version token", re.compile(r"\b(v1|v2)\b")),
]


ALLOWLIST_REGEX: List[re.Pattern[str]] = [
    re.compile(r"/v1/chat/completions"),  # OpenAI endpoint path
]


def iter_text_files(root: Path) -> Iterable[Path]:
    exts = {
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".json",
        ".py",
        ".ps1",
        ".r",
        ".rmd",
        ".tex",
        ".csv",
    }
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def is_allowlisted(line: str) -> bool:
    return any(p.search(line) for p in ALLOWLIST_REGEX)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit repro_kit for public-release safety.")
    ap.add_argument(
        "--root",
        default=None,
        help="Root directory to scan (defaults to repro_kit root).",
    )
    args = ap.parse_args()

    default_root = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve() if args.root else default_root
    if not root.exists():
        print(f"[AUDIT] Missing root: {root}", file=sys.stderr)
        return 1

    findings: List[Tuple[Path, int, str, str]] = []
    for path in iter_text_files(root):
        if path.name == "audit_public_package.py":
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:  # noqa: BLE001
            continue
        for i, line in enumerate(lines, start=1):
            if is_allowlisted(line):
                continue
            for label, pattern in BLOCK_PATTERNS:
                if pattern.search(line):
                    findings.append((path, i, label, line.strip()))

    report_path = root / "public_audit_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        if not findings:
            f.write("PASS: no blocked patterns found.\n")
        else:
            for path, line_no, label, text in findings:
                f.write(f"{path}:{line_no}: {label}: {text}\n")

    if findings:
        print(f"[AUDIT] FAILED with {len(findings)} finding(s). See {report_path}", file=sys.stderr)
        return 1

    print(f"[AUDIT] PASS. Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
