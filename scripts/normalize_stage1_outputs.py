from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from common import custom_id_parts, parse_json_content, write_jsonl


def extract_content(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "response_content" in row:
        # Already flattened raw response from run_stage1.py
        custom_id = str(row.get("custom_id", "UNKNOWN"))
        content = str(row.get("response_content", "") or "")
        status = str(row.get("status", "ok"))
        parse_error = str(row.get("parse_error", "") or "")
    else:
        # OpenAI batch-like envelope
        custom_id = str(row.get("custom_id", "UNKNOWN"))
        body = row.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if not choices:
            return None
        content = str(choices[0].get("message", {}).get("content", "") or "")
        status = "ok"
        parse_error = ""

    parts = custom_id_parts(custom_id)
    try:
        payload = parse_json_content(content)
    except Exception as e:  # noqa: BLE001
        payload = {}
        if not parse_error:
            parse_error = str(e)
        status = "error"

    return {
        "base_custom_id_stage_1": custom_id,
        "paper_id": parts["paper_id"],
        "stage1_iteration": parts["stage1_iteration"] or 1,
        "status": status,
        "parse_error": parse_error,
        "research_question": payload.get("research_question", ""),
        "causal_identification": payload.get("causal_identification", ""),
        "causal_claim": payload.get("causal_claim", ""),
        "claim_snippets": payload.get("claim_snippets", []),
        "title": payload.get("title", ""),
        "year_of_release": payload.get("year_of_release", None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize Stage 1 outputs into Stage 2 input format.")
    ap.add_argument("--input", required=True, help="Input JSONL file.")
    ap.add_argument("--output", required=True, help="Output normalized JSONL.")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    if not in_path.exists():
        print(f"[NORMALIZE_STAGE1] Missing input file: {in_path}", file=sys.stderr)
        return 1

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            norm = extract_content(obj)
            if norm is not None:
                rows.append(norm)

    if not rows:
        print("[NORMALIZE_STAGE1] No valid rows parsed from input.", file=sys.stderr)
        return 1

    write_jsonl(out_path, rows)
    print(f"[NORMALIZE_STAGE1] Wrote {len(rows)} normalized rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

