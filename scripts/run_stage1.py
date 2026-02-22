from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import (
    custom_id_parts,
    ensure_dir,
    get_message_content,
    get_openai_client,
    load_config,
    parse_json_content,
    read_text,
    resolve_path,
    run_chat_request,
    write_jsonl,
)


def build_requests(config_path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]], Path]:
    cfg = load_config(config_path)
    io_cfg = cfg["io"]
    stg = cfg["stage1"]

    input_dir = resolve_path(config_path, io_cfg["input_dir"])
    output_dir = resolve_path(config_path, io_cfg["output_dir"])
    ensure_dir(output_dir)

    file_glob = io_cfg.get("file_glob", "*.txt")
    files = sorted(input_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No input files found in {input_dir} matching {file_glob}")

    system_prompt = read_text(resolve_path(config_path, stg["system_prompt_file"]))
    with resolve_path(config_path, stg["response_schema_file"]).open("r", encoding="utf-8") as f:
        schema = json.load(f)

    iters = int(stg.get("iterations", 1))
    requests: List[Dict[str, Any]] = []
    for file_path in files:
        paper_text = read_text(file_path)
        for i in range(1, iters + 1):
            custom_id = f"{file_path.stem}__s1_i{i}"
            user_prompt = stg["user_prompt_template"].replace("<<PAPER_TEXT_EXTRACT>>", paper_text)
            body = {
                "model": stg["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": float(stg.get("temperature", 0.0)),
                "max_tokens": int(stg.get("max_tokens", 4096)),
                "response_format": schema,
            }
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
    return cfg, requests, output_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1 extraction runner.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Call the API directly. Without this flag, only request JSONL is generated.",
    )
    ap.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional cap on number of requests (for smoke tests).",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    try:
        _, requests, output_dir = build_requests(config_path)
    except Exception as e:  # noqa: BLE001
        print(f"[STAGE1] Failed to build requests: {e}", file=sys.stderr)
        return 1

    if args.max_requests is not None:
        requests = requests[: max(0, args.max_requests)]

    req_path = output_dir / "stage1_requests.jsonl"
    write_jsonl(req_path, requests)
    print(f"[STAGE1] Wrote request file: {req_path}")

    if not args.execute:
        print("[STAGE1] Dry mode complete. Use --execute to call the API.")
        return 0

    try:
        client = get_openai_client()
    except Exception as e:  # noqa: BLE001
        print(f"[STAGE1] {e}", file=sys.stderr)
        return 1

    raw_rows: List[Dict[str, Any]] = []
    norm_rows: List[Dict[str, Any]] = []
    for idx, req in enumerate(requests, start=1):
        cid = req["custom_id"]
        print(f"[STAGE1] Executing {idx}/{len(requests)}: {cid}")
        try:
            response = run_chat_request(client, req["body"])
            content = get_message_content(response)
            payload = parse_json_content(content)
            status = "ok"
            parse_error = ""
        except Exception as e:  # noqa: BLE001
            payload = {}
            content = ""
            status = "error"
            parse_error = str(e)

        parts = custom_id_parts(cid)
        raw_rows.append(
            {
                "custom_id": cid,
                "paper_id": parts["paper_id"],
                "stage1_iteration": parts["stage1_iteration"],
                "status": status,
                "parse_error": parse_error,
                "response_content": content,
            }
        )

        norm_rows.append(
            {
                "base_custom_id_stage_1": cid,
                "paper_id": parts["paper_id"],
                "stage1_iteration": parts["stage1_iteration"],
                "research_question": payload.get("research_question", ""),
                "causal_identification": payload.get("causal_identification", ""),
                "causal_claim": payload.get("causal_claim", ""),
                "claim_snippets": payload.get("claim_snippets", []),
                "title": payload.get("title", ""),
                "year_of_release": payload.get("year_of_release", None),
            }
        )

    raw_path = output_dir / "stage1_raw_responses.jsonl"
    out_path = output_dir / "stage1_outputs.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(out_path, norm_rows)
    print(f"[STAGE1] Wrote raw responses: {raw_path}")
    print(f"[STAGE1] Wrote normalized outputs: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

