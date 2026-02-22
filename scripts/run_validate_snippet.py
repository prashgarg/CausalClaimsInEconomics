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
    read_jsonl,
    read_text,
    resolve_path,
    run_chat_request,
    write_jsonl,
)


def build_requests(config_path: Path, stage1_path: Path) -> tuple[List[Dict[str, Any]], Path]:
    cfg = load_config(config_path)
    io_cfg = cfg["io"]
    stg = cfg["stage2_snippet_only"]
    out_dir = resolve_path(config_path, io_cfg["output_dir"])
    ensure_dir(out_dir)

    system_prompt = read_text(resolve_path(config_path, stg["system_prompt_file"]))
    with resolve_path(config_path, stg["response_schema_file"]).open("r", encoding="utf-8") as f:
        schema = json.load(f)

    s1_rows = read_jsonl(stage1_path)
    if not s1_rows:
        raise ValueError(f"No rows found in Stage 1 file: {stage1_path}")

    iters = int(stg.get("iterations", 1))
    requests: List[Dict[str, Any]] = []
    for row in s1_rows:
        paper_id = str(
            row.get("paper_id")
            or row.get("base_custom_id_stage_1", "UNKNOWN").split("__s1_i")[0]
        )
        s1_iter = int(row.get("stage1_iteration") or 1)
        claim_snippets = row.get("claim_snippets", [])
        if not isinstance(claim_snippets, list):
            claim_snippets = [str(claim_snippets)]
        snippets_text = "\n".join(str(x) for x in claim_snippets)

        for j in range(1, iters + 1):
            custom_id = f"{paper_id}__s1_i{s1_iter}__s2_i{j}"
            user_prompt = stg["user_prompt_template"].replace("<<CLAIM_SNIPPETS>>", snippets_text)
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

    return requests, out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Snippet-only Stage 2 extraction runner.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--stage1-jsonl",
        default=None,
        help="Normalized Stage 1 outputs JSONL. Defaults to <output_dir>/stage1_outputs.jsonl.",
    )
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
    cfg = load_config(config_path)
    out_dir = resolve_path(config_path, cfg["io"]["output_dir"])
    stage1_path = (
        Path(args.stage1_jsonl).resolve()
        if args.stage1_jsonl
        else out_dir / "stage1_outputs.jsonl"
    )
    if not stage1_path.exists():
        print(f"[STAGE2_SNIPPET] Missing Stage 1 file: {stage1_path}", file=sys.stderr)
        return 1

    try:
        requests, out_dir = build_requests(config_path, stage1_path)
    except Exception as e:  # noqa: BLE001
        print(f"[STAGE2_SNIPPET] Failed to build requests: {e}", file=sys.stderr)
        return 1

    if args.max_requests is not None:
        requests = requests[: max(0, args.max_requests)]

    req_path = out_dir / "stage2_snippet_requests.jsonl"
    write_jsonl(req_path, requests)
    print(f"[STAGE2_SNIPPET] Wrote request file: {req_path}")

    if not args.execute:
        print("[STAGE2_SNIPPET] Dry mode complete. Use --execute to call the API.")
        return 0

    try:
        client = get_openai_client()
    except Exception as e:  # noqa: BLE001
        print(f"[STAGE2_SNIPPET] {e}", file=sys.stderr)
        return 1

    raw_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    for idx, req in enumerate(requests, start=1):
        cid = req["custom_id"]
        parts = custom_id_parts(cid)
        print(f"[STAGE2_SNIPPET] Executing {idx}/{len(requests)}: {cid}")
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

        raw_rows.append(
            {
                "custom_id": cid,
                "paper_id": parts["paper_id"],
                "stage1_iteration": parts["stage1_iteration"],
                "stage2_iteration": parts["stage2_iteration"],
                "status": status,
                "parse_error": parse_error,
                "response_content": content,
            }
        )

        edges = payload.get("edges", [])
        if not isinstance(edges, list):
            edges = []
        for k, edge in enumerate(edges, start=1):
            if not isinstance(edge, dict):
                continue
            row = {
                "paper_id": parts["paper_id"],
                "base_custom_id_stage_1": f"{parts['paper_id']}__s1_i{parts['stage1_iteration']}",
                "stage1_iteration": parts["stage1_iteration"],
                "stage2_iteration": parts["stage2_iteration"],
                "edge_index_in_response": k,
            }
            row.update(edge)
            edge_rows.append(row)

    raw_path = out_dir / "stage2_snippet_raw_responses.jsonl"
    edges_path = out_dir / "stage2_snippet_edges_raw.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(edges_path, edge_rows)
    print(f"[STAGE2_SNIPPET] Wrote raw responses: {raw_path}")
    print(f"[STAGE2_SNIPPET] Wrote flattened edges: {edges_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

