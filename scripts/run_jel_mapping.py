from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import read_jsonl, write_jsonl


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_config(config_path: Path) -> Dict:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config format.")
    return cfg


def resolve_path(config_path: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 JEL mapping (embedding-based).")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--edges-jsonl",
        default=None,
        help="Input edges JSONL. Defaults to <output_dir>/stage2_edges_raw.jsonl.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <output_dir>/stage3_edges_with_jel.jsonl.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k JEL matches to keep for each node (default: 3).",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    io_cfg = cfg["io"]
    stg3 = cfg.get("stage3_jel_mapping", {})

    output_dir = resolve_path(config_path, io_cfg["output_dir"])
    edges_path = (
        Path(args.edges_jsonl).resolve()
        if args.edges_jsonl
        else output_dir / "stage2_edges_raw.jsonl"
    )
    out_path = (
        Path(args.output).resolve()
        if args.output
        else output_dir / "stage3_edges_with_jel.jsonl"
    )
    jel_path = resolve_path(config_path, stg3.get("jel_metadata_csv", "AEA_Information.csv"))
    model = stg3.get("embedding_model", "text-embedding-3-large")

    if not edges_path.exists():
        print(f"[STAGE3] Missing edges input: {edges_path}", file=sys.stderr)
        return 1
    if not jel_path.exists():
        print(f"[STAGE3] Missing JEL metadata CSV: {jel_path}", file=sys.stderr)
        return 1

    try:
        from openai import OpenAI
    except Exception:  # noqa: BLE001
        print("[STAGE3] Missing dependency: openai. Install with `pip install openai`.", file=sys.stderr)
        return 1

    if not __import__("os").environ.get("OPENAI_API_KEY"):
        print("[STAGE3] OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    client = OpenAI()
    edges = read_jsonl(edges_path)
    if not edges:
        print(f"[STAGE3] No rows in {edges_path}", file=sys.stderr)
        return 1

    jel_df = pd.read_csv(jel_path)
    required_cols = {"JEL Code", "Keywords"}
    if not required_cols.issubset(set(jel_df.columns)):
        print(
            f"[STAGE3] JEL metadata must include columns: {sorted(required_cols)}.",
            file=sys.stderr,
        )
        return 1

    jel_df = jel_df.dropna(subset=["JEL Code"]).copy()
    jel_df["desc"] = jel_df["Keywords"].fillna("").astype(str)
    jel_codes: List[str] = jel_df["JEL Code"].astype(str).tolist()
    jel_descs: List[str] = jel_df["desc"].tolist()

    print(f"[STAGE3] Embedding {len(jel_descs)} JEL descriptions...")
    jel_embed = client.embeddings.create(model=model, input=jel_descs).data
    jel_vecs = np.array([x.embedding for x in jel_embed], dtype=float)

    enriched = []
    for i, row in enumerate(edges, start=1):
        cause = str(row.get("cause", "")).strip()
        effect = str(row.get("effect", "")).strip()
        if i % 100 == 0:
            print(f"[STAGE3] Processed {i}/{len(edges)} edges")

        emb = client.embeddings.create(model=model, input=[cause, effect]).data
        cause_vec = np.array(emb[0].embedding, dtype=float)
        effect_vec = np.array(emb[1].embedding, dtype=float)

        cause_scores = np.array([cosine_sim(cause_vec, v) for v in jel_vecs], dtype=float)
        effect_scores = np.array([cosine_sim(effect_vec, v) for v in jel_vecs], dtype=float)

        c_idx = np.argsort(-cause_scores)[: max(1, args.top_k)]
        e_idx = np.argsort(-effect_scores)[: max(1, args.top_k)]

        out_row = dict(row)
        out_row["jel_cause"] = jel_codes[int(c_idx[0])]
        out_row["jel_effect"] = jel_codes[int(e_idx[0])]
        out_row["jel_cause_topk"] = [
            {"jel_code": jel_codes[int(idx)], "cosine_similarity": float(cause_scores[int(idx)])}
            for idx in c_idx
        ]
        out_row["jel_effect_topk"] = [
            {"jel_code": jel_codes[int(idx)], "cosine_similarity": float(effect_scores[int(idx)])}
            for idx in e_idx
        ]
        enriched.append(out_row)

    write_jsonl(out_path, enriched)
    print(f"[STAGE3] Wrote JEL-enriched edges: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

