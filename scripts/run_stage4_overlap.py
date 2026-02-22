from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

from common import clean_for_key, read_jsonl


def resolve_path(config_path: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config format.")
    return cfg


def run_key(row: Dict[str, Any]) -> str:
    s1 = int(row.get("stage1_iteration") or 1)
    s2 = int(row.get("stage2_iteration") or 1)
    return f"s1_{s1}__s2_{s2}"


def edge_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    cause = clean_for_key(str(row.get("cause", "")))
    effect = clean_for_key(str(row.get("effect", "")))
    method = clean_for_key(str(row.get("causal_inference_method", "")))
    rel = clean_for_key(str(row.get("type_of_relationship", "")))
    return (cause, effect, method, rel)


def to_bool_causal(method: str) -> int:
    m = method.strip().upper()
    causal_set = {"DID", "IV", "RCT", "RDD", "SYNTHETIC CONTROLS"}
    return int(m in causal_set)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4 edge-overlap aggregation.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--edges-jsonl",
        default=None,
        help="Input edges JSONL. Defaults to <output_dir>/stage2_edges_raw.jsonl.",
    )
    ap.add_argument(
        "--prefix",
        default="stage4",
        help="Prefix for output files in output_dir.",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    io_cfg = cfg["io"]
    stg4 = cfg.get("stage4", {})

    out_dir = resolve_path(config_path, io_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_path = (
        Path(args.edges_jsonl).resolve()
        if args.edges_jsonl
        else out_dir / "stage2_edges_raw.jsonl"
    )
    if not edges_path.exists():
        print(f"[STAGE4] Missing edges input: {edges_path}", file=sys.stderr)
        return 1

    rows = read_jsonl(edges_path)
    if not rows:
        print(f"[STAGE4] No rows found in {edges_path}", file=sys.stderr)
        return 1

    # overlap_index[paper_id][edge_key] -> set(run_ids)
    overlap_index: Dict[str, Dict[Tuple[str, str, str, str], Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    representative: Dict[Tuple[str, Tuple[str, str, str, str]], Dict[str, Any]] = {}
    paper_run_counts: Dict[str, Set[str]] = defaultdict(set)

    for row in rows:
        paper_id = str(row.get("paper_id") or "UNKNOWN")
        rkey = run_key(row)
        ekey = edge_key(row)
        paper_run_counts[paper_id].add(rkey)
        overlap_index[paper_id][ekey].add(rkey)
        representative[(paper_id, ekey)] = row

    threshold_list = stg4.get("report_thresholds", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    threshold_list = sorted({int(x) for x in threshold_list if int(x) >= 1})
    baseline_threshold = int(stg4.get("baseline_threshold", 4))

    aggregated_rows: List[Dict[str, Any]] = []
    for paper_id, edge_map in overlap_index.items():
        total_runs = len(paper_run_counts[paper_id])
        for ekey, runs in edge_map.items():
            base = dict(representative[(paper_id, ekey)])
            base["edge_overlap"] = len(runs)
            base["total_stage2_runs"] = total_runs
            base["edge_overlap_share"] = (
                float(len(runs)) / float(total_runs) if total_runs > 0 else 0.0
            )
            if "is_method_causal_inference" not in base:
                base["is_method_causal_inference"] = to_bool_causal(
                    str(base.get("causal_inference_method", ""))
                )
            aggregated_rows.append(base)

    counts_csv = out_dir / f"{args.prefix}_edge_overlap_counts.csv"
    if aggregated_rows:
        fieldnames = sorted({k for row in aggregated_rows for k in row.keys()})
        with counts_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(aggregated_rows)
    else:
        counts_csv.write_text("", encoding="utf-8")
    print(f"[STAGE4] Wrote overlap counts: {counts_csv}")

    # Threshold-specific edge files
    for thr in threshold_list:
        thr_rows = [r for r in aggregated_rows if int(r.get("edge_overlap", 0)) >= thr]
        out_csv = out_dir / f"{args.prefix}_edges_eo_ge{thr}.csv"
        if thr_rows:
            fieldnames = sorted({k for row in thr_rows for k in row.keys()})
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(thr_rows)
        else:
            out_csv.write_text("", encoding="utf-8")
        print(f"[STAGE4] Wrote EO>={thr} edges: {out_csv}")

    # Paper-level threshold summary
    summary_rows: List[Dict[str, Any]] = []
    by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        by_paper[str(row.get("paper_id") or "UNKNOWN")].append(row)

    for paper_id, rows_p in by_paper.items():
        for thr in threshold_list:
            keep = [r for r in rows_p if int(r.get("edge_overlap", 0)) >= thr]
            n_edges = len(keep)
            n_causal = sum(int(r.get("is_method_causal_inference", 0) == 1) for r in keep)
            share_causal = float(n_causal) / float(n_edges) if n_edges > 0 else 0.0
            summary_rows.append(
                {
                    "paper_id": paper_id,
                    "eo_threshold": thr,
                    "n_edges": n_edges,
                    "n_causal_edges": n_causal,
                    "share_causal_edges": share_causal,
                    "is_baseline_threshold": int(thr == baseline_threshold),
                }
            )

    summary_csv = out_dir / f"{args.prefix}_paper_level_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "paper_id",
                "eo_threshold",
                "n_edges",
                "n_causal_edges",
                "share_causal_edges",
                "is_baseline_threshold",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[STAGE4] Wrote paper-level summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

