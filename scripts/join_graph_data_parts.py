from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def concat_parquet(parts: list[Path], out_path: Path) -> None:
    if not parts:
        raise FileNotFoundError("No parquet parts found to combine.")
    tables = [pq.read_table(p) for p in parts]
    if len(tables) == 1:
        combined = tables[0]
    else:
        try:
            combined = pa.concat_tables(tables, promote_options="default")
        except TypeError:
            combined = pa.concat_tables(tables, promote=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, out_path, compression="zstd")
    print(f"[JOIN] wrote {out_path} from {len(parts)} part(s)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Join split graph parquet files in analysis_data into combined outputs."
    )
    ap.add_argument(
        "--analysis-data-dir",
        default="analysis_data",
        help="Path to analysis_data directory (default: analysis_data).",
    )
    ap.add_argument(
        "--output-dir",
        default="int_data",
        help="Directory where joined parquet outputs are written (default: int_data).",
    )
    ap.add_argument(
        "--skip-aggregated",
        action="store_true",
        help="Skip joining aggregated graph shards.",
    )
    ap.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip joining run-level graph shards.",
    )
    args = ap.parse_args()

    data_dir = Path(args.analysis_data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing analysis_data directory: {data_dir}")

    if not args.skip_aggregated:
        agg_parts = sorted(
            (data_dir / "core" / "graph_aggregated").glob("claim_graph_edge_metadata_part*.parquet")
        )
        if not agg_parts:
            raise FileNotFoundError(
                "No aggregated graph parts found under core/graph_aggregated/."
            )
        concat_parquet(
            agg_parts,
            out_dir / "claim_graph_all_nine_iter_union_aggregated_meta.parquet",
        )

    if not args.skip_runs:
        run_parts = sorted(
            (data_dir / "core" / "graph_runs").glob("graph_edges_run_*.parquet")
        )
        if not run_parts:
            raise FileNotFoundError("No run-level graph parts found under core/graph_runs/.")
        concat_parquet(run_parts, out_dir / "claim_graph_runs_all.parquet")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
