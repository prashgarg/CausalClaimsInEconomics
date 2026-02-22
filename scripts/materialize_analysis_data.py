from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[DATA] {src} -> {dst}")


def first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    joined = ", ".join(str(p) for p in paths)
    raise FileNotFoundError(f"None of the expected files exist: {joined}")


def join_parquet(parts: list[Path], dst: Path) -> None:
    if not parts:
        raise FileNotFoundError("No parquet parts were provided to join.")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "pyarrow is required to assemble split graph parquet files. "
            "Install with `pip install -r requirements_full.txt`."
        ) from e

    tables = [pq.read_table(p) for p in parts]
    if len(tables) == 1:
        combined = tables[0]
    else:
        try:
            combined = pa.concat_tables(tables, promote_options="default")
        except TypeError:
            combined = pa.concat_tables(tables, promote=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, dst, compression="zstd")
    print(f"[DATA] joined {len(parts)} part(s) -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Copy analysis_data bundle into expected working paths (int_data, results/tables, benchmarks)."
    )
    ap.add_argument(
        "--root",
        default=".",
        help="Repository root where int_data/ and results/ should be created.",
    )
    ap.add_argument(
        "--analysis-data-dir",
        default=None,
        help="Optional custom path to analysis_data/ (defaults to repro_kit/analysis_data).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    data_dir = Path(args.analysis_data_dir).resolve() if args.analysis_data_dir else package_root / "analysis_data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing analysis_data directory: {data_dir}")

    # Core files for figure/validation builders.
    graph_parts = sorted(
        (data_dir / "core" / "graph_aggregated").glob("claim_graph_edge_metadata_part*.parquet")
    )
    graph_legacy = data_dir / "core" / "claim_graph_edge_metadata.parquet"
    graph_dst = root / "int_data" / "claim_graph_all_nine_iter_union_aggregated_meta.parquet"
    if graph_parts:
        join_parquet(graph_parts, graph_dst)
    else:
        copy_file(graph_legacy, graph_dst)

    for src in sorted((data_dir / "core" / "graph_runs").glob("graph_edges_run_*.parquet")):
        copy_file(src, root / "int_data" / "graph_runs" / src.name)
    run_index = data_dir / "core" / "graph_runs" / "run_index.csv"
    if run_index.exists():
        copy_file(run_index, root / "int_data" / "graph_runs" / "run_index.csv")

    copy_file(
        first_existing([
            data_dir / "core" / "paper_level" / "stage1_paper_metadata.parquet",
            data_dir / "core" / "stage1_paper_metadata.parquet",
        ]),
        root / "int_data" / "claim_graph_s1_aggregated.parquet",
    )
    copy_file(
        first_existing([
            data_dir / "core" / "paper_level" / "paper_level_data_eo_ge4.parquet",
            data_dir / "core" / "paper_level_data_eo_ge4.parquet",
        ]),
        root / "int_data" / "paper_level_data_eo3p.parquet",
    )
    copy_file(
        first_existing([
            data_dir / "core" / "paper_level" / "paper_level_cites.csv",
            data_dir / "core" / "paper_level_cites.csv",
        ]),
        root / "int_data" / "paper_level_cites.csv",
    )
    copy_file(
        first_existing([
            data_dir / "core" / "paper_level" / "gap_filling_non_causal.parquet",
            data_dir / "core" / "gap_filling_non_causal.parquet",
        ]),
        root / "int_data" / "gap_filling_measures_non_causal_cooccurrence_undir_eo3p.parquet",
    )
    copy_file(
        first_existing([
            data_dir / "core" / "paper_level" / "gap_filling_causal.parquet",
            data_dir / "core" / "gap_filling_causal.parquet",
        ]),
        root / "int_data" / "gap_filling_measures_causal_cooccurrence_undir_eo3p.parquet",
    )

    eo_dir = data_dir / "core" / "paper_level" / "edge_overlap"
    if not eo_dir.exists():
        eo_dir = data_dir / "core" / "edge_overlap"
    for src in sorted(eo_dir.glob("paper_level_data_eo_ge*.parquet")):
        # Example source file: paper_level_data_eo_ge4.parquet
        stem = src.stem
        eo_token = stem.replace("paper_level_data_", "")
        dst = root / "int_data" / "edge_overlap_runs" / eo_token / src.name
        copy_file(src, dst)

    copy_file(
        data_dir / "core" / "benchmarks" / "plausibly_exogenous.xlsx",
        root / "int_data" / "4_plausibly_exogenous" / "list_20july2024.xlsx",
    )
    copy_file(
        data_dir / "core" / "benchmarks" / "brodeur_primary.csv",
        root / "abel_etal_EJ_replic_pack" / "data" / "bcn_public_v1_as_csv.csv",
    )
    copy_file(
        data_dir / "core" / "benchmarks" / "brodeur_with_wp.csv",
        root / "abel_etal_EJ_replic_pack" / "data" / "bcn_public_with_wp_v1_as_csv.csv",
    )

    for src in sorted((data_dir / "tables").glob("*.csv")):
        copy_file(src, root / "results" / "tables" / src.name)

    print("[DATA] analysis_data materialization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
