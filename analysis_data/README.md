# Analysis Data Bundle

This directory contains analysis-ready data for users who only want the data artifacts.

## Contents

- `tables/`: manuscript and validation output tables (`.csv`).
- `core/graph_aggregated/`: split aggregated graph shards (`claim_graph_edge_metadata_part*.parquet`).
- `core/graph_runs/`: nine run-level graph files (`graph_edges_run_01 ... graph_edges_run_09`) and `run_index.csv`.
- `core/paper_level/`: Stage 1 metadata, EO>=4 baseline paper-level file, citations, gap-filling files.
- `core/paper_level/edge_overlap/`: paper-level EO grid files (`EO >= 1 ... EO = 9`).
- `core/benchmarks/`: external benchmark files used in validation modules.

All distributed files are kept below `25 MiB` for standard GitHub upload compatibility.

## Data-Only Quick Start

If you only need data:

1. Download `analysis_data/`.
2. Read tables directly from `analysis_data/tables/`.

If you want to run scripts using this bundle:

1. From `repro_kit/`, run:

```bash
python scripts/materialize_analysis_data.py --root .
```

2. Then run:

```bash
python scripts/run_full_reproduction.py
```

If you only need joined graph files and not full materialization, run:

```bash
python scripts/join_graph_data_parts.py --analysis-data-dir analysis_data --output-dir int_data
```

## Integrity Manifest

Generate a checksum manifest:

```bash
python scripts/build_analysis_data_manifest.py
```

This writes `analysis_data/manifest.csv` with SHA-256 hashes and file sizes.
