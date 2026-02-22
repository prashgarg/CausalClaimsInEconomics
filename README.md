# Causal Claims Reproduction Package

This package supports two transparent workflows:

1. Reproduce all main and appendix exhibits from assembled data (no API calls).
2. Run Stage 1-4 extraction on new papers using your own API key.

No keys are stored in this package. Set credentials only through environment variables.

## Directory structure

- `configs/`: reproducible configuration files.
- `analysis_data/`: downloadable analysis-ready datasets and table outputs.
- `prompts/`: Stage 1/2 prompt text used in retrieval.
- `schemas/`: JSON response schemas for structured model outputs.
- `scripts/`: public runners (full reproduction, extraction pipeline, safety audit).
- `demo_input/`: place user paper text/PDF inputs here.
- `demo_output/`: created at runtime.

## Workflow A: Full paper reproduction from assembled data

Use this when you want all manuscript figures/tables without rerunning extraction.

### Prerequisites

- Python 3.10+
- R 4.3+ (only needed for R-based helper scripts)
- The assembled project data in `int_data/`
- Citation file in `int_data/` with columns `paper_id` and `coalesced_cites` (for citation-based exhibits)

### Install Python dependencies

```bash
pip install -r requirements_full.txt
```

### Run full reproduction

From the `repro_kit/` root:

```bash
python scripts/run_full_reproduction.py
```

Optional flags:

- `--skip-validation` skips Brodeur/Plausibly validation rebuild.
- `--allow-missing` skips missing helper scripts and continues.
- `--run-audit` runs the public-package safety audit at the end.
- `--rscript-exe "<path-to-Rscript>"` sets a custom Rscript binary.

`scripts/run_full_reproduction.py` executes helper scripts that are present in `scripts/` and reports any missing modules explicitly.

Expected helper script names for assembled-data builds:

- `build_method_figures.R`
- `build_edge_overlap_figures.R`
- `build_core_figures.py`
- `build_publication_predictor_figures.py`
- `validate_brodeur.R` (unless `--skip-validation`)
- `validate_exogenous_benchmark.py` (unless `--skip-validation`)
- `build_validation_tables.R` (unless `--skip-validation`)

## Data-only download

If you only want analysis datasets (without running extraction), use:

- `analysis_data/tables/` for ready-to-use output tables.
- `analysis_data/core/graph_aggregated/` for split aggregated graph shards.
- `analysis_data/core/graph_runs/` for nine run-level graph files.
- `analysis_data/core/paper_level/` for paper-level analysis inputs.
- `analysis_data/core/benchmarks/` for benchmark datasets.

All packaged files are below `25 MiB`.

Quick start:

```bash
python scripts/materialize_analysis_data.py --root .
```

If you only want rebuilt graph parquet files (without full materialization), run:

```bash
python scripts/join_graph_data_parts.py --analysis-data-dir analysis_data --output-dir int_data
```

This rebuilds:

- `int_data/claim_graph_all_nine_iter_union_aggregated_meta.parquet` from split aggregated shards.
- `int_data/claim_graph_runs_all.parquet` from the nine run-level files.

This stages `analysis_data` into expected runtime locations (`int_data/`, `results/tables/`, benchmark paths), so all build scripts can run directly.

To generate checksums and sizes:

```bash
python scripts/build_analysis_data_manifest.py
```

## Workflow B: Stage 1-4 extraction on new papers

Use this when you want to run the retrieval pipeline on your own paper(s).

### Step 1: Install extraction dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare input text

Option A (TXT already available):

- Place one or more `.txt` files in `demo_input/`.

Option B (PDF input):

```bash
python scripts/convert_pdf_to_text.py \
  --input demo_input \
  --output demo_input \
  --parser auto \
  --first-pages 30
```

### Step 3: Configure pipeline

- Edit `configs/config.yaml`.
- For a fresh config template, copy from `configs/config.example.yaml`.

### Step 4: Set API key in environment

PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
```

Bash:

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

### Step 5: Run Stage 1-4

Dry run (build JSONL requests only):

```bash
python scripts/run_extraction_pipeline.py \
  --config configs/config.yaml
```

Live API execution:

```bash
python scripts/run_extraction_pipeline.py \
  --config configs/config.yaml \
  --execute \
  --with-snippet \
  --with-stage3
```

Outputs are written to `demo_output/` (or your configured output path), including:

- `stage1_outputs.jsonl`
- `stage2_edges_raw.jsonl`
- `stage2_snippet_edges_raw.jsonl` (if enabled)
- `stage3_edges_with_jel.jsonl` (if enabled)
- `stage4_edge_overlap_counts.csv`
- `stage4_edges_eo_ge*.csv`
- `stage4_paper_level_summary.csv`

## Transparency and safety checks

Run this before release:

```bash
python scripts/audit_public_package.py
```

It fails if it detects:

- hardcoded API key literals,
- explicit key assignments in files,
- internal/private-note markers,
- legacy internal version labels in public-facing content.

## Notes

- This package is designed so users can skip extraction and reproduce exhibits directly from assembled data.
- API usage costs and latency depend on model choice, token usage, and iteration counts in `config.yaml`.
- Keep credentials out of all tracked files. Use environment variables only.
