from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[PIPELINE] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config.")
    return cfg


def resolve_path(config_path: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Stage 1-4 extraction pipeline.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Run live API calls. Without this flag, only request JSONL files are generated.",
    )
    ap.add_argument(
        "--with-stage3",
        action="store_true",
        help="Run Stage 3 JEL mapping after Stage 2.",
    )
    ap.add_argument(
        "--with-snippet",
        action="store_true",
        help="Run snippet-only validation extraction branch.",
    )
    ap.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional cap propagated to Stage 1/2 for smoke tests.",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output_dir = resolve_path(config_path, cfg["io"]["output_dir"])
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    py = sys.executable

    stage1_cmd = [py, str(script_dir / "run_stage1.py"), "--config", str(config_path)]
    stage2_cmd = [py, str(script_dir / "run_stage2.py"), "--config", str(config_path)]
    stage4_cmd = [py, str(script_dir / "run_stage4_overlap.py"), "--config", str(config_path)]
    snippet_cmd = [
        py,
        str(script_dir / "run_validate_snippet.py"),
        "--config",
        str(config_path),
    ]

    if args.execute:
        stage1_cmd.append("--execute")
        stage2_cmd.append("--execute")
        if args.with_snippet:
            snippet_cmd.append("--execute")
    if args.max_requests is not None:
        stage1_cmd += ["--max-requests", str(args.max_requests)]
        stage2_cmd += ["--max-requests", str(args.max_requests)]
        if args.with_snippet:
            snippet_cmd += ["--max-requests", str(args.max_requests)]

    try:
        run_cmd(stage1_cmd, cwd=package_root)
        if not args.execute:
            print(
                "[PIPELINE] Dry mode complete. Stage 1 request JSONL is ready. "
                "Next, submit Stage 1 requests, normalize the responses with "
                "`normalize_stage1_outputs.py`, and then run `run_stage2.py`."
            )
            return 0

        run_cmd(stage2_cmd, cwd=package_root)
        if args.with_snippet:
            run_cmd(snippet_cmd, cwd=package_root)

        edges_for_stage4 = output_dir / "stage2_edges_raw.jsonl"
        if args.with_stage3:
            stage3_cmd = [
                py,
                str(script_dir / "run_jel_mapping.py"),
                "--config",
                str(config_path),
                "--edges-jsonl",
                str(edges_for_stage4),
                "--output",
                str(output_dir / "stage3_edges_with_jel.jsonl"),
            ]
            run_cmd(stage3_cmd, cwd=package_root)
            edges_for_stage4 = output_dir / "stage3_edges_with_jel.jsonl"

        stage4_cmd += ["--edges-jsonl", str(edges_for_stage4)]
        run_cmd(stage4_cmd, cwd=package_root)
    except subprocess.CalledProcessError as e:
        print(f"[PIPELINE] Failed: {e}", file=sys.stderr)
        return 1

    print("[PIPELINE] Completed Stage 1-4 pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
