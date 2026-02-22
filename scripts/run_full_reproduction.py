from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple
import shutil


def run_step(name: str, cmd: List[str], cwd: Path) -> None:
    print(f"[REPRO] START {name}")
    print("[REPRO] CMD   " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)
    print(f"[REPRO] END   {name}")


def pick_existing_script(scripts_dir: Path, candidates: Sequence[str]) -> Path | None:
    for rel in candidates:
        path = scripts_dir / rel
        if path.exists():
            return path
    return None


def find_project_root(start_points: Sequence[Path]) -> Path | None:
    checked: set[Path] = set()
    for start in start_points:
        for base in [start, *start.parents]:
            if base in checked:
                continue
            checked.add(base)
            if (base / "int_data").exists():
                return base
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run assembled-data figure/validation build scripts bundled in this package."
        ),
    )
    ap.add_argument(
        "--project-root",
        default=None,
        help=(
            "Project root containing int_data/. "
            "If omitted, auto-detected from current and script parent directories."
        ),
    )
    ap.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable for Python scripts.",
    )
    ap.add_argument(
        "--rscript-exe",
        default="Rscript",
        help="Rscript executable for R scripts.",
    )
    ap.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Brodeur/Plausibly validation rebuild and table generation.",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Allow missing optional helper scripts. Missing scripts are reported and skipped."
        ),
    )
    ap.add_argument(
        "--run-audit",
        action="store_true",
        help="Run scripts/audit_public_package.py after successful execution.",
    )
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    scripts_dir = script_path.parent
    package_root = script_path.parents[1]
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = find_project_root([Path.cwd().resolve(), package_root])
        if project_root is None:
            print(
                "[REPRO] Could not auto-detect project root (missing int_data/). "
                "Provide --project-root <path>.",
                file=sys.stderr,
            )
            return 1

    py = args.python_exe
    rscript = args.rscript_exe

    step_specs: List[Tuple[str, str, List[str]]] = [
        (
            "Build method figures",
            "r",
            ["build_method_figures.R"],
        ),
        (
            "Build edge-overlap figures",
            "r",
            ["build_edge_overlap_figures.R"],
        ),
        (
            "Build core figures",
            "py",
            ["build_core_figures.py"],
        ),
        (
            "Build predictor figures",
            "py",
            ["build_publication_predictor_figures.py"],
        ),
    ]

    if not args.skip_validation:
        step_specs.extend(
            [
                (
                    "Validation Brodeur",
                    "r",
                    ["validate_brodeur.R"],
                ),
                (
                    "Validation exogenous benchmark",
                    "py",
                    ["validate_exogenous_benchmark.py"],
                ),
                (
                    "Build validation tables",
                    "r",
                    ["build_validation_tables.R"],
                ),
            ]
        )

    steps: List[Tuple[str, List[str], Path]] = []
    missing: List[str] = []
    for name, lang, candidates in step_specs:
        step_script = pick_existing_script(scripts_dir, candidates)
        if step_script is None:
            missing.append(f"{name}: expected one of {', '.join(candidates)}")
            continue
        if lang == "r":
            if shutil.which(rscript) is None:
                common_r = Path(r"C:\Program Files\R\R-4.3.0\bin\Rscript.exe")
                if common_r.exists():
                    rscript = str(common_r)
                else:
                    missing.append(
                        f"{name}: Rscript not found (provide --rscript-exe <path-to-Rscript>)"
                    )
                    continue
            exe = rscript
        else:
            exe = py
        steps.append((name, [exe, str(step_script)], project_root))

    if missing and not args.allow_missing:
        print("[REPRO] Missing required helper scripts:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print(
            "[REPRO] Re-run with --allow-missing to skip unavailable steps.",
            file=sys.stderr,
        )
        return 1

    if not steps:
        print(
            "[REPRO] No runnable assembled-data build steps found in scripts/.",
            file=sys.stderr,
        )
        return 1

    try:
        for name, cmd, cwd in steps:
            run_step(name, cmd, cwd)
    except subprocess.CalledProcessError as e:
        print(f"[REPRO] Failed at step: {e}", file=sys.stderr)
        return 1

    if missing:
        print("[REPRO] Skipped missing helper scripts:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)

    if args.run_audit:
        audit_script = scripts_dir / "audit_public_package.py"
        if not audit_script.exists():
            print(f"[REPRO] Missing audit script: {audit_script}", file=sys.stderr)
            return 1
        try:
            run_step("Public package audit", [py, str(audit_script)], package_root)
        except subprocess.CalledProcessError as e:
            print(f"[REPRO] Audit failed: {e}", file=sys.stderr)
            return 1

    print("[REPRO] Assembled-data reproduction run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
