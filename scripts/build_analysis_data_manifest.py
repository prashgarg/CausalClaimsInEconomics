from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "analysis_data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing analysis_data directory: {data_dir}")

    rows = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(data_dir).as_posix()
        size = p.stat().st_size
        rows.append(
            {
                "path": rel,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 4),
                "sha256": file_sha256(p),
            }
        )

    out = data_dir / "manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "size_bytes", "size_mb", "sha256"])
        w.writeheader()
        w.writerows(rows)

    total_bytes = sum(r["size_bytes"] for r in rows)
    print(f"Wrote {out}")
    print(f"Files: {len(rows)}")
    print(f"Total size: {total_bytes / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
