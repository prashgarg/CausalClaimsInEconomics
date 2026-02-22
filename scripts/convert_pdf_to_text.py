from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_pdftotext(pdf_path: Path, out_txt: Path, first_pages: int | None) -> None:
    cmd = ["pdftotext"]
    if first_pages is not None and first_pages > 0:
        cmd += ["-f", "1", "-l", str(first_pages)]
    cmd += [str(pdf_path), str(out_txt)]
    subprocess.run(cmd, check=True)


def run_pypdf(pdf_path: Path, out_txt: Path, first_pages: int | None) -> None:
    try:
        from pypdf import PdfReader
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("pypdf is not installed. Run `pip install pypdf`.") from e

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    keep = n_pages if first_pages is None else min(n_pages, max(0, first_pages))
    texts = []
    for i in range(keep):
        texts.append(reader.pages[i].extract_text() or "")
    out_txt.write_text("\n".join(texts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert PDFs to plain text.")
    ap.add_argument("--input", required=True, help="Input directory containing PDF files.")
    ap.add_argument("--output", required=True, help="Output directory for .txt files.")
    ap.add_argument(
        "--parser",
        choices=["auto", "pdftotext", "pypdf"],
        default="auto",
        help="Text extraction backend.",
    )
    ap.add_argument(
        "--first-pages",
        type=int,
        default=30,
        help="Number of pages to extract from start of each PDF (default: 30). Use 0 for full PDF.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    first_pages = None if args.first_pages <= 0 else args.first_pages

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[PDF2TXT] No PDFs found in {input_dir}", file=sys.stderr)
        return 1

    for pdf in pdfs:
        out_txt = output_dir / f"{pdf.stem}.txt"
        if args.parser in {"auto", "pdftotext"}:
            try:
                run_pdftotext(pdf, out_txt, first_pages)
                print(f"[PDF2TXT] Wrote {out_txt} via pdftotext")
                continue
            except Exception as e:  # noqa: BLE001
                if args.parser == "pdftotext":
                    print(f"[PDF2TXT] pdftotext failed for {pdf}: {e}", file=sys.stderr)
                    return 1
        try:
            run_pypdf(pdf, out_txt, first_pages)
            print(f"[PDF2TXT] Wrote {out_txt} via pypdf")
        except Exception as e:  # noqa: BLE001
            print(f"[PDF2TXT] pypdf failed for {pdf}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

