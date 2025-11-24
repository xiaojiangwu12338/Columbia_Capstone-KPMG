from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from healthcare_rag_llm.utils.io import read_yaml
from healthcare_rag_llm.chunking.pattern_chunking import asterisk_separate_chunking


def main():
    ap = argparse.ArgumentParser(description="Chunk documents using >=N repeated separators (no overlap)")
    ap.add_argument("--config", "-c", default="configs/ingest_parse.yaml",
                    help="YAML config with paths.processed and paths.chunked")
    ap.add_argument("--sep", type=str, default="*", help="Separator character (e.g., *, #, -)")
    ap.add_argument("--min-repeats", "-a", type=int, default=10,
                    help="Minimum consecutive repeats of the separator to split on")
    ap.add_argument("--max-chars", "-s", type=int, default=1200,
                    help="Maximum characters per chunk (segments longer than this are split)")
    ap.add_argument("--pattern", "-p", default="*.json",
                    help="Glob pattern for input JSON files inside processed dir")
    ap.add_argument("--quiet", action="store_true", help="Suppress logs")
    args = ap.parse_args()

    if not args.sep or len(args.sep) != 1:
        raise ValueError("--sep must be a single character")

    cfg = read_yaml(args.config)
    processed = Path(cfg["paths"]["processed"])
    chunked = Path(cfg["paths"]["chunked"])

    # This is the *actual* output directory for this script
    chunked_out = chunked / "asterisk_separate_chunking_result"

    print(f"[INFO] (sep='{args.sep}') Chunking from processed={processed} -> {chunked_out}")

    asterisk_separate_chunking(
        processed_dir=str(processed),
        chunked_dir=str(chunked_out),   # <- note: pass the subfolder here
        max_chunk_chars=args.max_chars,
        glob_pattern=args.pattern,
        min_repeats=args.min_repeats,
        separator_char=args.sep,
        verbose=not args.quiet,
    )
    print("[DONE] Separator-based chunking complete.")


if __name__ == "__main__":
    main()
