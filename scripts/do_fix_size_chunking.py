import argparse
from healthcare_rag_llm.chunking.fix_size_chunking import fix_size_chunking
from healthcare_rag_llm.utils.io import read_yaml
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Fixed-size chunking from processed JSON files")
    ap.add_argument("--config", "-c", default="configs/ingest_parse.yaml",
                    help="YAML config file (with processed and chunked paths)")
    ap.add_argument("--max-chars", "-s", type=int, default=1200,
                    help="Maximum characters per chunk (default: 1200)")
    ap.add_argument("--overlap", "-o", type=int, default=150,
                    help="Number of overlapping characters between chunks (default: 150)")
    ap.add_argument("--pattern", "-p", default="*.json",
                    help="Glob pattern for input files (default: *.json)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress verbose output")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    processed = Path(cfg["paths"]["processed"])
    chunked = Path(cfg["paths"]["chunked"])

    print(f"[INFO] Chunking from processed={processed} -> chunked={chunked}")
    print(f"[INFO] max_chunk_chars={args.max_chars}, overlap={args.overlap}, pattern={args.pattern}")
    fix_size_chunking(
        processed_dir=str(processed),
        chunked_dir=str(chunked),
        max_chunk_chars=args.max_chars,
        overlap=args.overlap,
        glob_pattern=args.pattern,
        verbose=not args.quiet,
    )
    print("[DONE] Fixed-size chunking complete.")


if __name__ == "__main__":
    main()
