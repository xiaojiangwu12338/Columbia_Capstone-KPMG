# scripts/do_chunking_semantic.py
from pathlib import Path
import argparse
import sys

# Ensure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from healthcare_rag_llm.utils.io import read_yaml
from healthcare_rag_llm.chunking.semantic_chunking import semantic_chunking


def main():
    ap = argparse.ArgumentParser(description="Embedding-based semantic chunking (local, NLTK + sentence-transformers)")
    ap.add_argument("--config", "-c", default="configs/ingest_parse.yaml",
                    help="YAML config with paths.processed and paths.chunked")
    ap.add_argument("--model", "-m", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer model name (local)")
    ap.add_argument("--unit", "-u", choices=["sentence", "paragraph"], default="paragraph",
                    help="Base segmentation unit")
    ap.add_argument("--threshold", "-t", type=float, default=0.55,
                    help="Similarity threshold for merging (0-1)")
    ap.add_argument("--hysteresis", type=float, default=0.02,
                    help="Similarity hysteresis to reduce jitter")
    ap.add_argument("--max-chars", "-s", type=int, default=1200,
                    help="Max characters per chunk (no overlap)")
    ap.add_argument("--pattern", "-p", default="*.json",
                    help="Glob pattern for input JSON files inside processed dir")
    ap.add_argument("--quiet", action="store_true", help="Suppress logs")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    processed = Path(cfg["paths"]["processed"])
    chunked = Path(cfg["paths"]["chunked"])

    chunked_out = chunked / "semantic_chunking_result"

    print(f"[INFO] (semantic) Chunking from processed={processed} -> {chunked}/semantic_chunking_result")

    semantic_chunking(
        processed_dir=str(processed),
        chunked_dir=str(chunked_out),
        model_name=args.model,
        unit=args.unit,
        similarity_threshold=args.threshold,
        max_chunk_chars=args.max_chars,
        glob_pattern=args.pattern,
        hysteresis=args.hysteresis,
        verbose=not args.quiet,
    )
    print("[DONE] Semantic chunking complete.")


if __name__ == "__main__":
    main()
