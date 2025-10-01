import argparse
from healthcare_rag_llm.chunking.fix_size_chunking import fix_size_chunking
from healthcare_rag_llm.utils.io import read_yaml
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Fixed-size chunking from processed JSON files")
    ap.add_argument("--config", "-c", default="configs/ingest_parse.yaml",
                    help="YAML config file (with processed and chunked paths)")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    processed = Path(cfg["paths"]["processed"])
    chunked = Path(cfg["paths"]["chunked"])

    print(f"[INFO] Chunking from processed={processed} -> chunked={chunked}")
    fix_size_chunking(
        processed_dir=str(processed),
        chunked_dir=str(chunked),
        chunk_size=1200,
        overlap=150,
        verbose=True,
    )
    print("[DONE] Fixed-size chunking complete.")


if __name__ == "__main__":
    main()
