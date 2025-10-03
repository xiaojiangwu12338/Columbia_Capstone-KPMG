# scripts/ingestion_data.py
from pathlib import Path
import argparse

# parsing pipeline: expects only .pdf and .docx (from raw or interim)
from healthcare_rag_llm.pipelines.ingest_parse import run_pipeline as parse_pipeline

# scripts/ingestion_data.py
from pathlib import Path
import argparse

# parsing pipeline: expects only .pdf and .docx (from raw or interim)
from healthcare_rag_llm.pipelines.ingest_parse import run_pipeline as parse_pipeline

# optional YAML config loader
try:
    from healthcare_rag_llm.utils.io import read_yaml
except Exception:
    read_yaml = None


def main():
    """
    End-to-end parsing (no .doc conversion here).
    Reads .pdf from raw/ and .docx from raw/ or interim/,
    writes structured JSON to processed/.
    """
    ap = argparse.ArgumentParser(description="Parse .pdf/.docx into structured JSON")
    ap.add_argument("--raw", "-r", default="data/raw", help="Raw input folder (.pdf/.docx)")
    ap.add_argument("--processed", "-p", default="data/processed", help="Output folder for JSON")
    ap.add_argument("--config", "-c", default="configs/ingest_parse.yaml", help="Optional YAML config with paths")
    args = ap.parse_args()

    # Load from YAML if provided (takes precedence)
    if args.config:
        if read_yaml is None:
            raise RuntimeError("Config requested but read_yaml is unavailable.")
        cfg = read_yaml(args.config)
        raw = Path(cfg["paths"]["raw"])
        processed = Path(cfg["paths"]["processed"])
    else:
        raw = Path(args.raw)
        processed = Path(args.processed)

    print(f"[INFO] Parsing from raw={raw}-> processed={processed}")
    parse_pipeline(raw_dir=str(raw), out_dir=str(processed),save_text=False,save_json=True)
    print("[DONE] Ingestion (parsing) complete.")


if __name__ == "__main__":
    main()
