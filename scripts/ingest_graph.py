"""
Run command:
python scripts/ingest_graph.py --chunk_dir data/chunks/asterisk_separate_chunking_result \
    --meta_file data/metadata/metadata_filled.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector
from healthcare_rag_llm.graph_builder.ingest_chunks import ingest_chunks

def load_metadata(meta_path: Path):
    """
    Read metadata_filled.csv into a dict keyed by file_name (doc_id).
    Columns expected:
        authority_name, authority_abbr, doc_title,
        file_name, source_url, effective_date, doc_type
    """
    mapping = {}
    if not meta_path.exists():
        print(f"[WARN] Metadata file not found: {meta_path}")
        return mapping

    with open(meta_path, "r", encoding="utf-8-sig") as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("file_name", "").strip()
            if not doc_id:
                continue
            mapping[doc_id] = {
                "authority": row.get("authority_name", "").strip() or "Unknown",
                "authority_abbr": row.get("authority_abbr", "").strip(),
                "title": row.get("doc_title", "").strip(),
                "url": row.get("source_url", "").strip(),
                "effective_date": row.get("effective_date", "").strip(),
                "doc_type": row.get("doc_type", "").strip().upper() or "PDF",
            }

    print(f"[INFO] Loaded {len(mapping)} metadata rows from {meta_path}")
    # sample = list(mapping.items())[:2]
    # print("  Examples:", sample)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Ingest chunked JSONL files into Neo4j graph.")
    parser.add_argument("--chunk_dir", required=True,
                        help="Directory containing *.chunks.jsonl (or a single .chunks.jsonl file)")
    parser.add_argument("--meta_file", default="data/metadata/metadata_filled.csv",
                        help="Metadata CSV path")
    args = parser.parse_args()

    # Schema init
    connector = Neo4jConnector()
    connector.init_schema()
    connector.close()

    meta = load_metadata(Path(args.meta_file))

    cpath = Path(args.chunk_dir)
    files = []
    if cpath.is_file() and cpath.suffix == ".jsonl":
        files = [cpath]
    else:
        files = list(cpath.glob("*.chunks.jsonl"))

    if not files:
        print(f"[WARN] No .chunks.jsonl files found in {cpath}")
        return

    for fp in files:
        print(f"Ingesting {fp} ...")
        ingest_chunks(str(fp), doc_metadata=meta, batch_size=50)

    print("✅ All chunks ingested into Neo4j.")

if __name__ == "__main__":
    main()
