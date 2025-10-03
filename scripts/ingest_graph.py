# scripts/ingest_graph.py
'''
python scripts/ingest_graph.py --chunk_dir data/chunks/asterisk_separate_chunking_result
'''
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector
from healthcare_rag_llm.graph_builder.ingest_chunks import ingest_chunks

def main():
    parser = argparse.ArgumentParser(description="Ingest chunked JSONL files into Neo4j graph.")
    parser.add_argument("--chunk_dir", type=str, required=True,
                        help="Directory containing .chunks.jsonl files")
    parser.add_argument("--meta_file", type=str, default=None,
                        help="Optional metadata JSON/CSV file for documents")
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    if not chunk_dir.exists():
        print(f"Directory not found: {chunk_dir}")
        sys.exit(1)

    # Step 1: initialize schema
    connector = Neo4jConnector()
    connector.init_schema()
    connector.close()

    # Step 2: iterate all chunks.jsonl files
    files = list(chunk_dir.glob("*.chunks.jsonl"))
    if not files:
        print(f"No .chunks.jsonl files found in {chunk_dir}")
        sys.exit(0)

    # Step 3: optional metadata
    doc_metadata = {}
    if args.meta_file:
        if args.meta_file.endswith(".json"):
            import json
            with open(args.meta_file, "r", encoding="utf-8") as f:
                doc_metadata = json.load(f)
        elif args.meta_file.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(args.meta_file)
            doc_metadata = {
                row["doc_id"]: {
                    "authority": row.get("authority"),
                    "doc_type": row.get("doc_type"),
                    "effective_date": row.get("effective_date")
                }
                for _, row in df.iterrows()
            }

    # Step 4: ingest every file
    for file in files:
        print(f"Ingesting {file} ...")
        ingest_chunks(str(file), doc_metadata)

    print("All chunks ingested into Neo4j.")

if __name__ == "__main__":
    main()
