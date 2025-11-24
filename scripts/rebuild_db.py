# scripts/run_all.py
from pathlib import Path
import os, shutil, subprocess

from healthcare_rag_llm.pipelines.ingest_parse import run_pipeline as parse_pipeline
from healthcare_rag_llm.chunking.pattern_chunking import asterisk_separate_chunking
from healthcare_rag_llm.chunking.semantic_chunking import semantic_chunking
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector


ROOT = Path(__file__).resolve().parents[1]

CHUNK_DIR = ROOT / "data" / "chunks" / "mixed_chunking_result"

RAW_MEDICAID = ROOT / "data" / "raw" / "Childrens Evolution of Care" / "State" / "Medicaid Updates"
OUT_MEDICAID = ROOT / "data" / "processed" / "medicaid update"

RAW_WAIVER = ROOT / "data" / "raw" / "Childrens Evolution of Care" / "State" / "Waivers" / "Childrens Waiver"
OUT_WAIVER = ROOT / "data" / "processed" / "children waiver"

INGEST_GRAPH_SCRIPT = ROOT / "scripts" / "ingest_graph.py"
METADATA_FILE = ROOT / "data" / "metadata" / "metadata_filled.csv"


def clear_folder(path):
    for name in os.listdir(path):
        full = path / name
        if full.is_file() or full.is_symlink():
            full.unlink()
        else:
            shutil.rmtree(full)


def reset_graph():
    connector = Neo4jConnector()
    with connector.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    connector.close()
    print("All nodes and relationships deleted from Neo4j.")


def main():
    clear_folder(CHUNK_DIR)

    # parse Medicaid
    parse_pipeline(
        raw_dir=str(RAW_MEDICAID),
        out_dir=str(OUT_MEDICAID),
        save_text=False,
        save_json=True
    )

    # parse Waiver
    parse_pipeline(
        raw_dir=str(RAW_WAIVER),
        out_dir=str(OUT_WAIVER),
        save_text=False,
        save_json=True
    )

    # chunk Medicaid
    asterisk_separate_chunking(
        processed_dir=str(OUT_MEDICAID),
        chunked_dir=str(CHUNK_DIR),
        max_chunk_chars=1200,
        glob_pattern="*.json",
        min_repeats=10,
        separator_char="*",
        verbose=True
    )

    # semantic chunk Waiver
    semantic_chunking(
        processed_dir=str(OUT_WAIVER),
        chunked_dir=str(CHUNK_DIR),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        unit="paragraph",
        similarity_threshold=0.55,
        max_chunk_chars=1200,
        glob_pattern="*.json",
        hysteresis=0.02,
        verbose=True
    )

    # reset graph
    reset_graph()

    # ingest graph using subprocess
    subprocess.run([
        "python",
        str(INGEST_GRAPH_SCRIPT),
        "--chunk_dir", str(CHUNK_DIR),
        "--meta_file", str(METADATA_FILE)
    ], check=True)

    print("DONE")


if __name__ == "__main__":
    main()
