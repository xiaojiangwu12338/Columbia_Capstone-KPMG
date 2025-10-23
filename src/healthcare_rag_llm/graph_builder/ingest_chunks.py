# src/healthcare_rag_llm/graph_builder/ingest_chunks.py
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector


def _write_batch(session, batch_rows: List[Dict[str, Any]]):
    """
    Batch write a set of records into Neo4j.

    Structure:
      (:Authority)-[:ISSUED]->(:Document)
        └─[:CONTAINS]->(:Page)
             ├─[:HAS_CHUNK]->(:Chunk {type:'text'})
             ├─[:HAS_TABLE]->(:Chunk {type:'table'})
             └─[:HAS_OCR]->(:Chunk {type:'ocr'})
    """
    session.run("""
    UNWIND $batch AS row
    // --- Authority + Document ---
    MERGE (a:Authority {name: row.authority})
      ON CREATE SET a.abbr = row.authority_abbr
    MERGE (d:Document {doc_id: row.doc_id})
      ON CREATE SET d.title = row.title,
                    d.url = row.url,
                    d.doc_type = row.doc_type,
                    d.effective_date = row.effective_date
    MERGE (a)-[:ISSUED]->(d)

    // --- Pages ---
    WITH row, d
    UNWIND row.pages AS pno
      MERGE (p:Page {uid: row.doc_id + ':' + toString(pno)})
        ON CREATE SET p.doc_id = row.doc_id, p.page_no = pno
      MERGE (d)-[:CONTAINS]->(p)

      // --- Chunk node (all types unified) ---
      MERGE (c:Chunk {chunk_id: row.chunk_id})
        ON CREATE SET c.text = row.text,
                      c.type = row.chunk_type,
                      c.pages = row.pages,
                      c.denseEmbedding = row.denseEmbedding

      // --- Relationship by chunk type ---
      FOREACH (_ IN CASE WHEN row.chunk_type = 'text' THEN [1] ELSE [] END |
        MERGE (p)-[:HAS_CHUNK]->(c)
      )
      FOREACH (_ IN CASE WHEN row.chunk_type = 'table' THEN [1] ELSE [] END |
        MERGE (p)-[:HAS_TABLE]->(c)
      )
      FOREACH (_ IN CASE WHEN row.chunk_type = 'ocr' THEN [1] ELSE [] END |
        MERGE (p)-[:HAS_OCR]->(c)
      )
    """, {"batch": batch_rows})


def ingest_chunks(
    jsonl_path: str,
    doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    batch_size: int = 30
):
    """
    Ingest a single JSONL chunk file into Neo4j.

    Features:
    - Automatically links Authority, Document, Page, Chunk
    - Unifies all chunk types (text/table/ocr)
    - Every chunk gets a denseEmbedding (for vector search)
    - Uses batch mode for stability and performance
    """
    embedder = HealthcareEmbedding()
    connector = Neo4jConnector()

    chunk_file = Path(jsonl_path)
    if not chunk_file.exists():
        raise FileNotFoundError(f"❌ File not found: {jsonl_path}")

    batch, chunk_count = [], 0

    with connector.driver.session() as session:
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Ingesting {chunk_file.name}"):
                record = json.loads(line)
                doc_id = record["doc_id"].strip().lower()
                chunk_id = record["chunk_id"]
                pages = record.get("pages", []) or [1]
                text = record.get("text", "").strip()

                # --- Determine chunk type ---
                if "::table" in chunk_id:
                    chunk_type = "table"
                elif "::ocr" in chunk_id:
                    chunk_type = "ocr"
                else:
                    chunk_type = "text"

                # --- Generate embedding (for all types) ---
                enc = embedder.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
                dense_vec = enc["dense_vecs"][0]
                dense_vec = dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)

                # --- Metadata from CSV ---
                meta = (doc_metadata or {}).get(doc_id, {})
                authority = meta.get("authority", "Unknown")
                authority_abbr = meta.get("authority_abbr", "")
                title = meta.get("title", "")
                url = meta.get("url", "")
                effective_date = meta.get("effective_date", "")
                doc_type = meta.get("doc_type", "PDF")

                batch.append({
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "doc_type": doc_type,
                    "effective_date": effective_date,
                    "authority": authority,
                    "authority_abbr": authority_abbr,
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "text": text,
                    "denseEmbedding": dense_vec,
                    "pages": pages,
                })
                chunk_count += 1

                # --- Batch write ---
                if len(batch) >= batch_size:
                    _write_batch(session, batch)
                    batch = []

            # --- Final batch ---
            if batch:
                _write_batch(session, batch)

    connector.close()
    print(f"✅ Ingested {chunk_count} chunks (including table/OCR) from {chunk_file.name}")