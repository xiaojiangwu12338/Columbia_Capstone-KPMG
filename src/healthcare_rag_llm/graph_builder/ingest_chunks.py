# src/healthcare_rag_llm/graph_builder/ingest_chunks.py
import json
from datetime import datetime, date
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector


def _parse_effective_date(value: Optional[str]) -> Optional[date]:
    """
    Convert various date string formats into a datetime.date object so Neo4j
    receives a proper DATE value.
    """
    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None

    # Try ISO first (covers most of our rows).
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        pass

    # Fall back to a handful of common US formats.
    candidates = [
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue

    return None


def _write_batch(session, batch_rows: List[Dict[str, Any]]):
    """
    Batch write a set of records into Neo4j.

    Structure:
      (:Authority)-[:ISSUED]->(:Document {category})
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
                    d.effective_date = row.effective_date,
                    d.category = row.category
      // Keep category up to date even if the Document already exists
      SET d.category = coalesce(row.category, d.category)

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

    Graph: Authority -> Document (category property) -> Page -> Chunk
    - Links Authority, Document, Page, Chunk
    - Unifies all chunk types (text/table/ocr)
    - Every chunk gets a denseEmbedding (for vector search)
    - Batch mode for stability/performance
    """
    embedder = HealthcareEmbedding()
    connector = Neo4jConnector()

    chunk_file = Path(jsonl_path)
    if not chunk_file.exists():
        raise FileNotFoundError(f"❌ File not found: {jsonl_path}")

    batch: List[Dict[str, Any]] = []
    chunk_count = 0

    with connector.driver.session() as session:
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Ingesting {chunk_file.name}"):
                record = json.loads(line)

                # --- Normalize core fields ---
                doc_id_raw = (record.get("doc_id") or "").strip()
                if not doc_id_raw:
                    continue
                doc_id = doc_id_raw.lower()  # important: metadata keys should be lowercased too

                chunk_id = record.get("chunk_id")
                pages = record.get("pages", []) or [1]
                text = (record.get("text") or "").strip()

                # --- Chunk type detection ---
                if isinstance(chunk_id, str) and "::table" in chunk_id:
                    chunk_type = "table"
                elif isinstance(chunk_id, str) and "::ocr" in chunk_id:
                    chunk_type = "ocr"
                else:
                    chunk_type = "text"

                # --- Category (document-level) from record, then metadata, then 'unknown' ---
                category = (record.get("category") or "").strip()
                if not category:
                    dm = (doc_metadata or {}).get(doc_id, {})
                    category = (dm.get("category") or "unknown").strip() or "unknown"

                # --- Generate embedding (for all types) ---
                enc = embedder.encode(
                    [text],
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False
                )
                dense_vec = enc["dense_vecs"][0]
                dense_vec = dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)

                # --- Document-level metadata (keyed by lowercased doc_id) ---
                meta = (doc_metadata or {}).get(doc_id, {})
                authority = meta.get("authority", "Unknown")
                authority_abbr = meta.get("authority_abbr", "")
                title = meta.get("title", "")
                url = meta.get("url", "")
                effective_date = _parse_effective_date(
                    record.get("effective_date") or meta.get("effective_date")
                )
                doc_type = meta.get("doc_type", "PDF")

                # --- Accumulate row for batch write ---
                batch.append({
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "doc_type": doc_type,
                    "effective_date": effective_date,
                    "authority": authority,
                    "authority_abbr": authority_abbr,
                    "category": category,          # <- passed to Document level
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "text": text,
                    "denseEmbedding": dense_vec,
                    "pages": pages,
                })
                chunk_count += 1

                # --- Flush batch ---
                if len(batch) >= batch_size:
                    _write_batch(session, batch)
                    batch = []

            # --- Final flush ---
            if batch:
                _write_batch(session, batch)

    connector.close()
    print(f"✅ Ingested {chunk_count} chunks (Document.category set) from {chunk_file.name}")
