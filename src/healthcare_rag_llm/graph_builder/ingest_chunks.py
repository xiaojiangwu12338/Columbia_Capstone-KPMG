# src/healthcare_rag_llm/graph_builder/ingest_chunks.py

import json
from pathlib import Path
from tqdm import tqdm
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from .neo4j_loader import Neo4jConnector


def ingest_chunks(jsonl_path: str, doc_metadata: dict = None, batch_size: int = 50):
    """
    Ingest chunks into Neo4j with ONLY dense embeddings.
    - denseEmbedding stored as float list (Neo4j vector indexable)
    - sparse/lexical & colbert embeddings are skipped (stored separately in FAISS if needed)
    """
    embedder = HealthcareEmbedding()
    connector = Neo4jConnector()

    chunk_count = 0
    current_doc = None
    batch = []

    with connector.driver.session() as session:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Ingesting {Path(jsonl_path).name}"):
                record = json.loads(line)
                doc_id = record["doc_id"]
                chunk_id = record["chunk_id"]
                pages = record.get("pages", [])
                text = record["text"]

                if current_doc is None:
                    current_doc = doc_id

                # Generate dense embedding
                embs = embedder.encode(text)
                dense_emb = embs.get("dense_vecs")
                if dense_emb is not None:
                    dense_emb = dense_emb.flatten().tolist()
                else:
                    dense_emb = []

                # Document metadata
                meta = doc_metadata.get(doc_id, {}) if doc_metadata else {}
                authority = meta.get("authority", "Unknown")
                doc_type = meta.get("doc_type", "Unknown")
                effective_date = meta.get("effective_date", None)

                # Collect batch
                batch.append({
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "effective_date": effective_date,
                    "authority": authority,
                    "chunk_id": chunk_id,
                    "text": text,
                    "denseEmbedding": dense_emb,
                    "pages": pages,
                })

                # Commit batch when threshold reached
                if len(batch) >= batch_size:
                    _write_batch(session, batch)
                    chunk_count += len(batch)
                    batch = []

            # Commit remaining batch
            if batch:
                _write_batch(session, batch)
                chunk_count += len(batch)

    connector.close()
    print(f"Ingested {chunk_count} chunks for document {current_doc} (from {Path(jsonl_path).name})")


def _write_batch(session, batch):
    """
    Write a batch of chunk/doc records into Neo4j (dense embeddings only).
    """
    session.run("""
    UNWIND $batch AS row
    MERGE (d:Document {doc_id:row.doc_id})
      ON CREATE SET d.doc_type=row.doc_type,
                    d.effective_date=row.effective_date,
                    d.authority=row.authority
    MERGE (c:Chunk {chunk_id:row.chunk_id})
      SET c.text=row.text,
          c.denseEmbedding=row.denseEmbedding,
          c.pages=row.pages
    MERGE (d)-[:HAS_CHUNK]->(c)
    """, {"batch": batch})