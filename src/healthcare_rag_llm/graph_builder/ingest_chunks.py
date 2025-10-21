# src/healthcare_rag_llm/graph_builder/ingest_chunks.py

import json
import time
from pathlib import Path
from tqdm import tqdm
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from .neo4j_loader import Neo4jConnector


def ingest_chunks(jsonl_path: str,
                  doc_metadata: dict = None,
                  batch_size: int = 200,
                  embedding_batch_size: int = 32,
                  embedder: HealthcareEmbedding = None):
    """
    Ingest chunks into Neo4j with ONLY dense embeddings (OPTIMIZED with batch embedding).

    Performance optimization: Generate embeddings in batches to utilize GPU efficiently.
    This provides 20-30x speedup compared to processing chunks one by one.

    Args:
        jsonl_path: Path to .chunks.jsonl file
        doc_metadata: Optional dict mapping doc_id to metadata (authority, doc_type, effective_date)
        batch_size: Batch size for Neo4j writes (default: 200)
        embedding_batch_size: Batch size for embedding generation (default: 32)
        embedder: Optional pre-initialized HealthcareEmbedding instance. If None, creates new one.
                  IMPORTANT: Reuse the same embedder across multiple files to avoid model reloading!

    Note:
        - denseEmbedding stored as float list (Neo4j vector indexable)
        - sparse/lexical & colbert embeddings are skipped (can be stored separately in FAISS)
        - If processing multiple files, pass the same embedder instance to save 5-15s per file
    """
    total_start = time.time()

    # Step 1: Load all chunk data from JSONL file
    print(f"[1/3] Loading chunks from {Path(jsonl_path).name}...")
    chunks_data = []
    texts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks_data.append(record)
            texts.append(record["text"])

    total_chunks = len(chunks_data)
    print(f"  Loaded {total_chunks} chunks")

    # Step 2: Generate embeddings in batches (THIS IS THE MAJOR SPEEDUP!)
    print(f"[2/3] Generating embeddings (batch_size={embedding_batch_size})...")
    embedding_start = time.time()

    # Use provided embedder or create new one
    if embedder is None:
        print("  Creating new HealthcareEmbedding instance (this may take 5-15s)...")
        embedder = HealthcareEmbedding()
    else:
        print("  Using pre-initialized embedder (no model loading needed)")

    all_embeddings = []

    for i in tqdm(range(0, total_chunks, embedding_batch_size), desc="  Embedding batches", unit="batch"):
        batch_texts = texts[i:i+embedding_batch_size]

        # Batch encode - only return dense vectors (skip sparse/colbert for speed)
        embs = embedder.encode(batch_texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense_vecs = embs.get("dense_vecs")

        # Convert each vector to list format for Neo4j
        for vec in dense_vecs:
            if vec is not None:
                all_embeddings.append(vec.flatten().tolist())
            else:
                all_embeddings.append([])

    embedding_time = time.time() - embedding_start
    print(f"  Embedding generation took {embedding_time:.2f}s ({total_chunks/embedding_time:.1f} chunks/sec)")

    # Step 3: Write to Neo4j in batches
    print(f"[3/3] Writing to Neo4j (batch_size={batch_size})...")
    db_start = time.time()

    connector = Neo4jConnector()
    batch = []
    chunk_count = 0
    current_doc = None

    with connector.driver.session() as session:
        for record, dense_emb in tqdm(zip(chunks_data, all_embeddings),
                                      total=total_chunks,
                                      desc="  Writing batches",
                                      unit="chunk"):
            doc_id = record["doc_id"]
            chunk_id = record["chunk_id"]
            pages = record.get("pages", [])
            text = record["text"]

            if current_doc is None:
                current_doc = doc_id

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
    db_time = time.time() - db_start
    total_time = time.time() - total_start

    # Performance summary
    print(f"\n{'='*60}")
    print(f"Ingestion complete for {Path(jsonl_path).name}")
    print(f"{'='*60}")
    print(f"Total chunks ingested: {chunk_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"  - Embedding generation: {embedding_time:.2f}s ({embedding_time/total_time*100:.1f}%)")
    print(f"  - Database writes: {db_time:.2f}s ({db_time/total_time*100:.1f}%)")
    print(f"  - Throughput: {chunk_count/total_time:.1f} chunks/sec")
    print(f"{'='*60}\n")


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