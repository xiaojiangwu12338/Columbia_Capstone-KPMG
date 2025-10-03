# src/healthcare_rag_llm/graph_builder/ingest_chunks.py

import json
from pathlib import Path
from tqdm import tqdm
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from .neo4j_loader import Neo4jConnector


def ingest_chunks(jsonl_path: str, doc_metadata: dict = None):
    """
    Ingest chunks into Neo4j with embeddings.
    jsonl_path: path to chunks.jsonl
    doc_metadata: optional dict keyed by doc_id with metadata
    """
    embedder = HealthcareEmbedding()
    connector = Neo4jConnector()

    chunk_count = 0
    current_doc = None

    with connector.driver.session() as session:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Ingesting {Path(jsonl_path).name}"):
                record = json.loads(line)
                doc_id = record["doc_id"]
                chunk_id = record["chunk_id"]
                pages = record.get("pages", [])
                text = record["text"]

                # track current doc
                if current_doc is None:
                    current_doc = doc_id

                # generate embedding
                emb = embedder.encode(text)
                if emb is not None:
                    emb = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                else:
                    emb = []

                # document metadata
                meta = doc_metadata.get(doc_id, {}) if doc_metadata else {}
                authority = meta.get("authority", "Unknown")
                doc_type = meta.get("doc_type", "Unknown")
                effective_date = meta.get("effective_date", None)

                print(f"Writing doc={doc_id}, chunk={chunk_id}, text_len={len(text)}")

                # write into Neo4j
                session.run("""
                MERGE (d:Document {doc_id:$doc_id})
                  ON CREATE SET d.doc_type=$doc_type,
                                d.effective_date=$effective_date,
                                d.authority=$authority
                MERGE (c:Chunk {chunk_id:$chunk_id})
                  SET c.text=$text,
                      c.textEmbedding=$embedding,
                      c.pages=$pages
                MERGE (d)-[:HAS_CHUNK]->(c)
                """, {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "effective_date": effective_date,
                    "authority": authority,
                    "chunk_id": chunk_id,
                    "text": text,
                    "embedding": emb,
                    "pages": pages
                })

                chunk_count += 1

    connector.close()
    print(f" Ingested {chunk_count} chunks for document {current_doc} (from {Path(jsonl_path).name})")
