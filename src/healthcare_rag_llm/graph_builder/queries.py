# src/healthcare_rag_llm/graph_builder/queries.py
from .neo4j_loader import Neo4jConnector
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding

def query_chunks(query_embedding, top_k=5):
    connector = Neo4jConnector()
    with connector.driver.session() as session:
        # result = session.run("""
        # CALL db.index.vector.queryNodes('chunk_vec', $k, $query_embedding)
        # YIELD node, score
        # MATCH (node)-[:ON_PAGE]->(p:Page)<-[:CONTAINS]-(d:Document)
        # OPTIONAL MATCH (d)<-[:ISSUED]-(a:Authority)
        # RETURN node.chunk_id AS chunk_id, node.text AS text,
        #        d.doc_id AS doc_id, d.doc_type AS doc_type,
        #        d.effective_date AS effective_date,
        #        a.name AS authority,
        #        p.page_no AS page, score
        # ORDER BY score ASC
        # """, {"query_embedding": query_embedding, "k": top_k})
        # return result.data()
        result = session.run("""
        CALL db.index.vector.queryNodes('chunk_vec', $k, $query_embedding)
        YIELD node, score
        MATCH (node)<-[:HAS_CHUNK]-(d:Document)
        RETURN node.chunk_id AS chunk_id, node.text AS text,
               d.doc_id AS doc_id, d.doc_type AS doc_type,
               d.effective_date AS effective_date,
               d.authority AS authority,
               score
        ORDER BY score ASC
        """, {"query_embedding": query_embedding, "k": top_k})
        return result.data()

if __name__ == "__main__":
    # 1. Encode the query
    embedder = HealthcareEmbedding()
    query = "What constitutes RRP referral requirements?"
    query_vec = embedder.encode([query])["dense_vecs"][0].tolist()

    # 2. Run the query
    results = query_chunks(query_vec, top_k=5)

    # 3. Print results
    print(f"Query: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"Result {i} (score={r['score']:.4f})")
        print(f"Doc ID: {r['doc_id']}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Text: {r['text'][:300]}...")  # truncate for readability
        print("---")
