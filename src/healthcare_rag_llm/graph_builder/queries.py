# src/healthcare_rag_llm/graph_builder/queries.py
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
import pandas as pd

def query_chunks(query_embedding, top_k=5):
    """
    Query chunks using vector similarity search in Neo4j
    
    Args:
        query_embedding: Vector embedding of the query text
        top_k: Number of top similar chunks to return (default: 5)
    
    Returns:
        List of matching chunks with metadata
    """
    connector = Neo4jConnector()
    with connector.driver.session() as session:
        # result = session.run("""
        # CALL db.index.vector.queryNodes('chunk_vec', $k, $query_embedding)
        # YIELD node, score
        # MATCH (node)<-[:HAS_CHUNK]-(d:Document)
        # RETURN node.chunk_id AS chunk_id, node.text AS text, node.pages AS pages,
        #        d.doc_id AS doc_id, d.doc_type AS doc_type,
        #        d.effective_date AS effective_date,
        #        d.authority AS authority,
        #        score
        # ORDER BY score ASC
        # """, {"query_embedding": query_embedding, "k": top_k})

        result = session.run("""
        CALL db.index.vector.queryNodes('chunk_vec', $k, $query_embedding)
        YIELD node, score
        MATCH (node)<-[:HAS_CHUNK]-(d:Document)
        RETURN node.chunk_id AS chunk_id, node.text AS text, node.pages AS pages,
               d.doc_id AS doc_id, d.doc_type AS doc_type,
               d.authority AS authority,
               score
        ORDER BY score ASC
        """, {"query_embedding": query_embedding, "k": top_k})
        return result.data()

def check_match_page_level(gt_doc_ids, gt_page_nos, results,only_highest_score=False):
    """
    Check if retrieval results match the ground truth documents
    
    Args:
        gt_doc_ids: List of ground truth document IDs
        gt_page_nos: List of ground truth page numbers
        results: List of retrieval results from query_chunks
    
    Returns:
        bool or None: True if match, False if no match, None if no ground truth
    """
    if only_highest_score:
        results = [result for result in results if result["score"] == max(result["score"] for result in results)]
    
    if not gt_doc_ids:  # If no ground truth documents
        return None  # Return None to indicate cannot evaluate
    
    doc_id_results = [result["doc_id"] for result in results]
    chunk_page_results = [result["pages"] for result in results]
    # Check document ID matching
    for i,gt_doc_id in enumerate(gt_doc_ids):
        if gt_doc_id not in doc_id_results:
            return False
        p = []
        for r in results:
            if r["doc_id"]  == gt_doc_id:
                for j in r["pages"]:
                    p.append(j)
        for gt_p in gt_page_nos[i]:
            if gt_p not in p:
                return False
    
    # Note: Current query results don't include page field, need to modify query for page matching
    # For now, only check document matching
    return True

def check_match_doc_level(gt_doc_ids, results,only_highest_score=False):
    if only_highest_score:
        results = [result for result in results if result["score"] == max(result["score"] for result in results)]
    if not gt_doc_ids:
        return None
    doc_id_results = [result["doc_id"] for result in results]
    for gt_doc_id in gt_doc_ids:
        if gt_doc_id not in doc_id_results:
            return False
    return True

if __name__ == "__main__":
    queries_for_test = {
    "test_query_1": {
        "Question": "When did redetrmination begin for the COVID-19 Public Health Emergency unwind in New York State",
        "document": {
            "mu_no6_mar23_pr.pdf": [2]
        }
    },
    "test_query_2": {
        "Question": "When did the public health emergency end?",
        "document": {
            "mu_no6_mar23_pr.pdf": [1]
        }
    },
    "test_query_3": {
        "Question": "When submitting a claim for Brixandi, how many units should be indicated on the claim?",
        "document": {
            "mu_no4_apr24_pr.pdf": [4]
        }
    },
    "test_query_4": {
        "Question": "What rate codes should FQHCs use to bill for audio only telehealth?",
        "document": {
            "mu_no3_feb23_speced_pr.pdf": [11]
        }
    },
    "test_query_5": {
        "Question": "Give me a chronological list of the commissioners and what year they first appeared in the medicaid updates.",
        "document": {}
    },
    "test_query_6": {
        "Question": "What are the requirements for appointment scheduling in the medicaid model contract for urgent care?",
        "document": {}
    },
    "test_query_7": {
        "Question": "When did the pharmacy carve out occur?",
        "document": {
            "mu_no04_mar21_speced_pr.pdf": [1],
            "mu_no11_oct22_speced_pr.pdf": [3]
        }
    },
    "test_query_8": {
        "Question": "What are the key components of the SCN program in the NYHER Waiver?",
        "document": {
            "mu_no02_feb25_pr.pdf":[3]
            }
    },
    "test_query_9": {
        "Question": "What constitutes RRP referral requirements?",
        "document": {
            "mu_no01_jan25_pr.pdf":[7]
            }
    },
    "test_query_10": {
        "Question": "what are the requirements for a referral for enrollment in the childrens waiver?",
        "document": {
            }
    },
    "test_query_11": {
        "Question": "What are REC services offered to NYS providers?",
        "document": {
            }
    }       
}



    # Create results DataFrame
    result_pd = pd.DataFrame(columns=["Query", "Ground Truth", "Match Flag_page_level", "Match Flag_doc_level","Top 5 Results",])
    
    # Initialize embedder
    embedder = HealthcareEmbedding()
    
    for query_name, query_info in queries_for_test.items():
        query = query_info["Question"]
        ground_truth = query_info["document"]

        ground_truth_document = list(ground_truth.keys())
        ground_truth_page_no = list(ground_truth.values())

        query_vec = embedder.encode([query])["dense_vecs"][0].tolist()
        top_5_results = query_chunks(query_vec, top_k=5)
        
        match_Flag_page_level = check_match_page_level(ground_truth_document, ground_truth_page_no, top_5_results)
        match_Flag_doc_level = check_match_doc_level(ground_truth_document, top_5_results)
        

        simplified_results = []
        for i, result in enumerate(top_5_results):
            simplified_results.append({
                "rank": i + 1,
                "doc_id": result["doc_id"],
                "pages": result["pages"],
                "score": result["score"]
            })

        new_row = pd.DataFrame({
            "Query": [query_name],
            "Ground Truth": [ground_truth],
            "Match Flag_page_level": [match_Flag_page_level],
            "Match Flag_doc_level": [match_Flag_doc_level],
            "Top 5 Results": [simplified_results]
        })
        result_pd = pd.concat([result_pd, new_row], ignore_index=True)
    # Save results
    result_pd.to_csv("result_pd.csv", index=False)
