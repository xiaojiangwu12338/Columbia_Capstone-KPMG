# src/healthcare_rag_llm/graph_builder/queries.py
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
import pandas as pd

def query_chunks(
    query_embedding,
    top_k=5,
    include_table=True,
    include_ocr=True,
    authority_names=None,
    doc_titles=None,
    doc_types=None,
    min_effective_date=None,
    max_effective_date=None,
    keywords=None
):
    """
    Vector search over Chunk.denseEmbedding, then traverse to Page, Document, and Authority.
    Optionally include 'table' or 'ocr' type chunks in the search.
    """
    connector = Neo4jConnector()
    with connector.driver.session() as session:
        # Build type filter dynamically
        type_filter = "WHERE c.type = 'text'"
        if include_table and include_ocr:
            type_filter = "WHERE c.type IN ['text', 'table', 'ocr']"
        elif include_table:
            type_filter = "WHERE c.type IN ['text', 'table']"
        elif include_ocr:
            type_filter = "WHERE c.type IN ['text', 'ocr']"

        extra_where = """
        WHERE
          ($authority_names IS NULL OR a.name IN $authority_names)
          AND ($doc_titles IS NULL OR d.title IN $doc_titles)
          AND ($doc_types IS NULL OR d.doc_type IN $doc_types)
          AND ($min_effective_date IS NULL OR d.effective_date >= $min_effective_date)
          AND ($max_effective_date IS NULL OR d.effective_date <= $max_effective_date)
        """

        query = f"""
        CALL db.index.vector.queryNodes('chunk_vec', $k, $query_embedding)
        YIELD node, score
        MATCH (c:Chunk {{chunk_id: node.chunk_id}})
        {type_filter}
        OPTIONAL MATCH (p:Page)-[:HAS_CHUNK|HAS_TABLE|HAS_OCR]->(c)
        OPTIONAL MATCH (p)<-[:CONTAINS]-(d:Document)<-[:ISSUED]-(a:Authority)
        {extra_where}
        RETURN
            c.chunk_id        AS chunk_id,
            c.text            AS text,
            c.type            AS chunk_type,
            d.doc_id          AS doc_id,
            d.title           AS title,
            d.url             AS url,
            d.doc_type        AS doc_type,
            d.effective_date  AS effective_date,
            a.name            AS authority,
            c.pages           AS pages,
            score
        ORDER BY score ASC
        """
        params = {
            "query_embedding": query_embedding,
            "k": top_k,
            "authority_names": authority_names,
            "doc_titles": doc_titles,
            "doc_types": doc_types,
            "min_effective_date": min_effective_date,
            "max_effective_date": max_effective_date,
        }
        result = session.run(query, params)
        data = result.data()

    connector.close()
    return data

def check_match_page_level(gt_doc_ids, gt_page_nos, results, only_highest_score=False):
    """
    Check if retrieval results match the ground truth documents and pages.
    """
    if not results:
        return None

    if only_highest_score:
        max_score = max(r["score"] for r in results)
        results = [r for r in results if r["score"] == max_score]

    if not gt_doc_ids:
        return None

    # Build helper lookups
    doc_page_map = {}
    for r in results:
        doc_page_map.setdefault(r["doc_id"], set()).add(r.get("page"))

    for i, gt_doc_id in enumerate(gt_doc_ids):
        if gt_doc_id not in doc_page_map:
            return False
        gt_pages = set(gt_page_nos[i])
        # Compare with retrieved page numbers
        if not gt_pages.issubset(doc_page_map[gt_doc_id]):
            return False
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
        },
        "Answer":"The process of redeterminiation will begin in April of 2023"
    },
    "test_query_2": {
        "Question": "When did the public health emergency end?",
        "document": {
            "mu_no6_mar23_pr.pdf": [1]
        },
        "Answer":"May 11, 2023: The COVID-19 Public Health Emergency ended"
    },
    "test_query_3": {
        "Question": "When submitting a claim for Brixandi, how many units should be indicated on the claim?",
        "document": {
            "mu_no4_apr24_pr.pdf": [4]
        },
        "Answer":"1 unit should be billed. This is required by CMS and claims should not include values in milligrams or milliliters "
    },
    "test_query_4": {
        "Question": "What rate codes should FQHCs use to bill for audio only telehealth?",
        "document": {
            "mu_no3_feb23_speced_pr.pdf": [11]
        },
        "Answer":"For audio-only telehealth services, Federally Qualified Health Centers (FQHCs) can bill the Prospective Payment System (PPS) rate code \"4012\" or \"4013\", depending on on-site presence"
    },
    "test_query_5": {
        "Question": "Give me a chronological list of the commissioners and what year they first appeared in the medicaid updates.",
        "document": {},
        "Answer": "Zucker Feb 2019, Bassett Nov 2021, McDonald Jan 2025"
    },
    "test_query_6": {
        "Question": "What are the requirements for appointment scheduling in the medicaid model contract for urgent care?",
        "document": {},
        "Answer": "For urgent care under the Medicaid Model Contract 15.2, appointment availability standards require appointments to be scheduled within twenty-four (24) hours of the request"
    },
    "test_query_7": {
        "Question": "When did the pharmacy carve out occur?",
        "document": {
            "mu_no04_mar21_speced_pr.pdf": [1],
            "mu_no11_oct22_speced_pr.pdf": [3]
        },
        "Answer":"The pharmacy carve-out, which transitions the pharmacy benefit from Medicaid Managed Care (MMC) to the Fee-for-Service (FFS) program, was initially set for April 1, 2021, but was delayed to April 1, 2023"
    },
    "test_query_8": {
        "Question": "What are the key components of the SCN program in the NYHER Waiver?",
        "document": {
            "mu_no02_feb25_pr.pdf":[3]
            },
        "Answer": "The SCN program connects NYS Medicaid members to services such as nutritional support, housing assistance, and transportation"
    },
    "test_query_9": {
        "Question": "What constitutes RRP referral requirements?",
        "document": {
            "mu_no01_jan25_pr.pdf":[7]
            },
        "Answer": "RP referral requirements involve a restricted recipient needing to obtain care, either directly or through a referral, from their assigned providers. Here's a breakdown of the requirements:\n• An assigned primary care provider (PCP) is responsible for providing direct medical care or coordinating care through a referral to another medical provider for specialty services.\n• The assigned PCP is also responsible for ordering all non-emergency transportation, laboratory, durable medical equipment (DME), and pharmacy services for the assigned restricted recipient.\n• A referral from the assigned PCP is needed for any non-emergency medical services when a specialist is required.\n• Claims submitted for a restricted recipient will be denied if information for the assigned PCP is not included on the claim as the referring provider.\n• Pharmacies filling prescriptions for a restricted recipient are required to coordinate with the assigned PCP to safely manage the delivery of medications, including verifying referrals if prescriptions were not written by the assigned PCP."
    },
    "test_query_10": {
        "Question": "what are the requirements for a referral for enrollment in the childrens waiver?",
        "document": {
            },
        "Answer": "Children from birth to age 21 may be eligible for certain Medicaid services. • Children/youth in foster care will be enrolled in Medicaid Managed Care Plans (MMCPs). • Children and Youth Evaluation Service (C-YES) conducts HCBS eligibility determinations and serves as the HCBS care coordination alternative to Health Home care management for Medicaid-enrolled children/youth who decline Health Home care management. To obtain a Referral Packet, call C-YES at 1-833-333-CYES (1-833-333-2937) or toll-free at 1-888-329-1541. • Children eligible for NYS Medicaid and Child Health Plus (CHPlus) will be given continuous coverage until six years of age, effective January 1, 2025. • Children/youth in the care of Voluntary Foster Care Agencies (VFCAs) or placed in foster homes certified by Local Departments of Social Services (LDSS) will be enrolled in Medicaid Managed Care Plans (MMCPs) on or after July 1, 2021. • For children/youth enrolling in Medicaid managed care, policy requirements for this transition, including continuity of care requirements, 29-I Health Facility rate information, and 29-I Health Facility services guidelines, can be located on the NYS Department of Health’s 29-I Health Facility (VFCA Transition) web page."
        },
    "test_query_11": {
        "Question": "What are REC services offered to NYS providers?",
        "document": {
            },
      "Answer": "NYS Regional Extension Centers (RECs) offer free support to help providers achieve Meaningful Use of CEHRT. These REC services include:\n• Answers to questions regarding the EHR Incentive Program and its requirements\n• Assistance with selecting and using Certified EHR Technology (CEHRT)\n• Help meeting program objectives\nNYS RECs offer free assistance for all practices and providers located within New York."
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
        top_5_results = query_chunks(query_vec, top_k=5, include_table=True, include_ocr=True)
        
        match_Flag_page_level = check_match_page_level(ground_truth_document, ground_truth_page_no, top_5_results)
        match_Flag_doc_level = check_match_doc_level(ground_truth_document, top_5_results)
        
        simplified_results = []
        for i, result in enumerate(top_5_results):
            simplified_results.append({
                "rank": i + 1,
                "doc_id": result["doc_id"],
                "page": result.get("page"),
                "score": result["score"],
                "chunk_type": result.get("chunk_type")
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
