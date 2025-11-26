from datetime import datetime, date
from typing import Optional, Union, Iterable, List, Dict, Any

import pandas as pd  # kept because it's used in the __main__ block

from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding


def _normalize_date_filter(
    value: Optional[Union[str, datetime, date]]
) -> Optional[date]:
    """
    Normalize a date-like value into a date object (or None).

    Accepted inputs:
      - None          -> None
      - date          -> same date (unchanged)
      - datetime      -> datetime.date()
      - str           -> parsed as ISO format or 'YYYY-MM-DD'
    """
    if value is None:
        return None

    # Already a date (but not a datetime subclass)
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    # Datetime -> date
    if isinstance(value, datetime):
        return value.date()

    # String formats
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        # First try Python's ISO parser (covers many variants).
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            pass

        # Fallback to a strict YYYY-MM-DD format.
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid date format: {raw}")

    # Any other type is treated as "no usable date"
    return None


def query_chunks(
    query_embedding,
    top_k: int = 5,
    include_table: bool = True,
    include_ocr: bool = True,
    authority_names: Optional[Iterable[str]] = None,
    doc_titles: Optional[Iterable[str]] = None,
    doc_types: Optional[Iterable[str]] = None,
    min_effective_date: Optional[Union[str, datetime, date]] = None,
    max_effective_date: Optional[Union[str, datetime, date]] = None,
    keywords=None,  # currently unused, kept for interface compatibility
) -> List[Dict[str, Any]]:
    """
    Perform a vector search over Chunk.denseEmbedding via the 'chunk_vec' index,
    then traverse to Page, Document, and Authority.

    Returns a list of result dicts, each with:
      - chunk_id, text, chunk_type, doc_id, title, url, doc_type,
        effective_date, authority, pages, score
    """
    connector = Neo4jConnector()
    try:
        with connector.driver.session() as session:
            # Determine which chunk types to consider
            if include_table and include_ocr:
                type_filter = "WHERE c.type IN ['text', 'table', 'ocr']"
            elif include_table:
                type_filter = "WHERE c.type IN ['text', 'table']"
            elif include_ocr:
                type_filter = "WHERE c.type IN ['text', 'ocr']"
            else:
                type_filter = "WHERE c.type = 'text'"

            # Check if we have any filters to apply
            has_filters = any([
                authority_names is not None,
                doc_titles is not None,
                doc_types is not None,
                min_effective_date is not None,
                max_effective_date is not None
            ])

            # Build filter conditions for documents/authorities
            # Use ($param IS NULL OR condition) pattern so None values don't filter
            doc_filter_conditions = []
            doc_filter_conditions.append("($authority_names IS NULL OR a.name IN $authority_names)")
            doc_filter_conditions.append("($doc_titles IS NULL OR d.title IN $doc_titles)")
            doc_filter_conditions.append("($doc_types IS NULL OR d.doc_type IN $doc_types)")
            doc_filter_conditions.append("($min_effective_date IS NULL OR d.effective_date >= $min_effective_date)")
            doc_filter_conditions.append("($max_effective_date IS NULL OR d.effective_date <= $max_effective_date)")
            
            doc_filter_where = "WHERE " + " AND ".join(doc_filter_conditions)

            # Rank ALL chunks first, then filter, then return top k
            # Query all chunks to ensure we rank all relevant chunks
            # Then filter by document/authority criteria, then return top k
            search_k = 999999  # Query all chunks (very large number to get all results)
            
            cypher = f"""
            // Step 1: Rank ALL chunks by vector similarity (query large number to get all relevant chunks)
            CALL db.index.vector.queryNodes('chunk_vec', $search_k, $query_embedding)
            YIELD node, score
            // Step 2: Match chunks and apply type filter
            MATCH (c:Chunk {{chunk_id: node.chunk_id}})
            {type_filter}
            // Step 3: Traverse to documents and authorities to get metadata for filtering
            MATCH (p:Page)-[:HAS_CHUNK|HAS_TABLE|HAS_OCR]->(c)
            MATCH (p)<-[:CONTAINS]-(d:Document)<-[:ISSUED]-(a:Authority)
            // Step 4: Filter by document/authority criteria (when filters are None, conditions evaluate to True)
            {doc_filter_where}
            // Step 5: Return top k from filtered and ranked results (highest scores first)
            RETURN DISTINCT
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
            ORDER BY score DESC
            LIMIT $top_k
            """

            # Always include all filter parameters (even if None) so Cypher NULL checks work
            params = {
                "query_embedding": query_embedding,
                "search_k": search_k,
                "top_k": top_k,
                "authority_names": list(authority_names) if authority_names is not None else None,
                "doc_titles": list(doc_titles) if doc_titles is not None else None,
                "doc_types": list(doc_types) if doc_types is not None else None,
                "min_effective_date": _normalize_date_filter(min_effective_date),
                "max_effective_date": _normalize_date_filter(max_effective_date),
            }

            result = session.run(cypher, params)
            data = result.data()
    finally:
        connector.close()

    return data


def check_match_page_level(
    gt_doc_ids: Optional[Iterable[str]],
    gt_page_nos: Iterable[Iterable[int]],
    results: Optional[List[Dict[str, Any]]],
    only_highest_score: bool = False,
) -> Optional[bool]:
    """
    Check if retrieval results match the ground truth at (doc_id, page) level.

    Inputs:
      - gt_doc_ids: list of ground-truth document IDs
      - gt_page_nos: list of lists of ground-truth page numbers
      - results: list of result dicts from query_chunks
      - only_highest_score: if True, only use the highest-scoring result(s)

    Output:
      - True / False if ground truth provided
      - None if no results or no ground truth doc IDs
    """
    if not results:
        return None

    if only_highest_score:
        max_score = max(r["score"] for r in results)
        results = [r for r in results if r["score"] == max_score]

    if not gt_doc_ids:
        return None

    # Build a mapping: doc_id -> set of retrieved pages
    doc_page_map: Dict[str, set] = {}
    for r in results:
        doc_id = r.get("doc_id")
        if not doc_id:
            continue

        # Try to interpret pages from different shapes ("page" or "pages")
        pages_field = r.get("pages")
        page_single = r.get("page")

        pages: set = set()
        if isinstance(pages_field, (list, tuple, set)):
            pages.update(pages_field)
        elif isinstance(page_single, (int, float)) and page_single is not None:
            pages.add(int(page_single))

        if not pages:
            continue

        if doc_id not in doc_page_map:
            doc_page_map[doc_id] = set()
        doc_page_map[doc_id].update(pages)

    # Compare each ground-truth doc and its pages to retrieved ones
    for idx, gt_doc_id in enumerate(gt_doc_ids):
        if gt_doc_id not in doc_page_map:
            return False

        expected_pages = set(gt_page_nos[idx])
        if not expected_pages.issubset(doc_page_map[gt_doc_id]):
            return False

    return True


def check_match_doc_level(
    gt_doc_ids: Optional[Iterable[str]],
    results: List[Dict[str, Any]],
    only_highest_score: bool = False,
) -> Optional[bool]:
    """
    Check if retrieval results contain all ground-truth document IDs.

    Inputs:
      - gt_doc_ids: iterable of ground-truth document IDs
      - results: list of result dicts from query_chunks
      - only_highest_score: if True, restrict to highest-scoring result(s)

    Output:
      - True / False if ground truth provided
      - None if gt_doc_ids is empty/None
    """
    if not gt_doc_ids:
        return None

    if not results:
        return False

    filtered_results = results
    if only_highest_score:
        max_score = max(r["score"] for r in results)
        filtered_results = [r for r in results if r["score"] == max_score]

    retrieved_doc_ids = {r.get("doc_id") for r in filtered_results if r.get("doc_id")}

    for gt_doc_id in gt_doc_ids:
        if gt_doc_id not in retrieved_doc_ids:
            return False

    return True
