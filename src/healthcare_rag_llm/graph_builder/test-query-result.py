"""
Test script for query_chunks function from queries.py

This script tests the query functionality by:
1. Creating a query embedding from a test question
2. Calling query_chunks to retrieve relevant chunks (with and without filters)
3. Printing the results in a formatted way
"""

import json
from typing import Optional, Iterable, Union
from datetime import datetime, date
from healthcare_rag_llm.graph_builder.queries import query_chunks
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding


def print_filters(
    authority_names: Optional[Iterable[str]] = None,
    doc_titles: Optional[Iterable[str]] = None,
    doc_types: Optional[Iterable[str]] = None,
    min_effective_date: Optional[Union[str, datetime, date]] = None,
    max_effective_date: Optional[Union[str, datetime, date]] = None,
    include_table: bool = True,
    include_ocr: bool = True,
):
    """Print filter information in a readable format."""
    print("Filters Applied:")
    if authority_names:
        print(f"  Authority Names: {list(authority_names)}")
    else:
        print(f"  Authority Names: None (no filter)")
    
    if doc_titles:
        print(f"  Document Titles: {list(doc_titles)}")
    else:
        print(f"  Document Titles: None (no filter)")
    
    if doc_types:
        print(f"  Document Types: {list(doc_types)}")
    else:
        print(f"  Document Types: None (no filter)")
    
    if min_effective_date:
        print(f"  Min Effective Date: {min_effective_date}")
    else:
        print(f"  Min Effective Date: None (no filter)")
    
    if max_effective_date:
        print(f"  Max Effective Date: {max_effective_date}")
    else:
        print(f"  Max Effective Date: None (no filter)")
    
    print(f"  Include Table: {include_table}")
    print(f"  Include OCR: {include_ocr}")
    print()


def test_query(
    query_text: str,
    top_k: int = 5,
    authority_names: Optional[Iterable[str]] = None,
    doc_titles: Optional[Iterable[str]] = None,
    doc_types: Optional[Iterable[str]] = None,
    min_effective_date: Optional[Union[str, datetime, date]] = None,
    max_effective_date: Optional[Union[str, datetime, date]] = None,
    include_table: bool = True,
    include_ocr: bool = True,
):
    """
    Test query_chunks with a given query text and optional filters.
    
    Args:
        query_text: The query/question to search for
        top_k: Number of top results to retrieve (default: 5)
        authority_names: Filter by authority names
        doc_titles: Filter by document titles
        doc_types: Filter by document types
        min_effective_date: Minimum effective date filter
        max_effective_date: Maximum effective date filter
        include_table: Whether to include table chunks
        include_ocr: Whether to include OCR chunks
    """
    print("=" * 80)
    print(f"Testing Query: {query_text}")
    print("=" * 80)
    print()
    
    # Print filter information
    print_filters(
        authority_names=authority_names,
        doc_titles=doc_titles,
        doc_types=doc_types,
        min_effective_date=min_effective_date,
        max_effective_date=max_effective_date,
        include_table=include_table,
        include_ocr=include_ocr,
    )
    
    # Initialize embedding model (only once if called multiple times)
    print("Initializing embedding model...")
    embedder = HealthcareEmbedding()
    print()
    
    # Encode query to get embedding vector
    print(f"Encoding query: '{query_text}'")
    query_vec = embedder.encode([query_text])["dense_vecs"][0].tolist()
    print(f"Embedding vector shape: {len(query_vec)} dimensions")
    print()
    
    # Query chunks with filters
    print(f"Querying chunks (top_k={top_k})...")
    results = query_chunks(
        query_vec,
        top_k=top_k,
        authority_names=authority_names,
        doc_titles=doc_titles,
        doc_types=doc_types,
        min_effective_date=min_effective_date,
        max_effective_date=max_effective_date,
        include_table=include_table,
        include_ocr=include_ocr,
    )
    print()
    
    # Print results
    print("=" * 80)
    print(f"Query Results ({len(results)} chunks found):")
    print("=" * 80)
    print()
    
    if not results:
        print("No results found.")
        print("=" * 80)
        return
    
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Score: {result.get('score', 'N/A')}")
        print(f"Chunk ID: {result.get('chunk_id', 'N/A')}")
        print(f"Chunk Type: {result.get('chunk_type', 'N/A')}")
        print(f"Text: {result.get('text', 'N/A')[:200]}..." if len(result.get('text', '')) > 200 else f"Text: {result.get('text', 'N/A')}")
        print(f"Document ID: {result.get('doc_id', 'N/A')}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Document Type: {result.get('doc_type', 'N/A')}")
        print(f"Effective Date: {result.get('effective_date', 'N/A')}")
        print(f"Authority: {result.get('authority', 'N/A')}")
        print(f"Pages: {result.get('pages', 'N/A')}")
        print(f"URL: {result.get('url', 'N/A')}")
        print()
    
    # Print summary
    print("=" * 80)
    print("Summary:")
    print(f"  Total results: {len(results)}")
    if results:
        print(f"  Best score: {min(r.get('score', float('inf')) for r in results)}")
        print(f"  Worst score: {max(r.get('score', float('-inf')) for r in results)}")
        unique_docs = set(r.get('doc_id') for r in results if r.get('doc_id'))
        print(f"  Unique documents: {len(unique_docs)}")
        
        # Verify filters (if applied)
        if authority_names:
            authorities_in_results = set(r.get('authority') for r in results if r.get('authority'))
            expected_authorities = set(authority_names)
            print(f"  Authorities in results: {authorities_in_results}")
            print(f"  All results match authority filter: {authorities_in_results.issubset(expected_authorities)}")
        
        if doc_titles:
            titles_in_results = set(r.get('title') for r in results if r.get('title'))
            expected_titles = set(doc_titles)
            print(f"  Titles in results: {titles_in_results}")
            print(f"  All results match title filter: {titles_in_results.issubset(expected_titles)}")
        
        if doc_types:
            types_in_results = set(r.get('doc_type') for r in results if r.get('doc_type'))
            expected_types = set(doc_types)
            print(f"  Document types in results: {types_in_results}")
            print(f"  All results match doc_type filter: {types_in_results.issubset(expected_types)}")
        
        if min_effective_date or max_effective_date:
            dates_in_results = [r.get('effective_date') for r in results if r.get('effective_date')]
            if dates_in_results:
                print(f"  Effective dates in results: {dates_in_results}")
                if min_effective_date:
                    min_date = min_effective_date if isinstance(min_effective_date, date) else datetime.fromisoformat(str(min_effective_date)).date()
                    all_after_min = all(
                        (d if isinstance(d, date) else datetime.fromisoformat(str(d)).date()) >= min_date 
                        for d in dates_in_results
                    )
                    print(f"  All dates >= min_effective_date: {all_after_min}")
                if max_effective_date:
                    max_date = max_effective_date if isinstance(max_effective_date, date) else datetime.fromisoformat(str(max_effective_date)).date()
                    all_before_max = all(
                        (d if isinstance(d, date) else datetime.fromisoformat(str(d)).date()) <= max_date 
                        for d in dates_in_results
                    )
                    print(f"  All dates <= max_effective_date: {all_before_max}")
    print("=" * 80)


def main():
    """Main function to run a single test query with all filters."""
    
    # Example test query
    test_query_text = "what are major medicaid update publichsed in may 2025"
    
    # Single test with all filter fields
    # Note: Adjust filter values based on your actual database content
    test_query(
        test_query_text,
        top_k=5,
        authority_names=None,
        doc_titles=None,
        doc_types=None,
        min_effective_date="2025-05-01",
        max_effective_date="2025-05-31",
        include_table=True,
        include_ocr=True
    )


if __name__ == "__main__":
    main()

