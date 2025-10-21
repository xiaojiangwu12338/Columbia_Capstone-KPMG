"""
Test retrieval performance without LLM (no API key needed)

This script tests only the embedding + retrieval + reranking pipeline,
without requiring an LLM API key.
"""

import json
from pathlib import Path
from tqdm import tqdm
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.graph_builder.queries import query_chunks
from healthcare_rag_llm.evaluate.evaluate import evaluate_results


def test_retrieval(
    testing_queries_path: str = "data/testing_queries/testing_query.json",
    output_path: str = "data/test_results/retrieval_only.json",
    top_k: int = 5
):
    """
    Test retrieval performance without LLM generation.

    This evaluates how well the system retrieves relevant documents,
    which is independent of LLM answer generation.
    """
    print("="*60)
    print("Testing Retrieval Performance (No LLM required)")
    print("="*60)

    # Load test queries
    with open(testing_queries_path, 'r', encoding='utf-8') as f:
        tests = json.load(f)

    print(f"\nLoaded {len(tests)} test queries")

    # Initialize embedder
    print("\nInitializing embedder...")
    embedder = HealthcareEmbedding()

    # Run retrieval for each query
    results = {}
    print(f"\nRunning retrieval (top_k={top_k})...\n")

    for query_id, payload in tqdm(tests.items(), desc="Queries"):
        question = payload.get("question", "")

        # Encode query
        query_vec = embedder.encode([question], return_dense=True,
                                     return_sparse=False, return_colbert_vecs=False)
        query_embedding = query_vec["dense_vecs"][0].tolist()

        # Retrieve chunks
        retrieved_chunks = query_chunks(query_embedding, top_k=top_k)

        # Store results
        results[f"test_{query_id}"] = {
            "query_id": query_id,
            "question": question,
            "top_k_chunks": retrieved_chunks,
            "answers": None,  # No LLM answer
            "document": None  # Will be evaluated based on chunks only
        }

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Evaluate retrieval accuracy
    print("\nEvaluating retrieval accuracy...")
    eval_results = evaluate_results(
        tested_result_path=output_path,
        ground_truth_path=testing_queries_path,
        output_path=output_path.replace(".json", "_evaluation.json")
    )

    print("\n" + "="*60)
    print("Retrieval Evaluation Results")
    print("="*60)
    print(f"Document-level accuracy: {eval_results['summary']['doc_level_accuracy']:.2%}")
    print(f"Page-level accuracy: {eval_results['summary']['page_level_accuracy']:.2%}")
    print(f"Total tests: {eval_results['summary']['total_tests']}")
    print("="*60)

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test retrieval without LLM")
    parser.add_argument("--queries", default="data/testing_queries/testing_query.json",
                        help="Path to test queries JSON")
    parser.add_argument("--output", default="data/test_results/retrieval_only.json",
                        help="Output path for results")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of chunks to retrieve")

    args = parser.parse_args()

    test_retrieval(
        testing_queries_path=args.queries,
        output_path=args.output,
        top_k=args.top_k
    )
