# src/healthcare_rag_llm/evaluate/llm_evaluate.py
"""
LLM-based evaluation for RAG system responses.
Evaluates answer quality using an LLM as a judge across multiple dimensions.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from healthcare_rag_llm.llm.llm_client import LLMClient


class LLMEvaluator:
    """
    LLM-based evaluator for RAG system responses.
    Uses an LLM as a judge to evaluate answer quality across multiple dimensions.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the LLM evaluator.

        Args:
            llm_client: LLMClient instance for running evaluations
        """
        self.llm_client = llm_client

    def evaluate_faithfulness(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is faithfully grounded in the provided chunks.
        Checks if all claims in the answer can be verified from the chunks.

        Args:
            query: User query
            answer: LLM-generated answer
            chunks: List of retrieved chunks used to generate the answer

        Returns:
            Dict with score (0-1) and reasoning
        """
        chunks_text = "\n\n".join([
            f"Chunk {i+1} (from {chunk.get('doc_id', 'unknown')}, page {chunk.get('page', '?')}):\n{chunk.get('text', '')}"
            for i, chunk in enumerate(chunks)
        ])

        prompt = f"""You are an expert evaluator assessing whether an answer is faithfully grounded in the provided source documents.

QUERY:
{query}

ANSWER TO EVALUATE:
{answer}

SOURCE CHUNKS:
{chunks_text}

TASK:
Evaluate if EVERY claim, statement, and piece of information in the ANSWER can be directly verified from the SOURCE CHUNKS.

EVALUATION CRITERIA:
1. Every factual claim must have direct support in the chunks
2. No hallucinated or inferred information
3. Citations must match actual content
4. Dates, numbers, and specific details must be exact

Rate the faithfulness on a scale of 0.0 to 1.0:
- 1.0: All claims fully supported by chunks
- 0.8-0.9: Most claims supported, minor unsupported details
- 0.5-0.7: Some claims supported, some unsupported
- 0.3-0.4: Many unsupported claims
- 0.0-0.2: Mostly or completely unsupported

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<detailed explanation of what is supported and what is not>",
    "unsupported_claims": ["<list any claims not found in chunks>"]
}}"""

        response = self.llm_client.chat(prompt)
        return self._parse_json_response(response, "faithfulness")

    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer directly addresses the query.

        Args:
            query: User query
            answer: LLM-generated answer

        Returns:
            Dict with score (0-1) and reasoning
        """
        prompt = f"""You are an expert evaluator assessing whether an answer is relevant to the query.

QUERY:
{query}

ANSWER:
{answer}

TASK:
Evaluate if the ANSWER directly and comprehensively addresses the QUERY.

EVALUATION CRITERIA:
1. Does the answer address the main question?
2. Does the answer stay on topic?
3. Is the answer complete for the query asked?
4. Does the answer avoid irrelevant information?

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Perfectly addresses the query
- 0.8-0.9: Addresses query well, minor gaps
- 0.5-0.7: Partially addresses query
- 0.3-0.4: Somewhat relevant but missing key points
- 0.0-0.2: Not relevant or off-topic

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation of relevance assessment>",
    "missing_aspects": ["<list any aspects of the query not addressed>"]
}}"""

        response = self.llm_client.chat(prompt)
        return self._parse_json_response(response, "answer_relevance")

    def evaluate_citation_quality(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality and accuracy of citations in the answer.

        Args:
            answer: LLM-generated answer with citations
            chunks: List of retrieved chunks

        Returns:
            Dict with score (0-1) and reasoning
        """
        chunks_info = "\n".join([
            f"Chunk {i+1}: doc_id={chunk.get('doc_id', 'unknown')}, page={chunk.get('page', '?')}"
            for i, chunk in enumerate(chunks)
        ])

        prompt = f"""You are an expert evaluator assessing citation quality in an answer.

ANSWER WITH CITATIONS:
{answer}

AVAILABLE CHUNKS:
{chunks_info}

EXPECTED CITATION FORMAT: [<doc_id:page>  <date>] or [<doc_id:page>]

TASK:
Evaluate the quality and accuracy of citations.

EVALUATION CRITERIA:
1. Are citations properly formatted?
2. Do cited documents/pages exist in the available chunks?
3. Are all factual claims properly cited?
4. Are citations accurate (correct page numbers, doc IDs)?

Rate the citation quality on a scale of 0.0 to 1.0:
- 1.0: All citations perfect and accurate
- 0.8-0.9: Citations mostly good, minor format issues
- 0.5-0.7: Some citation problems
- 0.3-0.4: Many citation issues
- 0.0-0.2: Poor or missing citations

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation of citation quality>",
    "issues": ["<list any citation problems>"]
}}"""

        response = self.llm_client.chat(prompt)
        return self._parse_json_response(response, "citation_quality")

    def evaluate_completeness(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is complete given the available information.

        Args:
            query: User query
            answer: LLM-generated answer
            chunks: List of retrieved chunks

        Returns:
            Dict with score (0-1) and reasoning
        """
        chunks_text = "\n\n".join([
            f"Chunk {i+1}:\n{chunk.get('text', '')}"
            for i, chunk in enumerate(chunks[:5])  # Limit to top 5 for context
        ])

        prompt = f"""You are an expert evaluator assessing answer completeness.

QUERY:
{query}

ANSWER:
{answer}

AVAILABLE CONTEXT (top chunks):
{chunks_text}

TASK:
Evaluate if the ANSWER provides complete information from the available context to address the query.

EVALUATION CRITERIA:
1. Does the answer effectively use all necessary and relevant information from the provided chunks (not necessarily every chunk)?
2. Are important details from chunks included?
3. Is critical context provided?
4. Does it acknowledge when information is insufficient?

Rate the completeness on a scale of 0.0 to 1.0:
- 1.0: Uses all relevant available information
- 0.8-0.9: Uses most relevant information
- 0.5-0.7: Missing some relevant details
- 0.3-0.4: Missing many relevant details
- 0.0-0.2: Very incomplete

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation of completeness>",
    "missing_info": ["<list important information from chunks not included>"]
}}"""

        response = self.llm_client.chat(prompt)
        return self._parse_json_response(response, "completeness")

    def evaluate_correctness(
        self,
        query: str,
        answer: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Evaluate correctness by comparing answer to ground truth.

        Args:
            query: User query
            answer: LLM-generated answer
            ground_truth: Ground truth answer

        Returns:
            Dict with score (0-1) and reasoning
        """
        prompt = f"""You are an expert evaluator assessing answer correctness against a ground truth.

QUERY:
{query}

GENERATED ANSWER:
{answer}

GROUND TRUTH:
{ground_truth}

TASK:
Evaluate if the GENERATED ANSWER is factually correct compared to the GROUND TRUTH.

EVALUATION CRITERIA:
1. Factual accuracy - are the facts correct?
2. Key information match - does it cover the same key points?
3. Semantic equivalence - does it convey the same meaning?
4. Note: Exact wording doesn't need to match, semantic meaning matters

Rate the correctness on a scale of 0.0 to 1.0:
- 1.0: Semantically equivalent, all facts correct
- 0.8-0.9: Mostly correct, minor differences
- 0.5-0.7: Partially correct
- 0.3-0.4: Significant errors
- 0.0-0.2: Mostly or completely incorrect

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation comparing to ground truth>",
    "differences": ["<list key differences from ground truth>"]
}}"""

        response = self.llm_client.chat(prompt)
        return self._parse_json_response(response, "correctness")

    def evaluate_comprehensive(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all dimensions.

        Args:
            query: User query
            answer: LLM-generated answer
            chunks: Retrieved chunks
            ground_truth: Optional ground truth answer

        Returns:
            Dict with all evaluation results and overall score
        """
        results = {
            "query": query,
            "answer": answer,
            "evaluations": {}
        }

        # Run all evaluations (3 core metrics only)
        print("  Evaluating faithfulness...")
        results["evaluations"]["faithfulness"] = self.evaluate_faithfulness(query, answer, chunks)

        print("  Evaluating answer relevance...")
        results["evaluations"]["answer_relevance"] = self.evaluate_answer_relevance(query, answer)

        # REMOVED: Citation Quality (redundant with faithfulness)
        # REMOVED: Completeness (redundant with answer relevance)

        # Optional: correctness if ground truth available
        if ground_truth:
            print("  Evaluating correctness...")
            results["evaluations"]["correctness"] = self.evaluate_correctness(query, answer, ground_truth)

        # Calculate overall score (weighted average)
        # Simplified to 3 core metrics: faithfulness, answer_relevance, correctness
        if ground_truth:
            weights = {
                "faithfulness": 0.30,
                "answer_relevance": 0.30,
                "correctness": 0.40
            }
        else:
            weights = {
                "faithfulness": 0.60,
                "answer_relevance": 0.40
            }

        overall_score = sum(
            results["evaluations"][metric]["score"] * weight
            for metric, weight in weights.items()
            if metric in results["evaluations"]
        )

        results["overall_score"] = round(overall_score, 3)
        results["weights_used"] = weights

        return results

    def _parse_json_response(self, response: str, metric_name: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM, with fallback handling.

        Args:
            response: LLM response string
            metric_name: Name of the metric being evaluated

        Returns:
            Parsed JSON dict or error dict
        """
        try:
            # Step 1: Clean control characters (keep newlines for structure)
            cleaned_response = response.replace('\t', ' ').replace('\r', '')

            # Step 2: Try to find JSON in the response
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = cleaned_response[start_idx:end_idx]

                # Step 3: Try parsing with standard json.loads
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Step 4: Try non-strict mode if standard parsing fails
                    print(f"  Standard JSON parsing failed for {metric_name}, trying strict=False...")
                    result = json.loads(json_str, strict=False)

                # Validate required fields
                if "score" not in result:
                    raise ValueError("Missing 'score' field")

                # Ensure score is in valid range
                result["score"] = max(0.0, min(1.0, float(result["score"])))

                return result
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            print(f"Warning: Failed to parse {metric_name} response: {e}")
            print(f"Response was: {response[:200]}...")
            return {
                "score": 0.0,
                "reasoning": f"Failed to parse LLM response: {str(e)}",
                "error": str(e),
                "raw_response": response  # Save full response for debugging (no limit)
            }


def evaluate_test_results(
    test_results_path: str,
    output_path: str,
    llm_client: LLMClient,
    ground_truth_path: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate all test results using LLM-based evaluation.

    Args:
        test_results_path: Path to test results JSON
        output_path: Path to save evaluation results
        llm_client: LLMClient for evaluation
        ground_truth_path: Optional path to ground truth data
        limit: Optional limit on number of tests to evaluate

    Returns:
        Evaluation results dict
    """
    print(f"Loading test results from: {test_results_path}")
    with open(test_results_path, 'r', encoding='utf-8') as f:
        test_results = json.load(f)

    # Load ground truth if provided
    ground_truth = {}
    if ground_truth_path and Path(ground_truth_path).exists():
        print(f"Loading ground truth from: {ground_truth_path}")
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            # Use the keys directly (e.g., "Test_query_1") - no need to convert
            # The keys already match the query_id in test_results
            ground_truth = gt_data if isinstance(gt_data, dict) else {}
            print(f"Loaded {len(ground_truth)} ground truth entries")

    evaluator = LLMEvaluator(llm_client)
    evaluation_results = {
        "metadata": {
            "test_results_file": test_results_path,
            "ground_truth_file": ground_truth_path,
            "evaluator_model": llm_client.model,
            "total_tests": len(test_results)
        },
        "results": {},
        "summary": {}
    }

    # Evaluate each test
    test_items = list(test_results.items())
    if limit:
        test_items = test_items[:limit]
        print(f"Limiting evaluation to {limit} tests")

    # Only track 3 core metrics
    scores_by_metric = {
        "faithfulness": [],
        "answer_relevance": [],
        "correctness": [],
        "overall": []
    }

    for i, (test_id, test_data) in enumerate(test_items, 1):
        print(f"\n[{i}/{len(test_items)}] Evaluating {test_id} (query: {test_data.get('query_id')})")

        # Extract query from test data
        query_id = test_data.get("query_id")
        answer = test_data.get("answers", "")
        chunks = test_data.get("top_k_chunks", [])

        # Get ground truth if available
        gt_answer = None
        if query_id in ground_truth:
            # Note: testing_query.json uses "Answer" (capital A), not "answer"
            gt_answer = ground_truth[query_id].get("Answer", None)
            if gt_answer:
                print(f"  Found ground truth for {query_id}")

        # Use query_content if available, otherwise fall back to query_id
        query = test_data.get("query_content", query_id)
        if query == query_id:
            print(f"  Warning: No 'query_content' field found, using query_id '{query_id}'")
            print(f"  Answer Relevance score may be inaccurate.")

        # Run comprehensive evaluation
        result = evaluator.evaluate_comprehensive(query, answer, chunks, gt_answer)
        evaluation_results["results"][test_id] = result

        # Collect scores (only 3 core metrics)
        for metric in ["faithfulness", "answer_relevance"]:
            if metric in result["evaluations"]:
                scores_by_metric[metric].append(result["evaluations"][metric]["score"])

        if "correctness" in result["evaluations"]:
            scores_by_metric["correctness"].append(result["evaluations"]["correctness"]["score"])

        scores_by_metric["overall"].append(result["overall_score"])

        print(f"  Overall score: {result['overall_score']:.3f}")

    # Calculate summary statistics
    evaluation_results["summary"] = {
        metric: {
            "mean": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "min": round(min(scores), 3) if scores else 0.0,
            "max": round(max(scores), 3) if scores else 0.0,
            "count": len(scores)
        }
        for metric, scores in scores_by_metric.items() if scores
    }

    # Save results
    print(f"\nSaving evaluation results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for metric, stats in evaluation_results["summary"].items():
        print(f"{metric:20s}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")
    print("="*60)

    return evaluation_results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python llm_evaluate.py <test_results.json> <output.json> [ground_truth.json] [limit]")
        sys.exit(1)

    test_results_path = sys.argv[1]
    output_path = sys.argv[2]
    ground_truth_path = sys.argv[3] if len(sys.argv) > 3 else None
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None

    # Initialize LLM client (using config from api_config.yaml)
    from healthcare_rag_llm.utils.api_config import load_api_config

    config = load_api_config()
    provider_config = config["api_providers"][config["default_provider"]]
    model_config = config["models"]["gpt-5"]

    llm_client = LLMClient(
        api_key=provider_config["api_key"],
        base_url=provider_config["base_url"],
        model="gpt-5",
        provider=provider_config["provider"]
    )

    # Run evaluation
    evaluate_test_results(
        test_results_path=test_results_path,
        output_path=output_path,
        llm_client=llm_client,
        ground_truth_path=ground_truth_path,
        limit=limit
    )
