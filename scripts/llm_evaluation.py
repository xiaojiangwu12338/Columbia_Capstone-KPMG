#!/usr/bin/env python3
"""
Script to run LLM-based evaluation on RAG test results.

This script evaluates the quality of LLM-generated answers using another LLM as a judge.
Evaluation dimensions include:
- Faithfulness: Is the answer grounded in the source chunks?
- Answer Relevance: Does the answer address the query?
- Citation Quality: Are citations accurate and properly formatted?
- Completeness: Does the answer use all relevant available information?
- Correctness: (optional) Does it match ground truth?

Usage:
    python scripts/llm_evaluation.py --test_results data/test_results/exp_001.json --output data/llm_eval_results/exp_001_llm_eval.json

    # With ground truth
    python scripts/llm_evaluation.py --test_results data/test_results/exp_001.json --output data/llm_eval_results/exp_001_llm_eval.json --ground_truth data/ground_truth.json

    # Limit to first 5 tests (for testing)
    python scripts/llm_evaluation.py --test_results data/test_results/exp_001.json --output data/llm_eval_results/exp_001_llm_eval.json --limit 5

    # Use specific model
    python scripts/llm_evaluation.py --test_results data/test_results/exp_001.json --output data/llm_eval_results/exp_001_llm_eval.json --model gpt-4
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from healthcare_rag_llm.evaluate.llm_evaluate import evaluate_test_results
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.utils.api_config import load_api_config


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-based evaluation on RAG test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/llm_evaluation.py -t data/test_results/exp_001.json -o data/llm_eval/exp_001.json

  # With ground truth comparison
  python scripts/llm_evaluation.py -t data/test_results/exp_001.json -o data/llm_eval/exp_001.json -g data/ground_truth.json

  # Test on first 3 items only
  python scripts/llm_evaluation.py -t data/test_results/exp_001.json -o data/llm_eval/exp_001.json --limit 3

  # Use specific model
  python scripts/llm_evaluation.py -t data/test_results/exp_001.json -o data/llm_eval/exp_001.json --model gpt-4
        """
    )

    parser.add_argument(
        "-t", "--test_results",
        required=True,
        help="Path to test results JSON file"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save LLM evaluation results"
    )

    parser.add_argument(
        "-g", "--ground_truth",
        default=None,
        help="Optional path to ground truth data for correctness evaluation"
    )

    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model to use for evaluation (default: gpt-5)"
    )

    parser.add_argument(
        "--provider",
        default=None,
        help="API provider to use (default: uses default from config)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tests to evaluate (useful for testing)"
    )

    parser.add_argument(
        "--config",
        default="configs/api_config.yaml",
        help="Path to API config file (default: configs/api_config.yaml)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.test_results).exists():
        print(f"Error: Test results file not found: {args.test_results}")
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load API configuration
    print(f"Loading API configuration from: {args.config}")
    try:
        config = load_api_config(args.config)
    except Exception as e:
        print(f"Error loading API config: {e}")
        sys.exit(1)

    # Determine provider
    provider_name = args.provider or config.get("default_provider", "bltcy")
    if provider_name not in config["api_providers"]:
        print(f"Error: Provider '{provider_name}' not found in config")
        print(f"Available providers: {list(config['api_providers'].keys())}")
        sys.exit(1)

    provider_config = config["api_providers"][provider_name]

    # Initialize LLM client
    print(f"Initializing LLM client: model={args.model}, provider={provider_name}")
    try:
        llm_client = LLMClient(
            api_key=provider_config["api_key"],
            base_url=provider_config.get("base_url"),
            model=args.model,
            provider=provider_config.get("provider", "openai")
        )
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        sys.exit(1)

    # Run evaluation
    print("\n" + "="*60)
    print("STARTING LLM-BASED EVALUATION")
    print("="*60)
    print(f"Test results: {args.test_results}")
    print(f"Output: {args.output}")
    print(f"Ground truth: {args.ground_truth or 'Not provided'}")
    print(f"Evaluator model: {args.model}")
    print(f"Limit: {args.limit or 'All tests'}")
    print("="*60 + "\n")

    try:
        results = evaluate_test_results(
            test_results_path=args.test_results,
            output_path=args.output,
            llm_client=llm_client,
            ground_truth_path=args.ground_truth,
            limit=args.limit
        )

        print(f"\nEvaluation complete! Results saved to: {args.output}")
        return 0

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
