import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from healthcare_rag_llm.utils.api_config import APIConfigManager, APIConfig


# =========================
# Config Dataclasses
# =========================

@dataclass
class ChunkingConfig:
    method: str  # "semantic", "fix_size", "asterisk", "mixed_chunking", etc.
    params: Dict[str, Any]


@dataclass
class RetrievalConfig:
    top_k: int
    rerank: bool = True
    alpha: float = 0.3


@dataclass
class LLMConfig:
    model: str
    api_config: APIConfig  # Use APIConfig from your utils


@dataclass
class ExperimentConfig:
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    version_id: str


# =========================
# Evaluation-Only Pipeline
# =========================

class EvaluatePipelineWithoutTesting:
    """
    Evaluation-only version of the pipeline.

    Assumes that test result JSON files already exist in:
        data/test_results/{version_id}.json

    This pipeline:
      - Loads existing test_results
      - Runs traditional evaluation
      - Optionally runs LLM-based evaluation
      - Aggregates results into a CSV (batch mode)
    """

    def __init__(
        self,
        testing_queries_path: str = "data/testing_queries/testing_query.json",
        output_dir: str = "data/evaluation_results",
        # === LLM Evaluation Parameters ===
        enable_llm_eval: bool = False,          # Enable LLM-based evaluation
        llm_eval_model: str = "gpt-5",          # Model for evaluation
        llm_eval_provider: str = None,          # API provider (None = use default)
        llm_eval_limit: Optional[int] = None,   # Limit number of tests to evaluate (None = all)
        llm_eval_timeout: int = 3600            # Timeout for LLM evaluation (seconds) - kept for parity
    ):
        self.testing_queries_path = testing_queries_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LLM Evaluation settings
        self.enable_llm_eval = enable_llm_eval
        self.llm_eval_limit = llm_eval_limit
        self.llm_eval_timeout = llm_eval_timeout
        self.llm_eval_client = None

        if self.enable_llm_eval:
            # Create LLM evaluation output directory
            Path("data/llm_eval_results").mkdir(parents=True, exist_ok=True)

            # Initialize LLM client for evaluation
            from healthcare_rag_llm.utils.api_config import load_api_config
            from healthcare_rag_llm.llm.llm_client import LLMClient

            config = load_api_config()
            provider_name = llm_eval_provider or config.get("default_provider", "bltcy")
            provider_config = config["api_providers"][provider_name]

            self.llm_eval_client = LLMClient(
                api_key=provider_config["api_key"],
                base_url=provider_config.get("base_url"),
                model=llm_eval_model,
                provider=provider_config.get("provider", "openai")
            )
            print(f"[LLM Eval] Initialized with model={llm_eval_model}, provider={provider_name}")
            if self.llm_eval_limit:
                print(f"[LLM Eval] Will evaluate only first {self.llm_eval_limit} tests per experiment")

    # -------------------------
    # Core per-experiment flow
    # -------------------------

    def run_experiment(self, experiment_config: ExperimentConfig):
        """
        Run a single experiment in evaluation-only mode.

        Steps:
          - Load existing test_results JSON for this version_id
          - Run evaluation on those results
        """
        print(f"Running evaluation-only experiment: {experiment_config.version_id}")

        result_path = f"data/test_results/{experiment_config.version_id}.json"

        if not os.path.exists(result_path):
            raise FileNotFoundError(
                f"Test result file not found for version_id={experiment_config.version_id}: {result_path}"
            )

        # Load existing test results
        with open(result_path, "r", encoding="utf-8") as f:
            test_results = json.load(f)

        # Run evaluation (traditional + optional LLM)
        evaluation_results = self._evaluate_results(experiment_config.version_id, test_results)

        return {
            "config": experiment_config,
            "test_results": test_results,
            "evaluation_results": evaluation_results
        }

    # -------------------------
    # Evaluation helpers
    # -------------------------

    def _evaluate_results(self, version_id: str, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run both traditional and LLM-based evaluation on existing test_results."""
        print(f"Evaluating results (evaluation-only mode): {version_id}")

        # === 1. Traditional Evaluation (always run) ===
        from healthcare_rag_llm.evaluate.evaluate import evaluate_results

        result_path = f"data/test_results/{version_id}.json"
        trad_output_path = f"{self.output_dir}/{version_id}_evaluation.json"

        traditional_results = evaluate_results(
            tested_result_path=result_path,
            ground_truth_path=self.testing_queries_path,
            output_path=trad_output_path
        )

        # === 2. LLM Evaluation (optional) ===
        llm_results = None
        if self.enable_llm_eval:
            print(f"  Running LLM-based evaluation...")
            llm_results = self._run_llm_evaluation(version_id, result_path)

        # === 3. Return combined results ===
        return {
            "traditional": traditional_results,
            "llm_based": llm_results
        }

    def _run_llm_evaluation(self, version_id: str, result_path: str) -> Optional[Dict[str, Any]]:
        """
        Run LLM-based evaluation on test results.

        Args:
            version_id: Experiment version ID
            result_path: Path to test results JSON

        Returns:
            LLM evaluation results dict, or None if failed
        """
        try:
            from healthcare_rag_llm.evaluate.llm_evaluate import evaluate_test_results

            llm_output_path = f"data/llm_eval_results/{version_id}_llm_evaluation.json"

            print(f"    Model: {self.llm_eval_client.model}")
            if self.llm_eval_limit:
                print(f"    Evaluating first {self.llm_eval_limit} tests only")

            llm_results = evaluate_test_results(
                test_results_path=result_path,
                output_path=llm_output_path,
                llm_client=self.llm_eval_client,
                ground_truth_path=self.testing_queries_path,
                limit=self.llm_eval_limit
            )

            print(f"    LLM evaluation complete: {llm_output_path}")
            return llm_results

        except Exception as e:
            print(f"    Warning: LLM evaluation failed: {e}")
            traceback.print_exc()
            return None

    def _extract_llm_metrics(self, llm_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract LLM evaluation metrics for CSV export.

        Args:
            llm_results: LLM evaluation results dict or None

        Returns:
            Dictionary with LLM metric columns
        """
        if llm_results is None or "summary" not in llm_results:
            return {
                "llm_faithfulness_mean": None,
                "llm_answer_relevance_mean": None,
                # "llm_citation_quality_mean": None,  # REMOVED
                # "llm_completeness_mean": None,      # REMOVED
                "llm_correctness_mean": None,
                "llm_overall_mean": None
            }

        summary = llm_results["summary"]
        return {
            "llm_faithfulness_mean": summary.get("faithfulness", {}).get("mean"),
            "llm_answer_relevance_mean": summary.get("answer_relevance", {}).get("mean"),
            # "llm_citation_quality_mean": summary.get("citation_quality", {}).get("mean"),  # REMOVED
            # "llm_completeness_mean": summary.get("completeness", {}).get("mean"),          # REMOVED
            "llm_correctness_mean": summary.get("correctness", {}).get("mean"),
            "llm_overall_mean": summary.get("overall", {}).get("mean")
        }

    # -------------------------
    # Batch runner
    # -------------------------

    def run_batch_experiments(self, configs: List[ExperimentConfig]) -> pd.DataFrame:
        """Run batch experiments (evaluation-only) with progress tracking and robust error handling."""
        results = []

        # Use tqdm for progress bar
        for config in tqdm(configs, desc="Running evaluations", unit="exp"):
            try:
                print(f"\n{'=' * 80}")
                print(f"Starting evaluation-only experiment: {config.version_id}")
                print(f"{'=' * 80}")

                result = self.run_experiment(config)

                results.append({
                    "version_id": config.version_id,
                    "status": "success",
                    "chunking_method": config.chunking.method,
                    "chunking_params": json.dumps(config.chunking.params),
                    "top_k": config.retrieval.top_k,
                    "rerank": config.retrieval.rerank,
                    "alpha": config.retrieval.alpha,
                    "llm_model": config.llm.model,
                    # Traditional metrics
                    "doc_accuracy": result["evaluation_results"]["traditional"]["summary"]["doc_level_accuracy"],
                    "page_accuracy": result["evaluation_results"]["traditional"]["summary"]["page_level_accuracy"],
                    "total_tests": result["evaluation_results"]["traditional"]["summary"]["total_tests"],
                    # LLM metrics (if enabled)
                    **self._extract_llm_metrics(result["evaluation_results"]["llm_based"]),
                    "error": None
                })
                print(f"✓ Evaluation-only experiment {config.version_id} completed successfully")

            except Exception as e:
                error_trace = traceback.format_exc()
                error_msg = str(e)
                print(f"✗ Evaluation-only experiment {config.version_id} failed: {error_msg}")
                print(f"Full traceback:\n{error_trace}")

                results.append(self._create_failed_result(config, error_msg))

        # Save results to CSV
        df = pd.DataFrame(results)
        csv_path = f"{self.output_dir}/batch_evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n{'=' * 80}")
        print(f"Evaluation-only results saved to: {csv_path}")
        print(f"{'=' * 80}")

        return df

    def _create_failed_result(self, config: ExperimentConfig, error_msg: str) -> Dict[str, Any]:
        """Create a result entry for a failed experiment"""
        return {
            "version_id": config.version_id,
            "status": "failed",
            "chunking_method": config.chunking.method,
            "chunking_params": json.dumps(config.chunking.params),
            "top_k": config.retrieval.top_k,
            "rerank": config.retrieval.rerank,
            "alpha": config.retrieval.alpha,
            "llm_model": config.llm.model,
            # Traditional metrics
            "doc_accuracy": None,
            "page_accuracy": None,
            "total_tests": None,
            # LLM metrics (3 core metrics only)
            "llm_faithfulness_mean": None,
            "llm_answer_relevance_mean": None,
            # "llm_citation_quality_mean": None,  # REMOVED
            # "llm_completeness_mean": None,      # REMOVED
            "llm_correctness_mean": None,
            "llm_overall_mean": None,
            "error": error_msg
        }


# =========================
# Main: define experiments
# =========================

def main():
    """Main function - Define experiment configurations and run evaluation-only pipeline."""

    # Initialize API configuration manager
    api_manager = APIConfigManager()

    # Define hyperparameter combinations to evaluate (same as original experiments)
    experiments: List[ExperimentConfig] = []

    # Chunking method configurations (kept to match version_ids)
    chunking_configs = [
        # Configuration 1: Smaller chunks, stricter semantic threshold
        ChunkingConfig("mixed_chunking", {
            "medicaid": {
                "max_chunk_chars": 1000,
                "glob_pattern": "*.json",
                "min_repeats": 10,
                "separator_char": "*"
            },
            "waiver": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "unit": "paragraph",
                "similarity_threshold": 0.60,
                "max_chunk_chars": 1000,
                "glob_pattern": "*.json",
                "hysteresis": 0.02
            }
        }),

        # Configuration 2: Default size (baseline from rebuild_db.py)
        ChunkingConfig("mixed_chunking", {
            "medicaid": {
                "max_chunk_chars": 1200,
                "glob_pattern": "*.json",
                "min_repeats": 10,
                "separator_char": "*"
            },
            "waiver": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "unit": "paragraph",
                "similarity_threshold": 0.55,
                "max_chunk_chars": 1200,
                "glob_pattern": "*.json",
                "hysteresis": 0.02
            }
        }),

        # Configuration 3: Larger chunks, more lenient semantic threshold
        ChunkingConfig("mixed_chunking", {
            "medicaid": {
                "max_chunk_chars": 1500,
                "glob_pattern": "*.json",
                "min_repeats": 10,
                "separator_char": "*"
            },
            "waiver": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "unit": "paragraph",
                "similarity_threshold": 0.50,
                "max_chunk_chars": 1500,
                "glob_pattern": "*.json",
                "hysteresis": 0.02
            }
        })
    ]

    # Retrieval configurations - Compare baseline vs reranking with different alphas
    retrieval_configs = [
        RetrievalConfig(top_k=5, rerank=False, alpha=0.0),   # BASELINE (No Reranking)
        RetrievalConfig(top_k=5, rerank=True, alpha=0.5),    # Balanced
        RetrievalConfig(top_k=5, rerank=True, alpha=0.7),    # More weight on dense
    ]

    # LLM configurations - Use API configuration manager
    llm_configs = [
        #LLMConfig("gpt-5", api_manager.get_model_config("gpt-5")),  # Fix: Use APIConfig
        LLMConfig("gemini-3-pro-preview", api_manager.get_model_config("gemini-3-pro-preview"))  # Fix: Use APIConfig
    ]

    # Generate all combinations (same version_id pattern as original)
    experiment_id = 0
    for chunking in chunking_configs:
        for retrieval in retrieval_configs:
            for llm in llm_configs:
                experiment_id += 1

                # Generate descriptive version_id with rerank info
                rerank_suffix = "noRerank" if not retrieval.rerank else f"rerank_a{retrieval.alpha:.1f}"
                version_id = f"exp_{experiment_id:03d}_{chunking.method}_k{retrieval.top_k}_{rerank_suffix}_{llm.model}"

                experiments.append(ExperimentConfig(
                    chunking=chunking,
                    retrieval=retrieval,
                    llm=llm,
                    version_id=version_id
                ))

    # Run batch evaluations (no testing)
    pipeline = EvaluatePipelineWithoutTesting(
        enable_llm_eval=True,
        llm_eval_model="gpt-5",
        llm_eval_limit=None
    )
    print(f"\n{'=' * 80}")
    print(f"Total experiments to evaluate (evaluation-only): {len(experiments)}")
    print(f"{'=' * 80}\n")

    results_df = pipeline.run_batch_experiments(experiments)

    print("\n=== Evaluation-Only Results Summary ===")
    print(results_df.to_string())

    # Filter only successful experiments for best configuration analysis
    successful_df = results_df[results_df['status'] == 'success']

    if len(successful_df) > 0:
        # Find best configurations
        best_doc_acc = successful_df.loc[successful_df['doc_accuracy'].idxmax()]
        best_page_acc = successful_df.loc[successful_df['page_accuracy'].idxmax()]

        print(f"\n{'=' * 80}")
        print(f"Best Results (Evaluation-Only):")
        print(f"{'=' * 80}")
        print(f"Best document accuracy: {best_doc_acc['version_id']}")
        print(f"  - Doc Accuracy: {best_doc_acc['doc_accuracy']:.3f}")
        print(f"  - Page Accuracy: {best_doc_acc['page_accuracy']:.3f}")
        print(f"  - Chunking: {best_doc_acc['chunking_method']}")
        print(f"  - Top-K: {best_doc_acc['top_k']}")
        print(f"  - Model: {best_doc_acc['llm_model']}")
        print()
        print(f"Best page accuracy: {best_page_acc['version_id']}")
        print(f"  - Page Accuracy: {best_page_acc['page_accuracy']:.3f}")
        print(f"  - Doc Accuracy: {best_page_acc['doc_accuracy']:.3f}")
        print(f"  - Chunking: {best_page_acc['chunking_method']}")
        print(f"  - Top-K: {best_page_acc['top_k']}")
        print(f"  - Model: {best_page_acc['llm_model']}")

        # === LLM Evaluation Best Results (if enabled) ===
        if 'llm_overall_mean' in successful_df.columns and successful_df['llm_overall_mean'].notna().any():
            best_llm = successful_df.loc[successful_df['llm_overall_mean'].idxmax()]
            print()
            print(f"Best LLM overall score: {best_llm['version_id']}")
            print(f"  - LLM Overall: {best_llm['llm_overall_mean']:.3f}")
            print(f"  - Faithfulness: {best_llm['llm_faithfulness_mean']:.3f}")
            print(f"  - Answer Relevance: {best_llm['llm_answer_relevance_mean']:.3f}")
            if 'llm_correctness_mean' in best_llm and best_llm['llm_correctness_mean'] is not None and not pd.isna(best_llm['llm_correctness_mean']):
                print(f"  - Correctness: {best_llm['llm_correctness_mean']:.3f}")
            print(f"  - Chunking: {best_llm['chunking_method']}")
            print(f"  - Top-K: {best_llm['top_k']}")

        print(f"{'=' * 80}")

        # Show success/failure statistics
        total = len(results_df)
        success = len(successful_df)
        failed = total - success
        print(f"\nEvaluation-Only Experiment Statistics:")
        print(f"  Total: {total}")
        print(f"  Successful: {success} ({success / total * 100:.1f}%)")
        print(f"  Failed: {failed} ({failed / total * 100:.1f}%)")
    else:
        print("\nWarning: All evaluation-only experiments failed. Check error messages above.")


if __name__ == "__main__":
    main()
