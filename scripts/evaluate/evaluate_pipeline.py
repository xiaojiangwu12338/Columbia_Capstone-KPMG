import json
import os
import subprocess
import sys
import traceback
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from healthcare_rag_llm.utils.api_config import APIConfigManager, APIConfig


@dataclass
class ChunkingConfig:
    method: str  # "semantic", "fix_size", "asterisk"
    params: Dict[str, Any]


@dataclass
class RetrievalConfig:
    top_k: int
    rerank: bool = True
    alpha: float = 0.3


@dataclass
class LLMConfig:
    model: str
    api_config: APIConfig  # Use APIConfig


@dataclass
class ExperimentConfig:
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    version_id: str


class EvaluatePipeline:
    def __init__(self,
                 testing_queries_path: str = "data/testing_queries/testing_query.json",
                 output_dir: str = "data/evaluation_results",
                 timeout_chunking: int = 1800,  # 30 minutes for chunking
                 timeout_ingest: int = 3600,    # 1 hour for Neo4j ingestion
                 timeout_test: int = 1800,      # 30 minutes for testing
                 # === LLM Evaluation Parameters ===
                 enable_llm_eval: bool = False,          # Enable LLM-based evaluation
                 llm_eval_model: str = "gpt-5",          # Model for evaluation
                 llm_eval_provider: str = None,          # API provider (None = use default)
                 llm_eval_limit: Optional[int] = None,   # Limit number of tests to evaluate (None = all)
                 llm_eval_timeout: int = 3600            # Timeout for LLM evaluation (seconds)
                 ):
        self.testing_queries_path = testing_queries_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create necessary directories
        Path("data/test_results").mkdir(parents=True, exist_ok=True)
        Path("data/chunks").mkdir(parents=True, exist_ok=True)

        # Timeout settings
        self.timeout_chunking = timeout_chunking
        self.timeout_ingest = timeout_ingest
        self.timeout_test = timeout_test

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

    def run_experiment(self, experiment_config: ExperimentConfig):  # Fix: Use correct parameter name
        # Run single experiment
        print(f"Running experiment: {experiment_config.version_id}")

        self._run_chunking(experiment_config.chunking)  # Fix: Use experiment_config

        self._load_to_neo4j(experiment_config.chunking.method)  # Fix: Use experiment_config

        test_results = self._run_testing(experiment_config)  # Fix: Use experiment_config

        evaluation_results = self._evaluate_results(experiment_config.version_id, test_results)  # Fix: Use experiment_config

        return {"config": experiment_config,  # Fix: Use experiment_config
                "test_results": test_results,
                "evaluation_results": evaluation_results}

    def _run_chunking(self, chunking_config: ChunkingConfig):
        print(f"Running chunking: {chunking_config.method}")

        if chunking_config.method == "semantic":
            self._run_semantic_chunking(chunking_config.params)
        elif chunking_config.method == "fix_size":
            self._run_fix_size_chunking(chunking_config.params)
        elif chunking_config.method == "asterisk":
            self._run_asterisk_chunking(chunking_config.params)
        elif chunking_config.method == "mixed_chunking":
            self._run_mixed_chunking(chunking_config.params)
        else:
            raise ValueError(f"Invalid chunking method: {chunking_config.method}")

    def _run_semantic_chunking(self, params: Dict[str, Any]):
        """Run semantic chunking with configurable parameters"""
        cmd = [
            "python", "scripts/do_semantic_chunking.py",
            "--model", params.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            "--threshold", str(params.get("threshold", 0.80)),
            "--max-chars", str(params.get("max_chars", 2000)),
            "--unit", params.get("unit", "sentence"),
            "--hysteresis", str(params.get("hysteresis", 0.02))
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, timeout=self.timeout_chunking)

    def _run_fix_size_chunking(self, params: Dict[str, Any]):
        """Run fix size chunking with configurable parameters"""
        cmd = [
            "python", "scripts/do_fix_size_chunking.py",
            "--max-chars", str(params.get("max_chars", 1200)),
            "--overlap", str(params.get("overlap", 150)),
            "--pattern", params.get("pattern", "*.json")
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, timeout=self.timeout_chunking)

    def _run_asterisk_chunking(self, params: Dict[str, Any]):
        """Run asterisk (pattern-based) chunking"""
        cmd = [
            "python", "scripts/do_asterisk_chunking.py"
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, timeout=self.timeout_chunking)

    def _run_mixed_chunking(self, params: Dict[str, Any]):
        """Run mixed chunking (asterisk for Medicaid, semantic for Waiver)"""
        from healthcare_rag_llm.pipelines.ingest_parse import run_pipeline as parse_pipeline
        from healthcare_rag_llm.chunking.pattern_chunking import asterisk_separate_chunking
        from healthcare_rag_llm.chunking.semantic_chunking import semantic_chunking
        import shutil

        # Get paths
        root = Path(__file__).resolve().parents[2]
        chunk_dir = root / "data" / "chunks" / "mixed_chunking_result"

        raw_medicaid = root / "data" / "raw" / "Childrens Evolution of Care" / "State" / "Medicaid Updates"
        out_medicaid = root / "data" / "processed" / "medicaid update"

        raw_waiver = root / "data" / "raw" / "Childrens Evolution of Care" / "State" / "Waivers" / "Childrens Waiver"
        out_waiver = root / "data" / "processed" / "children waiver"

        # Clear chunk directory
        print("  Clearing chunk directory...")
        if chunk_dir.exists():
            for item in chunk_dir.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                else:
                    shutil.rmtree(item)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Parse Medicaid
        print("  Parsing Medicaid Updates...")
        parse_pipeline(
            raw_dir=str(raw_medicaid),
            out_dir=str(out_medicaid),
            save_text=False,
            save_json=True
        )

        # Parse Waiver
        print("  Parsing Children's Waiver...")
        parse_pipeline(
            raw_dir=str(raw_waiver),
            out_dir=str(out_waiver),
            save_text=False,
            save_json=True
        )

        # Get chunking parameters from config (with defaults)
        medicaid_params = params.get("medicaid", {})
        waiver_params = params.get("waiver", {})

        # Chunk Medicaid with asterisk
        print("  Chunking Medicaid with asterisk method...")
        asterisk_separate_chunking(
            processed_dir=str(out_medicaid),
            chunked_dir=str(chunk_dir),
            max_chunk_chars=medicaid_params.get("max_chunk_chars", 1200),
            glob_pattern=medicaid_params.get("glob_pattern", "*.json"),
            min_repeats=medicaid_params.get("min_repeats", 10),
            separator_char=medicaid_params.get("separator_char", "*"),
            verbose=True
        )

        # Chunk Waiver with semantic
        print("  Chunking Children's Waiver with semantic method...")
        semantic_chunking(
            processed_dir=str(out_waiver),
            chunked_dir=str(chunk_dir),
            model_name=waiver_params.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            unit=waiver_params.get("unit", "paragraph"),
            similarity_threshold=waiver_params.get("similarity_threshold", 0.55),
            max_chunk_chars=waiver_params.get("max_chunk_chars", 1200),
            glob_pattern=waiver_params.get("glob_pattern", "*.json"),
            hysteresis=waiver_params.get("hysteresis", 0.02),
            verbose=True
        )

        print("  Mixed chunking completed!")

    def _load_to_neo4j(self, chunking_method: str):
        """Load chunking results to Neo4j (with database reset)"""
        print(f"Loading to Neo4j: {chunking_method}")

        # Step 1: Reset Neo4j to ensure clean state
        print("  Resetting Neo4j database...")
        reset_cmd = ["python", "scripts/reset_graph.py"]
        subprocess.run(reset_cmd, check=True, timeout=300)  # 5 minutes timeout

        # Step 2: Load new chunks
        print("  Ingesting chunks into Neo4j...")
        # Handle different chunking methods' directory names
        if chunking_method == "asterisk":
            chunk_dir = "data/chunks/asterisk_separate_chunking_result"
        elif chunking_method == "mixed_chunking":
            chunk_dir = "data/chunks/mixed_chunking_result"
        else:
            chunk_dir = f"data/chunks/{chunking_method}_chunking_result"

        ingest_cmd = [
            "python", "scripts/ingest_graph.py",
            "--chunk_dir", chunk_dir
        ]

        # Add metadata file for mixed_chunking (matches rebuild_db.py workflow)
        if chunking_method == "mixed_chunking":
            metadata_file = "data/metadata/metadata_filled.csv"
            ingest_cmd.extend(["--meta_file", metadata_file])

        subprocess.run(ingest_cmd, check=True, timeout=self.timeout_ingest)

    def _run_testing(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run testing with temporary script (auto-cleanup)"""
        print(f"Running test: {config.version_id}")

        # Create temporary test script
        test_script = self._create_test_script(config)

        try:
            # Run test with timeout
            result = subprocess.run(
                [sys.executable, test_script],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout_test
            )

            # Print stdout if any
            if result.stdout:
                print(result.stdout)

            # Read results
            result_path = f"data/test_results/{config.version_id}.json"
            with open(result_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except subprocess.CalledProcessError as e:
            # Print both stdout and stderr for debugging
            print(f"\n{'='*80}")
            print(f"Error running test script:")
            print(f"{'='*80}")
            if e.stdout:
                print("STDOUT:")
                print(e.stdout)
            if e.stderr:
                print("STDERR:")
                print(e.stderr)
            print(f"{'='*80}")
            raise
        finally:
            # Always clean up temporary file
            if test_script and os.path.exists(test_script):
                try:
                    os.remove(test_script)
                    print(f"  Cleaned up temporary script: {test_script}")
                except Exception as e:
                    print(f"  Warning: Failed to remove temporary script {test_script}: {e}")

    def _create_test_script(self, config: ExperimentConfig) -> str:
        """Create temporary test script with reranking support enabled"""
        # Use environment variable for API key instead of hardcoding
        script_content = f'''import os
from healthcare_rag_llm.testing.generate_test_result import RAGBatchTester
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.llm.llm_client import LLMClient

def main():
    # Get API key from environment variable for security
    api_key = os.environ.get("EVAL_API_KEY", "{config.llm.api_config.api_key}")

    llm_client = LLMClient(
        api_key=api_key,
        provider="{config.llm.api_config.provider}",
        base_url="{config.llm.api_config.base_url}",
        model="{config.llm.model}"
    )

    # Reranking is now integrated!
    tester = RAGBatchTester(
        system_prompt_path="configs/system_prompt.txt",
        testing_queries_path="{self.testing_queries_path}",
        output_dir="data/test_results",
        version_id="{config.version_id}",
        embedding_method=HealthcareEmbedding,
        llm_client=llm_client,
        top_k={config.retrieval.top_k},
        repeats=1,
        use_rerank={str(config.retrieval.rerank)},
        rerank_alpha={config.retrieval.alpha}
    )

    tester.run()

if __name__ == "__main__":
    main()
'''

        # Use temporary file with unique name
        script_path = f"temp_test_{config.version_id}.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        return script_path

    def _evaluate_results(self, version_id: str, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run both traditional and LLM-based evaluation"""
        print(f"Evaluating results: {version_id}")

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
            import traceback
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

    def run_batch_experiments(self, configs: List[ExperimentConfig]) -> pd.DataFrame:
        """Run batch experiments with progress tracking and robust error handling"""
        results = []

        # Use tqdm for progress bar
        for config in tqdm(configs, desc="Running experiments", unit="exp"):
            try:
                print(f"\n{'='*80}")
                print(f"Starting experiment: {config.version_id}")
                print(f"{'='*80}")

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
                print(f"✓ Experiment {config.version_id} completed successfully")

            except subprocess.TimeoutExpired as e:
                error_msg = f"Timeout after {e.timeout}s in command: {e.cmd}"
                print(f"✗ Experiment {config.version_id} failed: {error_msg}")
                results.append(self._create_failed_result(config, error_msg))

            except Exception as e:
                error_trace = traceback.format_exc()
                error_msg = str(e)
                print(f"✗ Experiment {config.version_id} failed: {error_msg}")
                print(f"Full traceback:\n{error_trace}")

                results.append(self._create_failed_result(config, error_msg))

        # Save results to CSV
        df = pd.DataFrame(results)
        csv_path = f"{self.output_dir}/batch_evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {csv_path}")
        print(f"{'='*80}")

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


def main():
    """Main function - Define experiment configurations"""

    # Initialize API configuration manager
    api_manager = APIConfigManager()

    # Define hyperparameter combinations to test
    experiments = []

    # Chunking method configurations
    chunking_configs = [
        # === Mixed Chunking Variations ===
        # Test different chunk sizes and semantic thresholds

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
        # === BASELINE (No Reranking) ===
        RetrievalConfig(top_k=5, rerank=False, alpha=0.0),
        RetrievalConfig(top_k=5, rerank=True, alpha=0.5),  # Balanced
        RetrievalConfig(top_k=5, rerank=True, alpha=0.7),  # More weight on dense

    ]

    # LLM configurations - Use API configuration manager
    llm_configs = [
        LLMConfig("gpt-5", api_manager.get_model_config("gpt-5")),  # Fix: Use APIConfig
        #LLMConfig("gpt-4", api_manager.get_model_config("gpt-4"))  # Fix: Use APIConfig
    ]

    # Generate all combinations
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

    # Run batch experiments
    # === Enable LLM Evaluation ===
    pipeline = EvaluatePipeline(
        enable_llm_eval=True,           
        llm_eval_model="gpt-5",        
        llm_eval_limit=None
    )
    print(f"\n{'='*80}")
    print(f"Total experiments to run: {len(experiments)}")
    print(f"{'='*80}\n")

    results_df = pipeline.run_batch_experiments(experiments)

    print("\n=== Experiment Results Summary ===")
    print(results_df.to_string())

    # Filter only successful experiments for best configuration analysis
    successful_df = results_df[results_df['status'] == 'success']

    if len(successful_df) > 0:
        # Find best configurations
        best_doc_acc = successful_df.loc[successful_df['doc_accuracy'].idxmax()]
        best_page_acc = successful_df.loc[successful_df['page_accuracy'].idxmax()]

        print(f"\n{'='*80}")
        print(f"Best Results:")
        print(f"{'='*80}")
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
            # REMOVED: Citation Quality and Completeness
            # print(f"  - Citation Quality: {best_llm['llm_citation_quality_mean']:.3f}")
            # print(f"  - Completeness: {best_llm['llm_completeness_mean']:.3f}")
            # Only print correctness if available (requires ground truth)
            if 'llm_correctness_mean' in best_llm and best_llm['llm_correctness_mean'] is not None and not pd.isna(best_llm['llm_correctness_mean']):
                print(f"  - Correctness: {best_llm['llm_correctness_mean']:.3f}")
            print(f"  - Chunking: {best_llm['chunking_method']}")
            print(f"  - Top-K: {best_llm['top_k']}")

        print(f"{'='*80}")

        # Show success/failure statistics
        total = len(results_df)
        success = len(successful_df)
        failed = total - success
        print(f"\nExperiment Statistics:")
        print(f"  Total: {total}")
        print(f"  Successful: {success} ({success/total*100:.1f}%)")
        print(f"  Failed: {failed} ({failed/total*100:.1f}%)")
    else:
        print("\n⚠ Warning: All experiments failed. Check error messages above.")


if __name__ == "__main__":
    main()