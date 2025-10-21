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
                 timeout_test: int = 1800       # 30 minutes for testing
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
        else:
            raise ValueError(f"Invalid chunking method: {chunking_config.method}")

    def _run_semantic_chunking(self, params: Dict[str, Any]):
        """Run semantic chunking with configurable parameters"""
        cmd = [
            "python", "scripts/do_semantic_chunking.py",
            "--model", params.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            "--threshold", str(params.get("threshold", 0.80)),
            "--max-chars", str(params.get("max_chars", 2000)),
            "--unit", params.get("unit", "sentence")
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, timeout=self.timeout_chunking)

    def _run_fix_size_chunking(self, params: Dict[str, Any]):
        """Run fix size chunking

        Note: Current implementation uses hardcoded values in do_fix_size_chunking.py
        Parameters in 'params' dict are not used until script is updated.
        """
        cmd = [
            "python", "scripts/do_fix_size_chunking.py"
        ]
        print(f"  Running: {' '.join(cmd)}")
        print(f"  Note: Using hardcoded chunk_size=1200, overlap=150 (params not supported yet)")
        subprocess.run(cmd, check=True, timeout=self.timeout_chunking)

    def _run_asterisk_chunking(self, params: Dict[str, Any]):
        """Run asterisk (pattern-based) chunking"""
        cmd = [
            "python", "scripts/do_asterisk_chunking.py"
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, timeout=self.timeout_chunking)

    def _load_to_neo4j(self, chunking_method: str):
        """Load chunking results to Neo4j (with database reset)"""
        print(f"Loading to Neo4j: {chunking_method}")

        # Step 1: Reset Neo4j to ensure clean state
        print("  Resetting Neo4j database...")
        reset_cmd = ["python", "scripts/reset_graph.py"]
        subprocess.run(reset_cmd, check=True, timeout=300)  # 5 minutes timeout

        # Step 2: Load new chunks
        print("  Ingesting chunks into Neo4j...")
        # Handle asterisk chunking's different directory name
        if chunking_method == "asterisk":
            chunk_dir = "data/chunks/asterisk_separate_chunking_result"
        else:
            chunk_dir = f"data/chunks/{chunking_method}_chunking_result"

        ingest_cmd = [
            "python", "scripts/ingest_graph.py",
            "--chunk_dir", chunk_dir
        ]
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

            # Read results
            result_path = f"data/test_results/{config.version_id}.json"
            with open(result_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        finally:
            # Always clean up temporary file
            if test_script and os.path.exists(test_script):
                try:
                    os.remove(test_script)
                    print(f"  Cleaned up temporary script: {test_script}")
                except Exception as e:
                    print(f"  Warning: Failed to remove temporary script {test_script}: {e}")

    def _create_test_script(self, config: ExperimentConfig) -> str:
        """Create temporary test script with environment variable for API key

        Note: Rerank parameters (config.retrieval.rerank, alpha) are defined but not yet
        used by RAGBatchTester. They will be utilized once reranking is integrated
        into the testing workflow.
        """
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

    # Note: rerank={config.retrieval.rerank}, alpha={config.retrieval.alpha}
    # will be used once RAGBatchTester supports reranking
    tester = RAGBatchTester(
        system_prompt_path="configs/system_prompt.txt",
        testing_queries_path="{self.testing_queries_path}",
        output_dir="data/test_results",
        version_id="{config.version_id}",
        embedding_method=HealthcareEmbedding,
        llm_client=llm_client,
        top_k={config.retrieval.top_k},
        repeats=1
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
        """Evaluate test results"""
        print(f"Evaluating results: {version_id}")

        from healthcare_rag_llm.evaluate.evaluate import evaluate_results

        result_path = f"data/test_results/{version_id}.json"
        output_path = f"{self.output_dir}/{version_id}_evaluation.json"

        return evaluate_results(
            tested_result_path=result_path,
            ground_truth_path=self.testing_queries_path,
            output_path=output_path
        )

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
                    "doc_accuracy": result["evaluation_results"]["summary"]["doc_level_accuracy"],
                    "page_accuracy": result["evaluation_results"]["summary"]["page_level_accuracy"],
                    "total_tests": result["evaluation_results"]["summary"]["total_tests"],
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
            "doc_accuracy": None,
            "page_accuracy": None,
            "total_tests": None,
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
        #ChunkingConfig("semantic", {"threshold": 0.80, "max_chars": 2000}),
        ChunkingConfig("semantic", {"threshold": 0.85, "max_chars": 1500}),
        ChunkingConfig("fix_size", {"chunk_size": 1200, "overlap": 150}),
        #ChunkingConfig("fix_size", {"chunk_size": 800, "overlap": 100}),
        ChunkingConfig("asterisk", {})
    ]

    # Retrieval configurations
    retrieval_configs = [
        #RetrievalConfig(top_k=3),
        RetrievalConfig(top_k=5),
        #RetrievalConfig(top_k=7)
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
                version_id = f"exp_{experiment_id:03d}_{chunking.method}_k{retrieval.top_k}_{llm.model}"

                experiments.append(ExperimentConfig(
                    chunking=chunking,
                    retrieval=retrieval,
                    llm=llm,
                    version_id=version_id
                ))

    # Run batch experiments
    pipeline = EvaluatePipeline()
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