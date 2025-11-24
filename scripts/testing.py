from healthcare_rag_llm.testing.generate_test_result import RAGBatchTester
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.llm.llm_client import LLMClient

def main():
    # ==== Configuration Section ====
    system_prompt_path = "configs/system_prompt.txt"
    testing_queries_path = "data/testing_queries/testing_query.json"
    output_dir = "data/test_results"
    version_id = "v1.0-demo"  # name of output JSON file (data/test_results/v1.0-demo.json)

    # Embedding and LLM setup
    embedding_method = HealthcareEmbedding
    llm_client = LLMClient(
        api_key="",                # Add API key if needed
        provider="openai",         # e.g., openai, anthropic, ollama,
        base_url="https://api.bltcy.ai/v1",
        model="gpt-5"        # model name used by your LLM provider
    )

    # Retrieval and test parameters
    top_k = 5          # number of chunks to retrieve
    repeats = 1        # number of test repetitions per question
    # ===============================

    tester = RAGBatchTester(
        system_prompt_path=system_prompt_path,
        testing_queries_path=testing_queries_path,
        output_dir=output_dir,
        version_id=version_id,
        embedding_method=embedding_method,
        llm_client=llm_client,
        top_k=top_k,
        repeats=repeats,
    )

    results = tester.run()
    print("Testing completed.")
    print(f"Results saved to: {tester.output_path}")
    print(f"Total tests executed: {len(results)}")


if __name__ == "__main__":
    main()
