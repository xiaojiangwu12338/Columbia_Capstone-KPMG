import os
from healthcare_rag_llm.testing.generate_test_result import RAGBatchTester
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.llm.llm_client import LLMClient

def main():
    # Get API key from environment variable for security
    api_key = os.environ.get("EVAL_API_KEY", "sk-uGxA1B2QS7davJiwh3MTDeNEFFSIPSZRRHi21k5GUUMA1Jnr")

    llm_client = LLMClient(
        api_key=api_key,
        provider="openai",
        base_url="https://api.bltcy.ai/v1",
        model="gpt-5"
    )

    # Reranking is now integrated!
    tester = RAGBatchTester(
        system_prompt_path="configs/system_prompt.txt",
        testing_queries_path="data/testing_queries/testing_query.json",
        output_dir="data/test_results",
        version_id="exp_002_mixed_chunking_k5_rerank_a0.5_gpt-5",
        embedding_method=HealthcareEmbedding,
        llm_client=llm_client,
        top_k=5,
        repeats=1,
        use_rerank=True,
        rerank_alpha=0.5
    )

    tester.run()

if __name__ == "__main__":
    main()
