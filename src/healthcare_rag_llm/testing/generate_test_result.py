from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.graph_builder.queries import query_chunks


class RAGBatchTester:
    """
    Batch tester that runs retrieval + generation over a set of test questions.

    Spec recap (from user):
      1) System prompt path (txt). Default: 'configs/system_prompt.txt'
      2) Testing queries path (json). Default: 'data/testing_queries/query_covid.json'
      3) Output directory. Default: 'data/test_results'
      4) User input version ID (string). Default: 'version undefine'
      5) Embedding method. Default: HealthcareEmbedding
      6) LLM model. Default: LLMClient(api_key="", provider="ollama", model="llama3.2:3b")
      7) Number of chunks to retrieve (top_k). Default: 5
      8) Number of times each test should repeat. Default: 5

    Input JSON example (we only use the test key and the `question`):
    {
        "test1": {"question": "when was covid", "document": {...}, "answer": "..."},
        "test2": {"question": "did covid end",  "document": {...}, "answer": "..."}
    }

    Output: A single JSON file named by the user-provided version ID inside the output directory,
    containing a JSON object keyed by sequential row names: test_id_1, test_id_2, ...

    Each row schema:
      {
        "query_id": <the input test key, e.g., "test1">,
        "long_version_id": "<embeddingmethod>-<LLMClient>-k-<top_k>-<numberrepeated>",
        "short_version_id": <the provided user version id>,
        "top_k_chunks": <the raw list returned by query_chunks>,
        "answers": <string parsed from the LLM 'answer' field>,
        "document": <object parsed from the LLM 'document' field mapping doc_id -> [pages]>
      }
    """

    def __init__(
        self,
        system_prompt_path: str = "configs/system_prompt.txt",
        testing_queries_path: str = "data/testing_queries/query_covid.json",
        output_dir: str = "data/test_results",
        version_id: str = "version undefine",
        embedding_method=HealthcareEmbedding,
        llm_client: Optional[LLMClient] = None,
        top_k: int = 5,
        repeats: int = 5,
        use_rerank: bool = False,
        rerank_alpha: float = 0.3,
    ) -> None:
        self.system_prompt_path = system_prompt_path
        self.testing_queries_path = testing_queries_path
        self.output_dir = output_dir
        self.version_id = version_id
        self.embedding_method = embedding_method
        self.top_k = int(top_k)
        self.repeats = int(repeats)
        self.use_rerank = use_rerank
        self.rerank_alpha = rerank_alpha

        # Instantiate embedding and LLM client with defaults if not provided
        self.embedder = self.embedding_method()
        self.llm_client = (
            llm_client
            if llm_client is not None
            else LLMClient(api_key="", provider="ollama", model="llama3.2:3b")
        )

        # Initialize reranker if needed
        self.reranker = None
        if self.use_rerank:
            from healthcare_rag_llm.reranking.reranker import Reranker, RerankConfig
            rerank_config = RerankConfig(
                combine_with_dense=True,
                alpha=self.rerank_alpha,
                text_key="text",
                dense_score_key="score"
            )
            self.reranker = Reranker(config=rerank_config)
            print(f"[INFO] Reranker initialized with alpha={self.rerank_alpha}")

        # Derive identifiers
        embedding_name = self.embedding_method.__name__
        llm_name = self.llm_client.__class__.__name__
        self.long_version_id = f"{embedding_name}-{llm_name}-k-{self.top_k}-{self.repeats}"
        self.short_version_id = self.version_id

        # Prepare output path
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f"{self.version_id}.json")

    def run(self) -> Dict[str, Any]:
        system_prompt = self._read_text(self.system_prompt_path)
        tests = self._read_json(self.testing_queries_path)

        results: Dict[str, Any] = {}
        row_counter = 0

        # Calculate total iterations
        total_iterations = len(tests) * self.repeats

        # Progress bar with time tracking
        print(f"\n{'='*60}")
        print(f"Starting RAG Batch Testing")
        print(f"Total queries: {len(tests)} | Repeats per query: {self.repeats}")
        print(f"Total iterations: {total_iterations}")
        print(f"{'='*60}\n")

        start_time = time.time()

        with tqdm(total=total_iterations, desc="Processing queries", unit="query") as pbar:
            for query_id, payload in tests.items():
                question = self._extract_question(payload, query_id)
                for _ in range(self.repeats):
                    row_counter += 1
                    row_name = f"test_id_{row_counter}"

                    # Update progress bar description
                    pbar.set_description(f"Processing {query_id}")

                    query_vec = self.embedder.encode([question])["dense_vecs"][0].tolist()

                    # Retrieve more chunks if reranking (to rerank and then select top_k)
                    retrieval_k = self.top_k * 3 if self.use_rerank else self.top_k
                    retrieved_chunks = query_chunks(query_vec, top_k=retrieval_k)

                    # Apply reranking if enabled
                    if self.use_rerank and self.reranker is not None:
                        retrieved_chunks = self.reranker.rerank_hits(question, retrieved_chunks)
                        # Take top_k after reranking
                        retrieved_chunks = retrieved_chunks[:self.top_k]

                    context = self._format_context_chunks(retrieved_chunks)
                    user_msg = self._build_user_message(question, context)

                    # Measure LLM call time
                    llm_start = time.time()
                    llm_text = self.llm_client.chat(
                        user_prompt=user_msg,
                        system_prompt=system_prompt,
                    )
                    llm_elapsed = time.time() - llm_start

                    # Update progress bar with LLM timing
                    pbar.set_postfix({"LLM_time": f"{llm_elapsed:.1f}s"})

                    # Expect strict JSON text from the LLM: {"answer": <str>, "document": {doc_id: [pages]}}
                    parsed_answer: Optional[str] = None
                    parsed_document: Optional[Dict[str, Any]] = None
                    try:
                        parsed = json.loads(llm_text)
                        if isinstance(parsed, dict):
                            parsed_answer = parsed.get("answer")
                            parsed_document = parsed.get("document")
                    except Exception:
                        # If parsing fails, keep raw text in answers and leave document as None
                        parsed_answer = llm_text
                        parsed_document = None

                    results[row_name] = {
                        "query_id": query_id,
                        "query_content":question,
                        "long_version_id": self.long_version_id,
                        "short_version_id": self.short_version_id,
                        "top_k_chunks": retrieved_chunks,
                        "answers": parsed_answer,
                        "document": parsed_document,
                    }

                    pbar.update(1)

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / total_iterations if total_iterations > 0 else 0

        print(f"\n{'='*60}")
        print(f"Testing completed!")
        print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
        print(f"Average time per query: {avg_time:.2f}s")
        print(f"Output saved to: {self.output_path}")
        print(f"{'='*60}\n")

        self._write_json(self.output_path, results)
        return results

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_question(payload: Any, query_id: str) -> str:
        if isinstance(payload, dict) and "question" in payload and isinstance(payload["question"], str):
            return payload["question"].strip()
        raise ValueError(f"Input JSON missing 'question' for test key: {query_id}")

    @staticmethod
    def _format_context_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for chunk in retrieved_chunks:
            doc_id = chunk.get("doc_id", "?")
            chunk_id = chunk.get("chunk_id", "?")
            pages = chunk.get("pages", "?")
            text = chunk.get("text", "")
            parts.append(
                f"[Document ID: {doc_id}] -[Chunk ID: {chunk_id}]-[pages: {pages}] - [Chunk Content: {text}]"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _build_user_message(question: str, context: str) -> str:
        # Instruct the LLM to return STRICT JSON with fields: answer (string) and document (object mapping doc_id -> [pages]).
        return (
            f"Question:\n{question}\n\n"
            f"Context Chunks (authoritative; cite only these):\n{context}\n\n"
            f"Chunks format:\n"
            f"[Document ID: <doc_id>] -[Chunk ID: <chunk_id>]-[pages: <pages>] - [Chunk Content: <chunk_content>]\n\n"
            f"You must answer in STRICT JSON with two fields only, no extra text before/after.\n"
            f"Schema:\n"
            f"{{\n  \"answer\": <string>,\n  \"document\": {{ <doc_id>: [<page_numbers:int>] }}\n}}\n\n"
        )

if __name__ == "__main__":
    tester = RAGBatchTester()
    tester.run()
