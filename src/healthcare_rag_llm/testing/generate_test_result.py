from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.llm.guardrail_response_wrapper import ResponseGenerator  # Use production wrapper
from healthcare_rag_llm.filters.load_metadata import build_filter_extractor  # Production filter


def _format_evidence_for_evaluation(evidence_dict: Dict[str, Any]) -> str:
    """
    Format evidence_dict into a text representation for LLM evaluation.
    This matches what users see in the frontend (app.py:214-228).

    Args:
        evidence_dict: Dictionary with evidence1-5 keys containing doc_info, quote, publish_date, url

    Returns:
        Formatted evidence string
    """
    if not evidence_dict:
        return "(No evidence cited)"

    lines = []
    for i, (key, ev) in enumerate(evidence_dict.items(), 1):
        doc_info = ev.get("doc_info", "Unknown")
        quote = ev.get("quote", "")
        publish_date = ev.get("publish_date", "N/A")
        url = ev.get("url", "N/A")

        # Format similar to frontend
        lines.append(f"{i}. {doc_info}")
        lines.append(f"   Published: {publish_date}")
        lines.append(f"   URL: {url}")
        lines.append(f'   Quote: "{quote}"')
        lines.append("")  # Blank line between evidence items

    return "\n".join(lines)


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

        # Instantiate LLM client with defaults if not provided
        self.llm_client = (
            llm_client
            if llm_client is not None
            else LLMClient(api_key="", provider="ollama", model="llama3.2:3b")
        )

        # Build filter extractor (matching production: app.py:81)
        filter_extractor = build_filter_extractor()

        # Use production ResponseGenerator with guardrail wrapper (app.py:24)
        # This ensures evaluation uses the exact same logic as production:
        # - Guardrail: checks if question is healthcare-related
        # - Delegates to response_gen_json.ResponseGenerator for RAG
        self.response_generator = ResponseGenerator(
            llm_client=self.llm_client,
            use_reranker=self.use_rerank,
            filter_extractor=filter_extractor,
            alpha=self.rerank_alpha  # Pass alpha parameter
        )

        if self.use_rerank:
            print(f"[INFO] Using production ResponseGenerator with reranker (alpha={self.rerank_alpha})")

        # Derive identifiers
        embedding_name = self.embedding_method.__name__
        llm_name = self.llm_client.__class__.__name__
        self.long_version_id = f"{embedding_name}-{llm_name}-k-{self.top_k}-{self.repeats}"
        self.short_version_id = self.version_id

        # Prepare output path
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f"{self.version_id}.json")

    def run(self) -> Dict[str, Any]:
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

                    # Use production ResponseGenerator (same logic as app.py:307)
                    # Measure LLM call time
                    llm_start = time.time()
                    result = self.response_generator.answer_question(
                        question=question,
                        top_k=self.top_k,
                        rerank_top_k=self.top_k * 3,
                        history=None  # No chat history in evaluation
                    )
                    llm_elapsed = time.time() - llm_start

                    # Update progress bar with LLM timing
                    pbar.set_postfix({"LLM_time": f"{llm_elapsed:.1f}s"})

                    # Extract results from production ResponseGenerator
                    # Result format: {question, answer, evidence_dict, retrieved_docs}
                    answer_text = result.get("answer", "")
                    evidence_dict = result.get("evidence_dict", {})
                    retrieved_chunks = result.get("retrieved_docs", [])

                    # Format evidence for evaluation (matching frontend display)
                    formatted_evidence = _format_evidence_for_evaluation(evidence_dict)

                    # Combine answer + evidence for LLM evaluation
                    # This matches what users see in frontend (app.py:222-228)
                    parsed_answer = f"""{answer_text}

Evidence:
{formatted_evidence}"""

                    # Build document dict for evaluation (doc_id -> [pages])
                    # Extract from retrieved_chunks that were actually used (has evidence)
                    parsed_document = {}
                    for evidence_key in evidence_dict.keys():
                        # Evidence keys are like "evidence1", "evidence2", etc.
                        # Match to chunks (CHUNK1 = index 0)
                        chunk_idx = int(evidence_key.replace("evidence", "")) - 1
                        if 0 <= chunk_idx < len(retrieved_chunks):
                            chunk = retrieved_chunks[chunk_idx]
                            doc_id = chunk.get("doc_id")
                            pages = chunk.get("pages", [])
                            if doc_id:
                                if doc_id not in parsed_document:
                                    parsed_document[doc_id] = []
                                # Add pages if not already present
                                if isinstance(pages, list):
                                    for p in pages:
                                        if p not in parsed_document[doc_id]:
                                            parsed_document[doc_id].append(p)
                                elif pages and pages not in parsed_document[doc_id]:
                                    parsed_document[doc_id].append(pages)

                    results[row_name] = {
                        "query_id": query_id,
                        "query_content": question,
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
        def default_serializer(obj):
            """Handle non-serializable objects"""
            # Handle Neo4j Date objects
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Date':
                return str(obj)  # Convert to ISO format string
            # Handle other datetime-like objects
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            # Default: convert to string
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=default_serializer)

    @staticmethod
    def _extract_question(payload: Any, query_id: str) -> str:
        if isinstance(payload, dict) and "question" in payload and isinstance(payload["question"], str):
            return payload["question"].strip()
        raise ValueError(f"Input JSON missing 'question' for test key: {query_id}")

    @staticmethod
    def _format_context_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks to match production format (response_gen_json.py).
        Each chunk is labeled CHUNK1, CHUNK2, etc. with full metadata.
        """
        parts: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            doc_id = chunk.get("doc_id", "?")
            title = chunk.get("title", "N/A")
            effective_date = chunk.get("effective_date", "N/A")
            authority = chunk.get("authority", "N/A")
            pages = chunk.get("pages", "?")
            text = chunk.get("text", "")

            parts.append(
                f"CHUNK{idx}\n"
                f"[Document Title: {title}]\n"
                f"[Effective Date: {effective_date}]\n"
                f"[Authority: {authority}]\n"
                f"[Document ID: {doc_id}]\n"
                f"[Pages: {pages}]\n"
                f"[Content: {text}]"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _build_user_message(question: str, context: str) -> str:
        """
        Build user message matching production format (response_gen_json.py).
        Uses the same JSON contract for chunk citations.
        """
        json_contract = """
Return ONLY valid JSON with EXACTLY these fields (no extra keys, no trailing text):
{
  "answer": "<complete sentence(s) answering the question>",
  "chunk1": 0|1,
  "chunk1string": "<verbatim quote from CHUNK1 if used, else empty string>",
  "chunk2": 0|1,
  "chunk2string": "<verbatim quote from CHUNK2 if used, else empty string>",
  "chunk3": 0|1,
  "chunk3string": "<verbatim quote from CHUNK3 if used, else empty string>",
  "chunk4": 0|1,
  "chunk4string": "<verbatim quote from CHUNK4 if used, else empty string>",
  "chunk5": 0|1,
  "chunk5string": "<verbatim quote from CHUNK5 if used, else empty string>"
}

Rules:
- Set chunkN = 1 ONLY if you used CHUNKN as evidence for the answer; otherwise 0.
- If chunkN = 1, chunkNstring MUST be a verbatim quote copied from CHUNKN.
- If chunkN = 0, chunkNstring MUST be "" (empty string).
- Use ONLY the provided CHUNKs; do not cite or quote anything else.
- If the answer is not fully supported by the provided CHUNKs, set an appropriate answer like:
  "Insufficient grounded evidence in the provided documents to answer." and briefly name what is missing.

The answer should be a natural language response as if you are speaking directly to the user.
Answer Formatting Rules:
- If the answer contains only ONE main point → write as a single concise paragraph (no bullet points).
- If the answer contains MORE THAN ONE independent point → format them as bullet points.
- Bullet point format: each point MUST begin with "- " and be separated by a newline ("\\n").
- Do NOT include any quotations, citations, filenames, page numbers, or dates inside the "answer" field. These belong only in chunkNstring.
"""

        return f"""You must answer using ONLY these context chunks:

{context}

Question:
{question}

Output contract:
{json_contract}""".strip()

if __name__ == "__main__":
    tester = RAGBatchTester()
    tester.run()
