#!/usr/bin/env python3
"""
Test script - prints exactly what is sent to LLM and what LLM returns.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.graph_builder.queries import query_chunks
from healthcare_rag_llm.reranking.reranker import apply_rerank_to_chunks
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.utils.api_config import APIConfigManager
from healthcare_rag_llm.llm.response_gen_json import SYSTEM_PROMPT
from healthcare_rag_llm.filters.load_metadata import build_filter_extractor

# ========== CONFIGURE YOUR TEST ==========
TEST_QUESTION = "When did redetermination begin for the COVID-19 Public Health Emergency unwind in New York State?"
TEST_MODEL = "gemini-3-pro-preview"
TOP_K = 5
USE_RERANK = False
ALPHA = 0.5
TEMPERATURE = 0.1
# ==========================================


def _validate_json_payload(data) -> bool:
    """
    Validate the strict shape:
      - dict with exactly the keys in _JSON_KEYS
      - "answer": str
      - chunkN: 0|1 or bool
      - chunkNstring: str
      - coherence: if chunkN == 0 then chunkNstring must be empty
    """
    _JSON_KEYS = [
    "answer",
    "chunk1", "chunk1string",
    "chunk2", "chunk2string",
    "chunk3", "chunk3string",
    "chunk4", "chunk4string",
    "chunk5", "chunk5string",
]
    if not isinstance(data, dict):
        print(type(data))
        print(0)
        return False
    if set(data.keys()) != set(_JSON_KEYS):
        print(1)
        return False

    if not isinstance(data.get("answer"), str):
        print(2)
        return False

    for i in range(1, 6):
        flag_key = f"chunk{i}"
        quote_key = f"chunk{i}string"

        flag = data.get(flag_key)
        if not isinstance(flag, (bool, int)):
            print(3)
            return False
        if isinstance(flag, int) and flag not in (0, 1):
            print(4)
            return False

        quote = data.get(quote_key)
        if not isinstance(quote, str):
            print(5)
            return False

        if (flag in (0, False)) and quote.strip() != "":
            print(6)
            return False

    return True


def build_user_prompt(question: str, chunks: list, top_k: int = 5) -> str:
    chunk_lines = []
    for i, chunk in enumerate(chunks[:top_k], start=1):
        chunk_lines.append(f"CHUNK{i}:\n{chunk.get('text', '')}\n")

    chunks_section = "\n".join(chunk_lines)

    json_contract = f"""
Return ONLY valid JSON with EXACTLY these fields (no extra keys, no trailing text):
{{
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
}}

Rules:
- Set chunkN = 1 ONLY if you used CHUNKN as evidence for the answer; otherwise 0.
- If chunkN = 1, chunkNstring MUST be a verbatim quote copied from CHUNKN.
- If chunkN = 0, chunkNstring MUST be "" (empty string).
- Use ONLY the provided CHUNKs; do not cite or quote anything else.
- If the answer is not fully supported by the provided CHUNKs, set an appropriate answer like:
  "Insufficient grounded evidence in the provided documents to answer." and briefly name what is missing.
"""

    return f"""
Context:
{chunks_section}

Question: {question}

{json_contract}
"""


def main():
    print(f"Testing: {TEST_MODEL} | Question: {TEST_QUESTION}\n")

    # Initialize filter extractor
    print("Initializing filter extractor...")
    filter_extractor = build_filter_extractor()

    # Extract filters from question
    print("Extracting filters from question...")
    filters = filter_extractor.extract(TEST_QUESTION)

    print("\n" + "=" * 80)
    print("EXTRACTED FILTERS:")
    print("=" * 80)
    print(json.dumps(filters, indent=2, ensure_ascii=False))
    print()

    # Initialize and retrieve
    embedding_model = HealthcareEmbedding()
    query_vec = embedding_model.encode([TEST_QUESTION])['dense_vecs'][0].tolist()

    initial_k = TOP_K * 2 if USE_RERANK else TOP_K
    chunks = query_chunks(
        query_vec,
        top_k=initial_k,
        authority_names=filters.get("authority_names"),
        doc_titles=filters.get("doc_titles"),
        doc_types=filters.get("doc_types"),
        min_effective_date=filters.get("min_effective_date"),
        max_effective_date=filters.get("max_effective_date"),
        keywords=filters.get("keywords"),
    )

    if USE_RERANK and chunks:
        chunks = apply_rerank_to_chunks(
            query=TEST_QUESTION,
            chunks=chunks,
            combine_with_dense=True,
            alpha=ALPHA,
            text_key="text",
            dense_score_key="score",
        )

    final_chunks = chunks[:TOP_K]
    user_prompt = build_user_prompt(TEST_QUESTION, final_chunks, top_k=TOP_K)

    # Initialize LLM
    config = APIConfigManager().get_model_config(TEST_MODEL)
    llm = LLMClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=TEST_MODEL,
        provider=config.provider
    )

    # ========== PRINT WHAT IS SENT TO LLM ==========
    print("=" * 80)
    print("SYSTEM PROMPT:")
    print("=" * 80)
    print(SYSTEM_PROMPT)

    print("\n" + "=" * 80)
    print("USER PROMPT:")
    print("=" * 80)
    print(user_prompt)

    # Call LLM
    response = llm.chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=TEMPERATURE
    )

    # ========== PRINT LLM RESPONSE ==========
    print("\n" + "=" * 80)
    print("LLM RESPONSE:")
    #print(_validate_json_payload(json.loads(response)))
    print("=" * 80)
    print(response)

    # Parse if JSON
    try:
        parsed = json.loads(response)
        print("\n" + "=" * 80)
        print("PARSED:")
        print("=" * 80)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except:
        pass


if __name__ == "__main__":
    main()
    payload = { 
  "answer": "New York State began the redetermination process for the COVID-19 Public Health Emergency unwind in April 2023.",
  "chunk1": 1,
  "chunk1string": "The Consolidated Appropriations Act of 2023 required states to begin the process of redetermining Medicaid eligibility for its members, which New York State (NYS) began in April 2023.",
  "chunk2": 0,
  "chunk2string": "",
  "chunk3": 0,
  "chunk3string": "",
  "chunk4": 0,
  "chunk4string": "",
  "chunk5": 0,
  "chunk5string": ""
}
