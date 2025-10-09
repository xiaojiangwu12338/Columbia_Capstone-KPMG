# src/healthcare_rag_llm/llm/response_generator.py

from typing import Dict, List
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.graph_builder.queries import query_chunks
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding


SYSTEM_PROMPT = """
You are a careful NYS Medicaid policy assistant used in a compliance workflow.
Answer ONLY using the provided context chunks.

Rules:
1) Do not speculate. If the answer is not fully supported, say:
   "Insufficient grounded evidence in the provided documents to answer." and name what is missing.
2) For each bullet, quote the exact line(s) relied upon and place a citation immediately after it:
   [<doc_id:page> — <date if available>].
3) Prefer the most recent guidance when conflicted, but note the conflict explicitly.
4) Keep dates, codes, dollar figures, NCPDP fields, and policy names exactly as written.
5) Be concise and decision-ready. Use bullets and a short "What this means" at the end.
6) If timing is relevant, highlight lines starting with "Effective".
"""

class ResponseGenerator:
    """
    End-to-end RAG (Retrieval-Augmented Generation) pipeline.
    Responsible for:
      1. Retrieving relevant context documents
      2. Building the final LLM prompt
      3. Generating the answer using the LLM
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize with a given LLM client."""
        self.llm_client = llm_client
        self.embedder = HealthcareEmbedding()  # 添加嵌入器

    def answer_question(self, question: str, top_k: int = 5) -> Dict:
        """
        Full question-answering pipeline.

        Steps:
          1. Retrieve relevant chunks (RAG retrieval)
          2. Construct context + prompt
          3. Generate final response using the LLM
        """
        # 1. 编码查询为向量
        query_vec = self.embedder.encode([question])["dense_vecs"][0].tolist()
        
        # 2. 使用向量进行检索
        retrieved_chunks = query_chunks(query_vec, top_k=top_k)
        
        # 3. 构建上下文
        context = "\n\n".join(
            [f"[Document ID: {chunk['doc_id']}] -[Chunk ID: {chunk['chunk_id']}]-[pages: {chunk['pages']}] - [Chunk Content: {chunk['text']}]" 
             for i, chunk in enumerate(retrieved_chunks)]
        )

        # 4. 构建用户消息
        user_msg = f"""
Question:
{question}

Context Chunks (authoritative; cite only these):
{context}

Chunks format:
[Document ID: <doc_id>] -[Chunk ID: <chunk_id>]-[pages: <pages>] - [Chunk Content: <chunk_content>]

Output sections (exactly):
- Answer
- Evidence (quoted)
- Caveats (if any)
Each bullet must have a citation like [doc or doc:page — Mon DD, YYYY].
""".strip()

        llm_response = self.llm_client.chat(
            user_prompt=user_msg,
            system_prompt=SYSTEM_PROMPT
        )

        return {
            "question": question,
            "answer": llm_response,
            "retrieved_docs": retrieved_chunks,
        }