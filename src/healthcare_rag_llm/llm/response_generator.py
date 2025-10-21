# src/healthcare_rag_llm/llm/response_generator.py

from typing import Dict, List
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.graph_builder.queries import query_chunks
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.reranking.reranker import apply_rerank_to_chunks

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

    def __init__(self, llm_client: LLMClient,system_prompt: str = SYSTEM_PROMPT, use_reranker: bool = True):
        """Initialize with a given LLM client."""
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.embedder = HealthcareEmbedding()  # Embedding
        self.use_reranker = use_reranker

    def answer_question(self, question: str, top_k: int = 5, rerank_top_k: int = 20) -> Dict:
        """
        Full question-answering pipeline.

        Steps:
          1. Retrieve relevant chunks (RAG retrieval)
          2. Construct context + prompt
          3. Generate final response using the LLM
        """
        # 1. Encode query as vector
        query_vec = self.embedder.encode([question])["dense_vecs"][0].tolist()
        
        # 2. Retrieve more chunks initially (for reranking)
        initial_k = rerank_top_k if self.use_reranker else top_k
        retrieved_chunks = query_chunks(query_vec, top_k=initial_k)

        # 3. Apply reranking if enabled
        if self.use_reranker and retrieved_chunks:
            retrieved_chunks = apply_rerank_to_chunks(
                query=question,
                chunks=retrieved_chunks,
                combine_with_dense=True,  
                alpha=0.3,  
                text_key="text",
                dense_score_key="score"
            )
        #4 Take top k chunks
        final_chunks = retrieved_chunks[:top_k]
        
        # 3. Context
        context = "\n\n".join(
            [f"[Document ID: {chunk['doc_id']}] -[Chunk ID: {chunk['chunk_id']}]-[pages: {chunk['pages']}] - [Chunk Content: {chunk['text']}]" 
             for chunk in final_chunks]
        )
        
        # 6. Generate response 
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
            system_prompt=self.system_prompt
        )

        return {
            "question": question,
            "answer": llm_response,
            "retrieved_docs": final_chunks,
        }