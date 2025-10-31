# src/healthcare_rag_llm/llm/response_gen_json.py


from typing import Dict, Any, List, Optional
import json
import csv
from pathlib import Path


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
2) Prefer the most recent guidance when conflicted, but note the conflict explicitly.
3) Keep dates, codes, dollar figures, NCPDP fields, and policy names exactly as written.
4) Be concise and decision-ready.
5) If timing is relevant, highlight lines starting with "Effective".
"""


# --------------------------------------------------------------------
# Minimal schema helpers (no external dependencies)
# --------------------------------------------------------------------
_JSON_KEYS = [
    "answer",
    "chunk1", "chunk1string",
    "chunk2", "chunk2string",
    "chunk3", "chunk3string",
    "chunk4", "chunk4string",
    "chunk5", "chunk5string",
]


def _validate_json_payload(data: Any) -> bool:
    """
    Validate the strict shape:
      - dict with exactly the keys in _JSON_KEYS
      - "answer": str
      - chunkN: 0|1 or bool
      - chunkNstring: str
      - coherence: if chunkN == 0 then chunkNstring must be empty
    """
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != set(_JSON_KEYS):
        return False


    if not isinstance(data.get("answer"), str):
        return False


    for i in range(1, 6):
        flag_key = f"chunk{i}"
        quote_key = f"chunk{i}string"


        flag = data.get(flag_key)
        if not isinstance(flag, (bool, int)):
            return False
        if isinstance(flag, int) and flag not in (0, 1):
            return False


        quote = data.get(quote_key)
        if not isinstance(quote, str):
            return False


        if (flag in (0, False)) and quote.strip() != "":
            return False


    return True




def _normalize_flags_to_ints(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize chunkN to 0/1 ints."""
    for i in range(1, 6):
        k = f"chunk{i}"
        data[k] = int(bool(data.get(k, 0)))
    return data




# ----------------------- Metadata helpers ----------------------------


def _read_metadata_csv(csv_path: Path) -> List[Dict[str, str]]:
    """
    Read metadata_filled.csv and return a list of rows (dicts).
    Expected header includes:
      authority_name,authority_abbr,doc_title,file_name,source_url,effective_date,doc_type
    """
    rows: List[Dict[str, str]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize keys used later
            r["file_name"] = (r.get("file_name") or "").strip()
            r["source_url"] = (r.get("source_url") or "").strip()
            r["effective_date"] = (r.get("effective_date") or "").strip()
            r["doc_title"] = (r.get("doc_title") or "").strip()
            rows.append(r)
    return rows




def _add_key(idx: Dict[str, Dict[str, str]], key: str, row: Dict[str, str]):
    if key:
        idx[key.lower()] = row




def _basename(s: str) -> str:
    try:
        return Path(s).name
    except Exception:
        return s




def _stem(s: str) -> str:
    try:
        return Path(s).stem
    except Exception:
        return s




def _build_metadata_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Index by multiple variants to maximize hit rate:
      - file_name, basename(file_name), stem(file_name)
      - doc_title
    """
    idx: Dict[str, Dict[str, str]] = {}
    for r in rows:
        fn = r.get("file_name") or ""
        dt = r.get("doc_title") or ""
        _add_key(idx, fn, r)
        _add_key(idx, _basename(fn), r)
        _add_key(idx, _stem(fn), r)
        _add_key(idx, dt, r)
    return idx




def _lookup_metadata_for_chunk(chunk: Dict[str, Any], meta_idx: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Try these (case-insensitive), in order, with raw, basename, and stem variants:
      1) chunk.file_name
      2) chunk.doc_title
      3) chunk.doc_id
    """
    candidates = []
    for key in ("file_name", "doc_title", "doc_id"):
        val = (chunk.get(key) or "").strip()
        if val:
            candidates.extend([val, _basename(val), _stem(val)])
    for cand in candidates:
        row = meta_idx.get(cand.lower())
        if row:
            return row
    return None




def _format_pages(pages_val: Any) -> str:
    """
    Normalize the pages field to the format: 'page 3, 4'.
    Accepts: list/tuple -> comma list; int -> single; "3-4" -> kept as-is but prefixed with 'page'.
    """
    if pages_val is None:
        return ""
    if isinstance(pages_val, (list, tuple)):
        flat = [str(p).strip() for p in pages_val if str(p).strip()]
        return f"page {', '.join(flat)}" if flat else ""
    if isinstance(pages_val, int):
        return f"page {pages_val}"
    s = str(pages_val).strip()
    return f"page {s}" if s else ""




def _format_manual_view(
    data: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    meta_idx: Dict[str, Dict[str, str]],
) -> str:
    """
    Final manual output shape:


    Answer:
    <answer>


    Evidence:
    <doc_title>, page 3, 4:
    <verbatim quote>
    effective on <YYYY-MM-DD>: <url>
    """
    lines: List[str] = []
    lines.append("Answer:")
    lines.append(data["answer"].strip())
    lines.append("")  # ensure at least one empty line after Answer block
    lines.append("Evidence:")


    any_ev = False
    for i in range(1, 6):
        if data[f"chunk{i}"] != 1:
            continue


        src_chunk = chunks[i - 1] if i - 1 < len(chunks) else {}
        meta_row = _lookup_metadata_for_chunk(src_chunk, meta_idx)


        # doc_title from metadata preferred; fallbacks if missing
        if meta_row and meta_row.get("doc_title"):
            doc_title = meta_row["doc_title"]
        elif (src_chunk.get("doc_title") or "").strip():
            doc_title = src_chunk["doc_title"].strip()
        else:
            # last resort: friendly filename/doc_id
            for key in ("file_name", "doc_id"):
                val = (src_chunk.get(key) or "").strip()
                if val:
                    doc_title = _basename(val) or val
                    break
            else:
                doc_title = f"chunk{i}"


        pages_part = _format_pages(src_chunk.get("pages"))
        quote = data.get(f"chunk{i}string", "").strip()
        if not quote:
            continue


        any_ev = True
        # "<doc_title>, page 3, 4:"  (omit pages if unavailable)
        header_suffix = f", {pages_part}:" if pages_part else ":"
        lines.append(f"{doc_title}{header_suffix}")
        lines.append(quote)


        # "effective on <date>: <url>" if present
        src_url = meta_row.get("source_url") if meta_row else ""
        eff_date = meta_row.get("effective_date") if meta_row else ""
        if eff_date or src_url:
            eff_label = eff_date if eff_date else "N/A"
            url_label = src_url if src_url else "N/A"
            lines.append(f"effective on {eff_label}: {url_label}")


    if not any_ev:
        lines.append("(none)")


    # Guarantee exactly one blank line between the Answer block and "Evidence:"
    text = "\n".join(lines)
    text = text.replace("\nEvidence:", "\n\nEvidence:")
    return text




# ----------------------- RAG class ----------------------------


class ResponseGenerator:
    """
    End-to-end RAG pipeline with:
      - Strict JSON from LLM (answer + chunk flags + verbatim quotes)
      - Metadata join to replace chunk labels with doc_title and add URL/effective date
      - Output string compatible with existing Streamlit app (no app changes)
    """


    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = SYSTEM_PROMPT,
        use_reranker: bool = True,
        metadata_csv_path: Optional[Path] = None,
    ):
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.embedder = HealthcareEmbedding()
        self.use_reranker = use_reranker


        # Default metadata path: <project_root>/data/metadata/metadata_filled.csv
        if metadata_csv_path is None:
            # .../project/src/healthcare_rag_llm/llm/response_gen_json.py
            # parents[0]=llm, [1]=healthcare_rag_llm, [2]=src, [3]=project root
            project_root = Path(__file__).resolve().parents[3]
            metadata_csv_path = project_root / "data" / "metadata" / "metadata_filled.csv"


        self._metadata_rows = _read_metadata_csv(metadata_csv_path)
        self._metadata_index = _build_metadata_index(self._metadata_rows)


    def answer_question(self, question: str, top_k: int = 5, rerank_top_k: int = 20) -> Dict:
        """
        Return dict with keys expected by the current app.py:
          - "answer": formatted string (Answer + blank line + Evidence section)
          - "retrieved_docs": list of chunks (for the sources section)
          - (extra) "answer_json": the structured JSON from the LLM
          - (extra) "answer_manual": same as "answer"
        """
        # 1) Encode query
        query_vec = self.embedder.encode([question])["dense_vecs"][0].tolist()


        # 2) Retrieve
        initial_k = rerank_top_k if self.use_reranker else top_k
        retrieved_chunks = query_chunks(query_vec, top_k=initial_k)


        # 3) Rerank
        if self.use_reranker and retrieved_chunks:
            retrieved_chunks = apply_rerank_to_chunks(
                query=question,
                chunks=retrieved_chunks,
                combine_with_dense=True,
                alpha=0.3,
                text_key="text",
                dense_score_key="score",
            )


        # 4) Take top-k
        final_chunks = (retrieved_chunks or [])[:top_k]


        # Label chunks for the model
        labeled = []
        for idx, ch in enumerate(final_chunks, start=1):
            labeled.append({
                "label": f"CHUNK{idx}",
                "doc_id": ch.get("doc_id"),
                "chunk_id": ch.get("chunk_id"),
                "pages": ch.get("pages"),
                "text": ch.get("text"),
                "file_name": ch.get("file_name"),
                "doc_title": ch.get("doc_title"),
            })


        # Build LLM context
        context = "\n\n".join(
            [
                f"{c['label']} "
                f"[Document ID: {c['doc_id']}] "
                f"[Chunk ID: {c['chunk_id']}] "
                f"[pages: {c['pages']}] "
                f"[Chunk Content: {c['text']}]"
                for c in labeled
            ]
        )


        # Strict JSON contract for model output
        json_contract = """
Return ONLY valid JSON with EXACTLY these fields (no extra keys, no trailing text):
{
  "answer": "<string>",
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
"""


        user_prompt = f"""
You must answer using ONLY these context chunks:


{context}


Question:
{question}


Output contract:
{json_contract}
""".strip()


        # 5) Call the model
        raw = self.llm_client.chat(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt
        )


        # 6) Parse -> validate -> repair
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None


        if not _validate_json_payload(parsed):
            repair_prompt = f"""
The previous output was not valid per the required JSON contract.


Fix it. Return ONLY the corrected JSON (no commentary). Ensure:
- exact keys: {_JSON_KEYS}
- "answer" is a string
- chunkN is 0 or 1
- if chunkN = 1, chunkNstring is a verbatim quote from CHUNKN; else "".


Previous output (verbatim):
{raw}
"""
            repaired = self.llm_client.chat(
                user_prompt=repair_prompt,
                system_prompt="You fix JSON to match the exact schema. Output only JSON."
            )
            try:
                parsed = json.loads(repaired)
            except Exception:
                parsed = None


        if not _validate_json_payload(parsed):
            parsed = {
                "answer": "Insufficient grounded evidence in the provided documents to answer.",
                "chunk1": 0, "chunk1string": "",
                "chunk2": 0, "chunk2string": "",
                "chunk3": 0, "chunk3string": "",
                "chunk4": 0, "chunk4string": "",
                "chunk5": 0, "chunk5string": "",
            }


        parsed = _normalize_flags_to_ints(parsed)


        # 7) Manual view (doc_title + pages + effective date + url), with a forced blank line before "Evidence:"
        manual_view = _format_manual_view(parsed, final_chunks, self._metadata_index)


        # 8) Keys expected by app.py (unchanged)
        return {
            "question": question,
            "answer": manual_view,
            "retrieved_docs": final_chunks,
            # extras (optional)
            "answer_json": parsed,
            "answer_manual": manual_view,
        }



