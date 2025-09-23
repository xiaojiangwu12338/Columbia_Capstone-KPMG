import os
import json
from pathlib import Path
import requests
import re

# ---------- Config ----------
DOCS_DIR = "docs"
CHUNKS_DIR = "fix-size_chunks"
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 100      # characters overlap
LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
SUMMARY_MAX_CHARS = 4000
SUMMARY_TIMEOUT_SEC = 300
# ---------------------------

def chunk_text_fixed(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, n = [], len(text)
    if n == 0:
        return chunks
    overlap = max(0, min(overlap, chunk_size - 1))
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = end - overlap
    return chunks

def call_llm(prompt: str, model: str = LLM_MODEL, url: str = LLM_URL, timeout: int = SUMMARY_TIMEOUT_SEC) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def clean_one_line(s: str) -> str:
    """Force a clean, single-line summary without prefaces."""
    s = s.strip()
    # Drop common lead-ins the model might add anyway
    s = re.sub(r"^(summary|tl;dr|document summary)\s*[:\-–]\s*", "", s, flags=re.I)
    s = s.strip("“”\"'`•- \n\t")
    s = " ".join(s.split())  # collapse whitespace/newlines
    if s and s[-1] not in ".!?":  # end with punctuation
        s += "."
    return s

def summarize_document(text: str, max_chars: int = SUMMARY_MAX_CHARS, model: str = LLM_MODEL) -> str:
    if not text:
        return ""
    truncated = text[:max_chars]
    # Prompt that forces direct summary output
    prompt = (
        "Write ONE concise sentence summarizing the following document's main purpose and scope. "
        "Output ONLY the sentence—no preface, labels, or extra words.\n\n"
        f"Document:\n{truncated}\n"
    )
    try:
        raw = call_llm(prompt, model=model)
        return clean_one_line(raw)
    except Exception as e:
        print(f"⚠️  Summarization failed: {e}")
        return ""

def chunk_documents(input_dir=DOCS_DIR, out_dir=CHUNKS_DIR, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, add_doc_summary=True):
    input_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for meta_file in input_path.glob("*.json"):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Skipping {meta_file.name} (bad JSON): {e}")
            continue

        full_text = (meta.get("full_text") or "").strip()
        if not full_text:
            print(f" Skipping {meta_file.name} (no full_text).")
            continue

        # Use existing summary if present; otherwise generate
        doc_summary = (meta.get("doc_summary") or "").strip()
        if add_doc_summary and not doc_summary:
            doc_summary = summarize_document(full_text)

        chunks = chunk_text_fixed(full_text, chunk_size, overlap)
        ocr_used = any(bool(p.get("ocr_fallback")) for p in meta.get("pages", []))

        records = []
        for i, chunk in enumerate(chunks, start=1):
            records.append({
                "doc_id": meta.get("file_name"),
                "chunk_id": f"{meta.get('file_name')}_chunk{i}",
                "chunk_index": i,
                "text": chunk,
                "doc_summary": doc_summary,          # flattened
                "source_file": meta.get("source_file"),
                "parsed_time": meta.get("parsed_time"),
                "watermarks": meta.get("watermarks", []),
                "tables": meta.get("tables", []),
                "ocr_used": ocr_used
            })

        out_file = out_path / f"{meta_file.stem}_chunks.jsonl"
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f" {meta_file.name}: {len(records)} chunks → {out_file}")
        except Exception as e:
            print(f"  Failed writing chunks for {meta_file.name}: {e}")

if __name__ == "__main__":
    chunk_documents()
