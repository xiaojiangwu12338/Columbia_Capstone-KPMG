from __future__ import annotations

from pathlib import Path
import json
import csv
import io
from typing import List, Dict, Optional, Tuple


def semantic_chunking(
    processed_dir: str,
    chunked_dir: str,
    max_chunk_chars: int = 5000,
    glob_pattern: str = "*.json",
    verbose: bool = True,
) -> None:
    """
    Semantic-based chunking with sentence or paragraph awareness.
    - Preserves paragraph and semantic boundaries when possible.
    - Each chunk <= max_chunk_chars.
    - Tables become standalone chunks (CSV formatted).

    NEW:
    - OCR fallback merge for page-level text.
    - OCR image chunks from ocr_fallback.
    """
    processed_dir = Path(processed_dir)
    chunked_base = Path(chunked_dir)
    output_dir = chunked_base / "semantic_chunking_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(processed_dir.glob(glob_pattern))
    if verbose:
        print(f"[INFO] Found {len(json_paths)} JSON files in {processed_dir}")

    for jp in json_paths:
        # Load metadata
        try:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            if verbose:
                print(f"[SKIP] {jp.name} (cannot read JSON: {e})")
            continue

        if meta.get("error"):
            if verbose:
                print(f"[SKIP] {jp.name} (error present: {meta['error']})")
            continue

        file_name = meta.get("file_name") or jp.name
        full_text: Optional[str] = meta.get("full_text")
        pages: Optional[List[Dict]] = meta.get("pages")

        if not full_text:
            if verbose:
                print(f"[SKIP] {jp.name} (empty full_text)")
            continue
        if not pages:
            if verbose:
                print(f"[SKIP] {jp.name} (missing pages)")
            continue

        # NEW: Merge OCR fallback text into pages
        for p in pages:
            text = (p.get("text") or "").strip()
            ocr_blocks = p.get("ocr_fallback") or []

            # Case 1: replace text if short or empty but OCR exists
            if len(text) < 50 and ocr_blocks:
                merged_text = "\n".join(b["text"] for b in ocr_blocks if b.get("text"))
                p["text"] = merged_text
                p["ocr_used"] = True

            # Case 2: append OCR supplement if both exist
            elif text and ocr_blocks:
                supplement = "\n".join(b["text"] for b in ocr_blocks if b.get("text"))
                if supplement.strip():
                    p["text"] = f"{text}\n\n[OCR Supplement]\n{supplement}"
                    p["ocr_used"] = True

        # NEW: rebuild full_text after merge
        meta["full_text"] = "\n\n".join(p.get("text", "") for p in pages)
        full_text = meta["full_text"]

        if not full_text.strip():
            if verbose:
                print(f"[SKIP] {jp.name} (empty after OCR merge)")
            continue

        # --- Semantic chunking logic (original part) ---
        # Example: split on double newlines or semantic cues.
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_chunk_chars:
                current += ("\n\n" if current else "") + para
            else:
                if current:
                    chunks.append(current)
                current = para
        if current:
            chunks.append(current)

        stem = Path(file_name).stem
        out_file = output_dir / f"{stem}.chunks.jsonl"

        buffered_chunks: List[Dict] = []
        chunk_idx = 0
        for text in chunks:
            rec = {
                "doc_id": file_name,
                "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}",
                "char_start": None,
                "char_end": None,
                "pages": [],  # optional: could map later
                "text": text,
                "chunk_type": "text"
            }
            buffered_chunks.append(rec)
            chunk_idx += 1

        # --- Table chunks (existing logic) ---
        tables = meta.get("tables") or []
        tables_added = 0
        for t in tables:
            page_no = t.get("page")
            table_data = t.get("table")
            if not isinstance(page_no, int) or not isinstance(table_data, list):
                continue
            csv_text = _table_to_csv(table_data)
            rec = {
                "doc_id": file_name,
                "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}::table",
                "char_start": None,
                "char_end": None,
                "pages": [page_no],
                "text": csv_text,
                "chunk_type": "table"
            }
            buffered_chunks.append(rec)
            chunk_idx += 1
            tables_added += 1

        # NEW: OCR image chunks
        ocr_chunks_added = 0
        for p in pages:
            for o in p.get("ocr_fallback", []):
                text_img = (o.get("text") or "").strip()
                if not text_img:
                    continue
                rec = {
                    "doc_id": file_name,
                    "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}::ocr",
                    "char_start": None,
                    "char_end": None,
                    "pages": [p.get("page")],
                    "text": text_img,
                    "chunk_type": "ocr_image",
                    "bbox": o.get("bbox")
                }
                buffered_chunks.append(rec)
                chunk_idx += 1
                ocr_chunks_added += 1

        # --- Write all chunks ---
        with open(out_file, "w", encoding="utf-8") as out_f:
            for rec in buffered_chunks:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if verbose:
            print(f"[OK]  {jp.name} -> {len(buffered_chunks)} chunks ({tables_added} table-chunks, {ocr_chunks_added} ocr-chunks) -> {out_file}")


def _table_to_csv(table_2d: List[List[Optional[str]]]) -> str:
    """
    Convert 2D table data into CSV string.
    """
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, dialect="excel", quoting=csv.QUOTE_MINIMAL)
    for row in table_2d:
        safe_row = [("" if (cell is None) else str(cell)) for cell in row]
        writer.writerow(safe_row)
    return buf.getvalue()