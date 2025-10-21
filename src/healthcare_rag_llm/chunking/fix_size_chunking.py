from __future__ import annotations

from pathlib import Path
import json
import csv
import io
from typing import List, Dict, Optional, Tuple


def fix_size_chunking(
    processed_dir: str,
    chunked_dir: str,
    max_chunk_chars: int = 5000,
    glob_pattern: str = "*.json",
    verbose: bool = True,
) -> None:
    """
    Fixed-size chunking (no overlap).
    - Each chunk has at most max_chunk_chars characters.
    - Table data from metadata["tables"] are exported as separate chunks (CSV-formatted).

    Output:
      <chunked_dir>/fix_size_chunking_result/<stem>.chunks.jsonl
    Record schema:
      {
        "doc_id": "<file_name>",
        "chunk_id": "<file_name>::0007",
        "char_start": <int>,   # inclusive, in original full_text
        "char_end": <int>,     # exclusive, in original full_text
        "pages": [<int>, ...],
        "text": "<substring>",
        "chunk_type": "text" or "table" or "ocr_image"   # NEW
      }
    """
    processed_dir = Path(processed_dir)
    chunked_base = Path(chunked_dir)
    output_dir = chunked_base / "fix_size_chunking_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(processed_dir.glob(glob_pattern))
    if verbose:
        print(f"[INFO] Found {len(json_paths)} JSON files in {processed_dir}")

    for jp in json_paths:
        # Load parsed metadata
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

        # NEW: Merge OCR fallback into text when applicable
        for p in pages:
            text = (p.get("text") or "").strip()
            ocr_blocks = p.get("ocr_fallback") or []

            # Case 1: replace text if too short but OCR exists
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

        # NEW: Rebuild full_text after OCR merge
        meta["full_text"] = "\n\n".join(p.get("text", "") for p in pages)
        full_text = meta["full_text"]

        # Skip empty text
        if not full_text.strip():
            if verbose:
                print(f"[SKIP] {jp.name} (no text after OCR merge)")
            continue

        # Build output file
        stem = Path(file_name).stem
        out_file = output_dir / f"{stem}.chunks.jsonl"
        buffered_chunks: List[Dict] = []
        chunk_idx = 0

        # --- Fixed-size text chunking ---
        cursor = 0
        total_len = len(full_text)
        while cursor < total_len:
            c_start = cursor
            c_end = min(cursor + max_chunk_chars, total_len)
            text = full_text[c_start:c_end]
            pages_list = _estimate_pages_for_span(pages, c_start, c_end, total_len)
            rec = {
                "doc_id": file_name,
                "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}",
                "char_start": c_start,
                "char_end": c_end,
                "pages": pages_list,
                "text": text,
                "chunk_type": "text"
            }
            buffered_chunks.append(rec)
            chunk_idx += 1
            cursor = c_end

        # --- Table chunks ---
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

        # NEW: OCR image chunks (from ocr_fallback regions)
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

        # Write chunks
        with open(out_file, "w", encoding="utf-8") as out_f:
            for rec in buffered_chunks:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if verbose:
            print(f"[OK]  {jp.name} -> {len(buffered_chunks)} chunks ({tables_added} table-chunks, {ocr_chunks_added} ocr-chunks) -> {out_file}")


def _table_to_csv(table_2d: List[List[Optional[str]]]) -> str:
    """
    Convert a 2D list (rows) into an RFC 4180-compliant CSV string:
    - Comma delimiter.
    - Double-quote fields when needed, escaping inner quotes by doubling.
    - None -> "".
    - Preserve embedded newlines inside quoted cells.
    """
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, dialect="excel", quoting=csv.QUOTE_MINIMAL)
    for row in table_2d:
        safe_row = [("" if (cell is None) else str(cell)) for cell in row]
        writer.writerow(safe_row)
    return buf.getvalue()


def _estimate_pages_for_span(
    pages: List[Dict], c_start: int, c_end: int, total_len: int
) -> List[int]:
    """
    Heuristic: map chunk span back to pages based on cumulative text lengths.
    """
    out = []
    cursor = 0
    for p in pages:
        page_no = int(p.get("page"))
        page_text = p.get("text", "") or ""
        length = len(page_text)
        page_start = cursor
        page_end = cursor + length
        cursor += length + 2  # approximate newline gap
        if page_end <= c_start:
            continue
        if page_start >= c_end:
            break
        out.append(page_no)
    return out
