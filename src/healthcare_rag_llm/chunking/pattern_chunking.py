from __future__ import annotations

from pathlib import Path
import json
import re
import csv
import io
from typing import List, Dict, Optional, Tuple


def asterisk_separate_chunking(
    processed_dir: str,
    chunked_dir: str,
    max_chunk_chars: int = 1200,
    glob_pattern: str = "*.json",
    min_repeats: int = 10,
    separator_char: str = "*",
    verbose: bool = True,
) -> None:
    """
    Chunk documents on runs of >= min_repeats of `separator_char`.
    - The separator runs are REMOVED from the output chunk text.
    - If a separator-delimited segment exceeds max_chunk_chars, split it by size (no overlap).
    - Keep raw text (no normalization).
    - Associate each chunk with the list of page numbers it spans.

    NEW 1.0:
    - If input JSON includes "tables": [...], each table becomes its own chunk:
        * pages: [<table.page>]
        * char_start/char_end: null
        * chunk_id: "<file_name>::0007::table"
        * text: CSV-formatted table (RFC 4180; comma; null→empty; embedded newlines preserved)
    - Output is interleaved in page order (text chunks before table chunks on same page).

    NEW 2.0:
    - OCR handling:
        * Page-level fallback if text missing or short
        * Mixed-page supplement merge
        * OCR fallback regions (per image) become independent chunks

    Output:
      <chunked_dir>/asterisk_separate_chunking_result/<stem>.chunks.jsonl
    Record schema:
      {
        "doc_id": "<file_name>",
        "chunk_id": "<file_name>::0007" or "<file_name>::0008::table",
        "char_start": <int|null>,   # inclusive, in original full_text
        "char_end": <int|null>,     # exclusive, in original full_text
        "pages": [<int>, ...],
        "text": "<substring with all separator runs removed>" OR "<csv>"
      }
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(chunked_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be > 0")

    # Build a regex for runs of the chosen separator character
    sep_token = re.escape(separator_char)
    sep_re = re.compile(fr"{sep_token}{{{int(min_repeats)},}}")

    json_paths = sorted(processed_dir.glob(glob_pattern))
    if verbose:
        print(f"[INFO] (sep='{separator_char}') Found {len(json_paths)} JSON files in {processed_dir}")

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
        category: str = (meta.get("category") or "unknown")

        if not full_text:
            if verbose:
                print(f"[SKIP] {jp.name} (empty full_text)")
            continue
        if not pages:
            if verbose:
                print(f"[SKIP] {jp.name} (missing pages)")
            continue

        # NEW: Page-level OCR merge and supplement logic
        for p in pages:
            text = (p.get("text") or "").strip()
            ocr_blocks = p.get("ocr_fallback") or []

            # Case 1: Replace whole page text if empty or very short but OCR exists
            if len(text) < 50 and ocr_blocks:
                merged_text = "\n".join(b["text"] for b in ocr_blocks if b.get("text"))
                p["text"] = merged_text
                p["ocr_used"] = True

            # Case 2: Append OCR supplement if both exist
            elif text and ocr_blocks:
                supplement = "\n".join(b["text"] for b in ocr_blocks if b.get("text"))
                if supplement.strip():
                    p["text"] = f"{text}\n\n[OCR Supplement]\n{supplement}"
                    p["ocr_used"] = True

        # NEW: Rebuild full_text after OCR merge
        meta["full_text"] = "\n\n".join(p.get("text", "") for p in pages)
        full_text = meta["full_text"]

        # Map pages to character spans in full_text
        page_spans = _build_page_spans_assuming_double_newlines(pages, full_text)
        if not page_spans or page_spans[-1][2] != len(full_text):
            if verbose:
                print(f"[WARN] {jp.name} page-span mismatch; falling back to search spans")
            page_spans = _build_page_spans_via_search(pages, full_text)
            if not page_spans:
                if verbose:
                    print(f"[SKIP] {jp.name} (could not map pages to full_text)")
                continue

        # Identify separator runs and build primary segments between them.
        segments: List[Tuple[int, int]] = []  # (seg_start, seg_end) in original full_text
        last_end = 0
        for m in sep_re.finditer(full_text):
            start, end = m.span()
            if start > last_end:
                segments.append((last_end, start))
            last_end = end
        if last_end < len(full_text):
            segments.append((last_end, len(full_text)))

        # Build text chunks (buffer first; we'll interleave with tables by page order)
        buffered_chunks: List[Tuple[int, int, Dict]] = []
        # Tuple: (first_page_for_sort, tie_breaker_int, record_dict)
        # tie_breaker: 0 for text, 1 for table — so tables come after text for the same page.

        chunk_idx = 0
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            offset = 0
            while offset < seg_len:
                c_start = seg_start + offset
                c_end = min(c_start + max_chunk_chars, seg_end)

                text = full_text[c_start:c_end]
                if separator_char in text:
                    text = sep_re.sub("", text)

                pages_list = _pages_overlapping_span(page_spans, c_start, c_end)
                first_page = pages_list[0] if pages_list else 10**9  # very large if none (shouldn't happen)

                rec = {
                    "doc_id": file_name,
                    "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}",
                    "char_start": c_start,
                    "char_end": c_end,
                    "pages": pages_list,
                    "text": text,
                    "chunk_type": "text",
                    "category": category,
                }
                buffered_chunks.append((first_page, 0, rec))
                chunk_idx += 1

                if c_end >= seg_end:
                    break
                offset += max_chunk_chars

        # Build table chunks
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
                "chunk_type": "table",
                "category": category,
            }
            buffered_chunks.append((page_no, 1, rec))
            chunk_idx += 1
            tables_added += 1

        # NEW: Add OCR image chunks (from ocr_fallback regions)
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
                    "bbox": o.get("bbox"),
                    "category": category,
                }
                buffered_chunks.append((p.get("page", 10**9), 2, rec))
                chunk_idx += 1
                ocr_chunks_added += 1

        # Sort by page order; text (0) before table (1), OCR (2) last
        buffered_chunks.sort(key=lambda x: (x[0], x[1]))

        # Emit all chunks in the chosen order
        stem = Path(file_name).stem
        out_file = output_dir / f"{stem}.chunks.jsonl"
        with open(out_file, "w", encoding="utf-8") as out_f:
            for _, __, rec in buffered_chunks:
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


# --- helpers (unchanged) ---
def _build_page_spans_assuming_double_newlines(
    pages: List[Dict], full_text: str
) -> List[Tuple[int, int, int]]:
    spans: List[Tuple[int, int, int]] = []
    cursor = 0
    total_len = len(full_text)

    for i, p in enumerate(pages):
        page_no = int(p.get("page", i + 1))
        page_text = p.get("text", "") or ""
        start = cursor
        end = start + len(page_text)
        if end > total_len:
            return []
        spans.append((page_no, start, end))
        cursor = end
        if i < len(pages) - 1:
            if cursor + 1 < total_len and full_text[cursor:cursor + 2] == "\n\n":
                cursor += 2
            else:
                return []
    if cursor != total_len:
        return []
    return spans


def _build_page_spans_via_search(
    pages: List[Dict], full_text: str
) -> List[Tuple[int, int, int]]:
    spans: List[Tuple[int, int, int]] = []
    cursor_hint = 0
    for i, p in enumerate(pages):
        page_no = int(p.get("page", i + 1))
        page_text = p.get("text", "") or ""
        if not page_text:
            spans.append((page_no, cursor_hint, cursor_hint))
            while cursor_hint < len(full_text) and full_text[cursor_hint] == "\n":
                cursor_hint += 1
            continue

        pos = full_text.find(page_text, cursor_hint)
        if pos == -1:
            return []
        start = pos
        end = pos + len(page_text)
        spans.append((page_no, start, end))
        cursor_hint = end
        while cursor_hint < len(full_text) and full_text[cursor_hint] == "\n":
            cursor_hint += 1
    return spans


def _pages_overlapping_span(
    page_spans: List[Tuple[int, int, int]], c_start: int, c_end: int
) -> List[int]:
    out: List[int] = []
    for page_no, p_start, p_end in page_spans:
        if p_end <= c_start:
            continue
        if p_start >= c_end:
            break
        out.append(page_no)
    return out

#     processed_dir = Path(processed_dir)
#     chunked_base = Path(chunked_dir)
#     output_dir = chunked_base / "asterisk_separate_chunking_result"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     if max_chunk_chars <= 0:
#         raise ValueError("max_chunk_chars must be > 0")

#     # Build a regex for runs of the chosen separator character
#     sep_token = re.escape(separator_char)
#     sep_re = re.compile(fr"{sep_token}{{{int(min_repeats)},}}")

#     json_paths = sorted(processed_dir.glob(glob_pattern))
#     if verbose:
#         print(f"[INFO] (sep='{separator_char}') Found {len(json_paths)} JSON files in {processed_dir}")

#     for jp in json_paths:
#         # Load parsed metadata
#         try:
#             with open(jp, "r", encoding="utf-8") as f:
#                 meta = json.load(f)
#         except Exception as e:
#             if verbose:
#                 print(f"[SKIP] {jp.name} (cannot read JSON: {e})")
#             continue

#         if meta.get("error"):
#             if verbose:
#                 print(f"[SKIP] {jp.name} (error present: {meta['error']})")
#             continue

#         file_name = meta.get("file_name") or jp.name
#         full_text: Optional[str] = meta.get("full_text")
#         pages: Optional[List[Dict]] = meta.get("pages")

#         if not full_text:
#             if verbose:
#                 print(f"[SKIP] {jp.name} (empty full_text)")
#             continue
#         if not pages:
#             if verbose:
#                 print(f"[SKIP] {jp.name} (missing pages)")
#             continue

#         # Map pages to character spans in full_text
#         page_spans = _build_page_spans_assuming_double_newlines(pages, full_text)
#         if not page_spans or page_spans[-1][2] != len(full_text):
#             if verbose:
#                 print(f"[WARN] {jp.name} page-span mismatch; falling back to search spans")
#             page_spans = _build_page_spans_via_search(pages, full_text)
#             if not page_spans:
#                 if verbose:
#                     print(f"[SKIP] {jp.name} (could not map pages to full_text)")
#                 continue

#         # Identify separator runs and build primary segments between them.
#         segments: List[Tuple[int, int]] = []  # (seg_start, seg_end) in original full_text
#         last_end = 0
#         for m in sep_re.finditer(full_text):
#             start, end = m.span()
#             if start > last_end:
#                 segments.append((last_end, start))
#             last_end = end
#         if last_end < len(full_text):
#             segments.append((last_end, len(full_text)))

#         # Build text chunks (buffer first; we'll interleave with tables by page order)
#         buffered_chunks: List[Tuple[int, int, Dict]] = []
#         # Tuple: (first_page_for_sort, tie_breaker_int, record_dict)
#         # tie_breaker: 0 for text, 1 for table — so tables come after text for the same page.

#         chunk_idx = 0
#         for seg_start, seg_end in segments:
#             seg_len = seg_end - seg_start
#             if seg_len <= 0:
#                 continue
#             offset = 0
#             while offset < seg_len:
#                 c_start = seg_start + offset
#                 c_end = min(c_start + max_chunk_chars, seg_end)

#                 text = full_text[c_start:c_end]
#                 if separator_char in text:
#                     text = sep_re.sub("", text)

#                 pages_list = _pages_overlapping_span(page_spans, c_start, c_end)
#                 first_page = pages_list[0] if pages_list else 10**9  # very large if none (shouldn't happen)

#                 rec = {
#                     "doc_id": file_name,
#                     "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}",
#                     "char_start": c_start,
#                     "char_end": c_end,
#                     "pages": pages_list,
#                     "text": text,
#                 }
#                 buffered_chunks.append((first_page, 0, rec))
#                 chunk_idx += 1

#                 if c_end >= seg_end:
#                     break
#                 offset += max_chunk_chars

#         # Build table chunks
#         tables = meta.get("tables") or []
#         tables_added = 0
#         for t in tables:
#             page_no = t.get("page")
#             table_data = t.get("table")
#             if not isinstance(page_no, int) or not isinstance(table_data, list):
#                 continue

#             csv_text = _table_to_csv(table_data)

#             rec = {
#                 "doc_id": file_name,
#                 "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}::table",
#                 "char_start": None,
#                 "char_end": None,
#                 "pages": [page_no],
#                 "text": csv_text,
#             }
#             buffered_chunks.append((page_no, 1, rec))
#             chunk_idx += 1
#             tables_added += 1

#         # Sort by page order; text (0) before table (1) within same page
#         buffered_chunks.sort(key=lambda x: (x[0], x[1]))

#         # Emit all chunks in the chosen order
#         stem = Path(file_name).stem
#         out_file = output_dir / f"{stem}.chunks.jsonl"
#         with open(out_file, "w", encoding="utf-8") as out_f:
#             for _, __, rec in buffered_chunks:
#                 out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

#         if verbose:
#             print(f"[OK]  {jp.name} -> {len(buffered_chunks)} chunks ({tables_added} table-chunks) -> {out_file}")


# def _table_to_csv(table_2d: List[List[Optional[str]]]) -> str:
#     """
#     Convert a 2D list (rows) into an RFC 4180-compliant CSV string:
#     - Comma delimiter.
#     - Double-quote fields when needed, escaping inner quotes by doubling.
#     - None -> "".
#     - Preserve embedded newlines inside quoted cells.
#     """
#     buf = io.StringIO(newline="")
#     writer = csv.writer(buf, dialect="excel", quoting=csv.QUOTE_MINIMAL)
#     for row in table_2d:
#         safe_row = [("" if (cell is None) else str(cell)) for cell in row]
#         writer.writerow(safe_row)
#     return buf.getvalue()


# # --- helpers (unchanged) ---
# def _build_page_spans_assuming_double_newlines(
#     pages: List[Dict], full_text: str
# ) -> List[Tuple[int, int, int]]:
#     spans: List[Tuple[int, int, int]] = []
#     cursor = 0
#     total_len = len(full_text)

#     for i, p in enumerate(pages):
#         page_no = int(p.get("page", i + 1))
#         page_text = p.get("text", "") or ""
#         start = cursor
#         end = start + len(page_text)
#         if end > total_len:
#             return []
#         spans.append((page_no, start, end))
#         cursor = end
#         if i < len(pages) - 1:
#             if cursor + 1 < total_len and full_text[cursor:cursor + 2] == "\n\n":
#                 cursor += 2
#             else:
#                 return []
#     if cursor != total_len:
#         return []
#     return spans


# def _build_page_spans_via_search(
#     pages: List[Dict], full_text: str
# ) -> List[Tuple[int, int, int]]:
#     spans: List[Tuple[int, int, int]] = []
#     cursor_hint = 0
#     for i, p in enumerate(pages):
#         page_no = int(p.get("page", i + 1))
#         page_text = p.get("text", "") or ""
#         if not page_text:
#             spans.append((page_no, cursor_hint, cursor_hint))
#             while cursor_hint < len(full_text) and full_text[cursor_hint] == "\n":
#                 cursor_hint += 1
#             continue

#         pos = full_text.find(page_text, cursor_hint)
#         if pos == -1:
#             return []
#         start = pos
#         end = pos + len(page_text)
#         spans.append((page_no, start, end))
#         cursor_hint = end
#         while cursor_hint < len(full_text) and full_text[cursor_hint] == "\n":
#             cursor_hint += 1
#     return spans


# def _pages_overlapping_span(
#     page_spans: List[Tuple[int, int, int]], c_start: int, c_end: int
# ) -> List[int]:
#     out: List[int] = []
#     for page_no, p_start, p_end in page_spans:
#         if p_end <= c_start:
#             continue
#         if p_start >= c_end:
#             break
#         out.append(page_no)
#     return out
