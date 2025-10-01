from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict, Optional


def fix_size_chunking(
    processed_dir: str,
    chunked_dir: str,
    chunk_size: int = 1200,
    overlap: int = 150,
    glob_pattern: str = "*.json",
    verbose: bool = True,
) -> None:
    """
    Chunk each parsed document JSON into fixed-size character windows.
    Results will be written to:
      <chunked_dir>/fix_size_chunking_result/<stem>.chunks.jsonl
    """
    processed_dir = Path(processed_dir)
    chunked_base = Path(chunked_dir)
    output_dir = chunked_base / "fix_size_chunking_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size}).")

    json_paths = sorted(processed_dir.glob(glob_pattern))
    if verbose:
        print(f"[INFO] Found {len(json_paths)} JSON files in {processed_dir}")

    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            if verbose:
                print(f"[SKIP] {jp.name} (cannot read JSON: {e})")
            continue

        if "error" in meta and meta["error"]:
            if verbose:
                print(f"[SKIP] {jp.name} (error present: {meta['error']})")
            continue

        file_name = meta.get("file_name") or jp.name
        full_text: Optional[str] = meta.get("full_text")
        pages: Optional[List[Dict]] = meta.get("pages")

        if not full_text or len(full_text) == 0:
            if verbose:
                print(f"[SKIP] {jp.name} (empty full_text)")
            continue
        if not pages:
            if verbose:
                print(f"[SKIP] {jp.name} (missing pages)")
            continue

        # page spans
        page_spans = _build_page_spans_assuming_double_newlines(pages, full_text)
        if not page_spans or page_spans[-1][2] != len(full_text):
            if verbose:
                print(f"[WARN] {jp.name} mismatch, falling back to search spans")
            page_spans = _build_page_spans_via_search(pages, full_text)
            if not page_spans:
                if verbose:
                    print(f"[SKIP] {jp.name} (could not map pages)")
                continue

        # chunk
        step = chunk_size - overlap
        n = len(full_text)
        stem = Path(file_name).stem
        out_file = output_dir / f"{stem}.chunks.jsonl"
        chunks_written = 0

        with open(out_file, "w", encoding="utf-8") as out_f:
            idx = 0
            chunk_idx = 0
            while idx < n:
                c_start = idx
                c_end = min(idx + chunk_size, n)
                text = full_text[c_start:c_end]
                pages_list = _pages_overlapping_span(page_spans, c_start, c_end)
                rec = {
                    "doc_id": file_name,
                    "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}",
                    "char_start": c_start,
                    "char_end": c_end,
                    "pages": pages_list,
                    "text": text,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chunks_written += 1
                if c_end == n:
                    break
                idx += step
                chunk_idx += 1

        if verbose:
            print(f"[OK] {jp.name} -> {chunks_written} chunks -> {out_file}")


# helpers remain the same
def _build_page_spans_assuming_double_newlines(pages, full_text):
    spans = []
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


def _build_page_spans_via_search(pages, full_text):
    spans = []
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


def _pages_overlapping_span(page_spans, c_start, c_end):
    out = []
    for page_no, p_start, p_end in page_spans:
        if p_end <= c_start:
            continue
        if p_start >= c_end:
            break
        out.append(page_no)
    return out
