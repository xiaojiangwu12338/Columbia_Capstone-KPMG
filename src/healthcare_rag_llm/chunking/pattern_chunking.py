from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Tuple

def asterisk_separate_chunking(
    processed_dir: str,
    chunked_dir: str,
    max_chunk_chars: int = 5000,
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

    Output:
      <chunked_dir>/asterisk_separate_chunking_result/<stem>.chunks.jsonl
    Record schema:
      {
        "doc_id": "<file_name>",
        "chunk_id": "<file_name>::0007",
        "char_start": <int>,   # inclusive, in original full_text
        "char_end": <int>,     # exclusive, in original full_text
        "pages": [<int>, ...],
        "text": "<substring with all separator runs removed>"
      }
    """
    processed_dir = Path(processed_dir)
    chunked_base = Path(chunked_dir)
    output_dir = chunked_base / "asterisk_separate_chunking_result"
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

        if not full_text:
            if verbose:
                print(f"[SKIP] {jp.name} (empty full_text)")
            continue
        if not pages:
            if verbose:
                print(f"[SKIP] {jp.name} (missing pages)")
            continue

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

        # Emit chunks: within each segment, further split by max_chunk_chars (no overlap).
        stem = Path(file_name).stem
        out_file = output_dir / f"{stem}.chunks.jsonl"
        chunks_written = 0
        with open(out_file, "w", encoding="utf-8") as out_f:
            chunk_idx = 0
            for seg_start, seg_end in segments:
                seg_len = seg_end - seg_start
                if seg_len <= 0:
                    continue
                offset = 0
                while offset < seg_len:
                    c_start = seg_start + offset
                    c_end = min(c_start + max_chunk_chars, seg_end)

                    # Extract text for the chunk; remove any residual separator runs
                    text = full_text[c_start:c_end]
                    if separator_char in text:
                        text = sep_re.sub("", text)

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
                    chunk_idx += 1

                    if c_end >= seg_end:
                        break
                    offset += max_chunk_chars

        if verbose:
            print(f"[OK]  {jp.name} -> {chunks_written} chunks -> {out_file}")


# --- helpers (reuse the ones already present in your module) ---
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
