# src/healthcare_rag_llm/chunking/semantic_chunking.py
from __future__ import annotations

from pathlib import Path
import json
import csv
import io
from typing import List, Dict, Optional, Tuple

# NLTK for sentence splitting
import nltk
from nltk.tokenize import sent_tokenize

# Local embeddings
from sentence_transformers import SentenceTransformer
import numpy as np


def semantic_chunking(
    processed_dir: str,
    chunked_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    unit: str = "sentence",                  # "sentence" or "paragraph"
    similarity_threshold: float = 0.80,      # merge if sim >= threshold (with small hysteresis)
    max_chunk_chars: int = 2000,             # hard limit; split by boundary if exceeded
    glob_pattern: str = "*.json",
    hysteresis: float = 0.02,                # allow sim >= (threshold - hysteresis) to merge
    verbose: bool = True,
) -> None:
    """
    Embedding-based semantic chunking with table-as-chunk support.

    Steps per document:
      - Segment into units ("sentence" via NLTK or "paragraph" via blank lines).
      - Embed each unit with a local SentenceTransformer.
      - Greedily merge adjacent units if similarity to the current chunk centroid >= (similarity_threshold - hysteresis).
      - Enforce max_chunk_chars by splitting at unit boundaries (no overlap).
      - Map chunk char spans to page numbers.

    NEW:
      - If input JSON includes "tables": [...], each table becomes its own chunk:
          * pages: [<table.page>]
          * char_start/char_end: null
          * chunk_id: "<file_name>::0007::table"
          * text: CSV-formatted table (RFC 4180; comma; null→empty; embedded newlines preserved)
      - Output is interleaved in page order (text chunks before table chunks on the same page).

    Output:
      <chunked_dir>/semantic_chunking_result/<stem>.chunks.jsonl
      Each text-chunk line schema:
        {
          "doc_id": "<file_name>",
          "chunk_id": "<file_name>::0007",
          "char_start": <int>,
          "char_end": <int>,
          "pages": [<int>, ...],
          "text": "<chunk text>",
          "sim_merge_scores": [<float>, ...],
          "sim_cohesion": <float>
        }
      Each table-chunk line schema (differences noted):
        {
          "doc_id": "<file_name>",
          "chunk_id": "<file_name>::0008::table",
          "char_start": null,
          "char_end": null,
          "pages": [<int>],
          "text": "<csv>",
          "sim_merge_scores": null,
          "sim_cohesion": null
        }
    """
    processed_dir = Path(processed_dir)
    chunked_base = Path(chunked_dir)
    output_dir = chunked_base / "semantic_chunking_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be > 0")
    if unit not in {"sentence", "paragraph"}:
        raise ValueError("unit must be 'sentence' or 'paragraph'")
    if not (0 < similarity_threshold <= 1):
        raise ValueError("similarity_threshold must be in (0, 1].")

    # Ensure NLTK punkt is available (for sentence tokenization)
    _ensure_nltk_punkt()

    # Load local embedding model
    if verbose:
        print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    json_paths = sorted(processed_dir.glob(glob_pattern))
    if verbose:
        print(f"[INFO] (semantic) Found {len(json_paths)} JSON files in {processed_dir}")

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

        # Map pages to character spans
        page_spans = _build_page_spans_assuming_double_newlines(pages, full_text)
        if not page_spans or page_spans[-1][2] != len(full_text):
            if verbose:
                print(f"[WARN] {jp.name} page-span mismatch; falling back to search spans")
            page_spans = _build_page_spans_via_search(pages, full_text)
            if not page_spans:
                if verbose:
                    print(f"[SKIP] {jp.name} (could not map pages to full_text)")
                continue

        # Segment into units (with character spans)
        units, unit_spans = _segment_units(full_text, mode=unit)
        if not units:
            if verbose:
                print(f"[SKIP] {jp.name} (no {unit}s found)")
            continue

        # Embed units
        embeddings = _embed_units(model, units)

        # Merge units semantically
        chunks_by_units, _ = _merge_units_semantically(
            unit_spans, embeddings, similarity_threshold, hysteresis
        )

        # Enforce max_chunk_chars by splitting large chunks at unit boundaries
        finalized_chunks = _enforce_char_limit(chunks_by_units, max_chunk_chars)

        # Buffer text chunks for interleaving with table chunks by page order
        buffered_chunks: List[Tuple[int, int, Dict]] = []  # (first_page, type_flag, record)
        # type_flag: 0=text, 1=table
        chunk_idx = 0

        for chunk_units in finalized_chunks:
            c_start = chunk_units[0][0]
            c_end = chunk_units[-1][1]
            text = full_text[c_start:c_end]
            pages_list = _pages_overlapping_span(page_spans, c_start, c_end)
            first_page = pages_list[0] if pages_list else 10**9

            # Compute sim metrics for this chunk
            sim_merge_scores, sim_cohesion = _chunk_similarity_metrics(
                chunk_units, unit_spans, embeddings
            )

            rec = {
                "doc_id": file_name,
                "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}",
                "char_start": c_start,
                "char_end": c_end,
                "pages": pages_list,
                "text": text,
                "sim_merge_scores": sim_merge_scores,
                "sim_cohesion": sim_cohesion,
            }
            buffered_chunks.append((first_page, 0, rec))
            chunk_idx += 1

        # Convert tables (if any) to CSV chunks and add to buffer
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
                "sim_merge_scores": None,
                "sim_cohesion": None,
            }
            buffered_chunks.append((page_no, 1, rec))
            chunk_idx += 1
            tables_added += 1

        # Interleave by page; text before table on the same page
        buffered_chunks.sort(key=lambda x: (x[0], x[1]))

        # Write out JSONL
        stem = Path(file_name).stem
        out_file = output_dir / f"{stem}.chunks.jsonl"
        with open(out_file, "w", encoding="utf-8") as out_f:
            for _, __, rec in buffered_chunks:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if verbose:
            print(f"[OK]  {jp.name} -> {len(buffered_chunks)} chunks ({tables_added} table-chunks) -> {out_file}")


# ------------------------- helpers -------------------------

def _ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def _segment_units(full_text: str, mode: str) -> "Tuple[List[str], List[Tuple[int, int]]]":
    """
    Return (units, unit_spans)
      units: list of strings
      unit_spans: list of (start_char, end_char) for each unit in full_text

    mode:
      - "sentence": NLTK sentence tokenizer
      - "paragraph": split on blank-line boundaries (>= 2 consecutive newlines)
    """
    units: List[str] = []
    spans: List[Tuple[int, int]] = []

    if mode == "paragraph":
        # Split on two or more consecutive newlines; track spans
        n = len(full_text)
        start = 0
        i = 0
        while i < n:
            if full_text[i] == "\n":
                j = i
                while j < n and full_text[j] == "\n":
                    j += 1
                if (j - i) >= 2:
                    if start < i:
                        unit = full_text[start:i]
                        units.append(unit)
                        spans.append((start, i))
                    start = j
                    i = j
                    continue
            i += 1
        if start < n:
            units.append(full_text[start:n])
            spans.append((start, n))
    else:  # "sentence"
        sents = sent_tokenize(full_text)
        cursor = 0
        for s in sents:
            pos = full_text.find(s, cursor)
            if pos == -1:
                continue
            start = pos
            end = pos + len(s)
            units.append(s)
            spans.append((start, end))
            cursor = end

    return units, spans


def _embed_units(model: SentenceTransformer, units: List[str]) -> np.ndarray:
    vecs = model.encode(units, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def _merge_units_semantically(
    unit_spans: "List[Tuple[int, int]]",
    embeddings: np.ndarray,
    threshold: float,
    hysteresis: float,
) -> "Tuple[List[List[Tuple[int, int]]], List[float]]":
    """
    Greedy merge: build chunks as lists of unit spans.
    Returns (chunks_by_units, merge_scores_flat) where merge_scores_flat is the list of sims that justified merges.
    """
    if len(unit_spans) == 0:
        return [], []

    chunks: List[List[Tuple[int, int]]] = []
    merge_scores: List[float] = []

    current_units = [unit_spans[0]]
    current_vecs = [embeddings[0]]
    current_centroid = embeddings[0].copy()

    for i in range(1, len(unit_spans)):
        v = embeddings[i]
        centroid_norm = current_centroid / (np.linalg.norm(current_centroid) + 1e-12)
        sim = float(np.clip(np.dot(centroid_norm, v), -1.0, 1.0))

        if sim >= (threshold - hysteresis):
            current_units.append(unit_spans[i])
            current_vecs.append(v)
            current_centroid = np.mean(np.vstack(current_vecs), axis=0)
            merge_scores.append(sim)
        else:
            chunks.append(current_units)
            current_units = [unit_spans[i]]
            current_vecs = [v]
            current_centroid = v.copy()

    if current_units:
        chunks.append(current_units)

    return chunks, merge_scores


def _enforce_char_limit(
    chunks_by_units: "List[List[Tuple[int, int]]]",
    max_chunk_chars: int,
) -> "List[List[Tuple[int, int]]]":
    """
    For any chunk whose char span exceeds max_chunk_chars, split it into sub-chunks
    at unit boundaries (no overlap).
    """
    finalized: List[List[Tuple[int, int]]] = []
    for chunk_units in chunks_by_units:
        if (chunk_units[-1][1] - chunk_units[0][0]) <= max_chunk_chars:
            finalized.append(chunk_units)
            continue

        current: List[Tuple[int, int]] = []
        current_start = None
        for span in chunk_units:
            if not current:
                current = [span]
                current_start = span[0]
            else:
                if (span[1] - current_start) <= max_chunk_chars:
                    current.append(span)
                else:
                    finalized.append(current)
                    current = [span]
                    current_start = span[0]

        if current:
            finalized.append(current)

    return finalized


def _chunk_similarity_metrics(
    chunk_units: "List[Tuple[int, int]]",
    unit_spans: "List[Tuple[int, int]]",
    embeddings: np.ndarray,
) -> "Tuple[List[float], float]":
    """
    Compute:
      - sim_merge_scores: cosine sims for adjacent units that were merged inside this chunk
      - sim_cohesion: average cosine similarity of all unit vectors to the chunk centroid
    """
    span_to_idx = {span: i for i, span in enumerate(unit_spans)}
    idxs = [span_to_idx[s] for s in chunk_units]
    vecs = embeddings[idxs]

    sim_merge_scores: List[float] = []
    for a, b in zip(vecs[:-1], vecs[1:]):
        sim_merge_scores.append(float(np.clip(np.dot(a, b), -1.0, 1.0)))

    centroid = np.mean(vecs, axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = np.dot(vecs, centroid_norm)
    sim_cohesion = float(np.clip(np.mean(sims), -1.0, 1.0))

    return sim_merge_scores, sim_cohesion


def _table_to_csv(table_2d: "List[List[Optional[str]]]" ) -> str:
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
