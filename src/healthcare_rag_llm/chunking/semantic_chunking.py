from __future__ import annotations

from pathlib import Path
import json
import csv
import io
from typing import List, Dict, Optional, Tuple
import warnings

# Semantic chunking dependencies
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def semantic_chunking(
    processed_dir: str,
    chunked_dir: str,
    max_chunk_chars: int = 5000,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    unit: str = "sentence",
    similarity_threshold: float = 0.80,
    hysteresis: float = 0.02,
    glob_pattern: str = "*.json",
    verbose: bool = True,
) -> None:
    """
    Semantic-based chunking using sentence-transformers embeddings.

    Algorithm:
    1. Tokenize text into sentences or paragraphs (based on 'unit')
    2. Generate embeddings for each unit using sentence-transformers
    3. Calculate cosine similarity between adjacent units
    4. Merge units based on similarity threshold with hysteresis:
       - similarity > threshold + hysteresis/2: merge
       - similarity < threshold - hysteresis/2: split
       - in between: maintain current state (reduces jitter)
    5. Ensure chunks don't exceed max_chunk_chars

    Args:
        processed_dir: Directory containing parsed JSON files
        chunked_dir: Base directory for chunk output
        max_chunk_chars: Maximum characters per chunk (default: 5000)
        model_name: Sentence-transformers model name (default: all-MiniLM-L6-v2)
        unit: Segmentation unit - "sentence" or "paragraph" (default: sentence)
        similarity_threshold: Cosine similarity threshold for merging (default: 0.80)
        hysteresis: Similarity hysteresis to reduce jitter (default: 0.02)
        glob_pattern: Pattern to match input files (default: "*.json")
        verbose: Print progress information (default: True)

    Output:
        <chunked_dir>/semantic_chunking_result/<stem>.chunks.jsonl

    NEW:
    - OCR fallback merge for page-level text.
    - OCR image chunks from ocr_fallback.
    """
    # Validate parameters
    if unit not in ["sentence", "paragraph"]:
        raise ValueError(f"unit must be 'sentence' or 'paragraph', got '{unit}'")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
    if hysteresis < 0:
        raise ValueError(f"hysteresis must be non-negative, got {hysteresis}")

    processed_dir = Path(processed_dir)
    output_dir = Path(chunked_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sentence-transformers model (once for all documents)
    if verbose:
        print(f"[INFO] Loading sentence-transformers model: {model_name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress model loading warnings
        model = SentenceTransformer(model_name)

    json_paths = sorted(processed_dir.glob(glob_pattern))
    if verbose:
        print(f"[INFO] Found {len(json_paths)} JSON files in {processed_dir}")
        print(f"[INFO] Chunking parameters: unit={unit}, threshold={similarity_threshold}, hysteresis={hysteresis}")

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
        category: str = (meta.get("category") or "unknown")

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

        # --- Semantic chunking logic ---
        # Step 1: Tokenize full_text into units (sentences or paragraphs) with position tracking
        if unit == "sentence":
            # Use NLTK sentence tokenizer with span information
            units = []
            unit_spans = []
            for start, end in nltk.tokenize.punkt.PunktSentenceTokenizer().span_tokenize(full_text):
                units.append(full_text[start:end])
                unit_spans.append((start, end))
        else:  # paragraph
            # Split on double newlines and track positions
            units = []
            unit_spans = []
            cursor = 0
            for para in full_text.split("\n\n"):
                para_stripped = para.strip()
                if para_stripped:
                    # Find position in original text
                    start = full_text.find(para_stripped, cursor)
                    if start != -1:
                        end = start + len(para_stripped)
                        units.append(para_stripped)
                        unit_spans.append((start, end))
                        cursor = end

        # Handle edge case: empty or single unit
        if not units:
            chunks_with_info = []
        elif len(units) == 1:
            chunks_with_info = [{"text": units[0], "start": unit_spans[0][0], "end": unit_spans[0][1]}]
        else:
            # Step 2: Generate embeddings for all units
            if verbose:
                print(f"  Generating embeddings for {len(units)} {unit}s...")
            unit_embeddings = model.encode(units, show_progress_bar=False, convert_to_numpy=True)

            # Step 3: Apply semantic chunking with hysteresis
            chunks_with_info = []
            current_chunk_units = [units[0]]
            current_chunk_unit_indices = [0]
            current_chunk_embeddings = [unit_embeddings[0]]

            upper_threshold = similarity_threshold + hysteresis / 2
            lower_threshold = similarity_threshold - hysteresis / 2

            for i in range(1, len(units)):
                # Calculate similarity between current chunk average and next unit
                current_avg_embedding = np.mean(current_chunk_embeddings, axis=0).reshape(1, -1)
                next_embedding = unit_embeddings[i].reshape(1, -1)
                similarity = cosine_similarity(current_avg_embedding, next_embedding)[0][0]

                # Hysteresis-based decision
                should_merge = similarity > upper_threshold
                should_split = similarity < lower_threshold

                # Check if adding would exceed max_chunk_chars
                potential_chunk = " ".join(current_chunk_units + [units[i]])
                would_exceed_limit = len(potential_chunk) > max_chunk_chars

                if should_split or would_exceed_limit:
                    # Save current chunk with position info
                    chunk_start = unit_spans[current_chunk_unit_indices[0]][0]
                    chunk_end = unit_spans[current_chunk_unit_indices[-1]][1]
                    chunks_with_info.append({
                        "text": " ".join(current_chunk_units),
                        "start": chunk_start,
                        "end": chunk_end
                    })
                    # Start new chunk
                    current_chunk_units = [units[i]]
                    current_chunk_unit_indices = [i]
                    current_chunk_embeddings = [unit_embeddings[i]]
                else:
                    # Merge (either high similarity or in hysteresis zone)
                    current_chunk_units.append(units[i])
                    current_chunk_unit_indices.append(i)
                    current_chunk_embeddings.append(unit_embeddings[i])

            # Don't forget the last chunk
            if current_chunk_units:
                chunk_start = unit_spans[current_chunk_unit_indices[0]][0]
                chunk_end = unit_spans[current_chunk_unit_indices[-1]][1]
                chunks_with_info.append({
                    "text": " ".join(current_chunk_units),
                    "start": chunk_start,
                    "end": chunk_end
                })

        stem = Path(file_name).stem
        out_file = output_dir / f"{stem}.chunks.jsonl"

        # Total length for page estimation
        total_len = len(full_text)

        buffered_chunks: List[Dict] = []
        chunk_idx = 0
        for chunk_info in chunks_with_info:
            # Estimate pages for this chunk span
            pages_list = _estimate_pages_for_span(pages, chunk_info["start"], chunk_info["end"], total_len)

            rec = {
                "doc_id": file_name,
                "chunk_id": f"{file_name}::{str(chunk_idx).zfill(4)}",
                "char_start": chunk_info["start"],
                "char_end": chunk_info["end"],
                "pages": pages_list,
                "text": chunk_info["text"],
                "chunk_type": "text",
                "category": category,
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
                "chunk_type": "table",
                "category": category,
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
                    "bbox": o.get("bbox"),
                    "category": category,
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