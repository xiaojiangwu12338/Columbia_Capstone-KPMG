# Chunking Module

This module provides multiple ways to split parsed documents into smaller chunks
for embeddings, retrieval, or LLM input.  
All methods read parsed JSON from `data/processed/` and write results under
the chunked directory defined in `configs/ingest_parse.yaml`.

---

## Methods

### 1. Fixed-Size (`fix_size_chunking.py`)
- Splits text into fixed-size character chunks.
- Tracks page numbers spanned.
- **Params:** `chunk_size`, `overlap` (optional).
- **Output:** `data/chunks/fix_size_chunking_result`.

### 2. Asterisk-Separated (`asterisk_chunking.py`)
- Splits when a delimiter (default: ≥10 `*`) is found or when max size is hit.
- Delimiter is removed in output.
- **Params:** `delimiter_pattern`, `min_repeat`, `max_chunk_size`.
- **Output:** `data/chunks/asterisk_chunking_result`.

### 3. Semantic (`semantic_chunking.py`)
- Uses embeddings to split at low-similarity points (sentence/paragraph level).
- Default model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- **Params:** `model_name`, `unit`, `similarity_threshold`, `max_chunk_size`.
- **Output:** `data/chunks/semantic_chunking_result`.

---

## Input / Output

- **Input:** JSON files with `full_text` and `pages` from `data/processed/`.
- **Output:** JSONL per method, one file per input doc.  
  Example entry:

```json
{
  "chunk_id": "file_0",
  "text": "chunk text ...",
  "pages": [1,2],
  "metadata": {
    "source_file": "data/processed/example.json",
    "chunk_index": 0,
    "similarity_score": 0.78
  }
}
Configuration
Paths are set in configs/ingest_parse.yaml:

yaml
Copy code
paths:
  raw: data/raw/...
  processed: data/processed
  chunked: data/chunks
Notes
Errors in a file (e.g. missing full_text) are logged and skipped.

Semantic chunking requires:

sentence-transformers (embeddings)

nltk with resources punkt and punkt_tab (sentence tokenization)