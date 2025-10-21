# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a healthcare RAG (Retrieval-Augmented Generation) system that analyzes New York State Medicaid policy documents and provides citation-grounded answers to compliance queries. The system combines document parsing, semantic chunking, Neo4j graph database with vector search, reranking, and LLM-based response generation.

**Tech Stack:** Python 3.9+, Neo4j 5.17+, BGE-M3 embeddings, OpenAI/Gemini/Ollama LLMs, Streamlit

## Commands

### Setup & Installation

```bash
# Initial setup
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -e .

# Start Neo4j database (required for most operations)
cd docker
docker compose up -d

# Test database connection
python scripts/test_neo4j.py
```

### Core Pipeline Commands

Run these in sequence for the full pipeline:

```bash
# 1. Parse documents (PDF/DOCX → JSON)
python scripts/ingestion_parse.py --config configs/ingest_parse.yaml

# 2. Chunk documents (choose one strategy)
python scripts/do_asterisk_chunking.py      # Delimiter-based (recommended)
python scripts/do_fix_size_chunking.py      # Fixed-size chunks
python scripts/do_semantic_chunking.py      # Embedding-based chunking

# 3. Ingest chunks into Neo4j graph
python scripts/ingest_graph.py --chunk_dir data/chunks/asterisk_chunking_result

# 4. Run web interface
streamlit run frontend/app.py
```

### Testing & Development

```bash
# Test LLM query
python scripts/llm_query.py

# Evaluate retrieval accuracy
python scripts/evaluate.py --tested_result results.json --ground_truth gt.json --output eval.json

# Reset Neo4j graph (keep container)
python scripts/reset_graph.py

# Reset Neo4j completely (delete all data and container)
cd docker
docker compose down -v
```

### Database Access

- Neo4j Browser: http://localhost:7474
- Credentials: Load from `docker/.env`
- Streamlit App: http://localhost:8501

## Architecture

### Data Flow

```
Raw Documents (data/raw/)
    ↓
[doc_parsing] → Structured JSON (data/processed/)
    ↓
[chunking] → JSONL chunks (data/chunks/)
    ↓
[embedding + graph_builder] → Neo4j Graph Database
    ↓
[User Query] → [Embedding] → [Vector Search] → [Reranking] → [LLM] → Citation-based Answer
```

### Key Components

**Document Processing:**
- `src/healthcare_rag_llm/doc_parsing/doc_parsing.py` - PDF/DOCX parser with OCR support
- Entry point: `scripts/ingestion_parse.py`

**Chunking Strategies:**
- `src/healthcare_rag_llm/chunking/fix_size_chunking.py` - Fixed character chunks
- `src/healthcare_rag_llm/chunking/pattern_chunking.py` - Delimiter-based (asterisk separators)
- `src/healthcare_rag_llm/chunking/semantic_chunking.py` - Embedding-aware boundary detection

**Embedding System:**
- Model: BAAI/bge-m3 (1024-dim dense vectors)
- Class: `HealthcareEmbedding` in `src/healthcare_rag_llm/embedding/HealthcareEmbedding.py`
- Auto-detects GPU support via torch

**Graph Database:**
- Schema: Document → Page → Chunk (relationships: CONTAINS, HAS_CHUNK)
- Vector index: `chunk_vec` on Chunk.denseEmbedding (cosine similarity)
- Connection: `Neo4jConnector` in `src/healthcare_rag_llm/graph_builder/neo4j_loader.py`

**Retrieval & Reranking:**
- Vector search: `src/healthcare_rag_llm/graph_builder/queries.py::query_chunks()`
- Reranking: `src/healthcare_rag_llm/reranking/reranker.py` (BAAI/bge-reranker-base cross-encoder)
- Hybrid scoring: `alpha * rerank_score + (1-alpha) * dense_score` (default alpha=0.3)

**LLM Interface:**
- Multi-provider support: OpenAI, Gemini, Ollama
- Class: `LLMClient` in `src/healthcare_rag_llm/llm/llm_client.py`
- Config: `configs/api_config.yaml` (providers, models, max_tokens)

**Response Generation:**
- Orchestrator: `ResponseGenerator` in `src/healthcare_rag_llm/llm/response_generator.py`
- System prompt: Citation-focused format in `configs/system_prompt.txt`
- Output: Answer with exact quotes and document/page citations

### Neo4j Graph Schema

```cypher
# Nodes
(Document {doc_id, doc_type, effective_date, authority})
(Page {page_no, uid})
(Chunk {chunk_id, text, denseEmbedding, pages})

# Relationships
(Document)-[:CONTAINS]->(Page)-[:HAS_CHUNK]->(Chunk)

# Constraints
UNIQUE: doc_id, page_uid, chunk_id

# Indexes
VECTOR INDEX chunk_vec: Chunk.denseEmbedding (1024 dim, cosine)
```

## Configuration Files

**`configs/ingest_parse.yaml`** - Paths for raw/processed/chunked data
**`configs/api_config.yaml`** - LLM providers (OpenAI, Bltcy, Anthropic), models, tokens
**`configs/system_prompt.txt`** - RAG system prompt with citation rules
**`docker/.env`** - Neo4j credentials (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

Use `docker/.env.example` as template for `.env` file.

## Data Handling

- **NEVER commit data files**: `data/raw/` and `data/processed/` are git-ignored
- Raw documents: Place in `data/raw/Childrens Evolution of Care/State/Medicaid Updates/`
- Parsed outputs: Auto-generated in `data/processed/`
- Chunks: Auto-generated in `data/chunks/{method}_result/`
- Share data via Google Drive/SharePoint, not git

## Git Workflow

- Branch naming: `feature/<description>` or `bugfix/<description>`
- Always sync with main before pushing: `git merge main`
- Create Pull Requests into `main` branch (require 1+ reviewer)
- Delete branches after merge
- Current branch: `feature/Evaluation` (evaluation pipeline work in progress)

## System Prompt & Compliance Focus

The system enforces **NYS Medicaid policy compliance** with strict citation requirements:

1. Answer ONLY using provided context chunks
2. Quote exact lines with citations: `[<doc_id:page> — <date>]`
3. Prefer recent guidance when conflicted
4. Preserve exact dates, codes, dollar figures
5. Keep decisions actionable and concise
6. Flag missing evidence explicitly

This ensures responses are grounded, traceable, and suitable for regulated healthcare environments.

## Code Conventions

- **JSONL format** for chunks: One JSON object per line with `chunk_id`, `text`, `pages`, `metadata`
- **Chunk IDs**: `{filename}_{index}`
- **Page numbers**: 1-indexed
- **Vector embeddings**: 1024-dimensional float arrays
- **Error handling**: Log and skip malformed documents (don't halt pipeline)

## Platform-Specific Notes

**Windows:**
- Use backslash `\` in config file paths (e.g., `data\raw\...`)
- Document conversion: Requires MS Word COM interface (pywin32) or LibreOffice
- OCR: Install Tesseract separately

**macOS/Linux:**
- Use forward slash `/` in paths
- Document conversion: Requires LibreOffice (`brew install libreoffice` or `apt-get`)
- OCR: Install via package manager (`brew install tesseract` or `apt-get`)

## Recent Development

**Active work on `feature/Evaluation` branch:**
- Evaluation pipeline for retrieval accuracy (document-level, page-level metrics)
- Reranking module with hybrid scoring
- Alternative response generators (`response_gen2.py` - strict citation mode)
- API configuration management system

**Modified files** (not yet committed):
- `src/healthcare_rag_llm/llm/response_generator.py`
- New: `configs/api_config.yaml`
- New: `src/healthcare_rag_llm/utils/api_config.py`
- New: `scripts/evaluate/`
