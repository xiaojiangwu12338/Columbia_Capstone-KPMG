# src/healthcare_rag_llm/rerank/reranker.py
from __future__ import annotations

"""
Cross-encoder reranking utilities for Healthcare RAG.

- Works with retrieval hits produced by your dense search (FAISS/Neo4j).
- Adds 'rerank_score', optional 'final_score' (if blending), and 'rank' to hits.
- Sorts the hits list in-place and returns it (same object).

Expected hit schema (minimum):
    {
        "text": "...",            # passage text (configurable via text_key)
        "score": 0.123,           # optional dense score for blending (dense_score_key)
        # other fields (preserved): "source"/"filename"/"doc_id", "page", "doc_date", ...
    }

Typical usage:
    from healthcare_rag_llm.rerank.reranker import apply_rerank_to_chunks
    chunks = dense_retrieve(query, top_k=50)
    apply_rerank_to_chunks(query, chunks, combine_with_dense=False)
    return chunks[:top_k]
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder


# -----------------------
# Defaults (env-overridable)
# -----------------------
DEFAULT_RERANKER = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
DEFAULT_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "64"))


@dataclass
class RerankConfig:
    """
    Configuration for the cross-encoder reranker.
    """
    model_name: str = DEFAULT_RERANKER
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    device: Optional[str] = None            
    text_key: str = "text"                 
    dense_score_key: str = "score"        
    combine_with_dense: bool = False       
    alpha: float = 0.5                      
    max_pairs: Optional[int] = None      


class Reranker:
    """
    Thin wrapper around sentence-transformers CrossEncoder for (query, passage) scoring.

    Contract with hits:
      - Each hit must have a text under config.text_key.
      - If combine_with_dense=True, each hit should also have a dense score under config.dense_score_key.
      - This class mutates hits to attach: 'rerank_score', optionally 'final_score', and 'rank'.
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=device,
        )

    @torch.inference_mode()
    def score_pairs(self, query: str, passages: List[str]) -> List[float]:
        """
        Score (query, passage_i) pairs with the cross-encoder.

        Returns list of floats aligned with 'passages'.
        If config.max_pairs is set and shorter than len(passages), only the
        first max_pairs are scored and the tail receives very low scores.
        """
        scores: List[float] = []
        idxs = (
            range(len(passages))
            if self.config.max_pairs is None
            else range(min(len(passages), self.config.max_pairs))
        )

        batch: List[Tuple[str, str]] = []
        for i in idxs:
            batch.append((query, passages[i]))
            if len(batch) == self.config.batch_size:
                out = self._model.predict(batch)
                scores.extend([float(x) for x in (out.tolist() if hasattr(out, "tolist") else out)])
                batch = []
        if batch:
            out = self._model.predict(batch)
            scores.extend([float(x) for x in (out.tolist() if hasattr(out, "tolist") else out)])

        # If we truncated for latency, pad remaining with very low scores
        if self.config.max_pairs is not None and len(passages) > self.config.max_pairs:
            scores.extend([-1e9] * (len(passages) - self.config.max_pairs))

        return scores

    def rerank_hits(self, query: str, hits: List[Dict]) -> List[Dict]:
        """
        Mutates + returns hits ordered by:
          - rerank_score only, OR
          - blended final_score = z(alpha * dense + (1-alpha) * rerank).

        Adds:
          - h['rerank_score'] (float)
          - h['final_score'] (float) if blending
          - h['rank'] (int starting at 1)
        """
        if not hits:
            return hits

        # Collect texts and mask invalid
        texts: List[str] = []
        valid_mask: List[bool] = []
        for h in hits:
            t = (h.get(self.config.text_key) or "").strip()
            texts.append(t)
            valid_mask.append(bool(t))

        if not any(valid_mask):
            # Nothing to score; keep order but annotate
            for i, h in enumerate(hits, 1):
                h.setdefault("rerank_score", float("-inf"))
                h["rank"] = i
            return hits

        # Score only valid passages, then scatter back
        valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
        valid_texts = [texts[i] for i in valid_indices]
        scores_valid = self.score_pairs(query, valid_texts)

        for idx, s in zip(valid_indices, scores_valid):
            hits[idx]["rerank_score"] = float(s)
        # Mark invalid texts with -inf to sink them
        for i, ok in enumerate(valid_mask):
            if not ok:
                hits[i]["rerank_score"] = float("-inf")

        # Optional blending with dense scores
        if self.config.combine_with_dense:
            ds = np.array(
                [float(h.get(self.config.dense_score_key, 0.0)) for h in hits],
                dtype=np.float32,
            )
            rs = np.array(
                [float(h.get("rerank_score", 0.0)) for h in hits],
                dtype=np.float32,
            )

            def z(x: np.ndarray) -> np.ndarray:
                if x.size == 0:
                    return x
                mu = float(x.mean())
                sd = float(x.std())
                if sd < 1e-6:
                    return np.zeros_like(x)
                return (x - mu) / sd

            final = self.config.alpha * z(ds) + (1.0 - self.config.alpha) * z(rs)
            for h, f in zip(hits, final):
                h["final_score"] = float(f)
            hits.sort(key=lambda x: x.get("final_score", float("-inf")), reverse=True)
        else:
            hits.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)

        # Reassign sequential ranks
        for i, h in enumerate(hits, 1):
            h["rank"] = i

        return hits

def apply_rerank_to_chunks(
    query: str,
    chunks: List[Dict],
    *,
    text_key: str = "text",
    dense_score_key: str = "score",
    combine_with_dense: bool = False,
    alpha: float = 0.5,
    model_name: Optional[str] = None,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> List[Dict]:
    """
    Drop-in helper to rerank your retrieved chunks.

    Parameters
    ----------
    query : str
        The user query.
    chunks : List[Dict]
        Retrieval results. Each must have passage text under `text_key`.
        If combine_with_dense=True, each should also have a dense score under `dense_score_key`.
    text_key : str
        Key containing passage text (default "text").
    dense_score_key : str
        Key containing dense ANN score for blending (default "score").
    combine_with_dense : bool
        If True, z-score blend dense and rerank scores; else use rerank only.
    alpha : float
        Dense-score weight in blending (0..1). Only used if combine_with_dense=True.
    model_name, max_length, batch_size, device, max_pairs :
        Optional overrides for configuration.

    Returns
    -------
    List[Dict]
        The same list object, sorted in-place, with scores & ranks attached.
    """
    cfg = RerankConfig(
        model_name=model_name or DEFAULT_RERANKER,
        max_length=max_length or DEFAULT_MAX_LENGTH,
        batch_size=batch_size or DEFAULT_BATCH_SIZE,
        device=device,
        text_key=text_key,
        dense_score_key=dense_score_key,
        combine_with_dense=combine_with_dense,
        alpha=alpha,
        max_pairs=max_pairs,
    )
    rr = Reranker(cfg)
    return rr.rerank_hits(query, chunks)


__all__ = ["RerankConfig", "Reranker", "apply_rerank_to_chunks"]


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Rerank chunks from a JSON/JSONL file.")
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument("--file", required=True, help="Path to JSON or JSONL with a list of hits.")
    parser.add_argument("--blend", action="store_true", help="Blend dense + rerank via z-scores.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dense weight for blending.")
    parser.add_argument("--topk", type=int, default=10, help="Show top-k after rerank.")
    args = parser.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    # Load hits: either a JSON array or JSONL lines of dicts
    hits: List[Dict] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    hits.append(json.loads(line))
    else:
        hits = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(hits, list):
            raise SystemExit("JSON must contain a list of hit objects.")

    apply_rerank_to_chunks(
        args.query,
        hits,
        combine_with_dense=args.blend,
        alpha=args.alpha,
    )

    for i, h in enumerate(hits[: args.topk], 1):
        text_preview = h.get('text', '')[:100].replace('\n', ' ')
        print(f"{i:>2}. rr={h.get('rerank_score'):.4f}"
              + (f", final={h.get('final_score'):.4f}" if "final_score" in h else "")
              + f"  |  {text_preview}")