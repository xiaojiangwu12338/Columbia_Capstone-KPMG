# src/healthcare_rag_llm/llm/strict_response.py
# Strict, citation-first RAG answerer that plugs into LLMClient + query_chunks.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from textwrap import dedent, shorten
from typing import Dict, List, Tuple
import re

from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.graph_builder.queries import query_chunks

try:
    # Optional pretty table for local debugging; module is already used in your codebase.
    from tabulate import tabulate  # type: ignore
except Exception:  # pragma: no cover
    tabulate = None  # type: ignore

def _utc_year() -> int:
    return datetime.now(timezone.utc).year


def _label(c: Dict) -> str:
    """Stable label used in inline citations."""
    name = c.get("source") or c.get("filename") or c.get("doc_id") or "doc"
    page = c.get("page")
    return f"{name}" + (f":{page}" if page is not None else "")


def _date(c: Dict) -> str:
    """Prefer explicit doc date if present; fall back to any heuristic date."""
    return c.get("doc_date") or c.get("date_guess") or ""


def _token_estimate(text: str) -> int:
    """
    Cheap token estimate (~4 chars/token). Good enough for budgeting when
    model/tokenizer objects are not available here.
    """
    return max(1, int(len(text) / 4))


def _budgeted_context(
    chunks: List[Dict],
    reserve_for_answer: int = 700,
    chunk_char_cap: int = 1200,
    max_ctx: int = 8192,
) -> Tuple[List[str], List[int], int, int]:
    """
    Assemble context blocks without exceeding the approximate model context.

    Returns:
        blocks:   printable context blocks with headers
        used_ix:  indices from the input chunks that were used
        used_tok: approx tokens used for context
        budget:   approx token budget for all context
    """
    budget = int(0.80 * max_ctx) - reserve_for_answer
    if budget < 0:
        budget = int(0.50 * max_ctx)

    blocks, used_ix = [], []
    used_tok = 0

    for i, c in enumerate(chunks):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        if len(text) > chunk_char_cap:
            text = text[:chunk_char_cap] + " …"

        block = f"[{_label(c)} — {_date(c)}]\n{text}"
        need = _token_estimate(block) + 6
        if used_tok + need > budget:
            continue

        blocks.append(block)
        used_ix.append(i)
        used_tok += need

    return blocks, used_ix, used_tok, budget

# System & user prompts

SYSTEM_PROMPT = dedent(f"""
You are a **Medicaid Policy Assistant**. You answer provider-facing questions using
**only** the supplied context chunks from official sources (e.g., "Medicaid Update"
bulletins, provider manuals, fee schedules). Your mission is to produce **correct,
decision-ready** answers with precise citations. Do **not** use outside knowledge.

Authoritative rules (follow all):
1) Use only the provided context. If the answer is not fully supported, write:
   "Insufficient grounded evidence in the provided documents to answer."
   Then state what is missing (page, policy section, or document type).
2) Verbatim Evidence: Every factual statement in **Answer** must be supported
   by quoted lines in **Evidence (quoted)**. Quote the *exact* line(s) and place a
   citation immediately after each quote in the format:
   [<doc or doc:page> — <Mon DD, YYYY>].
3) Effective dates first: If timing matters, identify and prioritize lines
   that begin with "Effective" or similar. Keep dates exactly as written.
4) Numbers, codes, and names: Preserve all dates, amounts, rate codes,
   NCPDP fields, policy names, and provider types exactly as written.
5) Conflicts: If two sources conflict, show both in Evidence, prefer the
   most recent item (by date) in **What this means**, and explicitly note the conflict.
6) Scope: Do not assume parity between Managed Care and Fee-For-Service.
   Only state what the text says for each; if silent, say it's not specified.
7) Output sections (exact labels and order, **each exactly once**):
   - Answer
   - Evidence (quoted)
   - Caveats (if any)
   - What this means
8) Citations: Every bullet in **Answer** and every quoted line in **Evidence**
   ends with a citation like [doc or doc:page — Mon DD, YYYY].
9) Tone & brevity: Be concise and decision-ready; prefer bullets.

Current year: {_utc_year()}.
""").strip()


USER_PROMPT_TEMPLATE = dedent("""
You are acting strictly as a Medicaid **Policy Assistant**.

Follow this workflow step-by-step:

**Step 1 — Read the Question (provider asks):**
{question}

**Step 2 — Consider ONLY these Context Chunks (authoritative; you may cite only these):**
{context_blocks}

**Step 3 — Plan your answer:**
- Identify exact lines that answer the question; prefer lines beginning with "Effective".
- Extract any dates, rate codes, NCPDP fields, or policy names verbatim.
- If multiple docs are relevant, compare their dates and note any conflict.

**Step 4 — Write your output with these EXACT sections and rules:**
- Answer
  • Short bullets with the concrete rule(s), date(s), and scope (FFS vs Managed Care if applicable).
  • Each bullet must be supported by quotes in **Evidence (quoted)** and end with a citation.
- Evidence (quoted)
  • Quote the exact line(s) used. Do not paraphrase.
  • Each quote ends with a citation in the format [doc or doc:page — Mon DD, YYYY].
- Caveats (if any)
  • List missing pages, ambiguous scope, or document conflicts. If none, write "None."
- What this means
  • 1–2 provider-friendly bullet(s) summarizing how to act. No new facts.

**Citation formatting rule (mandatory):**
The label and date for a citation come from each context block header:
`[<LABEL> — <DATE>]` where the header is printed as `[LABEL — DATE]`.

Important prohibitions:
- Do not invent guidance.
- Do not generalize across programs (FFS vs Managed Care) unless explicitly stated.
- Do not alter codes, dollar amounts, dates, or field names.

Now produce the 4 sections.
""").strip()

@dataclass
class StrictResponse:
    answer: str
    used_chunks: List[Dict]
    approx_context_tokens: int
    approx_context_budget: int


class StrictResponseGenerator:
    """
    Drop-in, stricter variant of ResponseGenerator that:
      - builds a deterministic, sectioned prompt,
      - enforces inline citations format,
      - budgets context size,
      - uses the project's LLMClient for provider-agnostic calls.
    """

    def __init__(self, llm_client: LLMClient, max_ctx: int = 8192):
        self.llm_client = llm_client
        self.max_ctx = max_ctx

    def answer_question(
        self,
        question: str,
        top_k: int = 12,
        show_table: bool = False,
    ) -> StrictResponse:
        # 1) Retrieve candidate chunks (project's retriever)
        #    NOTE: query_chunks returns a list[dict] with keys: text, source/filename, page, doc_date etc.
        chunks = query_chunks(question, top_k)

        # 2) Build budgeted context blocks
        blocks, used_ix, used_tok, budget = _budgeted_context(
            chunks, max_ctx=self.max_ctx
        )
        if not blocks:
            return StrictResponse(
                answer="Insufficient grounded evidence in the provided documents to answer.",
                used_chunks=[],
                approx_context_tokens=0,
                approx_context_budget=budget,
            )

        # 3) Compose the user message
        user_msg = USER_PROMPT_TEMPLATE.format(
            question=question,
            context_blocks=("\n\n---\n\n").join(blocks),
        )

        # 4) Call the LLM (provider-agnostic)
        ans = self.llm_client.chat(user_prompt=user_msg, system_prompt=SYSTEM_PROMPT)

        # 5) Optional: local debug info
        if show_table and tabulate is not None:  # pragma: no cover
            rows = []
            for r, i in enumerate(used_ix, 1):
                c = chunks[i]
                rows.append(
                    [r, _label(c), _date(c), shorten((c.get("text") or "").replace("\n", " "), 120)]
                )
            print("=" * 110)
            print("QUERY:", question)
            print(f"(context tokens ~ {used_tok}/{budget}, window ~= {self.max_ctx})\n")
            print(ans, "\n")
            print("--- sources ---")
            print(tabulate(rows, headers=["rank", "doc", "date", "preview"]))

        # 6) Light guardrails: check sections + citation pattern and append gentle nudge if missing
        required = ["Answer", "Evidence (quoted)", "Caveats (if any)", "What this means"]
        sections_ok = all(h in ans for h in required)

        # Example citation: [Jan25_pr:7 — Jan 25, 2025]
        citation_ok = bool(re.search(r"\[[^\]\n]+—\s*[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\]", ans))

        if not sections_ok or not citation_ok:
            ans += (
                "\n\n(Note: One or more required sections/citations appear to be missing. "
                "Consider increasing top_k, refining retrieval, or narrowing the question.)"
            )

        return StrictResponse(
            answer=ans,
            used_chunks=[chunks[i] for i in used_ix],
            approx_context_tokens=used_tok,
            approx_context_budget=budget,
        )