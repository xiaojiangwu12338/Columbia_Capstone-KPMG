# src/healthcare_rag_llm/filters/llm_filter_extractor.py

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.utils.api_config import APIConfigManager

# -------------------------------------------------------------------
# LLM prompts for date (publication date) extraction
# -------------------------------------------------------------------
DATE_SYSTEM_PROMPT = (
    "You extract DATE BOUNDS for filtering policy documents by their PUBLICATION DATE.\n"
    "\n"
    "Your job is to infer the earliest and latest publication dates that should be\n"
    "used as bounds when retrieving relevant policy documents for a user's question.\n"
    "\n"
    "Return a JSON object with exactly these fields:\n"
    " - \"min_publish_date\": earliest publication date (YYYY-MM-DD) or null\n"
    " - \"max_publish_date\": latest publication date (YYYY-MM-DD) or null\n"
    "\n"
    "Rules:\n"
    " - Dates must be in ISO format: YYYY-MM-DD.\n"
    " - Do not infer min or max dates from your own knowledge of events.\n"
    " - If the question implies only a start (e.g., \"after March 2023\"), set:\n"
    "   min_publish_date = first day of that month (e.g., 2023-03-01)\n"
    "   max_publish_date = null.\n"
    " - If the question implies only an end (e.g., \"before 2021\"), set:\n"
    "   max_publish_date = last plausible day of that year (e.g., 2021-12-31)\n"
    "   min_publish_date = null.\n"
    " - If it implies a rough range (e.g., \"during 2020\"), choose a reasonable\n"
    "   min_publish_date and max_publish_date that cover that period.\n"
    " - If no dates are mentioned or implied, set both to null.\n"
    "\n"
    "Return ONLY the JSON object. Do not include explanations or any extra text.\n"
    "\n"
    "--------------------\n"
    "FEW-SHOT EXAMPLES\n"
    "--------------------\n"
    "\n"
    "User question:\n"
    "\"What are major Medicaid policies still in effect in Jan 2025?\"\n"
    "Output:\n"
    "{\n"
    "  \"min_publish_date\": null,\n"
    "  \"max_publish_date\": \"2025-01-31\"\n"
    "}\n"
    "\n"
    "User question:\n"
    "\"What major Medicaid updates were published in Jan 2025?\"\n"
    "Output:\n"
    "{\n"
    "  \"min_publish_date\": \"2025-01-01\",\n"
    "  \"max_publish_date\": \"2025-01-31\"\n"
    "}\n"
    "\n"
    "User question:\n"
    "\"Show me policies released after March 2023.\"\n"
    "Output:\n"
    "{\n"
    "  \"min_publish_date\": \"2023-03-01\",\n"
    "  \"max_publish_date\": null\n"
    "}\n"
    "\n"
    "User question:\n"
    "\"Find all guidance issued before 2021.\"\n"
    "Output:\n"
    "{\n"
    "  \"min_publish_date\": null,\n"
    "  \"max_publish_date\": \"2021-12-31\"\n"
    "}\n"
    "\n"
    "User question:\n"
    "\"Summaries of rules published during 2020.\"\n"
    "Output:\n"
    "{\n"
    "  \"min_publish_date\": \"2020-01-01\",\n"
    "  \"max_publish_date\": \"2020-12-31\"\n"
    "}\n"
    "\n"
    "User question:\n"
    "\"What's the federal stance on Medicaid waivers?\"\n"
    "Output:\n"
    "{\n"
    "  \"min_publish_date\": null,\n"
    "  \"max_publish_date\": null\n"
    "}\n"
)




# IMPORTANT: double-brace the literal JSON example so .format() only fills {question}
USER_PROMPT_TEMPLATE = (
    "Extract the publication-date bounds implied by the following question.\n"
    "These bounds will be used to filter policy documents by their publication date.\n"
    "\n"
    "Return ONLY a JSON object with this exact structure:\n"
    "{{\n"
    "  \"min_publish_date\": \"YYYY-MM-DD\" or null,\n"
    "  \"max_publish_date\": \"YYYY-MM-DD\" or null\n"
    "}}\n"
    "\n"
    "Question:\n"
    "{question}\n"
)

# Simple ISO date pattern for validation
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LLMFilterExtractor:
    """
    Drop-in replacement for FilterExtractor that:
      - Uses the SAME logic as the original FilterExtractor to detect:
          * authority_names (from authority_map + doc_metadata)
          * keywords (from acronym_map)
          * doc_titles (from doc_metadata)
          * doc_types (from doc_metadata)
      - BUT uses an LLM to infer:
          * min_effective_date
          * max_effective_date
        based on policy document PUBLICATION dates.

    The only behavioral difference from FilterExtractor is how dates are parsed.
    """

    def __init__(
        self,
        authority_map: Dict[str, str],
        acronym_map: Dict[str, str],
        doc_metadata: List[Dict],
        llm_client: Optional[LLMClient] = None,
    ):
        # --- Same stored fields as original FilterExtractor ---
        self.authority_map = {k.lower(): v for k, v in (authority_map or {}).items()}
        self.acronym_map = {k.lower(): v for k, v in (acronym_map or {}).items()}

        # metadata_filled.csv fields
        self.authorities = {
            str(d.get("authority_abbr", "")).lower(): str(d.get("authority_name", ""))
            for d in (doc_metadata or [])
            if d.get("authority_abbr")
        }
        self.doc_titles = [
            str(d.get("doc_title", "")).lower()
            for d in (doc_metadata or [])
            if d.get("doc_title")
        ]
        self.doc_types = list({
            str(d.get("doc_type", "")).lower()
            for d in (doc_metadata or [])
            if d.get("doc_type")
        })

        # --- LLM client setup (reuse existing config if not provided) ---
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                api_config_manager = APIConfigManager()
                cfg = api_config_manager.get_default_config()
                self.llm_client = LLMClient(
                    api_key=cfg.api_key,
                    model="gpt-5",      # adjust if needed to match your config
                    provider=cfg.provider,
                    base_url=cfg.base_url,
                )
            except Exception:
                self.llm_client = None

    # ------------------------------------------------------------------
    # Internal LLM-based date extractor
    # ------------------------------------------------------------------
    def _extract_dates_with_llm(self, query: str) -> Dict[str, Optional[str]]:
        """
        Use the LLM to infer min/max publication dates for the query.
        Return dict possibly containing:
            - "min_effective_date"
            - "max_effective_date"
        or {} if nothing usable is found.
        """
        if self.llm_client is None:
            return {}

        messages = [
            {"role": "system", "content": DATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(question=query.strip()),
            },
        ]

        try:
            raw = self.llm_client.chat(messages=messages)
        except Exception:
            return {}

        # raw may be a string or a dict-like
        if isinstance(raw, str):
            content = raw
        else:
            try:
                content = raw.get("content", "")
            except Exception:
                content = ""

        content = content.strip()

        # Try to isolate JSON between first '{' and last '}' in case the model
        # still adds some extra text despite instructions.
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
        else:
            json_str = content

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {}

        if not isinstance(data, dict):
            return {}

        min_publish = data.get("min_publish_date")
        max_publish = data.get("max_publish_date")

        # Normalize and validate against ISO date format
        def _normalize_date(value) -> Optional[str]:
            if value is None:
                return None
            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if not value:
                return None
            # Reject anything that doesn't look like YYYY-MM-DD
            if not ISO_DATE_RE.match(value):
                return None
            return value

        min_publish = _normalize_date(min_publish)
        max_publish = _normalize_date(max_publish)

        # If both are None, no filters
        if min_publish is None and max_publish is None:
            return {}

        out: Dict[str, Optional[str]] = {}
        if min_publish is not None:
            out["min_effective_date"] = min_publish
        if max_publish is not None:
            out["max_effective_date"] = max_publish

        return out

    # ------------------------------------------------------------------
    # Public API: same as original FilterExtractor.extract
    # ------------------------------------------------------------------
    def extract(self, query: str) -> Dict:
        """
        Extract structured filters from a user query.

        Same behavior as original FilterExtractor for:
          - authority_names
          - keywords
          - doc_titles
          - doc_types

        Different behavior only in:
          - min_effective_date
          - max_effective_date  (now LLM-based, using publication dates)
        """
        q = query.lower()
        filters: Dict[str, Optional[str]] = {}

        # 1. Authority detection (same as original)
        authorities = []
        for abbr, full in self.authority_map.items():
            if abbr in q or full.lower() in q:
                authorities.append(full)
        for abbr, full in self.authorities.items():
            if abbr in q or full.lower() in q:
                authorities.append(full)
        if authorities:
            filters["authority_names"] = list(set(authorities))

        # 2. Program / acronym detection (same as original)
        # Use word boundaries to match acronyms as whole words only
        keywords = []
        for k, v in self.acronym_map.items():
            # Create a regex pattern with word boundaries to match whole words only
            pattern = r'\b' + re.escape(k.lower()) + r'\b'
            if re.search(pattern, q):
                keywords.append(v)
        if keywords:
            filters["keywords"] = list(set(keywords))

        # 3. Document title match (same as original)
        matched_titles = [t for t in self.doc_titles if t and t in q]
        if matched_titles:
            filters["doc_titles"] = matched_titles

        # 4. Document type detection (same as original)
        matched_types = [t for t in self.doc_types if t and t in q]
        if matched_types:
            filters["doc_types"] = matched_types

        # 5. Date parsing: now LLM-based instead of _parse_date_expr
        date_filters = self._extract_dates_with_llm(query)
        filters.update(date_filters)

        return filters
