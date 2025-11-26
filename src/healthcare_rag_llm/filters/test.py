# src/healthcare_rag_llm/filters/test.py

from __future__ import annotations

from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.utils.api_config import APIConfigManager

# Same prompt logic as in llm_filter_extractor.py
DATE_SYSTEM_PROMPT = (
    "You extract DATE BOUNDS for filtering policy documents by their PUBLICATION DATE.\n"
    "\n"
    "Your job is to infer the earliest and latest publication dates that should be\n"
    "used as bounds when retrieving relevant policy documents for a user's question.\n"
    "\n"
    "Return a JSON object with exactly these fields:\n"
    "  - \"min_publish_date\": earliest publication date (YYYY-MM-DD) or null\n"
    "  - \"max_publish_date\": latest publication date (YYYY-MM-DD) or null\n"
    "\n"
    "Rules:\n"
    "  - Dates must be in ISO format: YYYY-MM-DD.\n"
    "  - If the question implies only a start (e.g., \"after March 2023\"), set:\n"
    "        min_publish_date = first day of that month (e.g., 2023-03-01)\n"
    "        max_publish_date = null.\n"
    "  - If the question implies only an end (e.g., \"before 2021\"), set:\n"
    "        max_publish_date = last plausible day of that year (e.g., 2021-12-31)\n"
    "        min_publish_date = null.\n"
    "  - If it implies a rough range (e.g., \"during 2020\"), choose a reasonable\n"
    "    min_publish_date and max_publish_date that cover that period.\n"
    "  - If no dates are mentioned or implied, set both to null.\n"
    "\n"
    "Return ONLY the JSON object. Do not include explanations or any extra text."
)

# NOTE: literal braces for JSON are escaped as {{ and }} so .format() only fills {question}
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


def build_llm_client() -> LLMClient:
    api_config_manager = APIConfigManager()
    cfg = api_config_manager.get_default_config()
    return LLMClient(
        api_key=cfg.api_key,
        model="gpt-5",
        provider=cfg.provider,
        base_url=cfg.base_url,
    )


def main():
    client = build_llm_client()

    # You can change or add queries here
    test_queries = [
        "When did redetrmination begin for the COVID-19 Public Health Emergency unwind in New York State",
    ]

    for q in test_queries:
        messages = [
            {"role": "system", "content": DATE_SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=q.strip())},
        ]

        print("\n=================================================================")
        print("[QUERY]:", q)
        print("\n--- MESSAGES SENT TO LLM ---")
        print(messages)

        print("\n--- RAW LLM RESPONSE ---")
        try:
            raw = client.chat(messages=messages)
            # Print exactly what the client returns
            print(repr(raw))
        except Exception as e:
            print("LLM ERROR:", repr(e))


if __name__ == "__main__":
    main()
