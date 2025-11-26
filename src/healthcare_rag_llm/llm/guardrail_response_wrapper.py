# guardrails.py
from __future__ import annotations

from typing import Dict, List, Optional
import csv
from pathlib import Path
import re

# Reuse the existing components so nothing downstream changes.
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.llm.chat_history import ChatHistory
from healthcare_rag_llm.utils.prompt_config import load_system_prompt

# The original generator we're wrapping.
from healthcare_rag_llm.llm.response_gen_json import ResponseGenerator as BaseResponseGenerator


REJECTION_MESSAGE = (
    "This assistant is intended to support inquiries related to healthcare policy, "
    "program administration, and regulatory guidance, based on official publications "
    "and initiatives of the New York State Department of Health and other state-administered programs. "
    "It does not provide clinical, technical, or general advice beyond that scope. "
    "We encourage you to restate your question with a clear healthcare policy or program focus — "
    "for example, topics such as Medicaid coverage, waiver programs, provider enrollment, "
    "or compliance requirements."
)

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a strict classifier. "
    "Answer ONLY 'YES' or 'NO' (uppercase, no punctuation). "
    "Say 'YES' if the user's query is primarily about healthcare POLICY, "
    "such as laws, regulations, coverage mandates, payer rules, reimbursement policy, "
    "Medicare/Medicaid/insurer policies, coding/billing policy, compliance requirements, "
    "eligibility rules, prior auth policies, formulary coverage, HIPAA policy questions, "
    "or interpretations/changes to policy documents.\n"
    "\n"
    "IMPORTANT: Questions asking for definitions or explanations of healthcare policy terms, "
    "programs, or acronyms (e.g., 'What is PCMH?', 'Explain MLTC') should be classified as YES, "
    "as understanding policy terminology is essential to policy questions.\n"
    "\n"
    "Say 'NO' if it's clinical advice, diagnostics, treatments, drugs' mechanisms/dosing, "
    "general wellness, admin/IT topics without policy focus, or anything unrelated to healthcare policy."
)

# A few very lightweight positive/negative examples to anchor behavior
CLASSIFIER_USER_PREFIX = (
    "Classify the following user input. Reply ONLY YES or NO.\n\n"
    "Examples:\n"
    "Q: Does Medicare cover CGM for type 2 diabetes?\nA: YES\n"
    "Q: What are ICD-10 codes for type 2 diabetes with neuropathy?\nA: YES\n"  # coding policy counts
    "Q: What is PCMH?\nA: YES\n"  # asking for policy term definition
    "Q: Explain the MLTC program\nA: YES\n"  # asking about policy program
    "Q: Should I increase my lisinopril dose?\nA: NO\n"
    "Q: How do I treat strep throat?\nA: NO\n"
    "Q: What's the HIPAA rule on texting patients?\nA: YES\n"
    "Q: Build me a Flask app\nA: NO\n"
    "\n"
    "{acronym_context}"
    "Now classify:\n"
    "Q: {question}\nA:"
)


def _load_acronym_dict(csv_path: Path) -> Dict[str, str]:
    """
    Load acronym dictionary from CSV file.
    Returns dict mapping uppercase acronym -> meaning.
    """
    acronyms = {}
    if not csv_path.exists():
        return acronyms

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                acronym = row.get('acronym', '').strip().upper()
                meaning = row.get('full_term', '').strip()
                if acronym and meaning:
                    acronyms[acronym] = meaning
    except Exception:
        pass

    return acronyms


def _detect_acronyms_in_question(question: str, acronym_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Detect known acronyms in the user's question.
    Returns dict of found acronyms -> meanings.
    """
    found = {}
    # Split question into words and check each against acronym dict
    # Match whole words only (avoid partial matches)
    words = re.findall(r'\b[A-Z]+\b', question)

    for word in words:
        if word.upper() in acronym_dict:
            found[word] = acronym_dict[word]

    return found


class ResponseGenerator:
    """
    Guardrailed wrapper around the original ResponseGenerator.
    Constructor/signature intentionally identical to the base class so you
    can swap imports without changing call sites.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: Optional[str] = None,
        use_reranker: bool = True,
        filter_extractor=None,
        chat_history: ChatHistory = None,
        acronym_csv_path: Optional[Path] = None,
    ):
        # Mirror the original initialization exactly
        if system_prompt is None:
            system_prompt = load_system_prompt()

        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.use_reranker = use_reranker
        self.filter_extractor = filter_extractor
        self.chat_history = chat_history or ChatHistory()

        # Load acronym dictionary
        if acronym_csv_path is None:
            # Default path: <project_root>/data/supplement/NYSDOH Acronym List.csv
            # Assuming this file is 4 levels up from src/healthcare_rag_llm/llm/guardrail_response_wrapper.py
            project_root = Path(__file__).resolve().parents[3]
            acronym_csv_path = project_root / "data" / "metadata" / "acronym_map.csv"

        self.acronym_dict = _load_acronym_dict(acronym_csv_path)

        # Internal: delegate that does the real RAG work when allowed
        self._delegate = BaseResponseGenerator(
            llm_client=llm_client,
            system_prompt=system_prompt,
            use_reranker=use_reranker,
            filter_extractor=filter_extractor,
            chat_history=self.chat_history,
        )

    # ---- Public API mirrors the base class ----
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        rerank_top_k: int = 20,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        If the question is healthcare-policy-related (per LLM classifier), delegate
        to the original RAG pipeline. Otherwise, return a rejection message.

        Return shape matches the base class:
            {
              "question": str,
              "answer": str,
              "retrieved_docs": List[Dict]
            }
        """

        # 1) Classify without polluting the user-facing chat history
        is_policy = self._is_healthcare_policy_question(question)

        if not is_policy:
            # Do NOT add assistant content to chat history here;
            # caller can decide how to surface the rejection.
            return {
                "question": question,
                "answer": REJECTION_MESSAGE,
                "evidence_dict": {},
                "retrieved_docs": [],
            }

        # 2) Delegate to the standard pipeline
        return self._delegate.answer_question(
            question=question,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            history=history,
        )

    # ---- Internal helpers ----
    def _is_healthcare_policy_question(self, question: str) -> bool:
        """
        Uses the same LLM backend to classify. It is intentionally strict and
        expects ONLY 'YES' or 'NO'. Any other output is treated as NO.

        If the question contains known healthcare acronyms, those are dynamically
        added to the classifier prompt to improve accuracy.
        """
        # Detect acronyms in the question
        found_acronyms = _detect_acronyms_in_question(question, self.acronym_dict)

        # Build acronym context string if any were found
        acronym_context = ""
        if found_acronyms:
            acronym_lines = [f"- {acr}: {meaning}" for acr, meaning in found_acronyms.items()]
            acronym_context = (
                "Note: This question contains the following healthcare acronyms:\n"
                + "\n".join(acronym_lines)
                + "\n\n"
            )

        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CLASSIFIER_USER_PREFIX.format(
                    question=question.strip(),
                    acronym_context=acronym_context
                ),
            },
        ]

        try:
            raw = self.llm_client.chat(messages=messages)
        except Exception:
            # Fail closed: if we can't classify, do not proceed.
            return False

        if not isinstance(raw, str):
            # If your llm_client returns a dict, adapt as needed here (e.g., raw["content"])
            try:
                content = raw.get("content", "")
            except Exception:
                content = ""
        else:
            content = raw

        answer = content.strip().upper()
        # Accept common variants like "YES.", "YES\n"
        if answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False

        # Unknown output -> treat as NO
        return False
