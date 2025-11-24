"""
Test script to verify acronym detection is working correctly
"""
from pathlib import Path
from src.healthcare_rag_llm.llm.guardrail_response_wrapper import (
    _load_acronym_dict,
    _detect_acronyms_in_question
)

# Load the acronym dictionary
project_root = Path(__file__).resolve().parent
acronym_csv_path = project_root / "data" / "supplement" / "NYSDOH Acronym List.csv"

print(f"Loading acronyms from: {acronym_csv_path}")
print(f"File exists: {acronym_csv_path.exists()}\n")

acronym_dict = _load_acronym_dict(acronym_csv_path)
print(f"Loaded {len(acronym_dict)} acronyms\n")

# Test if PCMH is in the dictionary
if "PCMH" in acronym_dict:
    print(f"[OK] PCMH found: {acronym_dict['PCMH']}\n")
else:
    print("[FAIL] PCMH not found in dictionary\n")

# Test detection in various questions
test_questions = [
    "What is PCMH?",
    "What information does the March 2025 Update give concerning PCMH?",
    "Explain MLTC policy",
    "What does HIPAA mean?",
]

print("Testing acronym detection:\n")
for question in test_questions:
    found = _detect_acronyms_in_question(question, acronym_dict)
    print(f"Question: {question}")
    print(f"Found acronyms: {found}")
    print()