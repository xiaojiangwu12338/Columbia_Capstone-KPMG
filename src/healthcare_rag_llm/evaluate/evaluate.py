# src/healthcare_rag_llm/evaluate.py
import json
from typing import List, Dict, Any

def evaluate_results(
    tested_result_path: str,
    ground_truth_path: str,
    output_path: str,
    k_ranks: List[int] = None
) -> Dict[str, Any]:
    """
    Evaluate multiple test runs (test_id_1, test_id_2, ...) for one hyperparameter setting.
    Each test contains multiple top_k_chunks with metadata.
    """

    # === Load ===
    with open(tested_result_path, "r", encoding="utf-8") as f:
        tested_data = json.load(f)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    total_tests = 0
    doc_level_correct = 0
    page_level_correct = 0
    details = {}

    # === Loop over each test ===
    for test_id, entry in tested_data.items():
        qid = entry.get("query_id")
        if not qid or qid not in gt_data:
            continue

        total_tests += 1
        gt_entry = gt_data[qid]

        # --- Ground Truth ---
        gt_docs_dict = gt_entry["document"]
        gt_docs = set(gt_docs_dict.keys())
        gt_pages = {f"{doc}_{p}" for doc, ps in gt_docs_dict.items() for p in ps}

        # --- Predicted Docs from top_k_chunks ---
        top_k_chunks = entry.get("top_k_chunks", [])
        predicted_docs = set()
        predicted_pages = set()

        for chunk in top_k_chunks:
            doc_id = chunk.get("doc_id")
            pages = chunk.get("pages", [])
            if doc_id:
                predicted_docs.add(doc_id)
                for p in pages:
                    predicted_pages.add(f"{doc_id}_{p}")

        # --- Evaluation ---
        doc_correct = gt_docs.issubset(predicted_docs)
        page_correct = gt_pages.issubset(predicted_pages)

        if doc_correct:
            doc_level_correct += 1
        if page_correct:
            page_level_correct += 1

        # --- Save Detail Record ---
        details[test_id] = {
            "query_id": qid,
            "long_version_id": entry.get("long_version_id", ""),
            "short_version_id": entry.get("short_version_id", ""),
            "predicted_docs": list(predicted_docs),
            "predicted_top_chunks": top_k_chunks,
            "gt_docs": gt_docs_dict,                     # keep doc→pages
            "doc_level_correct": doc_correct,
            "page_level_correct": page_correct
        }

    # === Summary ===
    summary = {
        "total_tests": total_tests,
        "doc_level_correct": doc_level_correct,
        "page_level_correct": page_level_correct,
        "doc_level_accuracy": round(doc_level_correct / total_tests, 3) if total_tests else 0.0,
        "page_level_accuracy": round(page_level_correct / total_tests, 3) if total_tests else 0.0,
        "k_ranks": k_ranks
    }

    result = {"summary": summary, "details": details}

    # === Save ===
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"✅ Evaluation complete → {output_path}")
    print(f"📊 Doc acc: {summary['doc_level_accuracy']}, Page acc: {summary['page_level_accuracy']}")
    return result
