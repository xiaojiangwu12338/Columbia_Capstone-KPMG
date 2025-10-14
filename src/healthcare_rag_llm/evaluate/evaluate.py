# src/healthcare_rag_llm/evaluate.py

import json
from typing import List


def evaluate_results(
    tested_result_path: str,
    ground_truth_path: str,
    output_path: str,
    k_ranks: List[int] = [1, 3]
) -> None:
    """
    Evaluate multiple test runs (test_1, test_2, ...) under one hyperparameter setting.
    Each test corresponds to a single query.
    """

    with open(tested_result_path, "r", encoding="utf-8") as f:
        tested_data = json.load(f)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # For overall statistics
    total_tests = 0
    doc_level_correct = 0
    page_level_correct = 0

    detailed_results = {}

    for test_id, entry in tested_data.items():
        qid = entry.get("query_id")
        if not qid or qid not in gt_data:
            continue

        total_tests += 1
        gt_entry = gt_data[qid]

        # ground truth
        gt_docs = set(gt_entry["document"].keys())
        gt_pages = {f"{d}_{p}" for d, ps in gt_entry["document"].items() for p in ps}

        # prediction
        test_docs = set(entry.get("document", {}).keys())
        test_pages = {f"{d}_{p}" for d, ps in entry.get("document", {}).items() for p in ps}

        # ===== Document-level evaluation =====
        # Only when the predicted document contains all ground truth docs is it correct
        if gt_docs.issubset(test_docs):
            doc_level_correct += 1
            doc_correct = True
        else:
            doc_correct = False

        # ===== Page-level evaluation =====
        # Similarly, the prediction must cover all correct page numbers
        if gt_pages.issubset(test_pages):
            page_level_correct += 1
            page_correct = True
        else:
            page_correct = False

        detailed_results[test_id] = {
            "query_id": qid,
            "long_version_id": entry.get("long_version_id", ""),
            "short_version_id": entry.get("short_version_id", ""),
            "predicted_docs": list(test_docs),
            "gt_docs": list(gt_docs),
            "doc_level_correct": doc_correct,
            "page_level_correct": page_correct
        }

    # ===== Aggregate metrics =====
    doc_level_acc = round(doc_level_correct / total_tests, 3) if total_tests else 0
    page_level_acc = round(page_level_correct / total_tests, 3) if total_tests else 0

    summary = {
        "total_tests": total_tests,
        "doc_level_correct": doc_level_correct,
        "page_level_correct": page_level_correct,
        "doc_level_accuracy": doc_level_acc,
        "page_level_accuracy": page_level_acc
    }

    output = {
        "summary": summary,
        "details": detailed_results
    }

    # ===== Save =====
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete. Saved to {output_path}")
    print(f"Doc-level acc: {doc_level_acc}, Page-level acc: {page_level_acc}")

if __name__ == "__main__":
    evaluate_results(
        tested_result_path="data/test_results/test_result_1.json",
        ground_truth_path="data/testing_queries/testing_query.json",
        output_path="data/evaluation_results/testing_query_evaluation.json"
    )