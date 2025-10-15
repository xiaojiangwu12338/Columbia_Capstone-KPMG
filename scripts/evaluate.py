from healthcare_rag_llm.evaluate.evaluate import evaluate_results


def main():
    tested_result_path = "data/test_results/v1.0-demo.json"
    ground_truth_path = "data/testing_queries/testing_query.json"
    output_path = "data/evaluation_results/testing_query_evaluation.csv"

    result = evaluate_results(tested_result_path, ground_truth_path, output_path)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()