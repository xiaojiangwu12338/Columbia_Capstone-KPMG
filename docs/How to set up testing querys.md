# How to Set Up Testing Queries

This document explains how to create and configure testing queries for evaluating the RAG system.

## 1. Location

Place your testing query files in the following directory:
```
./data/testing_queries
```

## 2. File Format

Testing query files must be in **JSON format**.

### JSON Structure

Each file should contain a dictionary with test queries as keys. Each test query includes:

```json
{
  "test_query_1": {
    "question": "Your question for the LLM",
    "document": ["ground_truth_doc_1.pdf", "ground_truth_doc_2.pdf"],
    "Answer": "The reference ground truth answer"
  },
  "test_query_2": {
    "question": "Another question for the LLM",
    "document": ["relevant_doc.pdf"],
    "Answer": "Expected answer for evaluation"
  }
}
```

### Field Descriptions

- **Key** (e.g., `"test_query_1"`): Unique identifier for each test query
- **`question`**: The query/question to ask the LLM
- **`document`**: Array of ground truth document names that should be retrieved
- **`Answer`**: The reference answer used for evaluation

## 3. Example

```json
{
  "diabetes_treatment": {
    "question": "What are the recommended treatments for type 2 diabetes?",
    "document": ["diabetes_guidelines_2024.pdf", "endocrinology_handbook.pdf"],
    "Answer": "First-line treatment for type 2 diabetes includes lifestyle modifications (diet and exercise) and metformin as initial pharmacotherapy."
  },
  "hypertension_diagnosis": {
    "question": "What are the diagnostic criteria for hypertension?",
    "document": ["cardiovascular_guidelines.pdf"],
    "Answer": "Hypertension is diagnosed when blood pressure readings are consistently ≥130/80 mmHg."
  }
}
```

## Notes

- Ensure all document names in the `document` field match actual files in your knowledge base
- Use descriptive names for test query keys to easily identify them in evaluation results
- The `Answer` field should contain accurate, reference-standard responses for meaningful evaluation
