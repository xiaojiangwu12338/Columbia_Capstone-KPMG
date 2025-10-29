# LLM-Based Evaluation System

This module provides LLM-based evaluation for RAG system responses, using an LLM as a judge to assess answer quality across multiple dimensions.

## Overview

The LLM evaluation system evaluates generated answers on 5 key dimensions:

1. **Faithfulness** (weight: 35%): Are all claims in the answer supported by the retrieved chunks?
2. **Answer Relevance** (weight: 25%): Does the answer directly address the query?
3. **Citation Quality** (weight: 20%): Are citations accurate and properly formatted?
4. **Completeness** (weight: 20%): Does the answer use all relevant available information?
5. **Correctness** (weight: 30%, optional): Does the answer match ground truth?

Each dimension is scored 0.0-1.0, and an overall weighted score is calculated.

## Quick Start

### Command Line Usage

```bash
# Basic evaluation
python scripts/llm_evaluation.py \
    --test_results data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    --output data/llm_eval_results/exp_001_llm_eval.json

# With ground truth for correctness evaluation
python scripts/llm_evaluation.py \
    --test_results data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    --output data/llm_eval_results/exp_001_llm_eval.json \
    --ground_truth data/ground_truth.json

# Test on first 3 items only (for testing)
python scripts/llm_evaluation.py \
    --test_results data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    --output data/llm_eval_results/exp_001_llm_eval.json \
    --limit 3

# Use specific model
python scripts/llm_evaluation.py \
    --test_results data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    --output data/llm_eval_results/exp_001_llm_eval.json \
    --model gpt-4
```

### Programmatic Usage

```python
from healthcare_rag_llm.evaluate.llm_evaluate import LLMEvaluator
from healthcare_rag_llm.llm.llm_client import LLMClient

# Initialize LLM client
llm_client = LLMClient(
    api_key="your-api-key",
    base_url="https://api.bltcy.ai/v1",
    model="gpt-5",
    provider="openai"
)

# Create evaluator
evaluator = LLMEvaluator(llm_client)

# Example: Evaluate faithfulness
query = "When did NYS begin redetermining Medicaid eligibility?"
answer = "New York State began redetermining Medicaid eligibility in April 2023. [mu_no02_feb25_pr.pdf:3]"
chunks = [
    {
        "text": "The Consolidated Appropriations Act of 2023 required states to begin the process of redetermining Medicaid eligibility for its members, which New York State (NYS) began in April 2023.",
        "doc_id": "mu_no02_feb25_pr.pdf",
        "page": 3
    }
]

result = evaluator.evaluate_faithfulness(query, answer, chunks)
print(f"Faithfulness score: {result['score']}")
print(f"Reasoning: {result['reasoning']}")

# Comprehensive evaluation
comprehensive = evaluator.evaluate_comprehensive(
    query=query,
    answer=answer,
    chunks=chunks,
    ground_truth="NYS began in April 2023"  # optional
)
print(f"Overall score: {comprehensive['overall_score']}")
```

## Input Format

### Test Results Format

The test results JSON should have this structure:

```json
{
  "test_id_1": {
    "query_id": "Test_query_1",
    "top_k_chunks": [
      {
        "chunk_id": "mu_no02_feb25_pr.pdf::0033",
        "text": "...",
        "doc_id": "mu_no02_feb25_pr.pdf",
        "page": 3,
        "score": 0.816
      }
    ],
    "answers": "- \"The Consolidated Appropriations Act of 2023...\" [mu_no02_feb25_pr.pdf:3]"
  },
  "test_id_2": { ... }
}
```

### Ground Truth Format (Optional)

```json
{
  "Test_query_1": {
    "query_id": "Test_query_1",
    "answer": "NYS began redetermining Medicaid eligibility in April 2023.",
    "document": {
      "mu_no02_feb25_pr.pdf": [3]
    }
  }
}
```

## Output Format

The evaluation results JSON contains:

```json
{
  "metadata": {
    "test_results_file": "...",
    "evaluator_model": "gpt-5",
    "total_tests": 11
  },
  "results": {
    "test_id_1": {
      "query": "Test_query_1",
      "answer": "...",
      "evaluations": {
        "faithfulness": {
          "score": 0.95,
          "reasoning": "All claims are supported...",
          "unsupported_claims": []
        },
        "answer_relevance": {
          "score": 1.0,
          "reasoning": "Directly answers the query...",
          "missing_aspects": []
        },
        "citation_quality": {
          "score": 0.9,
          "reasoning": "Citations are accurate...",
          "issues": ["Minor formatting inconsistency"]
        },
        "completeness": {
          "score": 0.85,
          "reasoning": "Uses most relevant information...",
          "missing_info": []
        },
        "correctness": {
          "score": 1.0,
          "reasoning": "Matches ground truth semantically",
          "differences": []
        }
      },
      "overall_score": 0.932,
      "weights_used": {
        "faithfulness": 0.35,
        "answer_relevance": 0.25,
        "citation_quality": 0.20,
        "completeness": 0.20,
        "correctness": 0.30
      }
    }
  },
  "summary": {
    "faithfulness": {"mean": 0.876, "min": 0.75, "max": 0.95, "count": 11},
    "answer_relevance": {"mean": 0.891, "min": 0.80, "max": 1.0, "count": 11},
    "citation_quality": {"mean": 0.834, "min": 0.70, "max": 0.95, "count": 11},
    "completeness": {"mean": 0.812, "min": 0.65, "max": 0.90, "count": 11},
    "correctness": {"mean": 0.789, "min": 0.60, "max": 1.0, "count": 11},
    "overall": {"mean": 0.842, "min": 0.724, "max": 0.932, "count": 11}
  }
}
```

## Evaluation Metrics

### Faithfulness
Checks if all claims in the answer can be verified from the source chunks. High faithfulness means no hallucination or unsupported claims.

**Scoring:**
- 1.0: All claims fully supported
- 0.8-0.9: Most claims supported
- 0.5-0.7: Some unsupported claims
- 0.0-0.4: Many unsupported claims

### Answer Relevance
Evaluates whether the answer directly addresses the query without going off-topic.

**Scoring:**
- 1.0: Perfectly addresses query
- 0.8-0.9: Addresses query well
- 0.5-0.7: Partially addresses query
- 0.0-0.4: Off-topic or missing key points

### Citation Quality
Assesses citation accuracy and formatting.

**Scoring:**
- 1.0: All citations perfect
- 0.8-0.9: Citations mostly good
- 0.5-0.7: Some citation problems
- 0.0-0.4: Poor or missing citations

### Completeness
Checks if the answer uses all relevant information available in the chunks.

**Scoring:**
- 1.0: Uses all relevant information
- 0.8-0.9: Uses most relevant information
- 0.5-0.7: Missing some relevant details
- 0.0-0.4: Very incomplete

### Correctness (Optional)
Compares the answer to ground truth for factual correctness. Only available when ground truth is provided.

**Scoring:**
- 1.0: Semantically equivalent to ground truth
- 0.8-0.9: Mostly correct
- 0.5-0.7: Partially correct
- 0.0-0.4: Mostly incorrect

## Configuration

### API Configuration

Edit `configs/api_config.yaml`:

```yaml
api_providers:
  bltcy:
    api_key: "your-api-key"
    base_url: "https://api.bltcy.ai/v1"
    provider: "openai"

default_provider: "bltcy"

models:
  gpt-5:
    provider: "bltcy"
    max_tokens: 4000
```

### Adjusting Weights

The default weights are:
- Faithfulness: 35%
- Answer Relevance: 25%
- Citation Quality: 20%
- Completeness: 20%
- Correctness: 30% (when available)

To change weights, modify the `evaluate_comprehensive` method in `llm_evaluate.py`:

```python
weights = {
    "faithfulness": 0.40,        # Increase faithfulness weight
    "answer_relevance": 0.25,
    "citation_quality": 0.15,    # Decrease citation weight
    "completeness": 0.20,
    "correctness": 0.30 if ground_truth else 0.0
}
```

## Tips for Best Results

1. **Use a strong evaluator model**: GPT-4 or similar for best evaluation quality
2. **Provide ground truth**: Enables correctness evaluation
3. **Test on small subset first**: Use `--limit 3` to verify it works before full run
4. **Review edge cases**: Check low-scoring results manually to validate evaluation
5. **Monitor API costs**: LLM evaluation makes 4-5 API calls per test item

## Troubleshooting

### Issue: JSON parsing errors
**Solution**: The LLM sometimes returns malformed JSON. The evaluator has fallback handling that sets score to 0.0 and logs the error.

### Issue: Slow evaluation
**Solution**: Each test requires 4-5 LLM calls. Use `--limit` for faster testing, or parallelize if needed.

### Issue: Low scores across all metrics
**Solution**: Review the reasoning fields to understand why. May indicate issues with the answer generation system or the evaluation prompts.

## Architecture

```
llm_evaluate.py
├── LLMEvaluator
│   ├── evaluate_faithfulness()
│   ├── evaluate_answer_relevance()
│   ├── evaluate_citation_quality()
│   ├── evaluate_completeness()
│   ├── evaluate_correctness()
│   └── evaluate_comprehensive()
└── evaluate_test_results()
```

## Related Files

- [llm_evaluate.py](llm_evaluate.py) - Core evaluation logic
- [../../scripts/llm_evaluation.py](../../../scripts/llm_evaluation.py) - CLI script
- [../llm/llm_client.py](../llm/llm_client.py) - LLM API client
- [../utils/api_config.py](../utils/api_config.py) - API configuration loader
