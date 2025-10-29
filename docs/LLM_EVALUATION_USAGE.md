# LLM Evaluation System - Complete Usage Guide

## Overview

The LLM evaluation system uses an LLM as a judge to assess RAG system answer quality across 5 key dimensions:

1. **Faithfulness** (35%): All claims supported by source chunks?
2. **Answer Relevance** (25%): Directly addresses the query?
3. **Citation Quality** (20%): Citations accurate and properly formatted?
4. **Completeness** (20%): Uses all relevant available information?
5. **Correctness** (30%, optional): Matches ground truth?

## Prerequisites

### 1. Updated Test Results Format

Your test results must include the `query_content` field. Update `generate_test_result.py` to include:

```python
results[row_name] = {
    "query_id": query_id,
    "query_content": question,  # ✅ Essential for accurate evaluation
    "long_version_id": self.long_version_id,
    "short_version_id": self.short_version_id,
    "top_k_chunks": retrieved_chunks,
    "answers": parsed_answer,
    "document": parsed_document,
}
```

**Why this matters**: Without `query_content`, the evaluator uses `query_id` (e.g., "Test_query_1") as the query, which causes Answer Relevance scores to be inaccurate or 0.

### 2. API Configuration

Ensure `configs/api_config.yaml` is properly configured:

```yaml
api_providers:
  bltcy:
    api_key: "your-api-key-here"
    base_url: "https://api.bltcy.ai/v1"
    provider: "openai"

default_provider: "bltcy"

models:
  gpt-5:
    provider: "bltcy"
    max_tokens: 4000
```

## Usage Examples

### Example 1: Basic Evaluation (No Ground Truth)

```bash
# Evaluate all test results
python scripts/llm_evaluation.py \
    --test_results data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    --output data/llm_eval_results/exp_001_eval.json

# Or with short flags
python scripts/llm_evaluation.py \
    -t data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    -o data/llm_eval_results/exp_001_eval.json
```

**Expected output:**
```
============================================================
STARTING LLM-BASED EVALUATION
============================================================
Test results: data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json
Output: data/llm_eval_results/exp_001_eval.json
Ground truth: Not provided
Evaluator model: gpt-5
Limit: All tests
============================================================

Loading test results from: data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json

[1/11] Evaluating test_id_1 (query: Test_query_1)
  Evaluating faithfulness...
  Evaluating answer relevance...
  Evaluating citation quality...
  Evaluating completeness...
  Overall score: 0.825

...

============================================================
EVALUATION SUMMARY
============================================================
faithfulness        : mean=0.891, min=0.750, max=1.000
answer_relevance    : mean=0.856, min=0.700, max=1.000
citation_quality    : mean=0.823, min=0.650, max=1.000
completeness        : mean=0.778, min=0.600, max=0.950
overall             : mean=0.837, min=0.724, max=0.932
============================================================
```

### Example 2: With Ground Truth (Enables Correctness Evaluation)

```bash
python scripts/llm_evaluation.py \
    -t data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    -o data/llm_eval_results/exp_001_eval_with_gt.json \
    -g data/ground_truth/ground_truth.json
```

This adds **Correctness** evaluation by comparing to ground truth answers.

### Example 3: Test on Subset (Faster Testing)

```bash
# Test on first 3 queries only
python scripts/llm_evaluation.py \
    -t data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    -o data/llm_eval_results/exp_001_sample.json \
    --limit 3
```

### Example 4: Use Specific Model

```bash
# Use GPT-4 instead of default GPT-5
python scripts/llm_evaluation.py \
    -t data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    -o data/llm_eval_results/exp_001_eval_gpt4.json \
    --model gpt-4
```

### Example 5: Batch Evaluate Multiple Experiments

```bash
# Create a batch evaluation script
for exp in exp_001 exp_002 exp_003; do
    echo "Evaluating $exp..."
    python scripts/llm_evaluation.py \
        -t "data/test_results/${exp}_semantic_k5_noRerank_gpt-5.json" \
        -o "data/llm_eval_results/${exp}_eval.json"
done
```

## Output Format

The evaluation results JSON contains:

```json
{
  "metadata": {
    "test_results_file": "data/test_results/exp_001.json",
    "ground_truth_file": null,
    "evaluator_model": "gpt-5",
    "total_tests": 11
  },
  "results": {
    "test_id_1": {
      "query": "When did New York State begin redetermining Medicaid eligibility?",
      "answer": "NYS began in April 2023 [mu_no02_feb25_pr.pdf:3]",
      "evaluations": {
        "faithfulness": {
          "score": 1.0,
          "reasoning": "The claim is directly supported by the source...",
          "unsupported_claims": []
        },
        "answer_relevance": {
          "score": 0.95,
          "reasoning": "The answer directly addresses the query...",
          "missing_aspects": []
        },
        "citation_quality": {
          "score": 1.0,
          "reasoning": "Citation is properly formatted...",
          "issues": []
        },
        "completeness": {
          "score": 0.85,
          "reasoning": "Answer includes key information...",
          "missing_info": ["Additional context about PHE"]
        }
      },
      "overall_score": 0.932,
      "weights_used": {
        "faithfulness": 0.35,
        "answer_relevance": 0.25,
        "citation_quality": 0.20,
        "completeness": 0.20
      }
    }
  },
  "summary": {
    "faithfulness": {"mean": 0.891, "min": 0.75, "max": 1.0, "count": 11},
    "answer_relevance": {"mean": 0.856, "min": 0.70, "max": 1.0, "count": 11},
    "citation_quality": {"mean": 0.823, "min": 0.65, "max": 1.0, "count": 11},
    "completeness": {"mean": 0.778, "min": 0.60, "max": 0.95, "count": 11},
    "overall": {"mean": 0.837, "min": 0.724, "max": 0.932, "count": 11}
  }
}
```

## Understanding the Scores

### Score Ranges

| Score | Interpretation |
|-------|----------------|
| 0.9-1.0 | Excellent |
| 0.8-0.89 | Good |
| 0.7-0.79 | Fair |
| 0.6-0.69 | Needs Improvement |
| < 0.6 | Poor |

### Metric-Specific Guidance

**Faithfulness (most critical)**
- **1.0**: Every claim has direct source support
- **0.8-0.9**: Minor unsupported details
- **< 0.8**: Hallucinations detected

**Answer Relevance**
- **1.0**: Perfect answer to the question
- **0.8-0.9**: Addresses query, minor gaps
- **< 0.8**: Off-topic or incomplete

**Citation Quality**
- **1.0**: All citations accurate and well-formatted
- **0.8-0.9**: Minor formatting issues
- **< 0.8**: Missing or incorrect citations

**Completeness**
- **1.0**: Uses all relevant chunk information
- **0.8-0.9**: Uses most relevant info
- **< 0.8**: Missing important details

## Troubleshooting

### Issue: Answer Relevance scores are 0 or very low

**Cause**: Test results don't have `query_content` field

**Solution**: Update `generate_test_result.py` to include:
```python
"query_content": question,  # Add this line
```

Then regenerate test results.

### Issue: JSON parsing errors in results

**Cause**: LLM sometimes returns malformed JSON with control characters

**Effect**: The failing metric gets score 0.0, but evaluation continues

**Solution**: Already handled automatically. Check the `raw_response` field in results for details.

### Issue: Evaluation is too slow

**Solutions**:
1. Use `--limit N` to test on subset first
2. Use faster/cheaper model for evaluation
3. Consider evaluating only critical experiments

### Issue: Scores seem unrealistic

**Solution**:
1. Manually review some low-scoring results to validate
2. Check the `reasoning` field to understand LLM's assessment
3. Adjust evaluation prompts if needed (in `llm_evaluate.py`)

## Best Practices

1. **Always include `query_content`** in test results for accurate evaluation
2. **Test with `--limit 3`** first to verify everything works
3. **Review a sample** of high and low scoring results manually
4. **Use ground truth** when available for correctness evaluation
5. **Save evaluations** with descriptive names indicating experiment parameters
6. **Compare across experiments** to identify best configurations

## Integration with Existing Workflow

```bash
# 1. Generate test results (with query_content!)
python scripts/testing/generate_test_result.py \
    --version exp_001_semantic_k5_noRerank

# 2. Evaluate with LLM judge
python scripts/llm_evaluation.py \
    -t data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    -o data/llm_eval_results/exp_001_llm_eval.json

# 3. Also run traditional evaluation (document/page accuracy)
python scripts/evaluate.py \
    --tested_result data/test_results/exp_001_semantic_k5_noRerank_gpt-5.json \
    --ground_truth data/ground_truth.json \
    --output data/evaluation_results/exp_001_eval.json

# 4. Compare both evaluation types for comprehensive assessment
```

## Advanced: Programmatic Usage

```python
from healthcare_rag_llm.evaluate.llm_evaluate import LLMEvaluator, evaluate_test_results
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.utils.api_config import load_api_config

# Load config
config = load_api_config()
provider_config = config["api_providers"]["bltcy"]

# Initialize LLM
llm_client = LLMClient(
    api_key=provider_config["api_key"],
    base_url=provider_config["base_url"],
    model="gpt-5",
    provider="openai"
)

# Run evaluation
results = evaluate_test_results(
    test_results_path="data/test_results/exp_001.json",
    output_path="data/llm_eval_results/exp_001_eval.json",
    llm_client=llm_client,
    ground_truth_path="data/ground_truth.json",  # optional
    limit=None  # or set to limit
)

# Access results
print(f"Overall mean score: {results['summary']['overall']['mean']}")
```

## Summary

✅ **Your addition of `query_content` is essential and correct**

This ensures:
- Accurate Answer Relevance evaluation
- Meaningful evaluation scores
- Proper assessment of how well answers address actual questions

The LLM evaluation system is now ready to use with your updated test results format!
