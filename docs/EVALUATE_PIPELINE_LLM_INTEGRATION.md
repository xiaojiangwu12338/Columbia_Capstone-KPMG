# Evaluate Pipeline with LLM Evaluation Integration

## Overview

The `evaluate_pipeline.py` now supports **dual evaluation**:
1. **Traditional Evaluation**: Document/page accuracy (always runs)
2. **LLM Evaluation**: Faithfulness, relevance, citation quality, completeness (optional)

This allows comprehensive assessment of RAG system performance from both precision and quality perspectives.

---

## Quick Start

### Example 1: Default Mode (No LLM Evaluation)

```python
# In main() function
pipeline = EvaluatePipeline()
results_df = pipeline.run_batch_experiments(experiments)

# Output: Only traditional metrics in CSV
# - doc_accuracy
# - page_accuracy
# - llm_* columns will be None
```

**Use case**: Fast experiments, cost-sensitive scenarios

---

### Example 2: Enable LLM Evaluation (All Tests)

```python
pipeline = EvaluatePipeline(
    enable_llm_eval=True,
    llm_eval_model="gpt-5"
)
results_df = pipeline.run_batch_experiments(experiments)

# Output: Traditional + LLM metrics in CSV
# - doc_accuracy, page_accuracy
# - llm_faithfulness_mean
# - llm_answer_relevance_mean
# - llm_citation_quality_mean
# - llm_completeness_mean
# - llm_overall_mean
```

**Use case**: Comprehensive quality assessment

---

### Example 3: LLM Evaluation with Sampling (Recommended for Testing)

```python
pipeline = EvaluatePipeline(
    enable_llm_eval=True,
    llm_eval_model="gpt-5",
    llm_eval_limit=3  # Only evaluate first 3 queries per experiment
)
results_df = pipeline.run_batch_experiments(experiments)

# Faster: ~1-2 min per experiment vs 5-10 min
# Still provides representative quality scores
```

**Use case**:
- Quick quality checks during development
- Cost control (reduce API calls)
- Faster iteration

---

### Example 4: Use Different Evaluation Model

```python
pipeline = EvaluatePipeline(
    enable_llm_eval=True,
    llm_eval_model="gpt-4",
    llm_eval_provider="openai_official"  # Use official OpenAI API
)
results_df = pipeline.run_batch_experiments(experiments)
```

**Use case**: Higher quality evaluation, specific model preferences

---

## Configuration Parameters

### LLM Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_llm_eval` | bool | `False` | Enable LLM-based evaluation |
| `llm_eval_model` | str | `"gpt-5"` | Model to use for evaluation |
| `llm_eval_provider` | str | `None` | API provider (None = use default from config) |
| `llm_eval_limit` | int\|None | `None` | Limit tests to evaluate (None = all) |
| `llm_eval_timeout` | int | `3600` | Timeout for LLM eval in seconds |

---

## Output Files

### File Structure

```
data/
├── test_results/
│   └── exp_001_semantic_k5_noRerank_gpt-5.json    # Test results
├── evaluation_results/
│   ├── exp_001_semantic_k5_noRerank_gpt-5_evaluation.json  # Traditional eval
│   └── batch_evaluation_results.csv                        # All experiments summary
└── llm_eval_results/
    └── exp_001_semantic_k5_noRerank_gpt-5_llm_evaluation.json  # LLM eval (detailed)
```

### CSV Output Columns

**Always present** (traditional metrics):
- `version_id`, `status`, `chunking_method`, `top_k`, `rerank`, `alpha`
- `doc_accuracy`, `page_accuracy`, `total_tests`
- `llm_model`, `error`

**When LLM eval enabled**:
- `llm_faithfulness_mean` (0-1)
- `llm_answer_relevance_mean` (0-1)
- `llm_citation_quality_mean` (0-1)
- `llm_completeness_mean` (0-1)
- `llm_overall_mean` (0-1, weighted average)

---

## Understanding LLM Metrics

### Metric Definitions

| Metric | What it Measures | Weight | Good Score |
|--------|------------------|--------|------------|
| **Faithfulness** | All claims supported by chunks? | 35% | > 0.90 |
| **Answer Relevance** | Directly addresses the query? | 25% | > 0.85 |
| **Citation Quality** | Citations accurate & formatted? | 20% | > 0.80 |
| **Completeness** | Uses all relevant chunk info? | 20% | > 0.75 |
| **Overall** | Weighted average | - | > 0.80 |

### Interpreting Scores

```
Score Range    Interpretation
-----------    --------------
0.9 - 1.0      Excellent
0.8 - 0.89     Good
0.7 - 0.79     Fair
0.6 - 0.69     Needs Improvement
< 0.6          Poor
```

---

## Terminal Output Example

### Without LLM Evaluation

```bash
============================================================
Best Results:
============================================================
Best document accuracy: exp_002_fix_size_k5_rerank_a0.5_gpt-5
  - Doc Accuracy: 0.818
  - Page Accuracy: 0.636
  - Chunking: fix_size
  - Top-K: 5
  - Model: gpt-5

Best page accuracy: exp_002_fix_size_k5_rerank_a0.5_gpt-5
  - Page Accuracy: 0.636
  - Doc Accuracy: 0.818
  - Chunking: fix_size
  - Top-K: 5
  - Model: gpt-5
============================================================
```

### With LLM Evaluation Enabled

```bash
============================================================
Best Results:
============================================================
Best document accuracy: exp_002_fix_size_k5_rerank_a0.5_gpt-5
  - Doc Accuracy: 0.818
  - Page Accuracy: 0.636
  - Chunking: fix_size
  - Top-K: 5
  - Model: gpt-5

Best page accuracy: exp_002_fix_size_k5_rerank_a0.5_gpt-5
  - Page Accuracy: 0.636
  - Doc Accuracy: 0.818
  - Chunking: fix_size
  - Top-K: 5
  - Model: gpt-5

Best LLM overall score: exp_001_semantic_k5_noRerank_gpt-5
  - LLM Overall: 0.892
  - Faithfulness: 0.923
  - Answer Relevance: 0.912
  - Citation Quality: 0.867
  - Completeness: 0.834
  - Chunking: semantic
  - Top-K: 5
============================================================
```

---

## Performance Considerations

### Time Impact

| Scenario | Time per Experiment | API Calls |
|----------|-------------------|-----------|
| Traditional only | 5-10 min | ~11 (LLM generation) |
| + LLM eval (all 11 tests) | 15-25 min | ~66 total (11 + 55 eval) |
| + LLM eval (limit=3) | 7-12 min | ~26 total (11 + 15 eval) |

### Cost Impact (Approximate)

Assuming gpt-5 at ~$0.002 per query:
- Traditional only: ~$0.02 per experiment
- + LLM eval (all): ~$0.13 per experiment (+$0.11)
- + LLM eval (limit=3): ~$0.05 per experiment (+$0.03)

**Recommendation**: Use `llm_eval_limit=3` during development, full evaluation for final comparisons.

---

## Best Practices

### 1. Start Small

```python
# First run: Test on 1-2 experiments with limit=3
pipeline = EvaluatePipeline(
    enable_llm_eval=True,
    llm_eval_limit=3
)

# Create small experiment list
test_experiments = experiments[:2]
results_df = pipeline.run_batch_experiments(test_experiments)
```

### 2. Gradual Scaling

```python
# Development: limit=3, few experiments
pipeline = EvaluatePipeline(enable_llm_eval=True, llm_eval_limit=3)

# Testing: limit=5, more experiments
pipeline = EvaluatePipeline(enable_llm_eval=True, llm_eval_limit=5)

# Production: full evaluation, all experiments
pipeline = EvaluatePipeline(enable_llm_eval=True)
```

### 3. Error Handling

LLM evaluation failures don't stop the pipeline:
- Traditional eval always completes
- LLM eval errors are logged
- CSV shows `None` for failed LLM metrics
- Check console output for warnings

### 4. Comparing Configurations

```python
# Compare traditional vs LLM rankings
csv = pd.read_csv("data/evaluation_results/batch_evaluation_results.csv")

# Top by traditional metrics
top_trad = csv.nlargest(3, 'doc_accuracy')

# Top by LLM metrics
top_llm = csv.nlargest(3, 'llm_overall_mean')

# Are they the same? If not, investigate why!
```

---

## Troubleshooting

### Issue: LLM evaluation takes too long

**Solution**:
```python
pipeline = EvaluatePipeline(
    enable_llm_eval=True,
    llm_eval_limit=3,      # Reduce from 11 to 3
    llm_eval_timeout=1800  # Set timeout
)
```

### Issue: API rate limits

**Solution**:
- Use `llm_eval_limit` to reduce calls
- Add delays between experiments (modify pipeline code)
- Use cheaper/faster model for evaluation

### Issue: All LLM metrics are None

**Cause**: `enable_llm_eval=False` (default)

**Solution**:
```python
pipeline = EvaluatePipeline(enable_llm_eval=True)
```

### Issue: LLM eval failed for some experiments

**Check**:
1. Console output for error messages
2. `data/llm_eval_results/` for partial results
3. Ensure test results have `query_content` field

---

## Integration with Existing Workflow

```bash
# Step 1: Generate test results
# (Ensure generate_test_result.py includes query_content field)
python scripts/testing.py

# Step 2: Run evaluate_pipeline with LLM eval
python scripts/evaluate/evaluate_pipeline.py
# (Modify main() to set enable_llm_eval=True)

# Step 3: Analyze results
# - Check data/evaluation_results/batch_evaluation_results.csv
# - Compare traditional vs LLM rankings
# - Review detailed results in data/llm_eval_results/
```

---

## Advanced: Modifying Main Function

### Enable LLM Evaluation by Default

```python
def main():
    # ... experiment configuration ...

    # Change this line:
    pipeline = EvaluatePipeline()

    # To:
    pipeline = EvaluatePipeline(
        enable_llm_eval=True,
        llm_eval_model="gpt-5",
        llm_eval_limit=3  # or None for all
    )

    results_df = pipeline.run_batch_experiments(experiments)
    # ...
```

### Conditional LLM Evaluation

```python
import os

# Enable LLM eval via environment variable
enable_llm = os.environ.get("ENABLE_LLM_EVAL", "false").lower() == "true"

pipeline = EvaluatePipeline(
    enable_llm_eval=enable_llm,
    llm_eval_limit=int(os.environ.get("LLM_EVAL_LIMIT", "3"))
)

# Run with: ENABLE_LLM_EVAL=true python scripts/evaluate/evaluate_pipeline.py
```

---

## Summary

✅ **LLM evaluation is now fully integrated into evaluate_pipeline.py**

Key features:
- **Flexible**: Can be enabled/disabled via parameters
- **Scalable**: Can limit evaluation to N tests per experiment
- **Complete**: Captures 5 quality dimensions + overall score
- **Robust**: Failures don't stop the pipeline
- **Informative**: Best LLM results shown alongside traditional metrics

**Recommended workflow**:
1. Start with `enable_llm_eval=True, llm_eval_limit=3` for testing
2. Use full evaluation (`llm_eval_limit=None`) for final comparisons
3. Analyze both traditional and LLM metrics for comprehensive insights
