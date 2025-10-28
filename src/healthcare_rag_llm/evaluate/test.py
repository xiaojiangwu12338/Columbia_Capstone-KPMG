import json

def t(response: str, metric_name: str):
    """
    Parse JSON response from LLM, with fallback handling.

    Args:
        response: LLM response string
        metric_name: Name of the metric being evaluated

    Returns:
        Parsed JSON dict or error dict
    """
    try:
        # Try to find JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)

            # Validate required fields
            if "score" not in result:
                raise ValueError("Missing 'score' field")

            # Ensure score is in valid range
            result["score"] = max(0.0, min(1.0, float(result["score"])))

            return result
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:
        print(f"Warning: Failed to parse {metric_name} response: {e}")
        print(f"Response was: {response[:200]}...")
        return {
            "score": 0.0,
            "reasoning": f"Failed to parse LLM response: {str(e)}",
            "error": str(e),
            "raw_response": response[:500]
        }
    
word = "{\n    \"score\": 0.85,\n    \"reasoning\": \"All claims are cited to the correct document (mu_no4_apr24_pr.pdf), and each statement has a corresponding citation. The page numbers (3 and 5) are plausible for the content described. However, the citation format does not match the specified standard, and the available chunks do not list specific pages, so page-level verification cannot be confirmed.\",\n    \"issues\": [\n        \"Citation format deviates from the expected format; used an em dash and spelled-o"

t(response=word,metric_name="test")


