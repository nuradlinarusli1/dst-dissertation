import json
import re
from typing import Any, Dict

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse the FIRST JSON object found in an LLM output string.

    Handles:
    - Extra text before/after JSON
    - Markdown code fences ```json ... ```
    - Multiple JSON objects in one output (parses the first)
    - Whitespace/newlines

    Raises:
        ValueError: if no JSON object can be found/parsed.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model output is empty or not a string.")
    
    cleaned = text.replace("``json", "").replace("``", "").strip()

    start = cleaned.find("{")
    if start == -1:
        match = re.search(r"\{", cleaned)
        
        if not match:
            raise ValueError("No JSON object in model output.")
        start = match.start()

    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(cleaned[start:])
    except json.JSONDecoderError as e:
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse JSON from model output: {e}") from e
    
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object/dict (got different JSON type).")
    
    return obj

"""
def validate_schema(predicted: dict, expected_structure: dict):
    if set(predicted.keys()) != set(expected_structure.keys()):
        raise ValueError("Domain mismatch")

    for domain in expected_structure:
        if set(predicted[domain].keys()) != set(expected_structure[domain].keys()):
            raise ValueError("Slot mismatch")

    return True
"""
