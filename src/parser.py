# src/parser.py
# Utility for extracting and parsing JSON objects from LLM output strings.
# Used to process model responses during DST evaluation.

import json
import re
from typing import Any, Dict

def extract_json(text: str) -> Dict[str, Any]:
    
    # Extract and parse the FIRST JSON object found in an LLM output string.
    # Handles: markdown code fences, extra text before/after JSON, multiple JSON objects (parses first), and whitespace/newlines.
    # Raises ValueError if no valid JSON object can be found or parsed.

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model output is empty or not a string.")
    
    cleaned = text.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    if start == -1:
        match = re.search(r"\{.*?\}", cleaned)
        
        if not match:
            raise ValueError("No JSON object in model output.")
        start = match.start()

    decoder = json.JSONDecoder()

    try:
        obj, end = decoder.raw_decode(cleaned[start:])
    except json.JSONDecodeError as e:
        
        # If JSON parsing fails, attempt to extract the first JSON-like substring and parse it.
        brace_match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        
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
