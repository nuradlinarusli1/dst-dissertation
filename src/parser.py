import json
import re

def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from model output.
    Handles cases where extra text is added.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object in model output.")
    
    json_str = match.group(0)
    return json.loads(json_str)

"""
def validate_schema(predicted: dict, expected_structure: dict):
    if set(predicted.keys()) != set(expected_structure.keys()):
        raise ValueError("Domain mismatch")

    for domain in expected_structure:
        if set(predicted[domain].keys()) != set(expected_structure[domain].keys()):
            raise ValueError("Slot mismatch")

    return True
"""
