#src/api_helper.py

import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it to your OpenRouter API key.")

def call_model_api(prompt: str, model_id: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a dialogue state tracker (DST) for MultiWOZ 2.2. Output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt}
            
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")
    
    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response format: {json.dumps(data, indent=2)}") from e