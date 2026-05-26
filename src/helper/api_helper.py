#src/api_helper.py
# Helper function to call the OpenRouter API for DST prediction.
# Handles API requests, responses, and error handling for model calls.

import os
import requests
import json

# Load environmental variables from .env file (store API key safely)
from dotenv import load_dotenv
load_dotenv()

# API Key
OPENROUTER_API_KEY = your_api_key_here
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Raise an error if the API key is not set
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

def call_model_api(prompt: str, model_id: str) -> str:
    # Calls the OpenRouter API with the given prompt and model ID, and returns the response content.
    # Takes prompt and model_id as input, constructs the API request, and handles the response.
    # Returns the content of the model's response as a string. Raises errors for API failures or unexpected response formats.

    # HTTP headers and payload for the API request
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Request payload with system instructions and user prompt
    payload = {
        "model": model_id,
        "messages": [
            # System instruction to set the behaviour of the model
            {"role": "system", "content": "You are a dialogue state tracker (DST) for MultiWOZ 2.2. Output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt}
            
        ],
        "temperature": 0.0, # Controls randomness: 0.0 = deterministic, 1.0 = very random
        "max_tokens": 256
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")
    
    data = response.json()

    # Extract the content from the API response, handling potential errors in the response format
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response format: {json.dumps(data, indent=2)}") from e