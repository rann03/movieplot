import requests
import json
import os

API_KEY = os.getenv("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")

def get_llm_response(prompt, model="meta-llama/llama-3.2-1b-instruct:free"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=15
        )
        response.raise_for_status()
        completion = response.json()
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM request failed: {e}")
        return None