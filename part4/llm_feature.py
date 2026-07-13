import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

MODEL_NAME = "openrouter/auto"


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("Status Code:", response.status_code)
        return None

    return response.json()["choices"][0]["message"]["content"]

result = call_llm(
    system_prompt="You are a helpful assistant.",
    user_prompt="Reply with only the word: hello"
)

print(result)


