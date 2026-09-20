import os

import httpx


api_url = os.getenv("API_URL", "http://api:8000/test/relay")


def send_message(message):
    try:
        response = httpx.post(api_url, json={"message": message}, timeout=10.0)
        response.raise_for_status()
        return response.json()["answer"]
    except httpx.HTTPError:
        return "API is unavailable."


while True:
    try:
        message = input("User: ").strip()
    except EOFError:
        break

    if message.lower() in {"exit", "quit", "/exit", "/quit"}:
        break
    if not message:
        continue

    print(f"Agent: {send_message(message)}")
