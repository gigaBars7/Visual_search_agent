import os

import httpx


api_url = os.getenv("API_URL", "http://api:8000/test/relay")
working_folder_url = "http://api:8000/working-folder"


def send_message(message):
    try:
        message = message.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        response = httpx.post(api_url, json={"message": message}, timeout=180.0)
        response.raise_for_status()
        return response.json()["answer"]
    except (httpx.HTTPError, UnicodeError):
        return "API is unavailable."


def select_working_folder():
    print("Working folder (relative to Pictures, '.' for root): ", end="", flush=True)
    path = input().strip()

    try:
        response = httpx.post(
            working_folder_url,
            json={"path": path or "."},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        agent_response = getattr(error, "response", None)
        detail = (
            agent_response.json().get("detail")
        )
        print(f"Error: {detail}")
        return select_working_folder()

    print(f"Working folder: {response.json()['working_folder']}")


select_working_folder()

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
