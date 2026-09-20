import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


AGENT_URL = os.getenv("AGENT_URL", "http://agent:8001/test/action")

app = FastAPI(title="Test API")


class ChatRequest(BaseModel):
    message: str


@app.post("/test/relay")
async def relay_to_agent(request: ChatRequest) -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(AGENT_URL, json={"message": request.message})
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Agent service is unavailable") from error

    return {"answer": response.json()["agent_result"]}
