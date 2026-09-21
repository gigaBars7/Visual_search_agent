import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


AGENT_URL = "http://agent:8001/test/action"
AGENT_WORKING_FOLDER_URL = "http://agent:8001/working-folder"

app = FastAPI(title="Test API")


class ChatRequestScheme(BaseModel):
    message: str


class WorkingFolderRequestScheme(BaseModel):
    path: str = "."


@app.post("/test/relay")
async def relay_to_agent(request: ChatRequestScheme):
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(AGENT_URL, json={"message": request.message})
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Agent service is unavailable") from error

    return {"answer": response.json()["agent_result"]}


@app.post("/working-folder")
async def relay_working_folder(request: WorkingFolderRequestScheme):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                AGENT_WORKING_FOLDER_URL,
                json=request.model_dump(),
            )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Agent service is unavailable") from error

    if response.is_error:
        raise HTTPException(status_code=response.status_code)

    return response.json()
