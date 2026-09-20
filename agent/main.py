import os

from fastapi import FastAPI
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


app = FastAPI(title="Test Agent")

llm = ChatOpenAI(
    model=os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-8b"),
    base_url=os.getenv("LM_STUDIO_BASE_URL", "http://host.docker.internal:1234/v1"),
    api_key="lm-studio",
    temperature=0,
    timeout=180,
    max_retries=0,
)


class AgentRequest(BaseModel):
    message: str


@app.post("/test/action")
async def test_action(request: AgentRequest) -> dict[str, str]:
    try:
        response = await llm.ainvoke([HumanMessage(content=request.message)])
    except Exception as error:
        raise HTTPException(status_code=503, detail="LM Studio is unavailable") from error

    return {"agent_result": str(response.content)}
