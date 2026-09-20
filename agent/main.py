from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Test Agent")


class AgentRequest(BaseModel):
    message: str


@app.post("/test/action")
def test_action(request: AgentRequest) -> dict[str, str]:
    return {"agent_result": f'\u041e\u0442\u0432\u0435\u0442 \u043d\u0430 "{request.message}"'}
