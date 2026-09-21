import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from model.loader import load_model


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SYSTEM_PROMPT = """
Ты — агент, который отвечает только на основе результатов доступных tools.

Правила:
1. Для выполнения задачи всегда используй подходящий tool.
2. Не выдумывай результат.
3. Если tool вернул ошибку, передай пользователю эту ошибку и не пытайся
   выполнить задачу самостоятельно.
4. Если подходящего tool нет, честно сообщи, что не можешь выполнить запрос.
""".strip()


class AgentState(MessagesState):
    working_folder: str | None
    collection_name: str | None
    indexed: bool


class AgentRequestScheme(BaseModel):
    message: str


class WorkingFolderRequestScheme(BaseModel):
    path: str = "."


def resolve_working_folder(path):
    photos_root = Path(os.getenv("PHOTOS_ROOT", "/photos")).resolve()
    requested_path = Path(path)

    if requested_path.is_absolute():
        raise ValueError("Path must be relative to the photos root")

    working_folder = (photos_root / requested_path).resolve()
    try:
        working_folder.relative_to(photos_root)
    except ValueError as error:
        raise ValueError("Path must be inside the photos root") from error

    if not working_folder.is_dir():
        raise ValueError("Folder does not exist")

    return str(working_folder)


def list_images(state: AgentState):
  working_folder = state["working_folder"]
  return list(
      path
      for path in Path(working_folder).iterdir()
      if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
  )


def create_llm():
    return ChatOpenAI(
        model=os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-8b"),
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://host.docker.internal:1234/v1"),
        api_key="lm-studio",
        temperature=0,
        timeout=180,
        max_retries=0,
    )


def build_graph(llm, tools):
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: AgentState):
        response = await llm_with_tools.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    return builder.compile()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    client = MultiServerMCPClient(
        {
            "test_tools": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["mcp_server.py"],
            }
        }
    )
    tools = await client.get_tools()
    app.state.graph = build_graph(create_llm(), tools)
    app.state.agent_state = {
        "messages": [],
        "working_folder": None,
        "collection_name": None,
        "indexed": False,
    }
    app.state.state_lock = asyncio.Lock()
    yield


app = FastAPI(title="Test Agent", lifespan=lifespan)


@app.post("/working-folder")
async def set_working_folder(request: WorkingFolderRequestScheme) -> dict[str, object]:
    try:
        working_folder = resolve_working_folder(request.path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    async with app.state.state_lock:
        app.state.agent_state = {
            **app.state.agent_state,
            "working_folder": working_folder,
            "collection_name": None,
            "indexed": False,
        }

    return {
        "working_folder": working_folder
    }


@app.post("/invoke")
async def invoke(request: AgentRequestScheme):
    async with app.state.state_lock:
        current_state = app.state.agent_state
        request_state = {
            **current_state,
            "messages": [
                *current_state["messages"],
                HumanMessage(content=request.message),
            ],
        }

        try:
            result = await app.state.graph.ainvoke(request_state)
        except Exception as error:
            raise HTTPException(status_code=503, detail="Agent is unavailable") from error

        app.state.agent_state = {
            "messages": result["messages"],
            "working_folder": result.get("working_folder"),
            "collection_name": result.get("collection_name"),
            "indexed": result.get("indexed", False),
        }

    return {"agent_result": str(result["messages"][-1].content)}
