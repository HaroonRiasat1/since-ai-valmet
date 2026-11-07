"""HTTP API and static web app. Run: uvicorn backend.app:app"""
import logging
from dataclasses import asdict
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import agent, copilot, llm, vision
from backend.config import get_settings
from backend.routing import RouteError, nearest_next
from backend.scene import Scene

log = logging.getLogger(__name__)
Position = list[float]
MAX_IMAGE = 12_000_000   # characters of base64, about 9 MB


class RouteRequest(BaseModel):
    grid: str = "fine"
    position: Position | None = Field(default=None, min_length=3, max_length=3)
    start_cell: int | None = None
    target: str | None = None
    target_cell: int | None = None


class RoundRequest(BaseModel):
    grid: str = "fine"
    position: Position | None = Field(default=None, min_length=3, max_length=3)


class CopilotRequest(BaseModel):
    tag: str
    question: str = Field(default="", max_length=500)
    nearby: list[str] = []
    provider: str = "lmstudio"
    model: str | None = None


class ChatLine(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    grid: str = "fine"
    position: Position | None = Field(default=None, min_length=3, max_length=3)
    provider: str = "lmstudio"
    model: str | None = None
    history: list[ChatLine] = Field(default=[], max_length=12)


class LabelRequest(BaseModel):
    box_id: str
    image: str = Field(max_length=MAX_IMAGE)
    provider: str = "lmstudio"
    model: str | None = None


class DetectRequest(BaseModel):
    run: str
    view_id: str
    image: str = Field(max_length=MAX_IMAGE)
    pose: list[float] = Field(min_length=16, max_length=16)
    proj: list[float] = Field(min_length=16, max_length=16)
    width: int | None = None
    height: int | None = None
    provider: str = "lmstudio"
    model: str | None = None


@lru_cache
def get_scene() -> Scene:
    return Scene(get_settings().scene_path)


def start_position(scene, grid, request):
    if request.start_cell is not None:
        if grid.name != "cells":
            raise RouteError("start cells need the cell grid")
        index = grid.index_of(request.start_cell)
        if grid.sizes[index] == 0:
            raise RouteError(f"cell {request.start_cell} is blocked")
        return grid.world_of(index)
    return request.position or scene.default_start()


app = FastAPI(title="Field-Service Navigator")


@app.exception_handler(RouteError)
async def route_error(_, error: RouteError):
    return JSONResponse(status_code=422, content={"error": str(error)})


@app.exception_handler(llm.ProviderError)
async def provider_error(_, error: llm.ProviderError):
    log.warning("provider failed: %s", error)
    return JSONResponse(status_code=502, content={"error": str(error)})


@app.exception_handler(ValueError)
async def bad_value(_, error: ValueError):
    return JSONResponse(status_code=400, content={"error": str(error)})


@app.get("/api/health")
def health():
    scene = get_scene()
    return {"ok": True, "objects": len(scene.items), "assets": len(scene.by_tag), "grids": list(scene.grids)}


@app.get("/api/grids")
def grids():
    return [{"name": g.name, "label": g.label} for g in get_scene().grids.values()]


@app.post("/api/route")
def route(request: RouteRequest):
    scene = get_scene()
    grid = scene.grid(request.grid)
    position = start_position(scene, grid, request)
    if request.target_cell is not None:
        if grid.name != "cells":
            raise RouteError("target cells need the cell grid")
        return {**asdict(grid.route_to_cell(position, request.target_cell)), "target": None}
    if not request.target:
        raise HTTPException(status_code=422, detail="give a target or a target_cell")
    item = scene.item(request.target)
    return {**asdict(grid.route(position, item)), "target": item.get("tag") or item["id"]}


@app.post("/api/route/round")
def inspection_round(request: RoundRequest):
    scene = get_scene()
    grid = scene.grid(request.grid)
    todo = [a for a in scene.assets() if a["state"] != "healthy"]
    legs = nearest_next(grid, request.position or scene.default_start(), todo)
    return {"legs": [{"target": item["tag"], **asdict(leg)} for item, leg in legs]}


@app.get("/api/providers")
def list_providers():
    return [llm.status(p) for p in llm.providers().values()]


@app.post("/api/copilot/query")
def copilot_query(request: CopilotRequest):
    scene = get_scene()
    if request.tag not in scene.copilot:
        raise HTTPException(status_code=400, detail=f"unknown asset {request.tag}")
    return copilot.diagnose(scene, llm.get_provider(request.provider), request.tag, request.question.strip(), request.nearby, request.model)


@app.post("/api/agent/chat")
def agent_chat(request: AgentRequest):
    scene = get_scene()
    history = [line.model_dump() for line in request.history]
    return agent.respond(scene, scene.grid(request.grid), request.position or scene.default_start(), request.message.strip(),
                         llm.get_provider(request.provider), request.model, history)


@app.post("/api/vlm/label")
def vlm_label(request: LabelRequest):
    return vision.label(llm.get_provider(request.provider), get_settings().web_dir / "data", request.box_id, request.image, request.model)


@app.post("/api/vlm/detect")
def vlm_detect(request: DetectRequest):
    return vision.detect(llm.get_provider(request.provider), get_settings().web_dir / "data", request.model_dump())


@app.get("/api/assets")
def assets():
    keys = ("tag", "id", "asset_class", "state", "health", "issue", "urgency", "center", "matched")
    return [{k: a.get(k) for k in keys} for a in get_scene().assets()]


app.mount("/", StaticFiles(directory=get_settings().web_dir, html=True), name="web")
