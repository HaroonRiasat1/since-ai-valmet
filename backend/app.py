"""HTTP API and static web app. Run: uvicorn backend.app:app"""
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.routing import RouteError, nearest_next
from backend.scene import Scene

Position = list[float]


class RouteRequest(BaseModel):
    grid: str = "fine"
    position: Position | None = Field(default=None, min_length=3, max_length=3)
    start_cell: int | None = None
    target: str | None = None
    target_cell: int | None = None


class RoundRequest(BaseModel):
    grid: str = "fine"
    position: Position | None = Field(default=None, min_length=3, max_length=3)


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
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"error": str(error)})


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


@app.get("/api/assets")
def assets():
    keys = ("tag", "id", "asset_class", "state", "health", "issue", "urgency", "center", "matched")
    return [{k: a.get(k) for k in keys} for a in get_scene().assets()]


app.mount("/", StaticFiles(directory=get_settings().web_dir, html=True), name="web")
