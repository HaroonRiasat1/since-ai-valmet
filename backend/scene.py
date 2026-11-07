"""The built scene: objects, asset evidence and navigation grids, loaded once at start."""
import json
from pathlib import Path

from backend.routing import CellGrid, RouteError


class Scene:
    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf8"))
        self.data = data
        self.items = data["items"]
        self.copilot = data["copilot"]
        self.target_id = data["target_id"]
        self.by_tag = {i["tag"]: i for i in self.items if i.get("tag")}
        self.by_id = {i["id"]: i for i in self.items}
        floor = data["frame"]["floor"] * data["frame"]["down_sign"]
        if "cellgrid" not in data:
            raise RouteError("scene.json has no cell grid: rebuild the scene (README, 'Rebuilding the data')")
        self.grids = {"cells": CellGrid(data["cellgrid"], floor)}   # the only route grid

    def grid(self, name):
        if name == "fine":   # the committed web page still asks for the removed A* grid by default
            name = "cells"
        if name not in self.grids:
            raise RouteError(f"unknown grid {name}")
        return self.grids[name]

    def item(self, key):
        """Find an object by asset tag or object id."""
        found = self.by_tag.get(key) or self.by_id.get(key)
        if found is None:
            raise RouteError(f"unknown asset or object {key}")
        return found

    def assets(self):
        return list(self.by_tag.values())

    def default_start(self):
        return self.data["path"][0]

    def remove(self, key: str) -> dict:
        """Drop an object from the live scene and free its footprint on every grid."""
        item = self.by_id.get(key) or self.by_tag.get(key)
        if item is None:
            raise RouteError(f"unknown asset or object {key}")
        oid = item["id"]
        self.items = [i for i in self.items if i["id"] != oid]
        self.by_id.pop(oid, None)
        if item.get("tag"):
            self.by_tag.pop(item["tag"], None)
        for grid in self.grids.values():
            grid.free_box(item)
        return item
