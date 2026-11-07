"""The built scene: objects, asset evidence and navigation grids, loaded once at start."""
import json
from pathlib import Path

from backend.routing import CellGrid, FineGrid, RouteError


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
        self.grids = {"fine": FineGrid(data["navgrid"], floor)}
        if "cellgrid" in data:
            self.grids["cells"] = CellGrid(data["cellgrid"], floor)

    def grid(self, name):
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
