"""Route planning on the two navigation grids.

Fine grid: the 5 cm occupancy grid, planned with A* and shortened by line of sight.
Cell grid: the numbered 60 x 60 grid, planned with Dijkstra, with tight cells costing more.
"""
import base64
import heapq
from dataclasses import dataclass

import numpy as np

import navigator


class RouteError(ValueError):
    """The request cannot be routed: unknown cell, blocked cell or no way through."""


@dataclass
class Route:
    grid: str
    points: list[list[float]]
    length: float
    steps: int
    end_cell: int | None = None


def _decode(text, dtype=np.uint8):
    return np.frombuffer(base64.b64decode(text), dtype=dtype)


def _world(x, z, plane_y):
    return [round(float(x), 3), round(float(plane_y), 3), round(float(z), 3)]


def _length(points):
    pts = np.array(points)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


class FineGrid:
    """The 5 cm grid from navigator.py."""

    name = "fine"
    label = "Fine 5 cm (A*)"

    def __init__(self, navgrid, floor):
        shape = tuple(navgrid["shape"])
        self.floor = floor
        self.cell = navgrid["cell"]
        self.origin = np.array(navgrid["origin"])
        self.horiz = navgrid["horiz"]
        self.walk = _decode(navgrid["walk"]).reshape(shape).astype(bool)
        self.clear = _decode(navgrid["clear"]).reshape(shape).astype(float) * self.cell

    def cell_of(self, position):
        return tuple(np.floor((np.asarray(position, float)[self.horiz] - self.origin) / self.cell).astype(int))

    def world_of(self, cell):
        x, z = self.origin + (np.asarray(cell) + 0.5) * self.cell
        return _world(x, z, self.floor)

    def nearest_free(self, cell):
        free = np.argwhere(self.walk)
        if not len(free):
            raise RouteError("the fine grid has no walkable cells")
        return tuple(int(v) for v in free[np.argmin(((free - cell) ** 2).sum(axis=1))])

    def goal_for(self, item, standoff=0.4):
        free = np.argwhere(self.walk)
        centres = self.origin + (free + 0.5) * self.cell
        half = np.array(item["size"])[self.horiz] / 2
        gaps = np.maximum(np.abs(centres - np.array(item["center"])[self.horiz]) - half, 0)
        distance = np.hypot(gaps[:, 0], gaps[:, 1])
        return tuple(int(v) for v in free[np.argmin(np.abs(distance - standoff) + (distance < 0.05) * 10)])

    def plan(self, start, goal):
        raw = navigator.astar(self.walk, self.clear, start, goal)
        if raw is None:
            raise RouteError("no walkable route on the fine grid")
        points = [self.world_of(c) for c in navigator.simplify(self.walk, raw)]
        return Route(self.name, points, round(_length(points), 2), len(raw) - 1)

    def route(self, position, item):
        return self.plan(self.nearest_free(self.cell_of(position)), self.goal_for(item))


class CellGrid:
    """The numbered 60 x 60 grid. Cell 1 is the top-left of the top view, numbered row by row."""

    name = "cells"
    label = "Cells 60 x 60 (Dijkstra)"

    def __init__(self, cellgrid, floor):
        self.cells = cellgrid["cells"]
        self.step = cellgrid["step"]
        self.span = self.step * self.cells
        self.center = np.array([cellgrid["center_x"], cellgrid["center_z"]])
        self.z_up = cellgrid["z_up"]
        self.floor = floor
        self.sizes = _decode(cellgrid["sizes"]).reshape(self.cells, self.cells).astype(int)

    def index_of(self, number):
        if not 1 <= number <= self.cells ** 2:
            raise RouteError(f"cell {number} does not exist, cells run from 1 to {self.cells ** 2}")
        row, col = (number - 1) // self.cells, (number - 1) % self.cells
        return col, self.cells - 1 - row if self.z_up else row

    def number_of(self, cell):
        row = self.cells - 1 - cell[1] if self.z_up else cell[1]
        return int(row * self.cells + cell[0] + 1)

    def cell_of(self, position):
        p = np.asarray(position, float)[[0, 2]]
        return tuple(np.clip(np.floor((p - self.center + self.span / 2) / self.step).astype(int), 0, self.cells - 1))

    def world_of(self, cell):
        x, z = self.center - self.span / 2 + (np.asarray(cell) + 0.5) * self.step
        return _world(x, z, self.floor)

    def nearest_free(self, cell):
        free = np.argwhere(self.sizes > 0)
        if not len(free):
            raise RouteError("the cell grid has no free cells")
        return tuple(int(v) for v in free[np.argmin(((free - cell) ** 2).sum(axis=1))])

    def goal_for(self, item, standoff=0.4):
        free = np.argwhere(self.sizes > 0)
        centres = self.center - self.span / 2 + (free + 0.5) * self.step
        half = np.array(item["size"])[[0, 2]] / 2
        gaps = np.maximum(np.abs(centres - np.array(item["center"])[[0, 2]]) - half, 0)
        distance = np.hypot(gaps[:, 0], gaps[:, 1])
        return tuple(int(v) for v in free[np.argmin(np.abs(distance - standoff) + (distance < 0.02) * 10)])

    def plan(self, start, goal):
        best, parent, heap, order = {start: 0.0}, {start: None}, [(0.0, 0, start)], 0
        while heap:
            cost, _, cell = heapq.heappop(heap)
            if cell == goal:
                break
            if cost > best[cell]:
                continue
            x, z = cell
            for other in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
                if 0 <= other[0] < self.cells and 0 <= other[1] < self.cells and self.sizes[other] > 0:
                    total = cost + 1 + 1 / min(self.sizes[cell], self.sizes[other])
                    if total < best.get(other, np.inf):
                        best[other], parent[other] = total, cell
                        order += 1
                        heapq.heappush(heap, (total, order, other))
        if goal not in parent:
            raise RouteError("no walkable route on the cell grid")
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path.reverse()
        points = [self.world_of(c) for c in self._corners(path)]
        return Route(self.name, points, round(_length([self.world_of(c) for c in path]), 2), len(path) - 1, self.number_of(goal))

    @staticmethod
    def _corners(path):
        """Keep the cells where the path turns."""
        if len(path) < 3:
            return path
        kept = [path[0]]
        for before, cell, after in zip(path, path[1:], path[2:]):
            if (cell[0] - before[0], cell[1] - before[1]) != (after[0] - cell[0], after[1] - cell[1]):
                kept.append(cell)
        return kept + [path[-1]]

    def route(self, position, item):
        return self.plan(self.nearest_free(self.cell_of(position)), self.goal_for(item))

    def route_to_cell(self, position, number):
        goal = self.index_of(number)
        if self.sizes[goal] == 0:
            raise RouteError(f"cell {number} is blocked")
        return self.plan(self.nearest_free(self.cell_of(position)), goal)


def nearest_next(grid, position, items):
    """Visit items in nearest-next order. Returns (item, route) legs."""
    legs, todo, here = [], list(items), position
    while todo:
        options = []
        for item in todo:
            try:
                options.append((grid.route(here, item), item))
            except RouteError:
                continue
        if not options:
            break
        route, item = min(options, key=lambda pair: pair[0].length)
        legs.append((item, route))
        todo.remove(item)
        here = route.points[-1]
    return legs
