"""Route planning on the numbered 60 x 60 cell grid, with Dijkstra: tight cells cost more, so routes keep to open floor."""
import base64
import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage



SPRITE_CELLS = 3    # the walker's square footprint in grid cells (3 x 0.15 m = 0.45 m): routes only use cells where it fits
REACH = 0.9         # metres: a goal further than this from the asset does not count as reaching it


def sprite_sizes(blocked):
    """Widest odd square sprite, in cells, that fits centred on each cell (0 on blocked cells); the grid edge counts as
    blocked. Same result as pipeline/view_splat.py max_sprite_sizes, from a chessboard distance transform."""
    depth = ndimage.distance_transform_cdt(np.pad(~np.asarray(blocked, bool), 1), metric="chessboard")[1:-1, 1:-1]
    return np.where(depth > 0, 2 * depth - 1, 0)


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


class CellGrid:
    """The numbered 60 x 60 grid. Cell 1 is the top-left of the top view, numbered row by row."""

    name = "cells"
    label = "Cells 60 x 60 (Dijkstra)"

    def __init__(self, cellgrid, floor, sprite=SPRITE_CELLS):
        self.sprite = sprite   # footprint in cells; a cell is passable when sizes >= sprite
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

    def goal_for(self, item, cells, standoff=0.4):
        """Among cells, the one about standoff metres from the item's footprint, and its distance to it."""
        free = cells
        centres = self.center - self.span / 2 + (free + 0.5) * self.step
        half = np.array(item["size"])[[0, 2]] / 2
        gaps = np.maximum(np.abs(centres - np.array(item["center"])[[0, 2]]) - half, 0)
        distance = np.hypot(gaps[:, 0], gaps[:, 1])
        k = np.argmin(np.abs(distance - standoff) + (distance < 0.02) * 10)
        return tuple(int(v) for v in free[k]), float(distance[k])

    def passable(self):
        """Cells where the sprite's square footprint fits."""
        return self.sizes >= self.sprite

    def _too_narrow(self, what, gap=None):
        near = "" if gap is None else f"; the closest it gets is {gap:.1f} m"
        return RouteError(f"no walkable route for the sprite ({self.sprite} cells, {self.sprite * self.step:.2f} m wide) "
                          f"to {what}: the way is narrower than the sprite{near}")

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
                if 0 <= other[0] < self.cells and 0 <= other[1] < self.cells and self.sizes[other] >= self.sprite:
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

    def _starts(self, position, ok, areas):
        """Where the sprite can stand near position: passable cells within 1 m of the nearest one, by connected area."""
        cand = np.argwhere(ok)
        if not len(cand):
            raise self._too_narrow("anywhere")
        dist = np.hypot(*(cand - np.asarray(self.cell_of(position))).T) * self.step
        near, dist = cand[dist <= dist.min() + 1.0], dist[dist <= dist.min() + 1.0]
        best = {}
        for cell, d in zip(near, dist):
            area = areas[tuple(cell)]
            if area not in best or d < best[area][1]:
                best[area] = (tuple(int(v) for v in cell), d)
        return best   # area -> (start cell, distance)

    def route(self, position, item):
        """Route the sprite to the item, only through cells where it fits. Of the floor areas the sprite could be
        standing in, use the one that gets closest to the item."""
        ok = self.passable()
        areas, _ = ndimage.label(ok)
        options = []
        for area, (start, d) in self._starts(position, ok, areas).items():
            goal, gap = self.goal_for(item, np.argwhere(areas == area))
            options.append((gap > REACH, gap, d, start, goal))
        unreachable, gap, _, start, goal = min(options)
        if unreachable:
            raise self._too_narrow(item.get("tag") or item["id"], gap)
        return self.plan(start, goal)

    def route_to_cell(self, position, number):
        goal = self.index_of(number)
        if self.sizes[goal] == 0:
            raise RouteError(f"cell {number} is blocked")
        if self.sizes[goal] < self.sprite:
            raise RouteError(f"cell {number} is too narrow for the sprite ({self.sprite} cells wide, room for {self.sizes[goal]})")
        ok = self.passable()
        areas, _ = ndimage.label(ok)
        starts = self._starts(position, ok, areas)
        if areas[goal] not in starts:
            raise self._too_narrow(f"cell {number}")
        return self.plan(starts[areas[goal]][0], goal)

    def free_box(self, item, pad=0.03):
        """Re-open cell-grid tiles covered by an object's footprint."""
        center, size = np.asarray(item["center"], float), np.asarray(item["size"], float)
        low_x, high_x = center[0] - size[0] / 2 - pad, center[0] + size[0] / 2 + pad
        low_z, high_z = center[2] - size[2] / 2 - pad, center[2] + size[2] / 2 + pad
        for col in range(self.cells):
            for row in range(self.cells):
                pos = self.world_of((col, row))
                if low_x <= pos[0] <= high_x and low_z <= pos[2] <= high_z and self.sizes[col, row] == 0:
                    self.sizes[col, row] = 1
        self.sizes = sprite_sizes(self.sizes == 0)   # the freed floor changes how wide a sprite fits around it


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
