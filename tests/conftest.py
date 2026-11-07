import base64
import json

import numpy as np
import pytest

from backend.routing import sprite_sizes
from backend.scene import Scene


def _encode(array):
    return base64.b64encode(np.asarray(array, dtype=np.uint8).tobytes()).decode()


def make_scene(tmp_path):
    """A small room: a wall with one gap splits the floor, and one asset sits on each side."""
    fine = np.ones((40, 40), dtype=np.uint8)
    fine[20, :] = 0
    fine[20, 30:34] = 1
    clear = np.where(fine, 6, 0)
    wall = np.zeros((10, 10), dtype=bool)   # cell grid: the same wall, with a gap three cells wide (z 1.4 to 2.0)
    wall[5, :] = True
    wall[5, 7:10] = False
    sizes = sprite_sizes(wall)
    items = [
        {"id": "obj_a", "tag": "CV-001", "center": [0.3, 0, 0.3], "size": [0.1, 0.1, 0.1], "state": "degraded", "health": 50,
         "asset_class": "CONTROL_VALVE", "issue": "drift", "urgency": "high", "matched": True},
        {"id": "obj_b", "tag": "P-001", "center": [1.7, 0, 0.3], "size": [0.1, 0.1, 0.1], "state": "healthy", "health": 96,
         "asset_class": "PROCESS_PUMP", "area": "Utility Island", "issue": "none", "urgency": "routine", "matched": True},
    ]
    data = {
        "frame": {"up_axis": 1, "down_sign": 1, "floor": 1.0}, "items": items, "copilot": {}, "target_id": "obj_a",
        "path": [[0.1, 1.0, 0.1]],
        "navgrid": {"origin": [0.0, 0.0], "cell": 0.05, "horiz": [0, 2], "shape": [40, 40], "walk": _encode(fine), "clear": _encode(clear)},
        "cellgrid": {"cells": 10, "step": 0.2, "center_x": 1.0, "center_z": 1.0, "z_up": True, "sizes": _encode(sizes)},
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def scene(tmp_path):
    return Scene(make_scene(tmp_path))
