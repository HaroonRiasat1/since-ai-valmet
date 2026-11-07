import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_REPO))

from splat_objects.bim.structure import build_structure, _nearest_wall
from splat_objects.config import load_config
from splat_objects.scene import SceneFrame


class TestBimStructure(unittest.TestCase):
    def test_synthetic_room_four_walls(self) -> None:
        cfg = load_config(_REPO / "config_all_objects.yaml")
        scene_id = cfg["scene_id"]
        frame = SceneFrame.load(_REPO / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
        pts = np.zeros((0, 3))
        labels = {"floor": np.array([], dtype=bool), "ceiling": np.array([], dtype=bool), "wall": np.array([], dtype=bool), "heights": np.array([])}
        structure = build_structure(frame, [], pts, labels, cfg)
        self.assertEqual(len(structure["walls"]), 4)
        self.assertGreater(structure["ceiling_height_m"], 2.0)

    def test_door_snaps_wall_index(self) -> None:
        walls = [{"p0_m": [0, 0], "p1_m": [5, 0]}, {"p0_m": [5, 0], "p1_m": [5, 4]}]
        i = _nearest_wall(np.array([2.5, 0.1]), walls)
        self.assertEqual(i, 0)
