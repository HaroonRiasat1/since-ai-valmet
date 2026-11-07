import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_REPO))

from splat_objects.bim.coords import local_to_scan, scan_to_local, stable_guid
from splat_objects.config import load_config
from splat_objects.scene import SceneFrame


class TestBimCoords(unittest.TestCase):
    def test_round_trip(self) -> None:
        cfg = load_config(_REPO / "config_all_objects.yaml")
        scene_id = cfg["scene_id"]
        frame = SceneFrame.load(_REPO / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
        scale = float(cfg["bim"]["units_to_m"])
        xyz = np.array([[1.0, 0.5, -0.2]])
        abh = scan_to_local(frame, xyz, scale)
        back = local_to_scan(frame, abh, scale)
        self.assertTrue(np.allclose(xyz, back, atol=1e-6))

    def test_stable_guid(self) -> None:
        self.assertEqual(stable_guid("tag:CV-002"), stable_guid("tag:CV-002"))
        self.assertNotEqual(stable_guid("tag:CV-002"), stable_guid("tag:CV-003"))
