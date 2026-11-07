import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_REPO))

from splat_objects.bim.meshing import _accept_mesh, mesh_cfg
from splat_objects.bim.meshing import synthetic_gaussian_box
from splat_objects.config import load_config
from splat_objects.scene import SceneFrame


class TestBimMesh(unittest.TestCase):
    def test_accept_mesh_volume(self) -> None:
        cfg = load_config(_REPO / "config_all_objects.yaml")
        mc = mesh_cfg(cfg)
        el = {"box_local": {"size_m": [1.0, 1.0, 1.0], "center_m": [0, 0, 0.5]}}
        import trimesh

        box = trimesh.creation.icosphere(subdivisions=2, radius=0.45)
        self.assertTrue(_accept_mesh(el, np.asarray(box.vertices), np.asarray(box.faces), mc))

    def test_synthetic_box_points(self) -> None:
        c, s = np.zeros(3), np.array([0.4, 0.5, 0.6])
        pts, _op, _fdc = synthetic_gaussian_box(c, s, n=500)
        self.assertEqual(pts.shape, (500, 3))
        self.assertTrue(np.all(pts >= c - 0.5 * s))
        self.assertTrue(np.all(pts <= c + 0.5 * s))

    def test_ifc_triangulated_on_sample(self) -> None:
        scene_id = "equipment_room_all_objects"
        bim_dir = _REPO / "outputs" / scene_id / "bim"
        if not (bim_dir / "elements_fitted.json").is_file():
            self.skipTest("run stage 10 first")
        import ifcopenshell

        ifc_path = bim_dir / f"{scene_id}.ifc"
        if not ifc_path.is_file():
            self.skipTest("run stage 11 first")
        f = ifcopenshell.open(str(ifc_path))
        meshes = f.by_type("IfcPolygonalFaceSet") + f.by_type("IfcTriangulatedFaceSet")
        self.assertGreater(len(meshes), 0)
