import json
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_REPO))

from splat_objects.bim.ifc_writer import write_ifc
from splat_objects.config import load_config
from splat_objects.scene import SceneFrame


class TestBimIfc(unittest.TestCase):
    def test_minimal_writer(self) -> None:
        cfg = load_config(_REPO / "config_all_objects.yaml")
        scene_id = cfg["scene_id"]
        bim_dir = _REPO / "outputs" / scene_id / "bim"
        if not (bim_dir / "elements_fitted.json").is_file():
            self.skipTest("run stages 8–10 first")
        frame = SceneFrame.load(_REPO / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
        elements = json.loads((bim_dir / "elements_fitted.json").read_text())["elements"][:3]
        structure = json.loads((bim_dir / "structure.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            ifc_path = Path(td) / "t.ifc"
            guid_path = Path(td) / "guid_map.json"
            result = write_ifc(cfg, _REPO, scene_id, frame, elements, structure, ifc_path, guid_path)
            self.assertTrue(ifc_path.stat().st_size > 1000)
            self.assertIn("guid_map", result)
