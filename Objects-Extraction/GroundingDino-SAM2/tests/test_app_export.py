from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.app_export import (
    app_box,
    app_object_id,
    class_confidence,
    load_app_checker,
    same_class_overlaps,
    to_app_object,
)
from splat_objects.config import load_config, repo_root

APP_ROOT = (repo_root() / "../..").resolve()


class TestAppExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()
        cls.ao = load_app_checker(APP_ROOT)
        path = repo_root() / "outputs" / cls.cfg["scene_id"] / "objects" / "objects.json"
        if path.is_file():
            cls.records = json.loads(path.read_text(encoding="utf-8"))
            layer = json.loads((APP_ROOT / "data" / "semantic_layer.json").read_text(encoding="utf-8"))
            cls.layer_objs = layer["objects"]
        else:
            cls.records = []
            cls.layer_objs = []

    def test_app_object_id(self) -> None:
        self.assertEqual(app_object_id({"class_key": "valve", "name": "Valves_13"}), "valve_13")
        self.assertEqual(app_object_id({"class_key": "washing_machine", "name": "Washing_Machine_00"}), "washing_machine_00")
        self.assertEqual(app_object_id({"class_key": "chair", "name": "Chair_00"}), "chair_00")

    def test_box_rounded_to_four_decimals(self) -> None:
        rec = {
            "box": {
                "aabb_min": [0.9363420337438584, 0.26480737030506135, -2.1002293467521667],
                "aabb_max": [1.4945218920707701, 1.1377048909664154, -1.5511645376682284],
            }
        }
        box = app_box(rec)
        for key in ("min", "max"):
            for v in box[key]:
                self.assertEqual(v, round(v, 4))

    def test_class_confidence_buckets(self) -> None:
        high = {"n_detected_views": 43, "detect_ratio": 0.98}
        med = {"n_detected_views": 10, "detect_ratio": 0.6}
        low = {"n_detected_views": 2, "detect_ratio": 1.0}
        self.assertEqual(class_confidence(high, self.cfg), "high")
        self.assertEqual(class_confidence(med, self.cfg), "medium")
        self.assertEqual(class_confidence(low, self.cfg), "low")

    def test_overlap_note_on_synthetic_layer_hit(self) -> None:
        record = {
            "name": "Valves_07",
            "class_key": "valve",
            "confidence": 0.27,
            "n_detected_views": 3,
            "n_visible_views": 11,
            "n_gaussians": 634,
            "detect_ratio": 0.27,
            "box": {
                "aabb_min": [1.0, 0.0, 0.0],
                "aabb_max": [2.0, 1.0, 1.0],
            },
        }
        layer = [
            {
                "id": "cls_123",
                "class": "control valve",
                "box": {"min": [1.1, 0.1, 0.1], "max": [1.9, 0.9, 0.9]},
                "asset": {"master": {"tag": "CV-006"}},
            }
        ]
        obj = to_app_object(record, self.cfg, layer)
        self.assertIn("Overlaps cls_123", obj["note"])
        self.assertIn("CV-006", obj["note"])

    def test_same_class_valve_matches_control_valve(self) -> None:
        record = {
            "class_key": "valve",
            "box": {"aabb_min": [0, 0, 0], "aabb_max": [1, 1, 1]},
        }
        layer = [{"id": "x", "class": "control valve", "box": {"min": [0.05, 0.05, 0.05], "max": [0.95, 0.95, 0.95]}}]
        hits = same_class_overlaps(record, layer, 0.5)
        self.assertEqual(len(hits), 1)

    @unittest.skipUnless(APP_ROOT.joinpath("data/semantic_layer.json").is_file(), "Since-AI layer missing")
    def test_all_pipeline_objects_pass_app_normalise(self) -> None:
        self.assertGreater(len(self.records), 0)
        for i, rec in enumerate(self.records):
            raw = to_app_object(rec, self.cfg, self.layer_objs)
            norm = self.ao.normalise(raw, f"test #{i}")
            self.assertIn(raw["id"], norm["id"])
            self.assertEqual(norm["class"], raw["class"])


if __name__ == "__main__":
    unittest.main()
