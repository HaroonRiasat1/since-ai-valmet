from __future__ import annotations

import itertools
import json
import unittest

import numpy as np

from splat_objects.config import classes_from_prompts, load_config, repo_root, slug
from splat_objects.instances import Instance, cross_class_pair
from splat_objects.viz import PALETTE_BGR

ALL_OBJECTS = repo_root() / "config_all_objects.yaml"


class TestPromptClasses(unittest.TestCase):
    def test_all_objects_config_has_one_class_per_prompt(self) -> None:
        cfg = load_config(ALL_OBJECTS)
        prompts = json.loads((repo_root() / cfg["classes_file"]).read_text(encoding="utf-8"))
        self.assertEqual([c["prompt"] for c in cfg["classes"]], prompts)
        self.assertEqual(len({c["key"] for c in cfg["classes"]}), len(prompts))
        self.assertEqual(len({slug(c["display"]) for c in cfg["classes"]}), len(prompts))
        self.assertEqual(cfg["scene_id"], "equipment_room_all_objects")

    def test_overrides_defaults_and_inherited_settings(self) -> None:
        cfg = load_config(ALL_OBJECTS)
        by_key = {c["key"]: c for c in cfg["classes"]}
        self.assertEqual(by_key["chair"]["box_threshold"], 0.35)
        self.assertTrue(by_key["chair"]["floor_standing"])
        self.assertEqual(by_key["gauge"]["box_threshold"], 0.27)
        self.assertFalse(by_key["gauge"]["floor_standing"])
        self.assertEqual(cfg["instances"]["cross_class_dup_iou"], 0.5)
        base = load_config()
        self.assertEqual(cfg["instances"]["min_views"], base["instances"]["min_views"])
        self.assertEqual(cfg["render"], base["render"])
        self.assertNotIn("cross_class_dup_iou", base["instances"])
        self.assertEqual(len(base["classes"]), 5)

    def test_prompts_with_slashes_get_safe_names(self) -> None:
        (c,) = classes_from_prompts(["pipe/fittings"], {"box_threshold": 0.3}, {})
        self.assertEqual((c["key"], c["display"], c["prompt"]), ("pipe_fittings", "Pipe/fittings", "pipe/fittings"))
        self.assertEqual(slug(c["display"]), "Pipe_fittings")
        self.assertEqual(slug("Washing Machine"), "Washing_Machine")

    def test_bad_prompt_lists_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            classes_from_prompts(["pipe/fittings", "pipe fittings"], {}, {})
        with self.assertRaises(ValueError):
            classes_from_prompts(["valve"], {}, {"vavle": {"box_threshold": 0.2}})


class TestPalette(unittest.TestCase):
    def test_enough_distinct_colors(self) -> None:
        self.assertGreaterEqual(len(PALETTE_BGR), len(load_config(ALL_OBJECTS)["classes"]))
        self.assertEqual(PALETTE_BGR[0], (60, 200, 60))
        closest = min(np.linalg.norm(np.subtract(a, b)) for a, b in itertools.combinations(PALETTE_BGR, 2))
        self.assertGreaterEqual(closest, 40.0)


def _instance(class_id: int, members: range, confidence: float = 0.5) -> Instance:
    m = np.array(members, dtype=np.int64)
    return Instance(class_id=class_id, proposals=[], members=m, ratios=np.ones(len(m), np.float32), confidence=confidence)


class TestCrossClassPair(unittest.TestCase):
    def test_same_object_under_two_prompts_is_found(self) -> None:
        insts = [_instance(0, range(0, 100)), _instance(1, range(10, 110)), _instance(2, range(500, 520))]
        i, j, iou = cross_class_pair(insts, 1000, 0.5)
        self.assertEqual((i, j), (0, 1))
        self.assertAlmostEqual(iou, 90 / 110, places=5)

    def test_same_class_and_small_parts_are_left_alone(self) -> None:
        same_class = [_instance(0, range(0, 100)), _instance(0, range(0, 100))]
        self.assertIsNone(cross_class_pair(same_class, 1000, 0.5))
        part_of_large = [_instance(0, range(0, 1000)), _instance(1, range(0, 50))]
        self.assertIsNone(cross_class_pair(part_of_large, 1000, 0.5))


if __name__ == "__main__":
    unittest.main()
