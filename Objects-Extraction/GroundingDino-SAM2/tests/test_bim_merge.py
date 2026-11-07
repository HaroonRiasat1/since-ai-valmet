import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_REPO))

from splat_objects.bim.merge import _same_family
from splat_objects.config import load_config


class TestBimMerge(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(_REPO / "config_all_objects.yaml")

    def test_same_family_valves(self) -> None:
        self.assertTrue(_same_family(self.cfg, "control valve", "handwheel valve"))
        self.assertFalse(_same_family(self.cfg, "pipe", "control valve"))
        self.assertFalse(_same_family(self.cfg, "box", "chair"))
