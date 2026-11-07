import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_REPO))

from splat_objects.bim.layout import make_transform, rot_y


class TestBimLayout(unittest.TestCase):
    def test_translation_only(self) -> None:
        tf = make_transform({"pivot": [1, 2, 3], "delta": [0.5, 0, -0.5], "yaw_rad": 0.0})
        self.assertTrue(np.allclose(tf(np.array([1.0, 2.0, 3.0])), [1.5, 2.0, 2.5]))

    def test_rotation_about_pivot_keeps_pivot(self) -> None:
        tf = make_transform({"pivot": [1, 2, 3], "delta": [0, 0, 0], "yaw_rad": 1.2})
        self.assertTrue(np.allclose(tf(np.array([1.0, 2.0, 3.0])), [1, 2, 3]))
        p = tf(np.array([2.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(np.linalg.norm(p - np.array([1, 2, 3]))), 1.0)

    def test_matches_three_rotation_y(self) -> None:
        self.assertTrue(np.allclose(rot_y(np.pi / 2) @ np.array([1.0, 0, 0]), [0, 0, -1]))
