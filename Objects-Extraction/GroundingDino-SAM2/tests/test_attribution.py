from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx3d.cameras import Camera
from mlx3d.splatting import GaussianModel

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.lift import blend_weights, encode_w0, w0_lut
from splat_objects.render_mlx import SplatScene, render_rgb


def tiny_scene(means: np.ndarray, opacity: float, scale: float) -> SplatScene:
    n = len(means)
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1.0
    sh = np.zeros((n, 16, 3), np.float32)
    return SplatScene(
        means=mx.array(means.astype(np.float32)),
        quats=mx.array(quats),
        scales=mx.full((n, 3), scale),
        opacities=mx.full((n,), opacity),
        sh=mx.array(sh),
        sh_degree=3,
    )


class TestAttribution(unittest.TestCase):
    def test_weights_sum_to_rendered_alpha(self) -> None:
        mx.random.seed(0)
        model = GaussianModel.from_points(mx.random.normal((300, 3)) * 0.4, sh_degree=0)
        scene = SplatScene(
            means=model.params["means"],
            quats=model.params["quats"],
            scales=model.scales_act,
            opacities=mx.sigmoid(mx.full((300,), 1.5)),
            sh=mx.zeros((300, 16, 3)),
            sh_degree=3,
        )
        cam = Camera.look_at(eye=(0, 0, -3), at=(0, 0, 0), width=96, height=72, fov=60)
        _, alpha = render_rgb(scene, cam, antialias=True)
        mask = np.zeros((72, 96), np.float32)
        mask[10:50, 20:70] = 1.0
        w = blend_weights(scene, cam, np.stack([np.ones_like(mask), mask], axis=-1), antialias=True)
        self.assertAlmostEqual(float(w[:, 0].sum()), float(alpha.sum()), delta=1e-3 * float(alpha.sum()))
        self.assertAlmostEqual(float(w[:, 1].sum()), float((alpha * mask).sum()), delta=1e-3 * float(alpha.sum()))

    def test_occluded_gaussian_gets_no_weight(self) -> None:
        cam = Camera.look_at(eye=(0, 0, 0), at=(0, 0, 1), width=64, height=64, fov=40, znear=0.01)
        ones = np.ones((64, 64, 1), np.float32)
        hidden = np.array([[0.0, 0.0, 3.0]])
        alone = blend_weights(tiny_scene(hidden, opacity=0.9, scale=0.03), cam, ones, antialias=False)[0, 0]
        front = SplatScene(
            means=mx.array(np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], np.float32)),
            quats=mx.array(np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32)),
            scales=mx.array(np.array([[0.6] * 3, [0.03] * 3], np.float32)),
            opacities=mx.array(np.array([0.99, 0.9], np.float32)),
            sh=mx.zeros((2, 16, 3)),
            sh_degree=3,
        )
        w = blend_weights(front, cam, ones, antialias=False)
        self.assertGreater(alone, 1.0)
        self.assertLess(w[1, 0], 0.03 * alone)

    def test_w0_quantization_roundtrip(self) -> None:
        w0 = np.array([0.0, 0.5, 3.0, 120.0, 5000.0])
        back = w0_lut()[encode_w0(w0)]
        np.testing.assert_allclose(back[1:], w0[1:], rtol=0.05)
        self.assertEqual(back[0], 0.0)


if __name__ == "__main__":
    unittest.main()
