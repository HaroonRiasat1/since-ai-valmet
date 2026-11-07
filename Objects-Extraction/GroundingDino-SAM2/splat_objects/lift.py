from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx3d.cameras import Camera
from mlx3d.splatting import render_gaussian_features

from splat_objects.render_mlx import SplatScene

W0_LOG_SCALE = 16.0


def blend_weights(scene: SplatScene, cam: Camera, channels: np.ndarray, antialias: bool) -> np.ndarray:
    """w[i, k] = sum_p channels[p, k] * alpha_i(p) * T_i(p), via one VJP of feature compositing."""
    c = int(channels.shape[-1])

    def composite(feat: mx.array) -> mx.array:
        return render_gaussian_features(
            cam,
            scene.means,
            scene.quats,
            scene.scales,
            scene.opacities,
            feat,
            background=mx.zeros((c,), dtype=mx.float32),
            normalize=False,
            antialias=antialias,
        )["features"]

    _, (w,) = mx.vjp(
        composite,
        [mx.zeros((scene.n, c), dtype=mx.float32)],
        [mx.array(np.ascontiguousarray(channels, dtype=np.float32))],
    )
    mx.eval(w)
    return np.array(w)


def encode_w0(w0: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.log2(1.0 + np.maximum(w0, 0.0)) * W0_LOG_SCALE), 0, 255).astype(np.uint8)


def w0_lut() -> np.ndarray:
    return (np.exp2(np.arange(256, dtype=np.float64) / W0_LOG_SCALE) - 1.0).astype(np.float32)
