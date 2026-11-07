from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
from mlx3d.cameras import Camera
from mlx3d.splatting import GaussianModel, render_gaussian_depth, render_gaussians


@dataclass
class SplatScene:
    """Activated 3DGS parameters (opacities in [0, 1], scales as std devs)."""

    means: mx.array
    quats: mx.array
    scales: mx.array
    opacities: mx.array
    sh: mx.array
    sh_degree: int

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def load(cls, path: Path) -> SplatScene:
        m = GaussianModel.load_ply(str(path))
        return cls(
            means=m.params["means"],
            quats=m.params["quats"],
            scales=m.scales_act,
            opacities=m.opacities_act,
            sh=m.sh,
            sh_degree=int(m.sh_degree),
        )

    def subset(self, idx: np.ndarray) -> SplatScene:
        ii = mx.array(np.asarray(idx, dtype=np.int32))
        return SplatScene(
            means=self.means[ii],
            quats=self.quats[ii],
            scales=self.scales[ii],
            opacities=self.opacities[ii],
            sh=self.sh[ii],
            sh_degree=self.sh_degree,
        )


def render_rgb(
    scene: SplatScene, cam: Camera, antialias: bool, background: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray]:
    out = render_gaussians(
        cam,
        scene.means,
        scene.quats,
        scene.scales,
        scene.opacities,
        sh=scene.sh,
        sh_degree=scene.sh_degree,
        background=mx.array(background, dtype=mx.float32),
        antialias=antialias,
    )
    mx.eval(out["image"], out["alpha"])
    return np.array(out["image"]), np.array(out["alpha"])


def render_depth(scene: SplatScene, cam: Camera, antialias: bool) -> np.ndarray:
    out = render_gaussian_depth(cam, scene.means, scene.quats, scene.scales, scene.opacities, antialias=antialias)
    mx.eval(out["depth"])
    return np.array(out["depth"])


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_rgb(path: Path, rgb_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_depth16(path: Path, depth: np.ndarray, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.nan_to_num(depth.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    cv2.imwrite(str(path), np.clip(np.round(d * scale), 0, 65535).astype(np.uint16))


def save_alpha8(path: Path, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), to_uint8(alpha))


def sharpness(rgb_u8: np.ndarray) -> float:
    g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (512, 384), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())
