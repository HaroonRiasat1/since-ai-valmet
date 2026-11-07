from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
from mlx3d.cameras import Camera

from splat_objects.scene import SceneFrame, unit

_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def view_direction(frame: SceneFrame, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit forward vector; negative pitch looks down toward the floor."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    horiz = math.cos(y) * frame.e1 + math.sin(y) * frame.e2
    return unit(math.cos(p) * horiz + math.sin(p) * frame.up)


def look_camera(
    eye: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_y_deg: float,
    znear: float,
    zfar: float = 100.0,
) -> Camera:
    eye = np.asarray(eye, dtype=np.float64)
    at = eye + unit(forward)
    return Camera.look_at(
        eye=tuple(float(v) for v in eye),
        at=tuple(float(v) for v in at),
        up=tuple(float(v) for v in unit(up)),
        fov=float(fov_y_deg),
        width=int(width),
        height=int(height),
        znear=float(znear),
        zfar=float(zfar),
    )


def camera_record(cam: Camera) -> dict[str, Any]:
    R = np.array(cam.R, dtype=np.float64)
    t = np.array(cam.t, dtype=np.float64)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return {
        "width": int(cam.width),
        "height": int(cam.height),
        "fx": float(cam.fx),
        "fy": float(cam.fy),
        "cx": float(cam.cx),
        "cy": float(cam.cy),
        "znear": float(cam.znear),
        "zfar": float(cam.zfar),
        "K": [[float(cam.fx), 0.0, float(cam.cx)], [0.0, float(cam.fy), float(cam.cy)], [0.0, 0.0, 1.0]],
        "R_w2c": R.tolist(),
        "t_w2c": t.tolist(),
        "c2w_opencv": c2w.tolist(),
        "position": c2w[:3, 3].tolist(),
        "forward": R[2].tolist(),
    }


def camera_from_record(rec: dict[str, Any]) -> Camera:
    return Camera(
        R=mx.array(np.array(rec["R_w2c"], dtype=np.float32)),
        t=mx.array(np.array(rec["t_w2c"], dtype=np.float32)),
        fx=float(rec["fx"]),
        fy=float(rec["fy"]),
        cx=float(rec["cx"]),
        cy=float(rec["cy"]),
        width=int(rec["width"]),
        height=int(rec["height"]),
        znear=float(rec["znear"]),
        zfar=float(rec["zfar"]),
    )


def project_points(rec: dict[str, Any], xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coords (N, 2) and camera z (N,) for world points using a stored record."""
    R = np.array(rec["R_w2c"], dtype=np.float64)
    t = np.array(rec["t_w2c"], dtype=np.float64)
    pc = np.asarray(xyz, dtype=np.float64) @ R.T + t
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = rec["fx"] * pc[:, 0] / zs + rec["cx"]
    v = rec["fy"] * pc[:, 1] / zs + rec["cy"]
    return np.stack([u, v], axis=1), z


def nerfstudio_frame(rec: dict[str, Any], file_path: str) -> dict[str, Any]:
    c2w_gl = np.array(rec["c2w_opencv"], dtype=np.float64) @ _CV_TO_GL
    return {"file_path": file_path, "transform_matrix": c2w_gl.tolist()}
