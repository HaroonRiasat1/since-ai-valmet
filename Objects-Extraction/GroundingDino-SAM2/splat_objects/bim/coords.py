from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np

from splat_objects.scene import SceneFrame


def units_to_m(cfg: dict[str, Any]) -> float:
    return float(cfg.get("bim", {}).get("units_to_m", 1.0))


def scan_to_local(frame: SceneFrame, xyz: np.ndarray, scale: float) -> np.ndarray:
    """Scan coordinates -> gravity-local metres (a, b, h)."""
    return frame.local(np.asarray(xyz, dtype=np.float64)) * scale


def local_to_scan(frame: SceneFrame, abh_m: np.ndarray, scale: float) -> np.ndarray:
    return frame.world(np.asarray(abh_m, dtype=np.float64) / scale)


def aabb_scan_to_local(frame: SceneFrame, lo: list[float], hi: list[float], scale: float) -> dict[str, Any]:
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    pts = np.stack(
        [
            [lo_a[0], lo_a[1], lo_a[2]],
            [hi_a[0], lo_a[1], lo_a[2]],
            [hi_a[0], hi_a[1], lo_a[2]],
            [lo_a[0], hi_a[1], lo_a[2]],
            [lo_a[0], lo_a[1], hi_a[2]],
            [hi_a[0], lo_a[1], hi_a[2]],
            [hi_a[0], hi_a[1], hi_a[2]],
            [lo_a[0], hi_a[1], hi_a[2]],
        ],
        axis=0,
    )
    L = scan_to_local(frame, pts, scale)
    lo_m, hi_m = L.min(axis=0), L.max(axis=0)
    return {
        "lo_m": lo_m.tolist(),
        "hi_m": hi_m.tolist(),
        "center_m": (0.5 * (lo_m + hi_m)).tolist(),
        "size_m": (hi_m - lo_m).tolist(),
        "corners_m": L.tolist(),
    }


def oriented_box_local(frame: SceneFrame, record: dict[str, Any], scale: float) -> dict[str, Any]:
    """Record boxes are in scan coordinates; return the oriented box in local metres (a, b, h)."""
    box = record["box"]
    corners = scan_to_local(frame, np.asarray(box["corners"], dtype=np.float64), scale)
    e01, e12, e04 = corners[1] - corners[0], corners[2] - corners[1], corners[4] - corners[0]
    center = corners.mean(axis=0)
    size = np.array([np.linalg.norm(e01[:2]), np.linalg.norm(e12[:2]), abs(e04[2])])
    yaw = math.degrees(math.atan2(e01[1], e01[0]))
    return {
        "center_m": center.tolist(),
        "size_m": size.tolist(),
        "yaw_deg": float(yaw),
        "corners_m": corners.tolist(),
        "bottom_height_m": float(corners[:, 2].min()),
        "top_height_m": float(corners[:, 2].max()),
    }


def placement_matrix_from_box(box: dict[str, Any]) -> np.ndarray:
    """4x4 SI matrix: local X along footprint width, Z up, origin at box center."""
    c = np.asarray(box["center_m"], dtype=np.float64)
    yaw = math.radians(float(box.get("yaw_deg", 0.0)))
    cx, sx = math.cos(yaw), math.sin(yaw)
    x_axis = np.array([cx, sx, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    m = np.eye(4)
    m[:3, 0] = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    m[:3, 1] = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    m[:3, 2] = z_axis
    m[:3, 3] = c
    return m


def stable_guid(key: str) -> str:
    import ifcopenshell.guid

    raw = uuid.uuid5(uuid.NAMESPACE_URL, f"since-ai/bim/{key}")
    return ifcopenshell.guid.compress(raw.hex)


def class_spec(cfg: dict[str, Any], class_text: str) -> dict[str, Any]:
    m = cfg.get("bim", {}).get("class_map", {})
    key = class_text.strip().lower()
    return dict(m.get(key) or m.get("default") or {"ifc": "IfcBuildingElementProxy", "family": "loose", "shape": "box"})


def asset_class_spec(cfg: dict[str, Any], asset_class: str) -> dict[str, Any]:
    m = cfg.get("bim", {}).get("asset_class_ifc", {})
    return dict(m.get(asset_class) or class_spec(cfg, "default"))
