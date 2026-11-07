"""Apply a user layout (objects dragged/rotated in the web editor) to the BIM elements."""
from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from splat_objects.bim.coords import local_to_scan, scan_to_local
from splat_objects.scene import SceneFrame


def rot_y(theta: float) -> np.ndarray:
    """Same matrix three.js uses for object.rotation.y = theta."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def make_transform(entry: dict[str, Any]):
    pivot = np.asarray(entry["pivot"], dtype=np.float64)
    delta = np.asarray(entry["delta"], dtype=np.float64)
    r = rot_y(float(entry.get("yaw_rad", 0.0)))

    def apply(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return (p - pivot) @ r.T + pivot + delta

    return apply


def apply_layout(
    elements: list[dict[str, Any]],
    layout: dict[str, Any],
    guid_map: dict[str, str],
    frame: SceneFrame,
    scale: float,
    src_dir: Path,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Return moved copies of the elements; moved meshes are rewritten under out_dir/meshes."""
    moves = layout.get("objects", layout)
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    n_moved = 0
    for el in elements:
        el2 = copy.deepcopy(el)
        key = el.get("tag") or el.get("unconfirmed_tag") or f"{el['class']}:{el['id']}"
        entry = moves.get(guid_map.get(key, ""))
        rel = (el2.get("mesh") or {}).get("path")
        if entry is not None:
            n_moved += 1
            tf = make_transform(entry)
            if rel and (src_dir / rel).is_file():
                z = np.load(src_dir / rel)
                v_scan = local_to_scan(frame, z["vertices"].astype(np.float64), scale)
                v_new = scan_to_local(frame, tf(v_scan), scale)
                np.savez_compressed(out_dir / rel, vertices=v_new.astype(np.float32), faces=z["faces"], colours=z["colours"])
            _move_box(el2, tf, frame, scale)
        elif rel and (src_dir / rel).is_file():
            shutil.copy2(src_dir / rel, out_dir / rel)
        out.append(el2)
    return out, n_moved


def _move_box(el: dict[str, Any], tf, frame: SceneFrame, scale: float) -> None:
    box = el["box_local"]
    c = np.asarray(box["center_m"], dtype=np.float64)
    yaw = math.radians(float(box.get("yaw_deg", 0.0)))
    p1 = c + np.array([math.cos(yaw), math.sin(yaw), 0.0])
    q0 = scan_to_local(frame, tf(local_to_scan(frame, c, scale)), scale)
    q1 = scan_to_local(frame, tf(local_to_scan(frame, p1, scale)), scale)
    box["center_m"] = q0.tolist()
    box["yaw_deg"] = math.degrees(math.atan2(q1[1] - q0[1], q1[0] - q0[0]))
    if box.get("corners_m"):
        corners = np.asarray(box["corners_m"], dtype=np.float64)
        moved = scan_to_local(frame, tf(local_to_scan(frame, corners, scale)), scale)
        box["corners_m"] = moved.tolist()
        box["bottom_height_m"] = float(moved[:, 2].min())
        box["top_height_m"] = float(moved[:, 2].max())
