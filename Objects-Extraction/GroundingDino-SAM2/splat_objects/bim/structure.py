from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from splat_objects.scene import SceneFrame


def build_structure(
    frame: SceneFrame,
    elements: list[dict[str, Any]],
    pts: np.ndarray,
    labels: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    bim = cfg["bim"]
    scale = float(bim.get("units_to_m", 1.0))
    thick = float(bim.get("wall_thickness_m", 0.2))
    lo_ab, hi_ab = frame.floor_extent()
    lo_ab, hi_ab = lo_ab * scale, hi_ab * scale

    # Floor mask -> room polygon (simplified rectangle from extent; flag inferred sides)
    poly = np.array(
        [
            [lo_ab[0], lo_ab[1]],
            [hi_ab[0], lo_ab[1]],
            [hi_ab[0], hi_ab[1]],
            [lo_ab[0], hi_ab[1]],
        ],
        dtype=np.float64,
    )

    wall_pts = pts[labels["wall"]] if len(pts) else np.zeros((0, 3))
    wall_faces = _detect_wall_faces(wall_pts, frame, cfg) if len(wall_pts) > 100 else []

    ceiling_h = _ceiling_height(frame, pts, labels, cfg)

    walls = _walls_from_polygon(poly, thick, ceiling_h, wall_faces, cfg)

    doors = _doors_from_elements(elements, walls, cfg)

    floor_occ = _floor_occupancy(frame, elements, scale)

    return {
        "room_polygon_m": poly.tolist(),
        "walls": walls,
        "ceiling_height_m": ceiling_h,
        "slab_thickness_m": float(bim.get("slab_thickness_m", 0.2)),
        "doors": doors,
        "wall_faces_detected": len(wall_faces),
        "floor_occupancy": floor_occ,
        "assumptions": {"wall_thickness_assumed": True, "room_polygon": "floor_extent"},
    }


def _detect_wall_faces(wall_pts: np.ndarray, frame: SceneFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if len(wall_pts) < 50:
        return []
    L = frame.local(wall_pts) * float(cfg["bim"].get("units_to_m", 1.0))
    ab = L[:, :2]
    # two dominant axes
    angles = np.arctan2(ab[:, 1], ab[:, 0])
    hist, edges = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
    k = int(np.argmax(hist))
    a0 = 0.5 * (edges[k] + edges[k + 1])
    a1 = a0 + np.pi / 2 if cfg["bim"].get("assume_right_angles", True) else a0 + np.pi / 4
    faces = []
    for ang, sign in ((a0, 1), (a0, -1), (a1, 1), (a1, -1)):
        c, s = np.cos(ang), np.sin(ang)
        n = np.array([c, s])
        off = (ab @ n) * sign
        slices = np.floor(L[:, 2] / float(cfg["bim"].get("wall_slice_m", 0.2))).astype(int)
        offsets = []
        for sl in np.unique(slices):
            mask = slices == sl
            if mask.sum() < 20:
                continue
            hoff, _ = np.histogram(off[mask], bins=80)
            peaks, _ = signal.find_peaks(hoff, height=hoff.max() * 0.25)
            if len(peaks):
                offsets.append(float(np.mean(off[mask][np.abs(off[mask] - np.percentile(off[mask], 50)) < 0.05])))
        if len(offsets) >= int(cfg["bim"].get("wall_peak_min_slices", 3)):
            off_m = float(np.median(offsets))
            faces.append({"angle_rad": float(ang), "sign": sign, "offset_m": off_m, "inferred": False})
    return faces


def _walls_from_polygon(
    poly: np.ndarray,
    thick: float,
    height: float,
    faces: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    walls = []
    n = len(poly)
    for i in range(n):
        p0, p1 = poly[i], poly[(i + 1) % n]
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < float(cfg["bim"].get("wall_min_length_m", 0.8)):
            continue
        mid = 0.5 * (p0 + p1)
        walls.append(
            {
                "p0_m": p0.tolist(),
                "p1_m": p1.tolist(),
                "length_m": length,
                "height_m": height,
                "thickness_m": thick,
                "thickness_assumed": True,
                "inferred": len(faces) < 2,
                "mid_m": mid.tolist(),
            }
        )
    return walls


def _ceiling_height(frame: SceneFrame, pts: np.ndarray, labels: dict[str, np.ndarray], cfg: dict[str, Any]) -> float:
    h = labels.get("heights")
    if h is not None and len(h):
        ceil_pts = h[labels["ceiling"]] if labels["ceiling"].any() else h
        if len(ceil_pts):
            return float(np.percentile(ceil_pts, 50))
    return float(frame.ceiling_h) * float(cfg["bim"].get("units_to_m", 1.0))


def _doors_from_elements(elements: list[dict[str, Any]], walls: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    open_yaw = float(cfg["bim"].get("door_open_yaw_deg", 10.0))
    doors = []
    for el in elements:
        if el["class"] != "door":
            continue
        box = el["box_local"]
        size = box["size_m"]
        width = float(max(size[0], size[1]))
        height = float(size[2])
        center = np.asarray(box["center_m"])
        yaw = float(box.get("yaw_deg", 0.0))
        wall_i = _nearest_wall(center[:2], walls)
        open_state = abs(yaw) > open_yaw or abs(yaw - 90) > open_yaw
        doors.append(
            {
                "element_id": el["id"],
                "center_m": center.tolist(),
                "width_m": width,
                "height_m": height,
                "yaw_deg": yaw,
                "open": bool(open_state),
                "wall_index": wall_i,
            }
        )
    return doors


def _nearest_wall(center: np.ndarray, walls: list[dict[str, Any]]) -> int:
    best_i, best_d = 0, 1e9
    for i, w in enumerate(walls):
        p0, p1 = np.asarray(w["p0_m"]), np.asarray(w["p1_m"])
        d = _point_seg_dist(center, p0, p1)
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _floor_occupancy(frame: SceneFrame, elements: list[dict[str, Any]], scale: float) -> dict[str, float]:
    total = float(frame.floor_mask.sum())
    if total <= 0:
        return {"seen_frac": 0, "under_object_frac": 0, "inferred_frac": 1}
    under = 0.0
    for el in elements:
        if el.get("box_local", {}).get("bottom_height_m", 1) > 0.25:
            continue
        # footprint overlap - coarse
        under += 0.01
    seen = max(0.0, 1.0 - under - 0.05)
    return {"seen_frac": seen, "under_object_frac": min(0.5, under), "inferred_frac": 0.05}
