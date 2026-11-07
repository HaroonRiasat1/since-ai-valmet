from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from splat_objects.scene import SceneFrame


def backproject_views(
    render_root: Path,
    frame: SceneFrame,
    cfg: dict[str, Any],
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """World-scan points and unit normals from rendered depth (subsampled by stride)."""
    bim = cfg["bim"]
    scale = float(bim.get("units_to_m", 1.0))
    depth_scale = float(cfg["render"]["depth_png_scale"])
    alpha_min = float(bim.get("surface_alpha_min", 0.95))
    voxel = float(bim.get("surface_voxel_m", 0.02))
    cams = json.loads((render_root / "cameras.json").read_text(encoding="utf-8"))
    pts_list: list[np.ndarray] = []
    nrm_list: list[np.ndarray] = []

    for rec in cams[:: max(1, stride)]:
        dep_path = render_root / rec["depth"]
        alp_path = render_root / rec["alpha"]
        if not dep_path.is_file() or not alp_path.is_file():
            continue
        depth = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED).astype(np.float64) / depth_scale
        alpha = cv2.imread(str(alp_path), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        h, w = depth.shape
        R = np.asarray(rec["R_w2c"], dtype=np.float64)
        t = np.asarray(rec["t_w2c"], dtype=np.float64)
        fx, fy, cx, cy = rec["fx"], rec["fy"], rec["cx"], rec["cy"]
        ys, xs = np.mgrid[0:h:4, 0:w:4]
        d = depth[ys, xs]
        a = alpha[ys, xs]
        ok = (a >= alpha_min) & (d > 0.05) & (d < 50)
        if not np.any(ok):
            continue
        u = xs[ok].astype(np.float64)
        v = ys[ok].astype(np.float64)
        z = d[ok]
        x_c = (u - cx) * z / fx
        y_c = (v - cy) * z / fy
        pc = np.stack([x_c, y_c, z], axis=1)
        pw = pc @ R.T + t
        pts_list.append(pw)
        # normal from depth gradient on sparse grid
        dzdx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        nx = -dzdx[ys, xs][ok] / fx * z
        ny = -dzdy[ys, xs][ok] / fy * z
        nz = np.ones_like(z)
        n_c = np.stack([nx, ny, nz], axis=1)
        n_w = n_c @ R.T
        n_w /= np.linalg.norm(n_w, axis=1, keepdims=True) + 1e-9
        nrm_list.append(n_w)

    if not pts_list:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pts = np.concatenate(pts_list, axis=0)
    nrms = np.concatenate(nrm_list, axis=0)
    return _voxel_downsample(pts, nrms, voxel)


def _voxel_downsample(pts: np.ndarray, nrms: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) == 0:
        return pts, nrms
    idx = np.floor(pts / voxel).astype(np.int64)
    keys: dict[tuple[int, ...], list[int]] = {}
    for i, k in enumerate(map(tuple, idx)):
        keys.setdefault(k, []).append(i)
    out_p, out_n = [], []
    for ix in keys.values():
        out_p.append(pts[ix].mean(axis=0))
        n = nrms[ix].mean(axis=0)
        n /= np.linalg.norm(n) + 1e-9
        out_n.append(n)
    return np.asarray(out_p), np.asarray(out_n)


def label_points(
    frame: SceneFrame,
    pts: np.ndarray,
    nrms: np.ndarray,
    elements: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    scale = float(cfg["bim"].get("units_to_m", 1.0))
    h = frame.heights(pts)
    up = frame.up
    horiz = np.abs(nrms @ up) < 0.35
    floor = (np.abs(h) < 0.03) & (nrms @ up > 0.7)
    ceil_min = float(cfg["bim"].get("wall_height_max_m", 1.9)) + 0.15
    ceiling = (h > ceil_min) & (nrms @ up < -0.7)
    wall_h = (h >= float(cfg["bim"].get("wall_height_min_m", 0.3))) & (
        h <= float(cfg["bim"].get("wall_height_max_m", 1.9))
    )
    wall = horiz & wall_h
    margin = 0.05
    inside_obj = np.zeros(len(pts), dtype=bool)
    for el in elements:
        lo, hi = np.asarray(el["lo_m"]), np.asarray(el["hi_m"])
        # convert element local box to scan for masking - approximate via local frame
        c = np.asarray(el["box_local"]["center_m"])
        # points in local
        L = frame.local(pts) * scale
        ok = np.all(L >= lo - margin, axis=1) & np.all(L <= hi + margin, axis=1)
        inside_obj |= ok
    wall &= ~inside_obj
    ab = frame.local(pts) * scale
    ia, ib = frame.floor_index(ab[:, 0], ab[:, 1])
    ok_idx = (ia >= 0) & (ia < frame.floor_mask.shape[0]) & (ib >= 0) & (ib < frame.floor_mask.shape[1])
    near_floor = np.zeros(len(pts), dtype=bool)
    near_floor[ok_idx] = frame.floor_mask[ia[ok_idx], ib[ok_idx]]
    wall &= near_floor | (h < 0.1)
    return {"floor": floor, "ceiling": ceiling, "wall": wall, "heights": h}
