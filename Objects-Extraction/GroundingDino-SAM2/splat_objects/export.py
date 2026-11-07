from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from splat_objects.cameras import project_points
from splat_objects.ply_io import PlyGaussian, write_binary_gaussian_ply

SH_C0 = 0.28209479177387814
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def write_subset(ply: PlyGaussian, indices: np.ndarray, path: Path) -> None:
    write_binary_gaussian_ply(path, ply, indices=np.sort(np.asarray(indices, dtype=np.int64)))


def recolored(ply: PlyGaussian, groups: list[tuple[np.ndarray, np.ndarray]]) -> PlyGaussian:
    """Copy with each (indices, rgb01) group painted a flat, view-independent color."""
    rows = ply.raw_rows.copy()
    rest = [n for n in ply.property_names if n.startswith("f_rest_")]
    for idx, rgb in groups:
        dc = (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0
        for k in range(3):
            rows[f"f_dc_{k}"][idx] = dc[k]
        for name in rest:
            rows[name][idx] = 0.0
    return replace(ply, raw_rows=rows)


def draw_box(bgr: np.ndarray, cam_rec: dict, corners: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> bool:
    uv, z = project_points(cam_rec, np.asarray(corners))
    if np.any(z <= 0.05):
        return False
    for i, j in BOX_EDGES:
        cv2.line(bgr, tuple(int(v) for v in uv[i]), tuple(int(v) for v in uv[j]), color, thickness, cv2.LINE_AA)
    return True
