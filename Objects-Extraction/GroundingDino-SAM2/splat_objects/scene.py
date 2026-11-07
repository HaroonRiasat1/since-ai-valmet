from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from splat_objects.ply_io import PlyGaussian


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def disk(radius_cells: int) -> np.ndarray:
    r = max(int(radius_cells), 1)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= r * r


@dataclass
class SceneFrame:
    """Gravity-aligned frame: local coords (a, b, h) with h = height above the floor plane."""

    up: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    d: float
    ceiling_h: float
    top_h: float
    cell: float
    grid_origin: np.ndarray
    floor_mask: np.ndarray
    vox: float
    vox_origin: np.ndarray
    clearance: np.ndarray

    def local(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        return np.stack([xyz @ self.e1, xyz @ self.e2, xyz @ self.up + self.d], axis=-1)

    def world(self, abh: np.ndarray) -> np.ndarray:
        abh = np.asarray(abh, dtype=np.float64)
        return (
            abh[..., 0:1] * self.e1
            + abh[..., 1:2] * self.e2
            + (abh[..., 2:3] - self.d) * self.up
        )

    def heights(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz, dtype=np.float64) @ self.up + self.d

    def floor_index(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ia = np.floor((np.asarray(a) - self.grid_origin[0]) / self.cell).astype(np.int64)
        ib = np.floor((np.asarray(b) - self.grid_origin[1]) / self.cell).astype(np.int64)
        return ia, ib

    def interior_mask(self, margin_m: float) -> np.ndarray:
        r = int(round(margin_m / self.cell))
        if r <= 0:
            return self.floor_mask.copy()
        return ndimage.binary_erosion(self.floor_mask, structure=disk(r), border_value=0)

    def in_mask(self, mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ia, ib = self.floor_index(a, b)
        ok = (ia >= 0) & (ia < mask.shape[0]) & (ib >= 0) & (ib < mask.shape[1])
        out = np.zeros(np.shape(ia), dtype=bool)
        out[ok] = mask[ia[ok], ib[ok]]
        return out

    def clearance_at(self, abh: np.ndarray) -> np.ndarray:
        abh = np.atleast_2d(np.asarray(abh, dtype=np.float64))
        idx = np.floor((abh - self.vox_origin) / self.vox).astype(np.int64)
        dims = np.array(self.clearance.shape)
        ok = np.all((idx >= 0) & (idx < dims), axis=1)
        out = np.zeros(len(abh), dtype=np.float64)
        out[ok] = self.clearance[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out

    def floor_extent(self) -> tuple[np.ndarray, np.ndarray]:
        ii, jj = np.nonzero(self.floor_mask)
        lo = self.grid_origin + np.array([ii.min(), jj.min()]) * self.cell
        hi = self.grid_origin + (np.array([ii.max(), jj.max()]) + 1) * self.cell
        return lo, hi

    def meta(self) -> dict[str, Any]:
        lo, hi = self.floor_extent()
        return {
            "up": self.up.tolist(),
            "e1": self.e1.tolist(),
            "e2": self.e2.tolist(),
            "floor_plane": {"normal": self.up.tolist(), "d": self.d, "convention": "h = x.up + d"},
            "ceiling_height_m": self.ceiling_h,
            "top_height_m": self.top_h,
            "floor_extent_ab_min": lo.tolist(),
            "floor_extent_ab_max": hi.tolist(),
            "floor_area_m2": float(self.floor_mask.sum() * self.cell * self.cell),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            up=self.up,
            e1=self.e1,
            e2=self.e2,
            d=self.d,
            ceiling_h=self.ceiling_h,
            top_h=self.top_h,
            cell=self.cell,
            grid_origin=self.grid_origin,
            floor_mask=self.floor_mask,
            vox=self.vox,
            vox_origin=self.vox_origin,
            clearance=self.clearance.astype(np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> SceneFrame:
        z = np.load(path)
        return cls(
            up=z["up"],
            e1=z["e1"],
            e2=z["e2"],
            d=float(z["d"]),
            ceiling_h=float(z["ceiling_h"]),
            top_h=float(z["top_h"]),
            cell=float(z["cell"]),
            grid_origin=z["grid_origin"],
            floor_mask=z["floor_mask"].astype(bool),
            vox=float(z["vox"]),
            vox_origin=z["vox_origin"],
            clearance=z["clearance"],
        )


def _fit_plane_ransac(
    pts: np.ndarray,
    up_hint: np.ndarray,
    iters: int,
    thresh: float,
    max_tilt_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray]:
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best_n, best_d, best_count = up_hint.copy(), -float(np.median(pts @ up_hint)), -1
    for _ in range(iters):
        p0, p1, p2 = pts[rng.choice(len(pts), size=3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        if n @ up_hint < 0:
            n = -n
        if n @ up_hint < cos_tilt:
            continue
        d = -float(n @ p0)
        count = int(np.count_nonzero(np.abs(pts @ n + d) < thresh))
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    inliers = pts[np.abs(pts @ best_n + best_d) < thresh]
    c = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - c, full_matrices=False)
    n = vt[-1]
    if n @ up_hint < 0:
        n = -n
    return n, -float(n @ c), inliers


def estimate_frame(ply: PlyGaussian, cfg: dict[str, Any], seed: int = 0) -> SceneFrame:
    sc = cfg["scene"]
    rng = np.random.default_rng(seed)
    xyz = ply.xyz()
    vis = ply.opacity_sigmoid() >= float(sc["opacity_min"])
    P = xyz[vis]
    up_hint = unit(sc["up_hint"])
    bin_m = float(sc["hist_bin_m"])

    # Floor: densest height peak in the lowest 35% of the height range.
    s = P @ up_hint
    lo, hi = np.percentile(s, [0.2, 99.8])
    hist, edges = np.histogram(s, bins=np.arange(lo, hi + bin_m, bin_m))
    k = int(np.argmax(hist[: max(3, int(0.35 * len(hist)))]))
    s_floor = 0.5 * (edges[k] + edges[k + 1])
    band = P[np.abs(s - s_floor) < 0.06]
    if len(band) > 60_000:
        band = band[rng.choice(len(band), size=60_000, replace=False)]
    up, d, _ = _fit_plane_ransac(
        band,
        up_hint,
        iters=int(sc["floor_ransac_iters"]),
        thresh=float(sc["floor_ransac_thresh_m"]),
        max_tilt_deg=float(sc["floor_max_tilt_deg"]),
        rng=rng,
    )

    h_all = xyz @ up + d
    hv = h_all[vis]
    top_h = float(np.percentile(hv, 99.5))
    hist, edges = np.histogram(hv, bins=np.arange(0.5, top_h + bin_m, bin_m))
    start = int(0.6 * len(hist))
    k = start + int(np.argmax(hist[start:]))
    ceiling_h = float(0.5 * (edges[k] + edges[k + 1]))

    xw = np.array([1.0, 0.0, 0.0])
    if abs(xw @ up) > 0.9:
        xw = np.array([0.0, 0.0, 1.0])
    e1 = unit(xw - (xw @ up) * up)
    e2 = unit(np.cross(up, e1))

    # Floor-plan mask from floor-level splats.
    cell = float(sc["floor_cell_m"])
    A, B = P @ e1, P @ e2
    a_lo, a_hi = np.percentile(A, [0.5, 99.5])
    b_lo, b_hi = np.percentile(B, [0.5, 99.5])
    margin = 0.6
    grid_origin = np.array([a_lo - margin, b_lo - margin])
    na = int(np.ceil((a_hi - a_lo + 2 * margin) / cell))
    nb = int(np.ceil((b_hi - b_lo + 2 * margin) / cell))
    fl = np.abs(hv) < 0.06
    ia = np.floor((A[fl] - grid_origin[0]) / cell).astype(np.int64)
    ib = np.floor((B[fl] - grid_origin[1]) / cell).astype(np.int64)
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    counts = np.zeros((na, nb), dtype=np.int32)
    np.add.at(counts, (ia[ok], ib[ok]), 1)
    fm = counts >= 2
    fm = ndimage.binary_closing(fm, structure=disk(int(round(float(sc["floor_close_m"]) / cell))))
    lab, nlab = ndimage.label(fm)
    if nlab > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
        fm = lab == (1 + int(np.argmax(sizes)))
    fm = ndimage.binary_fill_holes(fm)

    # 3D occupancy in (a, b, h) and clearance to the nearest occupied voxel.
    vox = float(sc["voxel_m"])
    vox_origin = np.array([grid_origin[0], grid_origin[1], -0.3])
    dims = (
        int(np.ceil(na * cell / vox)),
        int(np.ceil(nb * cell / vox)),
        int(np.ceil((top_h + 0.6) / vox)),
    )
    L = np.stack([A, B, hv], axis=1)
    idx = np.floor((L - vox_origin) / vox).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array(dims)), axis=1)
    lin = np.ravel_multi_index(idx[ok].T, dims)
    occ = (np.bincount(lin, minlength=int(np.prod(dims))) >= int(sc["min_centers_per_voxel"])).reshape(dims)
    clearance = (ndimage.distance_transform_edt(~occ) * vox).astype(np.float32)

    return SceneFrame(
        up=up,
        e1=e1,
        e2=e2,
        d=float(d),
        ceiling_h=ceiling_h,
        top_h=top_h,
        cell=cell,
        grid_origin=grid_origin,
        floor_mask=fm,
        vox=vox,
        vox_origin=vox_origin,
        clearance=clearance,
    )


def load_or_estimate_frame(ply: PlyGaussian, cfg: dict[str, Any], frame_path: Path) -> SceneFrame:
    if frame_path.exists():
        return SceneFrame.load(frame_path)
    frame = estimate_frame(ply, cfg)
    frame.save(frame_path)
    return frame


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
