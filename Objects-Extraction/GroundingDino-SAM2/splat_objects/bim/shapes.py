from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from sklearn.decomposition import PCA

from splat_objects.bim.coords import class_spec, units_to_m
from splat_objects.bim.meshing import load_mesh_npz
from splat_objects.ply_io import read_binary_gaussian_ply
from splat_objects.scene import SceneFrame


def fit_elements(
    cfg: dict[str, Any],
    repo: Path,
    scene_id: str,
    frame: SceneFrame,
    elements: list[dict[str, Any]],
    structure: dict[str, Any],
    surface_pts: np.ndarray,
) -> list[dict[str, Any]]:
    scale = units_to_m(cfg)
    tol = float(cfg["bim"].get("fit_tolerance_m", 0.05))
    snap = float(cfg["bim"].get("floor_snap_m", 0.25))
    out: list[dict[str, Any]] = []
    pipes = [e for e in elements if class_spec(cfg, e["class"]).get("family") == "pipe"]

    for el in elements:
        spec = el.get("ifc_spec") or class_spec(cfg, el["class"])
        shape_k = spec.get("shape", "box")
        if shape_k == "door":
            continue
        shape, method = _build_shape(shape_k, el, pipes, structure, cfg)
        if shape_k == "box" and el["box_local"].get("bottom_height_m", 1) <= snap:
            shape = _snap_floor(shape)
        pts = _element_points(el, frame, repo, scene_id, cfg, surface_pts, scale)
        bim_dir = repo / "outputs" / scene_id / "bim"
        mesh_fit = _fit_from_saved_mesh(el, bim_dir, pts, tol)
        fit = mesh_fit if mesh_fit else (_fit_stats(pts, shape, tol) if len(pts) else {"within_frac": 0.0, "mean_m": 0.0})
        el2 = dict(el)
        el2["shape"] = shape
        el2["fit"] = {**fit, "method": method, "tolerance_m": tol}
        out.append(el2)
    return out


def _snap_floor(shape: dict[str, Any]) -> dict[str, Any]:
    sh = dict(shape)
    c = np.asarray(sh["center_m"], dtype=np.float64)
    half = np.asarray(sh["half_extents_m"], dtype=np.float64)
    c[2] = half[2]
    sh["center_m"] = c.tolist()
    return sh


def _build_shape(kind: str, el: dict[str, Any], pipes: list[dict], structure: dict, cfg: dict) -> tuple[dict[str, Any], str]:
    box = el["box_local"]
    c = np.asarray(box["center_m"], dtype=np.float64)
    size = np.asarray(box["size_m"], dtype=np.float64)
    if kind == "cylinder":
        ax, radius = _pca_cylinder(el)
        if ax is not None:
            return {
                "kind": "cylinder",
                "center_m": c.tolist(),
                "axis": ax.tolist(),
                "radius_m": float(radius),
                "height_m": float(size[2]),
            }, "pca_cylinder"
    if kind == "pipe":
        runs = _pipe_runs(size, c)
        if runs:
            return {"kind": "pipe_runs", "runs": runs}, "ransac_runs"
    if kind == "valve" and pipes:
        near = min(pipes, key=lambda p: float(np.linalg.norm(np.asarray(p["box_local"]["center_m"]) - c)))
        pc = np.asarray(near["box_local"]["center_m"])
        axis = pc - c
        axis /= np.linalg.norm(axis) + 1e-9
        return {
            "kind": "valve",
            "center_m": c.tolist(),
            "axis": axis.tolist(),
            "body_radius_m": float(min(size[:2]) * 0.25),
            "stem_m": float(size[2] * 0.4),
        }, "valve_proxy"
    half = 0.5 * size
    return {"kind": "box", "center_m": c.tolist(), "half_extents_m": half.tolist(), "yaw_deg": box.get("yaw_deg", 0.0)}, "oriented_box"


def _pca_cylinder(el: dict[str, Any]) -> tuple[np.ndarray | None, float]:
    corners = np.asarray(el["box_local"].get("corners_m") or [], dtype=np.float64)
    if len(corners) < 4:
        return None, 0.0
    pca = PCA(n_components=3)
    pca.fit(corners)
    axis = pca.components_[0]
    axis /= np.linalg.norm(axis) + 1e-9
    radius = float(np.mean(np.linalg.norm(corners - corners.mean(axis=0), axis=1)))
    return axis, radius * 0.35


def _pipe_runs(size: np.ndarray, center: np.ndarray) -> list[dict[str, Any]]:
    long_i = int(np.argmax(size[:2]))
    short = float(size[1 - long_i])
    if size[long_i] > short * 2:
        axis = np.zeros(3)
        axis[long_i] = 1.0
        return [{"center_m": center.tolist(), "axis": axis.tolist(), "radius_m": short * 0.35, "length_m": float(size[long_i])}]
    return []


def _element_points(
    el: dict[str, Any],
    frame: SceneFrame,
    repo: Path,
    scene_id: str,
    cfg: dict[str, Any],
    surface_pts: np.ndarray,
    scale: float,
) -> np.ndarray:
    if el["source"] == "grounded-sam-2" and el.get("ply"):
        ply_path = repo / "outputs" / scene_id / "objects" / el["ply"]
        if ply_path.is_file():
            g = read_binary_gaussian_ply(ply_path)
            op = g.opacity_sigmoid()
            m = op >= 0.3
            xyz = g.xyz()[m]
            return frame.local(xyz) * scale
    lo, hi = np.asarray(el["lo_m"]), np.asarray(el["hi_m"])
    if len(surface_pts):
        L = frame.local(surface_pts) * scale
        ok = np.all(L >= lo, axis=1) & np.all(L <= hi, axis=1)
        return L[ok]
    c = np.asarray(el["box_local"]["center_m"])
    return c.reshape(1, 3)


def _fit_from_saved_mesh(el: dict[str, Any], bim_dir: Path, pts: np.ndarray, tol: float) -> dict[str, float] | None:
    meta = el.get("mesh") or {}
    if meta.get("method") != "splat_surface" or not meta.get("path"):
        return None
    loaded = load_mesh_npz(bim_dir, meta["path"])
    if not loaded or len(pts) == 0:
        return None
    verts, faces, _ = loaded
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        _prox, dist, _ = trimesh.proximity.closest_point(mesh, pts)
        return {"within_frac": float(np.mean(dist <= tol)), "mean_m": float(np.mean(dist))}
    except Exception:
        return None


def _fit_stats(pts: np.ndarray, shape: dict[str, Any], tol: float) -> dict[str, float]:
    if len(pts) == 0:
        return {"within_frac": 0.0, "mean_m": 0.0}
    d = _distances(pts, shape)
    within = float(np.mean(d <= tol))
    return {"within_frac": within, "mean_m": float(np.mean(d))}


def _distances(pts: np.ndarray, shape: dict[str, Any]) -> np.ndarray:
    kind = shape.get("kind", "box")
    if kind == "pipe_runs" and shape.get("runs"):
        return np.min([_distances(pts, {**r, "kind": "cylinder"}) for r in shape["runs"]], axis=0)
    if kind == "valve" and "center_m" in shape:
        proxy = {**shape, "kind": "cylinder", "radius_m": shape.get("body_radius_m", 0.05)}
        return _distances(pts, proxy)
    if kind == "box":
        c = np.asarray(shape["center_m"])
        h = np.asarray(shape["half_extents_m"])
        q = np.abs(pts - c) - h
        q = np.maximum(q, 0)
        return np.linalg.norm(q, axis=1)
    if kind == "cylinder":
        c = np.asarray(shape["center_m"])
        ax = np.asarray(shape["axis"])
        r = float(shape.get("radius_m", shape.get("body_radius_m", 0.05)))
        v = pts - c
        along = v @ ax
        rad = np.linalg.norm(v - np.outer(along, ax), axis=1)
        return np.abs(rad - r)
    return np.linalg.norm(pts - np.asarray(shape["center_m"]), axis=1)
