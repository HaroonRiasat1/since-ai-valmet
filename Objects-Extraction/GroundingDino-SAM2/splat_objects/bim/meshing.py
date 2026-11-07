from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure

from splat_objects.bim.coords import scan_to_local, units_to_m
from splat_objects.ply_io import read_binary_gaussian_ply
from splat_objects.scene import SceneFrame

_SH_C0 = 0.28209479177387814


def mesh_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "voxel_m": 0.015,
        "max_voxels_per_axis": 96,
        "opacity_min": 0.3,
        "iso_percentile": 35.0,
        "blur_sigma_voxels": 0.9,
        "max_faces": 2000,
        "min_faces": 20,
        "colour": True,
        "volume_ratio_max": 4.0,
    }
    return {**defaults, **(cfg.get("bim", {}).get("mesh") or {})}


def mesh_fitted_elements(
    cfg: dict[str, Any],
    repo: Path,
    scene_id: str,
    frame: SceneFrame,
    elements: list[dict[str, Any]],
    bim_dir: Path,
) -> dict[str, int]:
    mc = mesh_cfg(cfg)
    if not mc["enabled"]:
        return {"mesh": 0, "fallback": 0, "skipped": len(elements)}
    scale = units_to_m(cfg)
    tol = float(cfg["bim"].get("fit_tolerance_m", 0.05))
    mesh_dir = bim_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    counts = {"mesh": 0, "fallback": 0, "skipped": 0}

    for el in elements:
        if el["source"] != "grounded-sam-2" or not el.get("ply"):
            counts["skipped"] += 1
            continue
        ply_path = repo / "outputs" / scene_id / "objects" / el["ply"]
        if not ply_path.is_file():
            counts["skipped"] += 1
            continue
        result = _mesh_from_ply(ply_path, frame, scale, mc)
        rel = f"meshes/{el['id']}.npz"
        out = bim_dir / rel
        if result is None:
            el["mesh"] = {"method": "box_fallback", "path": None}
            counts["fallback"] += 1
            continue
        verts_local, faces, colours, method, hausdorff = result
        if not _accept_mesh(el, verts_local, faces, mc):
            el["mesh"] = {"method": "box_fallback", "path": None}
            counts["fallback"] += 1
            continue
        np.savez_compressed(
            out,
            vertices=verts_local.astype(np.float32),
            faces=faces.astype(np.int32),
            colours=colours.astype(np.uint8),
        )
        mesh_tm = trimesh.Trimesh(vertices=verts_local, faces=faces, process=False)
        pts = _element_points_for_fit(el, frame, repo, scene_id, scale)
        fit = _fit_mesh(mesh_tm, pts, tol)
        el["mesh"] = {
            "method": method,
            "path": rel,
            "n_faces": int(len(faces)),
            "hausdorff_p95_m": float(hausdorff),
            "colour": bool(mc["colour"]),
        }
        el["fit"] = {**fit, "method": method, "tolerance_m": tol}
        counts["mesh"] += 1
    return counts


def _mesh_from_ply(
    ply_path: Path,
    frame: SceneFrame,
    scale: float,
    mc: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float] | None:
    ply = read_binary_gaussian_ply(ply_path)
    op = ply.opacity_sigmoid()
    mask = op >= float(mc["opacity_min"])
    if mask.sum() < 8:
        return None
    xyz = ply.xyz()[mask]
    op_m = op[mask]
    scales = ply.scales_exp()[mask]
    lo = xyz.min(axis=0) - float(mc["voxel_m"]) * 2
    hi = xyz.max(axis=0) + float(mc["voxel_m"]) * 2
    voxel = float(mc["voxel_m"])
    dims_f = np.ceil((hi - lo) / voxel).astype(int)
    max_ax = int(mc["max_voxels_per_axis"])
    if np.any(dims_f < 2):
        return None
    if np.any(dims_f > max_ax):
        voxel = float(np.max((hi - lo) / max_ax))
        dims_f = np.ceil((hi - lo) / voxel).astype(int)
    dims = tuple(int(x) for x in dims_f)
    grid = np.zeros(dims, dtype=np.float32)
    idx = np.floor((xyz - lo) / voxel).astype(int)
    ok = np.all((idx >= 0) & (idx < dims), axis=1)
    idx, op_m, scales = idx[ok], op_m[ok], scales[ok]
    xyz_ok = xyz[ok]
    for i, j, k, w, sc in zip(idx[:, 0], idx[:, 1], idx[:, 2], op_m, scales, strict=False):
        r = int(max(1, min(3, np.ceil(np.max(sc) / voxel))))
        i0, i1 = max(0, i - r), min(dims[0], i + r + 1)
        j0, j1 = max(0, j - r), min(dims[1], j + r + 1)
        k0, k1 = max(0, k - r), min(dims[2], k + r + 1)
        grid[i0:i1, j0:j1, k0:k1] += w
    if not grid.max():
        return None
    grid = gaussian_filter(grid, sigma=float(mc["blur_sigma_voxels"]))
    iso = float(np.percentile(grid[grid > 0], float(mc["iso_percentile"])))
    try:
        verts, faces, _normals, _vals = measure.marching_cubes(grid, level=iso)
    except (ValueError, RuntimeError):
        return None
    verts_scan = lo + verts * voxel
    verts_local = scan_to_local(frame, verts_scan, scale)
    mesh = trimesh.Trimesh(vertices=verts_local, faces=faces, process=False)
    mesh = _largest_component(mesh)
    if mesh is None or len(mesh.faces) < int(mc["min_faces"]):
        return None
    mesh = _decimate(mesh, int(mc["max_faces"]))
    colours = _vertex_colours(ply, xyz_ok, verts_scan, mc)
    haus = _hausdorff_p95(mesh, scan_to_local(frame, xyz_ok, scale))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), colours, "splat_surface", haus


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    parts = mesh.split(only_watertight=False)
    if not parts:
        return None
    return max(parts, key=lambda m: len(m.faces))


def _decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(max_faces)
    except Exception:
        return mesh


def _vertex_colours(ply, xyz: np.ndarray, verts_scan: np.ndarray, mc: dict[str, Any]) -> np.ndarray:
    n = len(verts_scan)
    grey = np.tile(np.array([180, 190, 200, 255], dtype=np.uint8), (n, 1))
    if not mc.get("colour"):
        return grey
    d = ply.raw_rows
    if not all(k in d.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        return grey
    fdc = np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], axis=1).astype(np.float64)
    rgb = np.clip(0.5 + _SH_C0 * fdc, 0, 1)
    tree = cKDTree(xyz)
    _, ix = tree.query(verts_scan)
    c = (rgb[ix] * 255).astype(np.uint8)
    return np.column_stack([c, np.full(len(c), 255, dtype=np.uint8)])


def _accept_mesh(el: dict[str, Any], verts: np.ndarray, faces: np.ndarray, mc: dict[str, Any]) -> bool:
    if len(faces) < int(mc["min_faces"]):
        return False
    try:
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if not tm.is_watertight:
            pass
        box = el["box_local"]
        box_vol = float(np.prod(box["size_m"]))
        mesh_vol = float(abs(tm.volume)) if len(tm.faces) else 0.0
        if box_vol > 1e-6 and mesh_vol > box_vol * float(mc["volume_ratio_max"]):
            return False
    except Exception:
        return False
    return True


def _hausdorff_p95(mesh: trimesh.Trimesh, pts_local: np.ndarray) -> float:
    if len(pts_local) == 0:
        return 0.0
    try:
        _prox, dist, _ = trimesh.proximity.closest_point(mesh, pts_local)
        return float(np.percentile(dist, 95))
    except Exception:
        return 0.0


def _fit_mesh(mesh: trimesh.Trimesh, pts: np.ndarray, tol: float) -> dict[str, float]:
    if len(pts) == 0:
        return {"within_frac": 0.0, "mean_m": 0.0}
    try:
        _prox, dist, _ = trimesh.proximity.closest_point(mesh, pts)
        return {"within_frac": float(np.mean(dist <= tol)), "mean_m": float(np.mean(dist))}
    except Exception:
        return {"within_frac": 0.0, "mean_m": 0.0}


def _element_points_for_fit(el: dict[str, Any], frame: SceneFrame, repo: Path, scene_id: str, scale: float) -> np.ndarray:
    if el.get("ply"):
        ply_path = repo / "outputs" / scene_id / "objects" / el["ply"]
        if ply_path.is_file():
            g = read_binary_gaussian_ply(ply_path)
            op = g.opacity_sigmoid()
            m = op >= 0.3
            return scan_to_local(frame, g.xyz()[m], scale)
    c = np.asarray(el["box_local"]["center_m"])
    return c.reshape(1, 3)


def load_mesh_npz(bim_dir: Path, rel: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not rel:
        return None
    path = bim_dir / rel
    if not path.is_file():
        return None
    z = np.load(path)
    return z["vertices"], z["faces"], z.get("colours", np.zeros((len(z["vertices"]), 4), dtype=np.uint8))


def synthetic_gaussian_box(
    center: np.ndarray,
    size: np.ndarray,
    n: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit cube of Gaussians for tests (scan coords)."""
    rng = rng or np.random.default_rng(0)
    pts = rng.uniform(-0.5, 0.5, (n, 3)) * size + center
    op = np.full(n, 2.0)
    fdc = rng.uniform(-0.2, 0.2, (n, 3))
    return pts, op, fdc
