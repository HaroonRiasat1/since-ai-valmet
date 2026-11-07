from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, generate_binary_structure
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
        "iso_level": 0.35,
        "solid_families": ["cabinet", "appliance", "tank", "pump", "motor", "loose", "sensor"],
        "solid_close_voxels": 5,
        "close_voxels": 1,
        "occupancy_factor": 0.6,
        "surface_sigma_voxels": 1.0,
        "blur_sigma_voxels": 0.6,
        "scale_cap_m": 0.06,
        "max_footprint_voxels": 6,
        "component_keep_frac": 0.15,
        "smooth_iters": 6,
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
        family = (el.get("ifc_spec") or {}).get("family", "")
        result = _mesh_from_ply(ply_path, frame, scale, mc, solid=family in set(mc.get("solid_families") or []))
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
    solid: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float] | None:
    ply = read_binary_gaussian_ply(ply_path)
    op = ply.opacity_sigmoid()
    mask = op >= float(mc["opacity_min"])
    if mask.sum() < 8:
        return None
    xyz = ply.xyz()[mask]
    op_m = op[mask]
    scales = ply.scales_exp()[mask]
    pad = max(float(mc["voxel_m"]) * 2, 2.0 * float(np.clip(scales.max(), 0, float(mc.get("scale_cap_m", 0.06)))))
    lo = xyz.min(axis=0) - pad
    hi = xyz.max(axis=0) + pad
    voxel = float(mc["voxel_m"])
    dims_f = np.ceil((hi - lo) / voxel).astype(int)
    max_ax = int(mc["max_voxels_per_axis"])
    if np.any(dims_f < 2):
        return None
    if np.any(dims_f > max_ax):
        voxel = float(np.max((hi - lo) / max_ax))
        dims_f = np.ceil((hi - lo) / voxel).astype(int)
    dims = tuple(int(x) for x in dims_f)
    keep_idx = np.nonzero(mask)[0]
    rot = _quats(ply)[keep_idx]
    xyz_ok = xyz
    grid = _gaussian_density(xyz, op_m, scales, rot, lo, voxel, dims, mc)
    if not grid.max():
        return None
    grid = gaussian_filter(grid, sigma=float(mc["blur_sigma_voxels"]))
    iso = float(mc.get("iso_level") or 0.0)
    if iso <= 0 or iso >= float(grid.max()):
        iso = float(np.percentile(grid[grid > 0], float(mc["iso_percentile"])))
    close_it = int(mc.get("solid_close_voxels", 5)) if solid else int(mc.get("close_voxels", 1))
    pad_n = close_it + 2
    occ = np.pad(grid > iso * float(mc.get("occupancy_factor", 0.6)), pad_n, mode="constant")
    if close_it > 0:
        occ = binary_closing(occ, structure=generate_binary_structure(3, 1), iterations=close_it)
    if solid:
        occ = binary_fill_holes(occ)
    field = gaussian_filter(occ.astype(np.float32), sigma=float(mc.get("surface_sigma_voxels", 1.0)))
    try:
        verts, faces, _normals, _vals = measure.marching_cubes(field, level=0.5)
    except (ValueError, RuntimeError):
        return None
    verts_scan = lo + (verts - pad_n + 0.5) * voxel
    verts_local = scan_to_local(frame, verts_scan, scale)
    mesh = trimesh.Trimesh(vertices=verts_local, faces=faces, process=False)
    mesh = _keep_components(mesh, float(mc.get("component_keep_frac", 0.15)))
    if mesh is None or len(mesh.faces) < int(mc["min_faces"]):
        return None
    if int(mc.get("smooth_iters", 0)) > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, iterations=int(mc["smooth_iters"]))
        except Exception:
            pass
    mesh = _decimate(mesh, int(mc["max_faces"]))
    colours = _vertex_colours(ply, xyz_ok, verts_scan, mc)
    haus = _hausdorff_p95(mesh, scan_to_local(frame, xyz_ok, scale))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), colours, "splat_surface", haus


def _quats(ply) -> np.ndarray:
    d = ply.raw_rows
    q = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], axis=1).astype(np.float64)
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)


def _rotmats(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def _gaussian_density(
    xyz: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    lo: np.ndarray,
    voxel: float,
    dims: tuple[int, int, int],
    mc: dict[str, Any],
) -> np.ndarray:
    """Sum of the true anisotropic Gaussians (Mahalanobis falloff) on the voxel grid."""
    grid = np.zeros(dims, dtype=np.float32)
    smax_cap = float(mc.get("scale_cap_m", 0.06))
    sc = np.clip(scales, voxel * 0.35, smax_cap)
    R = _rotmats(quats)
    inv_var = 1.0 / (sc * sc)
    max_r = int(mc.get("max_footprint_voxels", 6))
    for p, w, iv, r_m, s in zip(xyz, opacity, inv_var, R, sc, strict=False):
        rad = int(np.clip(np.ceil(2.5 * float(s.max()) / voxel), 1, max_r))
        c = np.floor((p - lo) / voxel).astype(int)
        lo_i = np.maximum(c - rad, 0)
        hi_i = np.minimum(c + rad + 1, dims)
        if np.any(hi_i <= lo_i):
            continue
        gx, gy, gz = np.meshgrid(
            np.arange(lo_i[0], hi_i[0]),
            np.arange(lo_i[1], hi_i[1]),
            np.arange(lo_i[2], hi_i[2]),
            indexing="ij",
        )
        centres = lo + (np.stack([gx, gy, gz], axis=-1) + 0.5) * voxel
        d = centres - p
        local = d @ r_m  # world offset into the Gaussian's axes
        m2 = np.sum(local * local * iv, axis=-1)
        grid[lo_i[0] : hi_i[0], lo_i[1] : hi_i[1], lo_i[2] : hi_i[2]] += (w * np.exp(-0.5 * m2) * (m2 < 6.25)).astype(np.float32)
    return grid


def _keep_components(mesh: trimesh.Trimesh, keep_frac: float) -> trimesh.Trimesh | None:
    parts = mesh.split(only_watertight=False)
    if not parts:
        return None
    biggest = max(len(m.faces) for m in parts)
    kept = [m for m in parts if len(m.faces) >= max(8, keep_frac * biggest)]
    return trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]


def _decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
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
        # Splat centres sit inside the object, so count them as covered when inside or within tol of the surface.
        sd = trimesh.proximity.signed_distance(mesh, pts)
        return {"within_frac": float(np.mean(sd > -tol)), "mean_m": float(np.mean(np.abs(sd)))}
    except Exception:
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
