from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from splat_objects.scene import SceneFrame


@dataclass
class Proposal:
    view: int
    det: int
    class_id: int
    score: float
    sam_score: float
    mask_px: int
    members: np.ndarray


@dataclass
class Instance:
    class_id: int
    proposals: list[Proposal]
    members: np.ndarray
    ratios: np.ndarray
    views: list[int] = field(default_factory=list)
    confidence: float = 0.0
    detect_ratio: float = 0.0
    n_visible_views: int = 0


def load_lifting(lift_dir: Path, n_views: int) -> tuple[np.ndarray, np.ndarray, list[Proposal]]:
    vis_rows: list[np.ndarray | None] = [None] * n_views
    w0q_rows: list[np.ndarray | None] = [None] * n_views
    props: list[Proposal] = []
    for fp in sorted(lift_dir.glob("*.npz")):
        z = np.load(fp)
        v = int(z["view_idx"])
        n = int(z["n_gaussians"])
        vis_rows[v] = np.unpackbits(z["vis_bits"])[:n].astype(bool)
        w0q_rows[v] = z["w0_q"]
        members, offsets = z["members"], z["offsets"]
        for j, meta in enumerate(json.loads(str(z["props_json"]))):
            props.append(
                Proposal(
                    view=v,
                    det=int(meta["det_idx"]),
                    class_id=int(meta["class_id"]),
                    score=float(meta["score"]),
                    sam_score=float(meta["sam_score"]),
                    mask_px=int(meta["mask_px"]),
                    members=np.asarray(members[offsets[j] : offsets[j + 1]], dtype=np.int64),
                )
            )
    missing = [i for i, r in enumerate(vis_rows) if r is None]
    if missing:
        raise RuntimeError(f"lifting results missing for views {missing[:10]}")
    return np.stack(vis_rows), np.stack(w0q_rows), props


def affinity_matrix(props: list[Proposal], vis: np.ndarray, min_covis: int) -> tuple[np.ndarray, np.ndarray]:
    """Visibility-aware overlap between proposals; `known` is False where views share too little."""
    n_g = vis.shape[1]
    p = len(props)
    rows = np.concatenate([np.full(len(q.members), i) for i, q in enumerate(props)])
    cols = np.concatenate([q.members for q in props])
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(p, n_g))
    inter = (M @ M.T).toarray()
    covis = np.stack([vis[:, q.members].sum(axis=1) for q in props])  # (P, V)
    views = np.array([q.view for q in props])
    c_ab = covis[:, views]  # c_ab[i, j] = |members_i visible in view_j|
    c_ba = c_ab.T
    known = (c_ab >= min_covis) & (c_ba >= min_covis)
    with np.errstate(divide="ignore", invalid="ignore"):
        aff = np.minimum(inter / np.maximum(c_ab, 1), inter / np.maximum(c_ba, 1))
    same_view = views[:, None] == views[None, :]
    aff[same_view] = 0.0
    known[same_view] = True
    np.fill_diagonal(known, False)
    return np.where(known, aff, 0.0), known


def agglomerate(props: list[Proposal], aff: np.ndarray, known: np.ndarray, threshold: float) -> list[list[int]]:
    """Average linkage over known pairs; clusters that share a view never merge."""
    n = len(props)
    S = np.where(known, aff, 0.0).astype(np.float64)
    K = known.astype(np.float64)
    views = np.array([q.view for q in props])
    shares = views[:, None] == views[None, :]
    active = np.ones(n, dtype=bool)
    members = {i: [i] for i in range(n)}
    while n > 1:
        avg = np.where(K > 0, S / np.maximum(K, 1.0), -1.0)
        avg[shares] = -1.0
        avg[~active, :] = -1.0
        avg[:, ~active] = -1.0
        i, j = np.unravel_index(int(np.argmax(avg)), avg.shape)
        if avg[i, j] < threshold:
            break
        S[i, :] += S[j, :]
        S[:, i] = S[i, :]
        K[i, :] += K[j, :]
        K[:, i] = K[i, :]
        shares[i, :] |= shares[j, :]
        shares[:, i] = shares[i, :]
        S[i, i] = K[i, i] = 0.0
        S[j, :] = S[:, j] = 0.0
        K[j, :] = K[:, j] = 0.0
        active[j] = False
        members[i].extend(members.pop(j))
    return [members[i] for i in sorted(members)]


def vote(
    cluster: list[Proposal],
    vis: np.ndarray,
    xyz: np.ndarray,
    core_ratio: float,
    grow_ratio: float,
    radius_scale: float,
    radius_min: float,
    radius_max: float,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Cross-view vote with hysteresis.

    ratio = (#views where the Gaussian is in the cluster's mask) / (#cluster views where it is visible).
    Gaussians with ratio >= core_ratio form the core; weaker ones (>= grow_ratio) are added only if
    they connect to the core in 3D, so attached parts that SAM sometimes leaves out are recovered.
    """
    views = sorted({q.view for q in cluster})
    cand = np.unique(np.concatenate([q.members for q in cluster]))
    pos = np.zeros(len(cand), dtype=np.int32)
    seen = np.zeros(len(cand), dtype=np.int32)
    for v in views:
        mem_v = np.unique(np.concatenate([q.members for q in cluster if q.view == v]))
        pos[np.searchsorted(cand, mem_v)] += 1
        seen += vis[v, cand]
    ratio = (pos / np.maximum(seen, 1)).astype(np.float32)
    enough = pos >= min(2, len(views))
    core = enough & (ratio >= core_ratio)
    weak = np.nonzero(enough & (ratio >= grow_ratio) & ~core)[0]
    if core.sum() >= 9 and len(weak):
        P = xyz[cand[core]]
        d, _ = cKDTree(P).query(P, k=9)
        radius = float(np.clip(radius_scale * np.median(d[:, 1:]), radius_min, radius_max))
        tree = cKDTree(xyz[cand[weak]])
        added = np.zeros(len(weak), dtype=bool)
        frontier = P
        for _ in range(max_iters):
            hits = tree.query_ball_point(frontier, r=radius)
            new = np.unique(np.fromiter((i for hs in hits for i in hs), dtype=np.int64))
            new = new[~added[new]] if len(new) else new
            if not len(new):
                break
            added[new] = True
            frontier = xyz[cand[weak[new]]]
        core[weak[added]] = True
    return cand[core], ratio[core], views


def detection_consistency(
    members: np.ndarray, views: list[int], vis: np.ndarray, w0q: np.ndarray, lut: np.ndarray, min_px: float, min_frac: float
) -> tuple[float, int]:
    area = lut[w0q[:, members]].sum(axis=1)
    frac = vis[:, members].mean(axis=1)
    visible = (area >= min_px) & (frac >= min_frac)
    visible[views] = True
    n_vis = int(visible.sum())
    return len(views) / max(n_vis, 1), n_vis


def min_scale_normals(quats_wxyz: np.ndarray, scales: np.ndarray) -> np.ndarray:
    q = quats_wxyz / (np.linalg.norm(quats_wxyz, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q.T
    cols = [
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1),
        np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1),
        np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ]
    k = np.argmin(scales, axis=1)
    out = np.empty((len(q), 3), dtype=np.float32)
    for axis in range(3):
        sel = k == axis
        out[sel] = cols[axis][sel]
    return out


def cleanup(
    members: np.ndarray,
    xyz: np.ndarray,
    heights: np.ndarray,
    normals: np.ndarray,
    smax: np.ndarray,
    up: np.ndarray,
    floor_standing: bool,
    ic: dict[str, Any],
) -> np.ndarray:
    if floor_standing and len(members):
        flat = (np.abs(heights[members]) < float(ic["floor_band_m"])) & (
            np.abs(normals[members] @ up) >= float(ic["floor_normal_cos"])
        )
        members = members[~flat]
    if len(members) < int(ic["dbscan_min_samples"]) * 3:
        return members
    P = xyz[members]
    lo, hi = np.percentile(P, [2, 98], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    members = members[smax[members] <= max(0.05, float(ic["large_splat_frac"]) * diag)]
    if len(members) < int(ic["dbscan_min_samples"]) * 3:
        return members
    P = xyz[members]
    d, _ = cKDTree(P).query(P, k=min(9, len(P)))
    eps = float(np.clip(float(ic["dbscan_eps_scale"]) * np.median(d[:, 1:]), ic["dbscan_eps_min_m"], ic["dbscan_eps_max_m"]))
    lab = DBSCAN(eps=eps, min_samples=int(ic["dbscan_min_samples"])).fit_predict(P)
    if (lab >= 0).any():
        sizes = np.bincount(lab[lab >= 0])
        main = int(np.argmax(sizes))
        main_pts = P[lab == main]
        main_tree = cKDTree(main_pts)
        e1 = np.array([1.0, 0.0, 0.0]) - up[0] * up
        e1 /= np.linalg.norm(e1) + 1e-12
        e2 = np.cross(up, e1)
        foot = np.stack([main_pts @ e1, main_pts @ e2], axis=1)
        f_lo, f_hi = foot.min(axis=0) - 0.05, foot.max(axis=0) + 0.05
        good = [main]
        for k in np.nonzero(sizes > 0)[0]:
            if k == main:
                continue
            comp = P[lab == k]
            if sizes[k] >= float(ic["keep_component_frac"]) * sizes[main]:
                good.append(int(k))
                continue
            if main_tree.query(comp, k=1)[0].min() <= float(ic["link_dist_m"]):
                good.append(int(k))
                continue
            if floor_standing:
                cf = np.stack([comp @ e1, comp @ e2], axis=1)
                inside = np.all((cf >= f_lo) & (cf <= f_hi), axis=1).mean()
                if inside >= 0.8 and np.median(heights[members][lab == k]) < np.median(heights[members][lab == main]):
                    good.append(int(k))
        members = members[np.isin(lab, good)]
    k = int(ic["sor_k"])
    if len(members) > k + 1:
        P = xyz[members]
        d, _ = cKDTree(P).query(P, k=k + 1)
        md = d[:, 1:].mean(axis=1)
        members = members[md <= md.mean() + float(ic["sor_std"]) * md.std()]
    return members


def local_box(frame: SceneFrame, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = frame.local(xyz)
    return np.percentile(L, 5, axis=0), np.percentile(L, 95, axis=0)


def box_containment(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    if np.any(hi <= lo):
        return 0.0
    inter = float(np.prod(hi - lo))
    va, vb = float(np.prod(a[1] - a[0])), float(np.prod(b[1] - b[0]))
    return inter / max(min(va, vb), 1e-12)


def oriented_box(frame: SceneFrame, xyz: np.ndarray) -> dict[str, Any]:
    L = frame.local(xyz)
    lo, hi = np.percentile(L, 1, axis=0), np.percentile(L, 99, axis=0)
    inner = L[np.all((L >= lo) & (L <= hi), axis=1)]
    if len(inner) < 5:
        inner = L
    (ca, cb), (wa, wb), ang = cv2.minAreaRect(inner[:, :2].astype(np.float32))
    base = cv2.boxPoints(((ca, cb), (wa, wb), ang)).astype(np.float64)
    h0, h1 = float(lo[2]), float(hi[2])
    corners = np.concatenate(
        [np.column_stack([base, np.full(4, h0)]), np.column_stack([base, np.full(4, h1)])], axis=0
    )
    world = frame.world(corners)
    P = np.asarray(xyz)
    return {
        "center": frame.world(np.array([ca, cb, 0.5 * (h0 + h1)])).tolist(),
        "size_m": [float(wa), float(wb), h1 - h0],
        "yaw_deg": float(ang),
        "corners": world.tolist(),
        "aabb_min": np.percentile(P, 1, axis=0).tolist(),
        "aabb_max": np.percentile(P, 99, axis=0).tolist(),
        "bottom_height_m": h0,
        "top_height_m": h1,
    }
