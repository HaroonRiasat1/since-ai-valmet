#!/usr/bin/env python3
"""Stage 2: place valid in-room cameras, render RGB/depth/alpha, save a reusable view set."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.cameras import camera_record, look_camera, nerfstudio_frame, project_points, view_direction
from splat_objects.config import load_config, resolve_repo_path
from splat_objects.ply_io import read_binary_gaussian_ply
from splat_objects.render_mlx import (
    SplatScene,
    render_depth,
    render_rgb,
    save_alpha8,
    save_depth16,
    save_rgb,
    sharpness,
    to_uint8,
)
from splat_objects.scene import load_or_estimate_frame, write_json
from splat_objects.viz import contact_sheets, draw_label

LEVEL_COLORS = {"eye": (60, 220, 60), "low": (255, 200, 0)}


def farthest_point_sampling(pts: np.ndarray, k: int, start: int) -> np.ndarray:
    chosen = [int(start)]
    dist = np.linalg.norm(pts - pts[start], axis=1)
    while len(chosen) < min(k, len(pts)):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(chosen, dtype=np.int64)


def candidate_positions(frame, cfg) -> list[tuple[dict, np.ndarray]]:
    rc = cfg["render"]
    interior = frame.interior_mask(float(cfg["scene"]["wall_margin_m"]))
    lo, hi = frame.floor_extent()
    step = float(rc["grid_step_m"])
    A, B = np.meshgrid(np.arange(lo[0] + step / 2, hi[0], step), np.arange(lo[1] + step / 2, hi[1], step), indexing="ij")
    A, B = A.ravel(), B.ravel()
    inside = frame.in_mask(interior, A, B)
    center = np.array([A[inside].mean(), B[inside].mean()])
    out: list[tuple[dict, np.ndarray]] = []
    for level in rc["levels"]:
        h = min(float(level["height_m"]), frame.ceiling_h - 0.25)
        abh = np.stack([A, B, np.full_like(A, h)], axis=1)
        ok = inside & (frame.clearance_at(abh) >= float(level["clearance_m"]))
        pts = abh[ok]
        if len(pts) == 0:
            print(f"level {level['name']}: no valid positions")
            continue
        start = int(np.argmin(np.linalg.norm(pts[:, :2] - center, axis=1)))
        sel = farthest_point_sampling(pts[:, :2], int(level["positions"]), start)
        print(f"level {level['name']}: h={h:.2f}m valid_cells={len(pts)} selected={len(sel)}")
        out.extend((level, pts[i]) for i in sel)
    return out


def draw_camera_map(topdown_bgr: np.ndarray, td_rec: dict, frame, candidates: list[dict]) -> np.ndarray:
    img = topdown_bgr.copy()
    for c in candidates:
        p = np.array(c["camera"]["position"])
        fwd = np.array(c["camera"]["forward"])
        horiz = fwd - (fwd @ frame.up) * frame.up
        horiz /= np.linalg.norm(horiz) + 1e-12
        (p0, p1), _ = project_points(td_rec, np.stack([p, p + 0.45 * horiz]))
        p0i, p1i = tuple(int(v) for v in p0), tuple(int(v) for v in p1)
        col = LEVEL_COLORS.get(c["level"], (255, 255, 255)) if c["accepted"] else (40, 40, 230)
        if c["accepted"]:
            cv2.arrowedLine(img, p0i, p1i, col, 2, cv2.LINE_AA, tipLength=0.35)
        else:
            cv2.line(img, p0i, p1i, col, 1, cv2.LINE_AA)
        cv2.circle(img, p0i, 4, col, -1, cv2.LINE_AA)
    draw_label(img, "green: 1.6 m views  yellow: 1.0 m views  red: rejected", 8, 24, (255, 255, 255), 0.55)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    ap.add_argument("--force", action="store_true", help="re-render even if the view set exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rc, vc = cfg["render"], cfg["render"]["validate"]
    scene_id = cfg["scene_id"]
    out_root = _REPO / "renders" / scene_id
    overview = _REPO / "outputs" / scene_id / "scene_overview"
    if (out_root / "cameras.json").exists() and not args.force:
        print(f"Reusing existing view set at {out_root} (use --force to re-render)")
        return
    if out_root.exists():
        shutil.rmtree(out_root)
    for sub in ("images", "depth", "alpha"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    ply_path = resolve_repo_path(cfg, cfg["input_ply"])
    ply = read_binary_gaussian_ply(ply_path)
    frame = load_or_estimate_frame(ply, cfg, overview / "scene_frame.npz")
    scene = SplatScene.load(ply_path)
    antialias = bool(rc["antialias"])
    W, H = int(rc["width"]), int(rc["height"])

    positions = candidate_positions(frame, cfg)
    accepted: list[dict] = []
    candidates: list[dict] = []
    tiles: list[tuple[str, np.ndarray]] = []
    t0 = time.time()
    for pos_idx, (level, abh) in enumerate(positions):
        eye = frame.world(abh)
        for yi in range(int(rc["yaws"])):
            yaw = 360.0 * yi / int(rc["yaws"])
            fwd = view_direction(frame, yaw, float(level["pitch_deg"]))
            cam = look_camera(eye, fwd, frame.up, W, H, float(rc["fov_y_deg"]), float(rc["znear"]))
            rgb, alpha = render_rgb(scene, cam, antialias)
            depth = render_depth(scene, cam, antialias)
            u8 = to_uint8(rgb)
            solid = alpha > 0.5
            metrics = {
                "alpha_low_frac": float(np.mean(~solid)),
                "near_frac": float(np.mean(solid & (depth < float(vc["near_depth_m"])))),
                "median_depth_m": float(np.median(depth[solid])) if solid.any() else 0.0,
                "sharpness": sharpness(u8),
            }
            reasons = []
            if metrics["alpha_low_frac"] > float(vc["max_alpha_low_frac"]):
                reasons.append("holes")
            if metrics["near_frac"] > float(vc["max_near_frac"]):
                reasons.append("too_close")
            if metrics["median_depth_m"] < float(vc["min_median_depth_m"]):
                reasons.append("shallow")
            if metrics["sharpness"] < float(vc["min_sharpness"]):
                reasons.append("blurry")
            rec = camera_record(cam)
            cand = {
                "position_index": pos_idx,
                "level": level["name"],
                "height_m": float(abh[2]),
                "yaw_deg": yaw,
                "pitch_deg": float(level["pitch_deg"]),
                "metrics": metrics,
                "accepted": not reasons,
                "reject_reasons": reasons,
                "camera": rec,
            }
            candidates.append(cand)
            if reasons:
                continue
            vid = len(accepted)
            stem = f"v{vid:04d}"
            save_rgb(out_root / "images" / f"{stem}.png", u8)
            save_depth16(out_root / "depth" / f"{stem}.png", depth, float(rc["depth_png_scale"]))
            save_alpha8(out_root / "alpha" / f"{stem}.png", alpha)
            accepted.append(
                {
                    "id": vid,
                    "image": f"images/{stem}.png",
                    "depth": f"depth/{stem}.png",
                    "alpha": f"alpha/{stem}.png",
                    **{k: cand[k] for k in ("position_index", "level", "height_m", "yaw_deg", "pitch_deg", "metrics")},
                    **rec,
                }
            )
            tiles.append((f"{stem} {level['name']} yaw{int(yaw)}", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)))
    dt = time.time() - t0

    if not accepted:
        raise RuntimeError("No views passed validation; relax render.validate thresholds.")

    write_json(out_root / "cameras.json", accepted)
    write_json(out_root / "candidates.json", candidates)
    write_json(
        out_root / "transforms.json",
        {
            "camera_model": "OPENCV",
            "fl_x": accepted[0]["fx"],
            "fl_y": accepted[0]["fy"],
            "cx": accepted[0]["cx"],
            "cy": accepted[0]["cy"],
            "w": W,
            "h": H,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "frames": [nerfstudio_frame(a, a["image"]) for a in accepted],
        },
    )
    shutil.copy(overview / "scene_meta.json", out_root / "scene_meta.json")
    write_json(
        out_root / "render_info.json",
        {
            "source_ply": str(ply_path),
            "renderer": "mlx3d render_gaussians (SH degree 3)",
            "antialias": antialias,
            "depth_png_scale": float(rc["depth_png_scale"]),
            "depth_kind": "expected camera-z depth",
            "conventions": "R_w2c/t_w2c: p_cam = R p_world + t (OpenCV: x right, y down, z forward)",
            "n_views": len(accepted),
            "n_candidates": len(candidates),
        },
    )
    contact_sheets(tiles, out_root / "contact_sheets", "renders")

    td_rec = json.loads((overview / "topdown_camera.json").read_text(encoding="utf-8"))
    td = cv2.imread(str(overview / "topdown_plain.png"))
    cv2.imwrite(str(overview / "cameras_topdown.png"), draw_camera_map(td, td_rec, frame, candidates))

    rejects: dict[str, int] = {}
    for c in candidates:
        for r in c["reject_reasons"]:
            rejects[r] = rejects.get(r, 0) + 1
    print(f"Rendered {len(candidates)} candidates in {dt:.1f}s; accepted {len(accepted)}; rejected by reason {rejects}")
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
