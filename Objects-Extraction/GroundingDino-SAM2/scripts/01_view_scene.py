#!/usr/bin/env python3
"""Stage 1: analyze the scene (floor, up axis, free space) and render overview images."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.cameras import camera_record, look_camera, project_points
from splat_objects.config import load_config, resolve_repo_path
from splat_objects.ply_io import read_binary_gaussian_ply
from splat_objects.render_mlx import SplatScene, render_rgb, save_rgb, to_uint8
from splat_objects.scene import estimate_frame, write_json
from splat_objects.viz import draw_label


def topdown_camera(frame, width: int = 1400, fov_y_deg: float = 40.0):
    lo, hi = frame.floor_extent()
    span = hi - lo
    height = int(round(width * span[1] / span[0]))
    center = frame.world(np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.0]))
    tan_y = math.tan(math.radians(fov_y_deg) / 2)
    tan_x = tan_y * width / height
    dist = 1.06 * max(0.5 * span[1] / tan_y, 0.5 * span[0] / tan_x)
    eye = center + dist * frame.up
    cam = look_camera(eye, -frame.up, frame.e2, width, height, fov_y_deg, znear=0.5, zfar=dist + 10.0)
    return cam, lo, hi


def draw_grid(bgr: np.ndarray, rec: dict, frame, lo: np.ndarray, hi: np.ndarray) -> None:
    for a in range(int(math.ceil(lo[0])), int(math.floor(hi[0])) + 1):
        pts = frame.world(np.array([[a, lo[1], 0.0], [a, hi[1], 0.0]]))
        (p0, p1), _ = project_points(rec, pts)
        cv2.line(bgr, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (0, 255, 255), 1, cv2.LINE_AA)
        draw_label(bgr, f"a={a}", int(p1[0]) + 2, int(p1[1]) + 16, (0, 255, 255), 0.45)
    for b in range(int(math.ceil(lo[1])), int(math.floor(hi[1])) + 1):
        pts = frame.world(np.array([[lo[0], b, 0.0], [hi[0], b, 0.0]]))
        (p0, p1), _ = project_points(rec, pts)
        cv2.line(bgr, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (255, 255, 0), 1, cv2.LINE_AA)
        draw_label(bgr, f"b={b}", int(p0[0]) + 2, int(p0[1]) - 2, (255, 255, 0), 0.45)


def corner_eyes(frame, interior: np.ndarray, heights=(1.8, 1.6, 1.4)) -> list[np.ndarray]:
    ii, jj = np.nonzero(interior)
    ab = frame.grid_origin + (np.stack([ii, jj], axis=1) + 0.5) * frame.cell
    lo, hi = ab.min(axis=0), ab.max(axis=0)
    eyes = []
    for corner in [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]:
        order = np.argsort(np.linalg.norm(ab - np.array(corner), axis=1))
        chosen = None
        for k in order[:4000]:
            for h in heights:
                if h > frame.ceiling_h - 0.25:
                    continue
                p = np.array([ab[k, 0], ab[k, 1], h])
                if frame.clearance_at(p)[0] >= 0.35:
                    chosen = p
                    break
            if chosen is not None:
                break
        if chosen is not None:
            eyes.append(chosen)
    return eyes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ply_path = resolve_repo_path(cfg, cfg["input_ply"])
    out_dir = _REPO / "outputs" / cfg["scene_id"] / "scene_overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    ply = read_binary_gaussian_ply(ply_path)
    frame = estimate_frame(ply, cfg)
    frame.save(out_dir / "scene_frame.npz")
    meta = {"input_ply": str(ply_path), "n_gaussians": ply.n, **frame.meta()}
    write_json(out_dir / "scene_meta.json", meta)
    print(
        f"up={np.round(frame.up, 4).tolist()} ceiling_h={frame.ceiling_h:.2f}m top_h={frame.top_h:.2f}m "
        f"floor_area={meta['floor_area_m2']:.1f}m2"
    )

    scene = SplatScene.load(ply_path)
    antialias = bool(cfg["render"]["antialias"])

    # Floor plan: drop ceiling and overhead pipes so the floor is visible from above.
    h = frame.heights(ply.xyz())
    low = np.nonzero(h < float(cfg["scene"]["topdown_max_height_m"]))[0]
    cam, lo, hi = topdown_camera(frame)
    rgb, _ = render_rgb(scene.subset(low), cam, antialias)
    rec = camera_record(cam)
    write_json(out_dir / "topdown_camera.json", rec)
    bgr = cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "topdown_plain.png"), bgr)
    draw_grid(bgr, rec, frame, lo, hi)
    cv2.imwrite(str(out_dir / "topdown.png"), bgr)

    interior = frame.interior_mask(float(cfg["scene"]["wall_margin_m"]))
    vis_mask = np.zeros(frame.floor_mask.shape + (3,), dtype=np.uint8)
    vis_mask[frame.floor_mask] = (90, 90, 90)
    vis_mask[interior] = (60, 200, 60)
    cv2.imwrite(str(out_dir / "floor_mask.png"), np.flipud(np.transpose(vis_mask, (1, 0, 2))))

    center = frame.world(np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5]))
    rc = cfg["render"]
    for k, abh in enumerate(corner_eyes(frame, interior)):
        eye = frame.world(abh)
        cam = look_camera(eye, center - eye, frame.up, rc["width"], rc["height"], rc["fov_y_deg"], rc["znear"])
        rgb, _ = render_rgb(scene, cam, antialias)
        save_rgb(out_dir / f"overview_{k}.png", to_uint8(rgb))
        print(f"overview_{k}: eye(a,b,h)={np.round(abh, 2).tolist()}")
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
