#!/usr/bin/env python3
"""Stage 6: write one splat .ply per object, a manifest, and visual previews."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.cameras import camera_from_record, project_points
from splat_objects.config import load_config, resolve_repo_path, slug
from splat_objects.export import draw_box, recolored, write_subset
from splat_objects.ply_io import read_binary_gaussian_ply, write_binary_gaussian_ply
from splat_objects.render_mlx import SplatScene, load_rgb, render_rgb, to_uint8
from splat_objects.scene import write_json
from splat_objects.viz import class_color_bgr, class_color_rgb01, contact_sheets, draw_label


def crop_box(alpha: np.ndarray, fallback: list[float], pad_frac: float = 0.25, min_side: int = 220) -> tuple[int, int, int, int]:
    h, w = alpha.shape
    ys, xs = np.nonzero(alpha > 0.15)
    if len(xs) > 20:
        x0, x1 = np.percentile(xs, [0.5, 99.5])
        y0, y1 = np.percentile(ys, [0.5, 99.5])
    else:
        x0, y0, x1, y1 = fallback
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    side_x = max(min_side, (x1 - x0) * (1 + 2 * pad_frac))
    side_y = max(min_side, (y1 - y0) * (1 + 2 * pad_frac))
    x0, x1 = int(max(0, cx - side_x / 2)), int(min(w, cx + side_x / 2))
    y0, y1 = int(max(0, cy - side_y / 2)), int(min(h, cy + side_y / 2))
    return x0, y0, x1, y1


def fit_height(img: np.ndarray, height: int) -> np.ndarray:
    scale = height / img.shape[0]
    return cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), height), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scene_id = cfg["scene_id"]
    render_root = _REPO / "renders" / scene_id
    overview = _REPO / "outputs" / scene_id / "scene_overview"
    inst_dir = _REPO / "outputs" / scene_id / "instances"
    det_dir = _REPO / "outputs" / scene_id / "detections"
    out_dir = _REPO / "outputs" / scene_id / "objects"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "previews").mkdir(parents=True)

    ply_path = resolve_repo_path(cfg, cfg["input_ply"])
    ply = read_binary_gaussian_ply(ply_path)
    scene = SplatScene.load(ply_path)
    cams = json.loads((render_root / "cameras.json").read_text(encoding="utf-8"))
    records = json.loads((inst_dir / "instances.json").read_text(encoding="utf-8"))
    members = np.load(inst_dir / "members.npz")
    antialias = bool(cfg["render"]["antialias"])

    claimed = np.zeros(ply.n, dtype=bool)
    groups = []
    for r in records:
        idx = members[r["name"]]
        path = out_dir / slug(r["label"]) / f'{r["name"]}.ply'
        write_subset(ply, idx, path)
        r["ply"] = str(path.relative_to(out_dir))
        claimed[idx] = True
        groups.append((idx, class_color_rgb01(r["class_id"])))
    write_binary_gaussian_ply(out_dir / "scene_remainder.ply", ply, indices=np.nonzero(~claimed)[0])
    write_binary_gaussian_ply(out_dir / "scene_segmented_colored.ply", recolored(ply, groups))

    tiles = []
    for r in records:
        cam_rec = cams[r["best_view"]]
        full = cv2.cvtColor(load_rgb(render_root / cam_rec["image"]), cv2.COLOR_RGB2BGR)
        obj_rgb, obj_alpha = render_rgb(scene.subset(members[r["name"]]), camera_from_record(cam_rec), antialias, (0.12, 0.12, 0.12))
        det = json.loads((det_dir / f'{Path(cam_rec["image"]).stem}.json').read_text(encoding="utf-8"))
        x0, y0, x1, y1 = crop_box(obj_alpha, det["annotations"][r["best_det"]]["bbox"])
        color = class_color_bgr(r["class_id"])
        draw_box(full, cam_rec, np.array(r["box"]["corners"]), color)
        left = full[y0:y1, x0:x1]
        right = cv2.cvtColor(to_uint8(obj_rgb), cv2.COLOR_RGB2BGR)[y0:y1, x0:x1]
        tile = np.concatenate([fit_height(left, 360), np.full((360, 6, 3), 255, np.uint8), fit_height(right, 360)], axis=1)
        cap = f'{r["name"]}  conf {r["confidence"]:.2f}  views {r["n_detected_views"]}  {r["n_gaussians"]} splats'
        draw_label(tile, cap, 4, 22, color, 0.6)
        cv2.imwrite(str(out_dir / "previews" / f'{r["name"]}.jpg'), tile, [cv2.IMWRITE_JPEG_QUALITY, 90])
        tiles.append((r["name"], tile))
    contact_sheets(tiles, out_dir / "previews", "gallery", cols=3, rows=4, thumb=(560, 280))

    td_rec = json.loads((overview / "topdown_camera.json").read_text(encoding="utf-8"))
    td = cv2.imread(str(overview / "topdown_plain.png"))
    for r in records:
        bottom = np.array(r["box"]["corners"])[:4]
        uv, _ = project_points(td_rec, bottom)
        color = class_color_bgr(r["class_id"])
        cv2.polylines(td, [uv.astype(np.int32).reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        draw_label(td, r["name"], int(uv[:, 0].min()), int(uv[:, 1].min()) - 2, color, 0.4)
    cv2.imwrite(str(out_dir / "topdown_boxes.png"), td)

    write_json(out_dir / "objects.json", records)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    write_json(out_dir / "summary.json", {"objects_per_class": counts, "n_objects": len(records), "source_ply": str(ply_path)})
    print(f"Exported {len(records)} objects to {out_dir}: {counts}")


if __name__ == "__main__":
    main()
