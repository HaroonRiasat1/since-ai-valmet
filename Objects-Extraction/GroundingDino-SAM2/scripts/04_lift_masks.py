#!/usr/bin/env python3
"""Stage 4: lift every SAM mask to the Gaussians that actually produced its pixels."""
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

from splat_objects.cameras import camera_from_record
from splat_objects.config import load_config, resolve_repo_path
from splat_objects.detect import rle_to_mask
from splat_objects.lift import blend_weights, encode_w0
from splat_objects.ply_io import read_binary_gaussian_ply
from splat_objects.render_mlx import SplatScene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    lc, dc = cfg["lift"], cfg["detection"]
    scene_id = cfg["scene_id"]
    render_root = _REPO / "renders" / scene_id
    det_dir = _REPO / "outputs" / scene_id / "detections"
    out_dir = _REPO / "outputs" / scene_id / "lifting"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    ply_path = resolve_repo_path(cfg, cfg["input_ply"])
    smax = read_binary_gaussian_ply(ply_path).scales_exp().max(axis=1)
    scene = SplatScene.load(ply_path)
    cams = json.loads((render_root / "cameras.json").read_text(encoding="utf-8"))
    thresholds = [float(c.get("box_threshold", dc["default_box_threshold"])) for c in cfg["classes"]]
    antialias = bool(lc["antialias"])
    vis_min, in_frac = float(lc["visible_weight_min"]), float(lc["in_mask_frac"])
    small = smax <= float(lc["max_gaussian_scale_m"])
    per_pass = max(1, int(lc["max_channels_per_pass"]) - 1)
    kernel = np.ones((3, 3), np.uint8)

    n_props = 0
    t0 = time.time()
    for vidx, cam in enumerate(cams):
        stem = Path(cam["image"]).stem
        payload = json.loads((det_dir / f"{stem}.json").read_text(encoding="utf-8"))
        anns = [
            (k, a) for k, a in enumerate(payload["annotations"]) if a["score"] >= thresholds[int(a["class_id"])]
        ]
        masks = []
        for _, a in anns:
            m = rle_to_mask(a["segmentation"])
            if int(lc["mask_erode_px"]) > 0:
                m = cv2.erode(m.astype(np.uint8), kernel, iterations=int(lc["mask_erode_px"])).astype(bool)
            masks.append(m)

        cam_mx = camera_from_record(cam)
        H, W = int(cam["height"]), int(cam["width"])
        w0 = None
        wk = []
        for start in range(0, max(1, len(masks)), per_pass):
            chunk = masks[start : start + per_pass]
            ch = np.zeros((H, W, 1 + len(chunk)), dtype=np.float32)
            ch[..., 0] = 1.0
            for j, m in enumerate(chunk):
                ch[..., 1 + j] = m
            w = blend_weights(scene, cam_mx, ch, antialias)
            if w0 is None:
                w0 = w[:, 0]
            wk.append(w[:, 1:])
        wk = np.concatenate(wk, axis=1) if wk else np.zeros((scene.n, 0), np.float32)
        vis = w0 >= vis_min

        members, offsets, props = [], [0], []
        for j, (det_idx, a) in enumerate(anns):
            col = wk[:, j]
            mem = np.nonzero(vis & small & (col >= in_frac * w0))[0].astype(np.int32)
            members.append(mem)
            offsets.append(offsets[-1] + len(mem))
            props.append(
                {
                    "det_idx": det_idx,
                    "class_id": int(a["class_id"]),
                    "class_key": a["class_key"],
                    "label": a["class_name"],
                    "score": float(a["score"]),
                    "sam_score": float(a["sam_score"]),
                    "bbox": a["bbox"],
                    "mask_px": int(a["mask_px"]),
                    "mask_weight": float(col.sum()),
                    "n_members": int(len(mem)),
                }
            )
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            view_idx=vidx,
            view_id=int(cam["id"]),
            n_gaussians=scene.n,
            vis_bits=np.packbits(vis),
            w0_q=encode_w0(w0),
            members=np.concatenate(members) if members else np.zeros(0, np.int32),
            offsets=np.array(offsets, dtype=np.int64),
            props_json=json.dumps(props),
        )
        n_props += len(props)
        if vidx % 20 == 0 or vidx == len(cams) - 1:
            print(
                f"[{vidx + 1}/{len(cams)}] {stem}: visible={int(vis.sum())} proposals={len(props)} "
                f"members={[int(len(m)) for m in members]} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    print(f"Lifted {n_props} masks from {len(cams)} views in {time.time() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
