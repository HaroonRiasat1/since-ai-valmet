#!/usr/bin/env python3
"""Stage 3: Grounding DINO + SAM 2.1 on every saved render."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.config import load_config
from splat_objects.detect import GroundedSAM2, detections_payload
from splat_objects.render_mlx import load_rgb
from splat_objects.scene import write_json
from splat_objects.viz import contact_sheets, overlay_detections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scene_id = cfg["scene_id"]
    render_root = _REPO / "renders" / scene_id
    cams = json.loads((render_root / "cameras.json").read_text(encoding="utf-8"))
    out_dir = _REPO / "outputs" / scene_id / "detections"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "annotated").mkdir(parents=True)

    model = GroundedSAM2(cfg, _REPO)
    print(f"device={model.device} prompt={model.text!r}")

    counts = {c["display"]: 0 for c in cfg["classes"]}
    per_view: dict[str, dict[str, int]] = {}
    tiles = []
    t0 = time.time()
    for cam in cams:
        stem = Path(cam["image"]).stem
        rgb = load_rgb(render_root / cam["image"])
        dets = model.detect(rgb)
        write_json(out_dir / f"{stem}.json", detections_payload(cam["image"], dets, rgb.shape[1], rgb.shape[0]))
        vc: dict[str, int] = {}
        for d in dets:
            counts[d["label"]] += 1
            vc[d["label"]] = vc.get(d["label"], 0) + 1
        per_view[stem] = vc
        vis = overlay_detections(rgb, dets)
        cv2.imwrite(str(out_dir / "annotated" / f"{stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        tiles.append((f"{stem} n={len(dets)}", vis))
    contact_sheets(tiles, out_dir / "contact_sheets", "detections")
    write_json(out_dir / "summary.json", {"detections_per_class": counts, "per_view": per_view})
    print(f"{len(cams)} views in {time.time() - t0:.1f}s; detections per class: {counts}")


if __name__ == "__main__":
    main()
