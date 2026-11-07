#!/usr/bin/env python3
"""Stage 9: back-project depth maps and infer walls, floor, ceiling and doors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.bim.structure import build_structure
from splat_objects.bim.surface import backproject_views, label_points
from splat_objects.config import load_config, repo_root
from splat_objects.scene import SceneFrame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    ap.add_argument("--stride", type=int, default=2, help="camera stride for faster surface build")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scene_id = cfg["scene_id"]
    repo = repo_root()
    frame = SceneFrame.load(repo / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
    bim_dir = repo / "outputs" / scene_id / "bim"
    data = json.loads((bim_dir / "elements.json").read_text(encoding="utf-8"))
    elements = data["elements"]
    render_root = repo / "renders" / scene_id
    pts, nrms = backproject_views(render_root, frame, cfg, stride=args.stride)
    labels = label_points(frame, pts, nrms, elements, cfg) if len(pts) else {"floor": np.array([]), "ceiling": np.array([]), "wall": np.array([]), "heights": np.array([])}
    structure = build_structure(frame, elements, pts, labels, cfg)
    (bim_dir / "structure.json").write_text(json.dumps(structure, indent=2) + "\n", encoding="utf-8")
    _topdown_debug(repo, scene_id, structure)
    print(f"Structure: {len(structure['walls'])} walls, ceiling {structure['ceiling_height_m']:.2f} m")


def _topdown_debug(repo: Path, scene_id: str, structure: dict) -> None:
    img_path = repo / "outputs" / scene_id / "scene_overview" / "topdown_plain.png"
    if not img_path.is_file():
        return
    img = cv2.imread(str(img_path))
    scale = float(structure.get("assumptions", {}).get("scale", 1.0))
    for w in structure.get("walls", []):
        p0 = tuple(int(x) for x in np.asarray(w["p0_m"])[:2] * 50 + 200)  # rough overlay
        p1 = tuple(int(x) for x in np.asarray(w["p1_m"])[:2] * 50 + 200)
        cv2.line(img, p0, p1, (0, 255, 255), 2)
    out = repo / "outputs" / scene_id / "bim" / "structure_topdown.png"
    cv2.imwrite(str(out), img)


if __name__ == "__main__":
    main()
