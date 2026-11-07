#!/usr/bin/env python3
"""Stage 10: fit parametric shapes and compute fit statistics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.bim.meshing import mesh_fitted_elements
from splat_objects.bim.shapes import fit_elements
from splat_objects.bim.surface import backproject_views
from splat_objects.config import load_config, repo_root
from splat_objects.scene import SceneFrame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scene_id = cfg["scene_id"]
    repo = repo_root()
    frame = SceneFrame.load(repo / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
    bim_dir = repo / "outputs" / scene_id / "bim"
    elements = json.loads((bim_dir / "elements.json").read_text(encoding="utf-8"))["elements"]
    structure = json.loads((bim_dir / "structure.json").read_text(encoding="utf-8"))
    render_root = repo / "renders" / scene_id
    pts, _ = backproject_views(render_root, frame, cfg, stride=3)
    fitted = fit_elements(cfg, repo, scene_id, frame, elements, structure, pts)
    mesh_stats = mesh_fitted_elements(cfg, repo, scene_id, frame, fitted, bim_dir)
    (bim_dir / "elements_fitted.json").write_text(json.dumps({"elements": fitted}, indent=2) + "\n", encoding="utf-8")
    avg = float(np.mean([e.get("fit", {}).get("within_frac", 0) for e in fitted])) if fitted else 0.0
    print(f"Fitted {len(fitted)} elements; mean within-5cm frac {avg:.2%}")
    print(f"  meshes: {mesh_stats}")


if __name__ == "__main__":
    main()
