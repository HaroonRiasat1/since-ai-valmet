#!/usr/bin/env python3
"""Stage 13 (optional): rebuild IFC + GLB from a layout.json exported by web/layout.html."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.bim.glb import export_glb
from splat_objects.bim.ifc_writer import write_ifc
from splat_objects.bim.layout import apply_layout
from splat_objects.config import load_config, repo_root
from splat_objects.scene import SceneFrame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    ap.add_argument("--layout", type=Path, required=True, help="layout.json downloaded from the layout editor")
    ap.add_argument("--out-name", default="layout", help="output folder name under outputs/<scene>/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scene_id = cfg["scene_id"]
    repo = repo_root()
    scale = float(cfg["bim"].get("units_to_m", 1.0))
    frame = SceneFrame.load(repo / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
    src = repo / "outputs" / scene_id / "bim"
    out = repo / "outputs" / scene_id / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    elements = json.loads((src / "elements_fitted.json").read_text(encoding="utf-8"))["elements"]
    structure = json.loads((src / "structure.json").read_text(encoding="utf-8"))
    guid_map = json.loads((src / "guid_map.json").read_text(encoding="utf-8"))

    moved, n = apply_layout(elements, layout, guid_map, frame, scale, src, out)
    ifc_path = out / f"{scene_id}_layout.ifc"
    result = write_ifc(cfg, repo, scene_id, frame, moved, structure, ifc_path, out / "guid_map.json", bim_dir=out)
    result["guid_map"] = json.loads((out / "guid_map.json").read_text(encoding="utf-8"))
    export_glb(cfg, frame, moved, structure, result, out / "equipment_room_layout.glb", out / "index.json")
    (out / "layout_applied.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    print(f"Applied layout: {n} objects moved -> {ifc_path}")
    print(f"GLB: {out / 'equipment_room_layout.glb'}")


if __name__ == "__main__":
    main()
