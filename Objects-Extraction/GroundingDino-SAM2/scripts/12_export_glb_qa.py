#!/usr/bin/env python3
"""Stage 12: export scan-frame GLB + index for the web viewer and QA reports."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.bim.glb import export_glb
from splat_objects.bim.qa import write_qa
from splat_objects.config import load_config, repo_root
from splat_objects.scene import SceneFrame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scene_id = cfg["scene_id"]
    repo = repo_root()
    bim = cfg["bim"]
    frame = SceneFrame.load(repo / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
    bim_dir = repo / "outputs" / scene_id / "bim"
    elements = json.loads((bim_dir / "elements_fitted.json").read_text(encoding="utf-8"))["elements"]
    structure = json.loads((bim_dir / "structure.json").read_text(encoding="utf-8"))
    merge_stats = json.loads((bim_dir / "elements.json").read_text(encoding="utf-8"))["stats"]
    ifc_result = json.loads((bim_dir / "ifc_result.json").read_text(encoding="utf-8"))
    guid_map = json.loads((bim_dir / "guid_map.json").read_text(encoding="utf-8"))
    ifc_result["guid_map"] = guid_map

    glb_name = "equipment_room.glb"
    glb_path = bim_dir / glb_name
    index_path = bim_dir / "index.json"
    export_glb(cfg, frame, elements, structure, ifc_result, glb_path, index_path)
    write_qa(bim_dir, elements, structure, merge_stats, ifc_result)

    web_dir = (repo / bim["app_root"]).resolve() / bim["web_bim_dir"]
    web_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(glb_path, web_dir / glb_name)
    shutil.copy2(index_path, web_dir / "index.json")
    print(f"Copied GLB + index -> {web_dir}")


if __name__ == "__main__":
    main()
