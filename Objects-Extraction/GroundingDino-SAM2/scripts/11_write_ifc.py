#!/usr/bin/env python3
"""Stage 11: write IFC4 with properties, groups and validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.bim.ifc_writer import write_ifc
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
    elements = json.loads((bim_dir / "elements_fitted.json").read_text(encoding="utf-8"))["elements"]
    structure = json.loads((bim_dir / "structure.json").read_text(encoding="utf-8"))
    merge_stats = json.loads((bim_dir / "elements.json").read_text(encoding="utf-8")).get("stats", {})
    ifc_path = bim_dir / f"{scene_id}.ifc"
    guid_path = bim_dir / "guid_map.json"
    ids_path = repo / "bim_requirements.ids"
    result = write_ifc(cfg, repo, scene_id, frame, elements, structure, ifc_path, guid_path)
    (bim_dir / "ifc_result.json").write_text(json.dumps({k: v for k, v in result.items() if k != "elements"}, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {ifc_path} ({result['n_products']} products); validation ok={result['validation'].get('ok')}")


if __name__ == "__main__":
    main()
