#!/usr/bin/env python3
"""Stage 8: merge GSAM objects, semantic-layer assets and plant tags into bim/elements.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.bim.merge import merge_elements
from splat_objects.config import load_config, repo_root
from splat_objects.scene import SceneFrame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scene_id = cfg["scene_id"]
    frame = SceneFrame.load(repo_root() / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
    elements, stats = merge_elements(cfg, repo_root(), scene_id, frame)
    out = repo_root() / "outputs" / scene_id / "bim"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"elements": elements, "stats": stats}
    (out / "elements.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(elements)} elements -> {out / 'elements.json'}")


if __name__ == "__main__":
    main()
