r"""Save the objects whose Excel asset matches their type as separate .ply files.

Run:    .venv\Scripts\python export_assets.py
Output: assets/<tag>_<class>.ply   one Gaussian splat file per asset, cropped to its 3D box
        assets/all_assets.ply      all of them together, in their positions in the room
        assets/assets.csv          what each file is

View one with: .venv\Scripts\python view_splat.py --object "assets/P-005_process_pump.ply"
"""
import csv
import json
from pathlib import Path

import numpy as np

import extract
import view_splat as vs

ROOT = vs.ROOT
SCAN = vs.PLY.stem
FOLDER = ROOT / "extraction" / SCAN
OUT = ROOT / "assets"


def crop_to_box(v, low, high):
    """Keep the splats inside a box."""
    pos = np.stack([v[k].astype("f8") for k in "xyz"], axis=1)
    return ((pos >= low) & (pos <= high)).all(axis=1)


def main():
    OUT.mkdir(exist_ok=True)
    objects = [o for o in json.loads((FOLDER / "semantic_layer.json").read_text())["objects"]
               if o["mapping_method"] == "class_match"]
    rows, parts = [], []
    for obj in sorted(objects, key=lambda o: o["asset"]["master"]["tag"]):
        v = vs.read_ply(FOLDER / obj["ply"])
        kept = v[crop_to_box(v, np.array(obj["box"]["min"]), np.array(obj["box"]["max"]))]
        tag = obj["asset"]["master"]["tag"]
        name = f"{tag}_{obj['class'].replace(' ', '_')}.ply"
        extract.save_ply(OUT / name, kept, np.arange(len(kept)))
        parts.append(kept)
        state = obj["asset"]["state"]
        rows.append([name, tag, obj["class"], state["condition_state"], state["health_score"], len(kept), obj["id"],
                     obj["class_confidence"], *obj["box"]["size"]])
    every = np.concatenate(parts)
    extract.save_ply(OUT / "all_assets.ply", every, np.arange(len(every)))
    with open(OUT / "assets.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["file", "tag", "class", "condition", "health", "splats", "object", "label_confidence",
                      "size_x", "size_y", "size_z"])
        out.writerows(rows)
    print(f"Saved {len(rows)} assets to {OUT}, {sum(r[5] for r in rows):,} splats in total")


if __name__ == "__main__":
    # Self-check: only points inside the box are kept.
    demo = np.zeros(2, dtype=[(k, "<f4") for k in "xyz"])
    demo["x"][1] = 5
    assert crop_to_box(demo, np.zeros(3) - 1, np.ones(3)).tolist() == [True, False]
    main()
