r"""Rebuild the asset .ply files at higher quality, using multi-view voting.

Run:    .venv\Scripts\python refine_assets.py
Output: assets_hq/<tag>_<class>.ply, all_assets.ply and assets.csv

For each asset this takes every splat in its box from the full scan, outlines the asset with SAM in
several clear views, and keeps the splats that fall inside the outline in most views that see them.
Needs extract.py and semantic.py to have run first.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Sam2Model, Sam2Processor

import extract
import semantic
import view_splat as vs

ROOT = vs.ROOT
FOLDER = ROOT / "extraction" / vs.PLY.stem
OUT = ROOT / "assets_hq"

BOX_PADDING = 0.06        # include splats this far outside the box, in scan units
VIEW_COUNT = 12           # views used per asset
VIEW_RANGE = 0.25, 2.2    # allowed camera distance, in scan units
BOX_SHARE = 0.02, 0.6     # allowed share of the frame covered by the asset
KEEP_SHARE = 0.6          # keep a splat inside the outline in this share of the views seeing it
MIN_VOTES = 3             # a splat must be seen in this many views
MAX_BLOCKERS = 15         # views with more splats in the sight line are skipped
MIN_ALPHA = 0.1           # fainter splats look like haze once the object stands alone
MAX_SPREAD = 0.07         # larger splats are background blur, in scan units


def load_views():
    """All captured views with their view matrices, camera positions and image paths."""
    views = []
    for name in ("frames", "frames_close"):
        meta = json.loads((FOLDER / name / "frames.json").read_text())
        for frame in meta["frames"]:
            view = np.array(frame["view"])
            views.append({"path": FOLDER / name / frame["file"], "view": view,
                          "eye": -view[:3, :3].T @ view[:3, 3]})
    size = tuple(meta["size"])
    return views, size, size[1] / 2 / np.tan(meta["fov_y"] / 2)


def box_corners(low, high):
    return np.array([[(low, high)[i >> b & 1][b] for b in range(3)] for i in range(8)])


def pixel_box(corners, view, focal, size):
    """Frame rectangle of a 3D box, or None when it is not fully in view at a usable distance."""
    px, py, depth = extract.project(corners, view, focal, size)
    if depth.min() < VIEW_RANGE[0] or depth.max() > VIEW_RANGE[1]:
        return None
    if px.min() < 0 or py.min() < 0 or px.max() >= size[0] or py.max() >= size[1]:
        return None
    share = (px.max() - px.min()) * (py.max() - py.min()) / (size[0] * size[1])
    if not BOX_SHARE[0] <= share <= BOX_SHARE[1]:
        return None
    return float(px.min()), float(py.min()), float(px.max()), float(py.max()), share


def pick_views(low, high, views, focal, size, solid):
    """The largest unblocked views of a box, spread over different camera positions."""
    corners = box_corners(low, high)
    center = (low + high) / 2
    found = []
    for view in views:
        box = pixel_box(corners, view["view"], focal, size)
        if box is not None:
            found.append((box[4], view, box[:4]))
    found.sort(key=lambda item: -item[0])
    nearby = solid[np.linalg.norm(solid - center, axis=1) < VIEW_RANGE[1] + 0.3]
    half = np.linalg.norm(high - low) / 2
    chosen = []
    for share, view, box in found:
        if len(chosen) == VIEW_COUNT:
            break
        if any(np.linalg.norm(view["eye"] - other["eye"]) < 0.2 for other, _ in chosen):
            continue  # keep the views spread out, so the wall behind gets voted away
        stop = center + (view["eye"] - center) / np.linalg.norm(view["eye"] - center) * half
        if semantic.blockers(view["eye"], stop, nearby, 0.1) <= MAX_BLOCKERS:
            chosen.append((view, box))
    return chosen


@torch.inference_mode()
def outline(image, box, processor, model):
    """SAM's outline of the object in a box, at full frame size."""
    inputs = processor(images=image, input_boxes=[[list(box)]], return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        mask = model(**inputs, multimask_output=False).pred_masks[0, :, 0]
    mask = F.interpolate(mask[None].float(), size=image.size[::-1], mode="bilinear")[0, 0] > 0
    return mask.cpu().numpy()


def vote(points, chosen, focal, size, masks):
    """Return how many views saw each point, and how many had it inside the outline."""
    seen, inside = np.zeros(len(points), dtype=int), np.zeros(len(points), dtype=int)
    for (view, _), mask in zip(chosen, masks):
        px, py, depth = extract.project(points, view["view"], focal, size)
        visible = (depth > 0.05) & (px >= 0) & (px < size[0]) & (py >= 0) & (py < size[1])
        ids = np.flatnonzero(visible)
        seen[ids] += 1
        inside[ids[mask[py[ids].astype(int), px[ids].astype(int)]]] += 1
    return seen, inside


def main():
    OUT.mkdir(exist_ok=True)
    layer = json.loads((FOLDER / "semantic_layer.json").read_text())
    objects = sorted((o for o in layer["objects"] if o["mapping_method"] == "class_match"),
                     key=lambda o: o["asset"]["master"]["tag"])
    v, plane, band, up_sign, heights = extract.load_scan(vs.PLY)
    pos = np.stack([v[k].astype("f8") for k in "xyz"], axis=1)
    spread = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).max(axis=1)
    visible = (v["opacity"] >= np.log(MIN_ALPHA / (1 - MIN_ALPHA))) & (spread <= MAX_SPREAD)
    above_floor = heights > vs.object_clearance(heights[v["opacity"] >= extract.SOLID], band)
    solid = pos[v["opacity"] >= extract.SOLID]
    views, size, focal = load_views()
    print(f"{len(views)} views available", flush=True)

    processor = Sam2Processor.from_pretrained(extract.SAM_MODEL)
    model = Sam2Model.from_pretrained(extract.SAM_MODEL).to("cuda").eval()
    rows, parts = [], []
    for obj in objects:
        low, high = np.array(obj["box"]["min"]), np.array(obj["box"]["max"])
        chosen = pick_views(low, high, views, focal, size, solid)
        tag = obj["asset"]["master"]["tag"]
        name = f"{tag}_{obj['class'].replace(' ', '_')}.ply"
        in_box = ((pos >= low - BOX_PADDING) & (pos <= high + BOX_PADDING)).all(axis=1) & visible & above_floor
        ids = np.flatnonzero(in_box)
        if len(chosen) < MIN_VOTES:
            print(f"{tag}: only {len(chosen)} clear views, keeping everything in its box", flush=True)
            kept = ids
        else:
            masks = [outline(Image.open(view["path"]).convert("RGB"), box, processor, model) for view, box in chosen]
            seen, inside = vote(pos[ids], chosen, focal, size, masks)
            keep = (seen >= MIN_VOTES) & (inside >= KEEP_SHARE * seen)
            kept = ids[keep]
            if len(kept) >= extract.MIN_SPLATS:
                kept = kept[extract.connected_part(pos[kept])]
        # Never go below the existing file: if the vote lost the object, keep what was already there.
        earlier = vs.read_ply(ROOT / "assets" / name)
        if len(kept) < len(earlier):
            print(f"{tag}: vote kept {len(kept)}, fewer than the {len(earlier)} already saved, so using those", flush=True)
            kept = np.flatnonzero(((pos >= low) & (pos <= high)).all(axis=1) & np.isin(v["x"], earlier["x"]))
            method = "existing"
        else:
            method = "voted"
        extract.save_ply(OUT / name, v, kept)
        parts.append(kept)
        state = obj["asset"]["state"]
        rows.append([name, tag, obj["class"], state["condition_state"], state["health_score"], len(kept), len(ids),
                     len(chosen), method, obj["id"], obj["class_confidence"]])
        print(f"{tag}: {len(ids)} in box, {len(kept)} kept, {len(chosen)} views", flush=True)
    extract.save_ply(OUT / "all_assets.ply", v, np.unique(np.concatenate(parts)))
    with open(OUT / "assets.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["file", "tag", "class", "condition", "health", "splats", "splats_in_box", "views", "method",
                      "object", "label_confidence"])
        out.writerows(rows)
    print(f"Saved {len(rows)} assets, {sum(r[5] for r in rows):,} splats in total")


if __name__ == "__main__":
    # Self-check: only a point inside the outline in most views survives the vote rule.
    demo_seen, demo_inside = np.array([4, 4, 2]), np.array([4, 1, 2])
    demo_keep = (demo_seen >= MIN_VOTES) & (demo_inside >= KEEP_SHARE * demo_seen)
    assert demo_keep.tolist() == [True, False, False]
    assert len(box_corners(np.zeros(3), np.ones(3))) == 8
    main()
