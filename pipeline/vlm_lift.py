"""
Lifts the viewer's "VLM detect" run (2D boxes per rendered frame) to 3D boxes and fuses them across
views into an items file, so assets are found by the VLM instead of the placeholder blob detector.

    python pipeline/vlm_lift.py Equipment_Room.ply web/data/vlm_views data/items_vlm.json [run]

Per detection: project the opaque splats with the frame's camera, take the front surface inside the
inner half of the 2D box as the object's depth, keep the splats of the full 2D box within a depth band
around it, and box them (5-95th percentiles). Detections of the same class whose centres are close are
merged; a cluster needs 2+ views or confidence >= 0.8, so one-off hallucinations drop out.
"""
import glob, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import navigator as nv

ply, views_dir, out_path = sys.argv[1:4]
runs = sorted(d for d in os.listdir(views_dir) if os.path.isdir(os.path.join(views_dir, d)))
run = sys.argv[4] if len(sys.argv) > 4 else runs[-1]            # latest run by default
views = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(views_dir, run, "*.json")))]
print(f"run {run}: {len(views)} views, {sum(len(v['detections']) for v in views)} 2D detections")

xyz, op = nv.load_splats(ply)
hgt, floor = nv.height_above_floor(xyz, op)
keep = (op > 0.5) & (hgt > 0.03) & (hgt < 1.9)                 # no floor, no ceiling
P = xyz[keep]
P1 = np.c_[P, np.ones(len(P))]
HZ = [a for a in range(3) if a != nv.UP]

dets = []
for v in views:
    M = np.array(v["pose"]).reshape(4, 4).T                      # three.js matrices are column-major
    Pr = np.array(v["proj"]).reshape(4, 4).T
    cam = P1 @ np.linalg.inv(M).T
    clip = cam @ Pr.T
    ok = clip[:, 3] > 0.05
    u = np.full(len(P), -1.0); w = np.full(len(P), -1.0)
    u[ok] = (clip[ok, 0] / clip[ok, 3] + 1) / 2
    w[ok] = (1 - clip[ok, 1] / clip[ok, 3]) / 2
    depth = -cam[:, 2]
    tan_h = 1 / Pr[0, 0]                                         # tan(horizontal fov / 2)
    for d in v["detections"]:
        x1, y1, x2, y2 = np.array(d["bbox_2d"]) / 1000
        if x2 - x1 < 0.01 or y2 - y1 < 0.01 or (x2 - x1) * (y2 - y1) > 0.6:   # degenerate or whole-frame boxes
            continue
        cx, cy, hx, hy = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
        inner = ok & (abs(u - cx) <= hx) & (abs(w - cy) <= hy)
        if inner.sum() < 30:
            continue
        d0 = np.percentile(depth[inner], 20)                     # front surface of the object
        width = (x2 - x1) * 2 * d0 * tan_h                       # its rough size from the box's angular width
        full = ok & (u >= x1) & (u <= x2) & (w >= y1) & (w <= y2)
        band = full & (depth > d0 - 0.15) & (depth < d0 + max(0.3, width))
        if band.sum() < 30:
            continue
        lo, hi = np.percentile(P[band], 5, 0), np.percentile(P[band], 95, 0)
        size = hi - lo
        if size[HZ].max() > 3.0:                                 # a wall or a whole row, not one asset
            continue
        dets.append(dict(label=d["label"], asset_class=d["asset_class"], confidence=d["confidence"],
                         center=(lo + hi) / 2, size=size, view=v["view_id"]))
print(f"lifted {len(dets)} detections to 3D")

clusters = []
for d in sorted(dets, key=lambda d: -d["confidence"]):
    for c in clusters:
        ref = c[0]
        if ref["asset_class"] == d["asset_class"] and \
           np.linalg.norm((ref["center"] - d["center"])[HZ]) < max(0.4, 0.5 * ref["size"][HZ].max()):
            c.append(d)
            break
    else:
        clusters.append([d])

items = []
for c in clusters:
    nviews = len({d["view"] for d in c})
    conf = float(np.mean([d["confidence"] for d in c]))
    if nviews < 2 and conf < 0.8:
        continue
    center, size = np.median([d["center"] for d in c], 0), np.median([d["size"] for d in c], 0)
    # never below the floor: in the .ply frame the floor is the bottom of the box
    bottom = (center[nv.UP] + nv.DOWN_SIGN * size[nv.UP] / 2) * nv.DOWN_SIGN
    if bottom > floor:
        center[nv.UP] -= nv.DOWN_SIGN * (bottom - floor)
    items.append(dict(label=Counter(d["label"] for d in c).most_common(1)[0][0], asset_class=c[0]["asset_class"],
                      confidence=round(conf, 2), views=nviews, source="vlm",
                      center=center.round(3).tolist(), size=size.round(3).tolist()))
items.sort(key=lambda i: (i["center"][HZ[0]], i["center"][HZ[1]]))
for k, it in enumerate(items, 1):
    it["id"] = f"VLM-{k:03d}"
items = [{"id": i.pop("id"), **i} for i in items]
json.dump(items, open(out_path, "w"), indent=1)
print(f"{len(items)} objects -> {out_path}:", dict(Counter(i["asset_class"] for i in items)))
