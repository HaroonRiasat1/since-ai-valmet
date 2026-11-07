#!/usr/bin/env python3
"""Stage 5: match lifted masks across views into 3D instances, vote, clean up and box them."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.config import load_config, resolve_repo_path
from splat_objects.instances import (
    Instance,
    affinity_matrix,
    agglomerate,
    box_containment,
    cleanup,
    detection_consistency,
    load_lifting,
    local_box,
    min_scale_normals,
    oriented_box,
    vote,
)
from splat_objects.lift import w0_lut
from splat_objects.ply_io import read_binary_gaussian_ply
from splat_objects.scene import SceneFrame, write_json


def merge_pair(a: Instance, b: Instance) -> Instance:
    members = np.union1d(a.members, b.members)
    ratios = np.zeros(len(members), dtype=np.float32)
    for inst in (a, b):
        idx = np.searchsorted(members, inst.members)
        ratios[idx] = np.maximum(ratios[idx], inst.ratios)
    props = a.proposals + b.proposals
    return Instance(
        class_id=a.class_id,
        proposals=props,
        members=members,
        ratios=ratios,
        views=sorted(set(a.views) | set(b.views)),
        confidence=float(np.mean([q.score for q in props])),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ic = cfg["instances"]
    scene_id = cfg["scene_id"]
    classes = cfg["classes"]
    lift_dir = _REPO / "outputs" / scene_id / "lifting"
    out_dir = _REPO / "outputs" / scene_id / "instances"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cams = json.loads((_REPO / "renders" / scene_id / "cameras.json").read_text(encoding="utf-8"))

    t0 = time.time()
    ply = read_binary_gaussian_ply(resolve_repo_path(cfg, cfg["input_ply"]))
    frame = SceneFrame.load(_REPO / "outputs" / scene_id / "scene_overview" / "scene_frame.npz")
    xyz = ply.xyz()
    heights = frame.heights(xyz)
    scales = ply.scales_exp()
    smax = scales.max(axis=1)
    rows = ply.raw_rows
    normals = min_scale_normals(np.stack([rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"]], axis=1), scales)
    vis, w0q, props = load_lifting(lift_dir, len(cams))
    lut = w0_lut()
    props = [p for p in props if len(p.members) >= int(ic["min_proposal_gaussians"])]

    def finish(inst: Instance) -> Instance | None:
        floor_standing = bool(classes[inst.class_id].get("floor_standing", False))
        keep = cleanup(inst.members, xyz, heights, normals, smax, frame.up, floor_standing, ic)
        if len(keep) < int(ic["min_gaussians"]):
            return None
        inst.ratios = inst.ratios[np.searchsorted(inst.members, keep)]
        inst.members = keep
        inst.detect_ratio, inst.n_visible_views = detection_consistency(
            keep, inst.views, vis, w0q, lut, float(ic["min_visible_px"]), float(ic["min_visible_member_frac"])
        )
        return inst

    instances: list[Instance] = []
    for cid, c in enumerate(classes):
        cp = [p for p in props if p.class_id == cid]
        if not cp:
            print(f"{c['display']}: no proposals")
            continue
        aff, known = affinity_matrix(cp, vis, int(ic["min_covisible"]))
        clusters = agglomerate(cp, aff, known, float(ic["merge_affinity"]))
        kept = dropped_views = dropped_ratio = 0
        for cl in clusters:
            group = [cp[i] for i in cl]
            if len({q.view for q in group}) < int(ic["min_views"]):
                dropped_views += 1
                continue
            members, ratios, views = vote(
                group,
                vis,
                xyz,
                core_ratio=float(ic["core_vote_ratio"]),
                grow_ratio=float(ic["grow_vote_ratio"]),
                radius_scale=float(ic["grow_radius_scale"]),
                radius_min=float(ic["grow_radius_min_m"]),
                radius_max=float(ic["grow_radius_max_m"]),
                max_iters=int(ic["grow_max_iters"]),
            )
            inst = finish(
                Instance(
                    class_id=cid,
                    proposals=group,
                    members=members,
                    ratios=ratios,
                    views=views,
                    confidence=float(np.mean([q.score for q in group])),
                )
            )
            if inst is None:
                continue
            if inst.detect_ratio < float(ic["min_detect_ratio"]):
                dropped_ratio += 1
                continue
            instances.append(inst)
            kept += 1
        print(
            f"{c['display']}: proposals={len(cp)} clusters={len(clusters)} kept={kept} "
            f"(single-view={dropped_views}, inconsistent={dropped_ratio})"
        )

    # Merge same-class duplicates (e.g. one object split into front/back clusters).
    merged = True
    while merged:
        merged = False
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                a, b = instances[i], instances[j]
                if a.class_id != b.class_id:
                    continue
                inter = np.intersect1d(a.members, b.members, assume_unique=True).size
                overlap = inter / max(1, min(len(a.members), len(b.members)))
                contain = box_containment(local_box(frame, xyz[a.members]), local_box(frame, xyz[b.members]))
                if overlap >= float(ic["dup_overlap"]) or contain >= float(ic["dup_box_containment"]):
                    m = finish(merge_pair(a, b))
                    instances[i] = m if m is not None else a
                    del instances[j]
                    merged = True
                    break
            if merged:
                break

    # Each Gaussian belongs to at most one object: highest vote ratio x confidence wins.
    owner = np.full(ply.n, -1, dtype=np.int64)
    best = np.zeros(ply.n, dtype=np.float32)
    for k, inst in enumerate(instances):
        s = inst.ratios * inst.confidence
        better = s > best[inst.members]
        owner[inst.members[better]] = k
        best[inst.members[better]] = s[better]
    final: list[Instance] = []
    for k, inst in enumerate(instances):
        keep = owner[inst.members] == k
        inst.members, inst.ratios = inst.members[keep], inst.ratios[keep]
        inst = finish(inst) if len(inst.members) >= int(ic["min_gaussians"]) else None
        if inst is not None:
            final.append(inst)

    final.sort(key=lambda x: (x.class_id, -x.confidence))
    records, member_arrays, per_class = [], {}, {}
    for inst in final:
        c = classes[inst.class_id]
        P = xyz[inst.members]
        box = oriented_box(frame, P)
        height = box["top_height_m"] - box["bottom_height_m"]
        reasons = []
        if len(inst.members) < int(c.get("min_gaussians", ic["min_gaussians"])):
            reasons.append(f"n={len(inst.members)}")
        if height < float(c.get("min_height_m", 0.0)):
            reasons.append(f"height={height:.2f}m")
        if box["top_height_m"] < float(c.get("min_top_m", -1e9)):
            reasons.append(f"top={box['top_height_m']:.2f}m")
        if max(box["size_m"]) > float(c.get("max_size_m", 1e9)):
            reasons.append(f"size={max(box['size_m']):.2f}m")
        if reasons:
            print(f"  drop {c['display']} at {np.round(P.mean(axis=0), 2).tolist()}: {', '.join(reasons)}")
            continue
        k = per_class.get(c["display"], 0)
        per_class[c["display"]] = k + 1
        name = f'{c["display"].replace(" ", "_")}_{k:02d}'
        best = max(inst.proposals, key=lambda q: q.mask_px)
        records.append(
            {
                "id": len(records),
                "name": name,
                "class_id": inst.class_id,
                "class_key": c["key"],
                "label": c["display"],
                "confidence": round(inst.confidence, 4),
                "detect_ratio": round(inst.detect_ratio, 3),
                "n_detected_views": len(inst.views),
                "n_visible_views": inst.n_visible_views,
                "view_ids": [int(cams[v]["id"]) for v in inst.views],
                "n_gaussians": int(len(inst.members)),
                "centroid": P.mean(axis=0).tolist(),
                "box": box,
                "best_view": int(best.view),
                "best_det": int(best.det),
                "proposals": [{"view": int(q.view), "det": int(q.det), "score": round(q.score, 4)} for q in inst.proposals],
            }
        )
        member_arrays[name] = inst.members.astype(np.int64)

    write_json(out_dir / "instances.json", records)
    np.savez_compressed(out_dir / "members.npz", **member_arrays)
    print(f"Built {len(records)} instances in {time.time() - t0:.1f}s: {per_class}")


if __name__ == "__main__":
    main()
