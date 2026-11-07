#!/usr/bin/env python3
"""Rebuild docs/assets (the figures used in docs/README.md) from the pipeline outputs.

Figures carry no class names: masks, boxes and splats are colored by class, and objects are numbered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from splat_objects.cameras import camera_from_record, project_points
from splat_objects.config import load_config, resolve_repo_path
from splat_objects.detect import rle_to_mask
from splat_objects.export import draw_box
from splat_objects.ply_io import read_binary_gaussian_ply
from splat_objects.render_mlx import SplatScene, render_rgb, to_uint8
from splat_objects.viz import class_color_bgr, overlay_detections

ASSETS = Path(__file__).resolve().parent / "assets"
FONTS = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf", "/Library/Fonts/Arial.ttf"]


def font(size: int) -> ImageFont.ImageFont:
    for p in FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def read(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save(name: str, bgr: np.ndarray, max_w: int = 1600) -> None:
    if bgr.shape[1] > max_w:
        bgr = cv2.resize(bgr, (max_w, round(bgr.shape[0] * max_w / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    path = ASSETS / name
    if path.suffix == ".png":
        cv2.imwrite(str(path), bgr)
    else:
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", path.relative_to(REPO), bgr.shape[1], "x", bgr.shape[0])


def fit(bgr: np.ndarray, w: int, h: int, bg=(32, 32, 32)) -> np.ndarray:
    s = min(w / bgr.shape[1], h / bgr.shape[0])
    img = cv2.resize(bgr, (max(1, round(bgr.shape[1] * s)), max(1, round(bgr.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), bg, np.uint8)
    y, x = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
    return canvas


def captioned(bgr: np.ndarray, text: str, size: int = 22) -> np.ndarray:
    strip = Image.new("RGB", (bgr.shape[1], size + 18), (255, 255, 255))
    ImageDraw.Draw(strip).text((8, 8), text, fill=(30, 30, 30), font=font(size))
    return np.concatenate([bgr, cv2.cvtColor(np.array(strip), cv2.COLOR_RGB2BGR)], axis=0)


def row(panels: list[np.ndarray], gap: int = 10) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    padded = [np.concatenate([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)]) for p in panels]
    sep = np.full((h, gap, 3), 255, np.uint8)
    out = [padded[0]]
    for p in padded[1:]:
        out += [sep, p]
    return np.concatenate(out, axis=1)


def stack(rows: list[np.ndarray], gap: int = 10) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    padded = [np.concatenate([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)], axis=1) for r in rows]
    out = [padded[0]]
    for r in padded[1:]:
        out += [np.full((gap, w, 3), 255, np.uint8), r]
    return np.concatenate(out, axis=0)


def grid(tiles: list[np.ndarray], cols: int, gap: int = 10) -> np.ndarray:
    return stack([row(tiles[i : i + cols], gap) for i in range(0, len(tiles), cols)], gap)


def crop_to_alpha(alpha: np.ndarray, pad: float = 0.3, min_side: int = 240) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(alpha > 0.15)
    h, w = alpha.shape
    if len(xs) < 20:
        return 0, 0, w, h
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy = max(min_side, (x1 - x0) * (1 + 2 * pad)), max(min_side, (y1 - y0) * (1 + 2 * pad))
    return int(max(0, cx - sx / 2)), int(max(0, cy - sy / 2)), int(min(w, cx + sx / 2)), int(min(h, cy + sy / 2))


def render_bgr(scene: SplatScene, cam_rec: dict, bg=(0.12, 0.12, 0.12)) -> tuple[np.ndarray, np.ndarray]:
    rgb, alpha = render_rgb(scene, camera_from_record(cam_rec), True, bg)
    return cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2BGR), alpha


def pipeline_figure(cards: list[tuple[str, np.ndarray, str]]) -> np.ndarray:
    cw, ih, head, cap, gap, margin = 420, 315, 58, 92, 64, 36
    W = margin * 2 + len(cards) * cw + (len(cards) - 1) * gap
    H = margin * 2 + 64 + head + ih + cap
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((margin, margin - 6), "Object extraction from a 3D Gaussian Splatting scene", fill=(20, 20, 20), font=font(34))
    top = margin + 64
    for k, (title, img, caption) in enumerate(cards):
        x = margin + k * (cw + gap)
        d.rounded_rectangle([x, top, x + cw, top + head + ih + cap], radius=14, outline=(190, 190, 190), width=2, fill=(248, 248, 248))
        d.text((x + 16, top + 14), title, fill=(20, 20, 20), font=font(26))
        tile = cv2.cvtColor(fit(img, cw - 20, ih - 10), cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(tile), (x + 10, top + head))
        d.multiline_text((x + 16, top + head + ih + 8), caption, fill=(60, 60, 60), font=font(19), spacing=6)
        if k < len(cards) - 1:
            ay = top + head + ih // 2
            ax0, ax1 = x + cw + 10, x + cw + gap - 10
            d.line([ax0, ay, ax1 - 14, ay], fill=(80, 80, 80), width=6)
            d.polygon([(ax1, ay), (ax1 - 18, ay - 13), (ax1 - 18, ay + 13)], fill=(80, 80, 80))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


class Figures:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        scene_id = cfg["scene_id"]
        self.renders = REPO / "renders" / scene_id
        self.out = REPO / "outputs" / scene_id
        self.cams = json.loads((self.renders / "cameras.json").read_text(encoding="utf-8"))
        self.by_stem = {Path(c["image"]).stem: c for c in self.cams}
        self.objects = json.loads((self.out / "objects" / "objects.json").read_text(encoding="utf-8"))
        self.members = np.load(self.out / "instances" / "members.npz")
        dc = cfg["detection"]
        self.thresholds = [float(c.get("box_threshold", dc["default_box_threshold"])) for c in cfg["classes"]]
        self.dets = {
            stem: [a for a in json.loads((self.out / "detections" / f"{stem}.json").read_text(encoding="utf-8"))["annotations"]
                   if a["score"] >= self.thresholds[int(a["class_id"])]]
            for stem in self.by_stem
        }

    def frames(self, score: dict[str, float], n: int) -> list[str]:
        """The n highest-scoring frames, each from a different camera position."""
        out, used = [], set()
        for stem in sorted(score, key=lambda s: (-score[s], s)):
            pos = self.by_stem[stem]["position_index"]
            if pos not in used:
                out.append(stem)
                used.add(pos)
            if len(out) == n:
                break
        return out

    def objects_by_support(self, n: int, skip_classes: set[int], min_gaussians: int = 300) -> list[dict]:
        """Well-supported objects, one per class."""
        out, used = [], set(skip_classes)
        for o in sorted(self.objects, key=lambda o: (-o["n_detected_views"] * o["confidence"], o["id"])):
            if o["class_id"] not in used and o["n_gaussians"] >= min_gaussians:
                out.append(o)
                used.add(o["class_id"])
            if len(out) == n:
                break
        return out

    def overlay(self, stem: str, only: int | None = None, color_class: int | None = None) -> np.ndarray:
        rgb = cv2.cvtColor(read(self.renders / self.by_stem[stem]["image"]), cv2.COLOR_BGR2RGB)
        payload = json.loads((self.out / "detections" / f"{stem}.json").read_text(encoding="utf-8"))
        anns = [payload["annotations"][only]] if only is not None else self.dets[stem]
        dets = [
            {"class_id": int(a["class_id"]) if color_class is None else color_class, "label": "", "score": a["score"],
             "box_xyxy": a["bbox"], "mask": rle_to_mask(a["segmentation"])}
            for a in anns
        ]
        return overlay_detections(rgb, dets, labels=False)

    def topdown_boxes(self) -> np.ndarray:
        ov = self.out / "scene_overview"
        td_rec = json.loads((ov / "topdown_camera.json").read_text(encoding="utf-8"))
        td = read(ov / "topdown_plain.png")
        for o in self.objects:
            uv, _ = project_points(td_rec, np.array(o["box"]["corners"])[:4])
            cv2.polylines(td, [uv.astype(np.int32).reshape(-1, 1, 2)], True, class_color_bgr(o["class_id"]), 2, cv2.LINE_AA)
        return td


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "config_all_objects.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ASSETS.mkdir(parents=True, exist_ok=True)
    f = Figures(cfg)
    cams, by_stem, objects, members = f.cams, f.by_stem, f.objects, f.members
    ov = f.out / "scene_overview"

    det_count = {s: float(len(a)) for s, a in f.dets.items()}
    seen = {s: float(sum(1 for o in objects if by_stem[s]["id"] in o["view_ids"])) for s in by_stem}
    detection_views = f.frames(det_count, 2)
    segment_views = f.frames(seen, 2)

    # Stage 0: raw scene
    overview = read(ov / "overview_0.png")
    save(
        "stage0_raw_scene.jpg",
        row([captioned(fit(overview, 760, 570), "In-room render of the raw splat"),
             captioned(fit(read(ov / "topdown_plain.png"), 700, 570), "Top-down render, ceiling removed")]),
    )

    # Stage 1: scene analysis
    mask = read(ov / "floor_mask.png")
    mask = cv2.resize(mask, (mask.shape[1] * 3, mask.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    save(
        "stage1_scene_analysis.jpg",
        row([captioned(fit(read(ov / "topdown.png"), 700, 570), "Floor plan in the gravity frame (1 m grid)"),
             captioned(fit(mask, 700, 570, (255, 255, 255)), "Floor region (grey) and camera-safe interior (green)")]),
    )

    # Stage 2: cameras and rendered frames
    save("stage2_camera_map.jpg", read(ov / "cameras_topdown.png"), 1200)
    save("stage2_rendered_frames.jpg", read(f.renders / "contact_sheets" / "renders_00.jpg"))

    # Stage 3: detections (masks and boxes colored by class, no labels)
    save("stage3_detections.jpg", row([
        captioned(fit(f.overlay(s), 760, 570), f"{s}: {len(f.dets[s])} Grounding DINO boxes + SAM 2.1 masks")
        for s in detection_views
    ]))
    sheet_stems = [Path(cams[i]["image"]).stem for i in np.linspace(0, len(cams) - 1, 24).round().astype(int)]
    save("stage3_detection_sheet.jpg",
         grid([captioned(fit(f.overlay(s), 320, 240), f"{s}: {len(f.dets[s])} masks", 16) for s in sheet_stems], 6, 6), 1920)

    ply_path = resolve_repo_path(cfg, cfg["input_ply"])
    scene = SplatScene.load(ply_path)
    centers_all = read_binary_gaussian_ply(ply_path).xyz()

    # Stage 4: one mask lifted to Gaussians vs. the fused 3D instance
    big = [o for o in objects if o["n_gaussians"] >= 2000] or objects
    obj = max(big, key=lambda o: (o["n_detected_views"], -o["id"]))
    cam_a = cams[obj["best_view"]]
    stem_a = Path(cam_a["image"]).stem
    fwd_a = np.array(cam_a["forward"])
    centers = centers_all[members[obj["name"]]]

    def fully_in_frame(c: dict, margin: float = 40.0) -> bool:
        uv, z = project_points(c, centers)
        inside = (z > 0) & (uv[:, 0] > margin) & (uv[:, 0] < c["width"] - margin)
        inside &= (uv[:, 1] > margin) & (uv[:, 1] < c["height"] - margin)
        return bool(inside.mean() > 0.98)

    others = [c for c in cams if c["id"] in obj["view_ids"] and c["id"] != cam_a["id"]]
    cand = [c for c in others if fully_in_frame(c)] or others or [cam_a]
    cam_b = min(cand, key=lambda c: float(np.dot(fwd_a, np.array(c["forward"]))))
    z = np.load(f.out / "lifting" / f"{stem_a}.npz")
    metas = json.loads(str(z["props_json"]))
    j = next(k for k, meta in enumerate(metas) if meta["det_idx"] == obj["best_det"])
    single = scene.subset(z["members"][z["offsets"][j] : z["offsets"][j + 1]].astype(np.int64))
    fused = scene.subset(members[obj["name"]])
    fused_a, alpha_a = render_bgr(fused, cam_a)
    fused_b, alpha_b = render_bgr(fused, cam_b)
    single_a, _ = render_bgr(single, cam_a)
    single_b, _ = render_bgr(single, cam_b)
    ba, bb = crop_to_alpha(alpha_a), crop_to_alpha(alpha_b)

    def cut(img: np.ndarray, box) -> np.ndarray:
        x0, y0, x1, y1 = box
        return fit(img[y0:y1, x0:x1], 460, 400)

    mask_a = f.overlay(stem_a, only=obj["best_det"], color_class=obj["class_id"])
    save(
        "stage4_lifting.jpg",
        row([captioned(cut(mask_a, ba), f"(a) SAM 2.1 mask in {stem_a}", 20),
             captioned(cut(single_a, ba), "(b) Gaussians lifted from that mask", 20),
             captioned(cut(single_b, bb), f"(c) same Gaussians seen from {Path(cam_b['image']).stem}", 20),
             captioned(cut(fused_b, bb), f"(d) fused over {obj['n_detected_views']} views", 20)]),
    )

    # Stage 5: instances on the floor plan + colored segmentation
    save("stage5_instances_topdown.jpg", f.topdown_boxes(), 1200)
    colored = SplatScene.load(f.out / "objects" / "scene_segmented_colored.ply")
    pairs = []
    for s in segment_views:
        rec = by_stem[s]
        seg, _ = render_bgr(colored, rec, (0.0, 0.0, 0.0))
        pairs.append(row([captioned(fit(read(f.renders / rec["image"]), 700, 525), f"{s}: rendered frame"),
                          captioned(fit(seg, 700, 525), f"{s}: extracted objects painted by class")]))
    save("stage5_segmentation.jpg", stack(pairs))

    # Stage 6: isolated object splats, then each object in its best frame next to its isolated splat
    mosaic_objs = f.objects_by_support(6, set())
    tiles = []
    for o in mosaic_objs:
        img, alpha = render_bgr(scene.subset(members[o["name"]]), cams[o["best_view"]])
        x0, y0, x1, y1 = crop_to_alpha(alpha, pad=0.12, min_side=120)
        tiles.append(captioned(fit(img[y0:y1, x0:x1], 360, 320), f'object {o["id"]} ({o["n_gaussians"]} splats)', 19))
    mosaic = grid(tiles, 3)
    save("stage6_object_splats.jpg", mosaic)

    previews = []
    for o in f.objects_by_support(6, {o["class_id"] for o in mosaic_objs}) or mosaic_objs:
        rec = cams[o["best_view"]]
        full = read(f.renders / rec["image"])
        draw_box(full, rec, np.array(o["box"]["corners"]), class_color_bgr(o["class_id"]))
        img, alpha = render_bgr(scene.subset(members[o["name"]]), rec)
        x0, y0, x1, y1 = crop_to_alpha(alpha, pad=0.25, min_side=220)
        pair = row([fit(full[y0:y1, x0:x1], 300, 300), fit(img[y0:y1, x0:x1], 300, 300)], 4)
        previews.append(captioned(pair, f'object {o["id"]}: {o["n_detected_views"]} views, score {o["confidence"]:.2f}', 18))
    save("stage6_previews.jpg", grid(previews, 3))

    # Pipeline overview figure
    spread = f.frames(det_count, 4)
    frames = [read(f.renders / by_stem[s]["image"]) for s in spread]
    frames_grid = stack([row([fit(frames[0], 400, 300), fit(frames[1], 400, 300)], 6),
                         row([fit(frames[2], 400, 300), fit(frames[3], 400, 300)], 6)], 6)
    seg_card, _ = render_bgr(colored, by_stem[segment_views[0]], (0.0, 0.0, 0.0))
    n_pos = len({c["position_index"] for c in cams})
    cards = [
        ("0  Raw scene", overview, f"{ply_path.name}\n{scene.n:,} Gaussians, SH degree {scene.sh_degree}"),
        ("1  Scene analysis", read(ov / "topdown.png"), "floor plane, up axis, floor plan,\nceiling height, clearance map"),
        ("2  Rendered frames", frames_grid, f"{n_pos} positions x {cfg['render']['yaws']} directions\n{len(cams)} validated RGB-D frames"),
        ("3  Detections", f.overlay(detection_views[0]), f"Grounding DINO boxes + SAM 2.1\nmasks for {len(cfg['classes'])} text prompts"),
        ("4  Lift and fuse", seg_card, "mask -> Gaussian weights (VJP)\ncross-view voting, cleanup"),
        ("5  Final outputs", mosaic, f"{len(objects)} object splats (.ply)\n+ objects.json with 3D boxes"),
    ]
    save("pipeline_overview.png", pipeline_figure(cards), 2800)


if __name__ == "__main__":
    main()
