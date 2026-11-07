#!/usr/bin/env python3
"""Rebuild docs/assets (the figures used in docs/README.md) from the pipeline outputs."""
from __future__ import annotations

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
from splat_objects.ply_io import read_binary_gaussian_ply
from splat_objects.render_mlx import SplatScene, render_rgb, to_uint8

SCENE = "equipment_room"
RENDERS = REPO / "renders" / SCENE
OUT = REPO / "outputs" / SCENE
ASSETS = Path(__file__).resolve().parent / "assets"
FONTS = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf", "/Library/Fonts/Arial.ttf"]
SEGMENT_VIEWS = ["v0017", "v0110"]
DETECTION_VIEWS = ["v0110", "v0006"]
MOSAIC_OBJECTS = ["Washing_Machine_00", "Chair_00", "Chair_01", "Ladders_01", "Doors_00", "Valves_13"]
LIFT_OBJECT = "Washing_Machine_00"


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


def crop_to_alpha(rgb_bgr: np.ndarray, alpha: np.ndarray, pad: float = 0.3, min_side: int = 240) -> tuple[int, int, int, int]:
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


def proposal_members(stem: str, det_idx: int) -> np.ndarray:
    z = np.load(OUT / "lifting" / f"{stem}.npz")
    for j, meta in enumerate(json.loads(str(z["props_json"]))):
        if meta["det_idx"] == det_idx:
            return z["members"][z["offsets"][j] : z["offsets"][j + 1]].astype(np.int64)
    raise KeyError((stem, det_idx))


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


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "config.yaml")
    cams = json.loads((RENDERS / "cameras.json").read_text(encoding="utf-8"))
    by_stem = {Path(c["image"]).stem: c for c in cams}
    objects = {o["name"]: o for o in json.loads((OUT / "objects" / "objects.json").read_text(encoding="utf-8"))}
    members = np.load(OUT / "instances" / "members.npz")
    ov = OUT / "scene_overview"

    # Stage 0: raw scene
    overview = read(ov / "overview_0.png")
    topdown_plain = read(ov / "topdown_plain.png")
    save(
        "stage0_raw_scene.jpg",
        row([captioned(fit(overview, 760, 570), "In-room render of the raw splat"),
             captioned(fit(topdown_plain, 700, 570), "Top-down render, ceiling removed")]),
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
    save("stage2_rendered_frames.jpg", read(RENDERS / "contact_sheets" / "renders_00.jpg"))

    # Stage 3: detections
    det = [captioned(fit(read(OUT / "detections" / "annotated" / f"{s}.jpg"), 760, 570), f"{s}: Grounding DINO box + SAM 2.1 mask")
           for s in DETECTION_VIEWS]
    save("stage3_detections.jpg", row(det))
    save("stage3_detection_sheet.jpg", read(OUT / "detections" / "contact_sheets" / "detections_00.jpg"))

    ply_path = resolve_repo_path(cfg, cfg["input_ply"])
    scene = SplatScene.load(ply_path)

    # Stage 4: one mask lifted to Gaussians vs. the fused 3D instance
    obj = objects[LIFT_OBJECT]
    cam_a = cams[obj["best_view"]]
    stem_a = Path(cam_a["image"]).stem
    fwd_a = np.array(cam_a["forward"])
    centers = read_binary_gaussian_ply(ply_path).xyz()[members[obj["name"]]]

    def fully_in_frame(c: dict, margin: float = 40.0) -> bool:
        uv, z = project_points(c, centers)
        inside = (z > 0) & (uv[:, 0] > margin) & (uv[:, 0] < c["width"] - margin)
        inside &= (uv[:, 1] > margin) & (uv[:, 1] < c["height"] - margin)
        return bool(inside.mean() > 0.98)

    cand = [c for c in cams if c["id"] in obj["view_ids"] and fully_in_frame(c)]
    cam_b = min(cand, key=lambda c: float(np.dot(fwd_a, np.array(c["forward"]))))
    single = scene.subset(proposal_members(stem_a, obj["best_det"]))
    fused = scene.subset(members[obj["name"]])
    fused_a, alpha_a = render_bgr(fused, cam_a)
    fused_b, alpha_b = render_bgr(fused, cam_b)
    single_a, _ = render_bgr(single, cam_a)
    single_b, _ = render_bgr(single, cam_b)
    ba, bb = crop_to_alpha(fused_a, alpha_a), crop_to_alpha(fused_b, alpha_b)

    def cut(img: np.ndarray, box) -> np.ndarray:
        x0, y0, x1, y1 = box
        return fit(img[y0:y1, x0:x1], 460, 400)

    annotated_a = read(OUT / "detections" / "annotated" / f"{stem_a}.jpg")
    save(
        "stage4_lifting.jpg",
        row([captioned(cut(annotated_a, ba), f"(a) SAM 2.1 mask in {stem_a}", 20),
             captioned(cut(single_a, ba), "(b) Gaussians lifted from that mask", 20),
             captioned(cut(single_b, bb), f"(c) same Gaussians seen from {Path(cam_b['image']).stem}", 20),
             captioned(cut(fused_b, bb), f"(d) fused over {obj['n_detected_views']} views", 20)]),
    )

    # Stage 5: instances on the floor plan + colored segmentation
    save("stage5_instances_topdown.jpg", read(OUT / "objects" / "topdown_boxes.png"), 1200)
    colored = SplatScene.load(OUT / "objects" / "scene_segmented_colored.ply")
    pairs = []
    for s in SEGMENT_VIEWS:
        rec = by_stem[s]
        orig = read(RENDERS / rec["image"])
        seg, _ = render_bgr(colored, rec, (0.0, 0.0, 0.0))
        pairs.append(row([captioned(fit(orig, 700, 525), f"{s}: rendered frame"),
                          captioned(fit(seg, 700, 525), f"{s}: extracted objects painted by class")]))
    save("stage5_segmentation.jpg", stack(pairs))

    # Stage 6: isolated object splats
    tiles = []
    for name in MOSAIC_OBJECTS:
        o = objects[name]
        rec = cams[o["best_view"]]
        img, alpha = render_bgr(scene.subset(members[name]), rec)
        x0, y0, x1, y1 = crop_to_alpha(img, alpha, pad=0.12, min_side=120)
        tiles.append(captioned(fit(img[y0:y1, x0:x1], 360, 320), f'{name} ({o["n_gaussians"]} splats)', 19))
    mosaic = stack([row(tiles[:3]), row(tiles[3:])])
    save("stage6_object_splats.jpg", mosaic)
    save("stage6_previews.jpg", read(OUT / "objects" / "previews" / "gallery_00.jpg"))

    # Pipeline overview figure
    frames = [read(RENDERS / by_stem[s]["image"]) for s in ["v0017", "v0006", "v0109", "v0102"]]
    frames_grid = stack([row([fit(frames[0], 400, 300), fit(frames[1], 400, 300)], 6),
                         row([fit(frames[2], 400, 300), fit(frames[3], 400, 300)], 6)], 6)
    seg_card, _ = render_bgr(colored, by_stem[SEGMENT_VIEWS[0]], (0.0, 0.0, 0.0))
    cards = [
        ("0  Raw scene", overview, "Equipment Room.ply\n1,676,481 Gaussians, SH degree 3"),
        ("1  Scene analysis", read(ov / "topdown.png"), "floor plane, up axis, floor plan,\nceiling height, clearance map"),
        ("2  Rendered frames", frames_grid, "26 positions x 8 directions\n147 validated RGB-D frames"),
        ("3  Detections", read(OUT / "detections" / "annotated" / "v0110.jpg"), "Grounding DINO boxes\n+ SAM 2.1 masks per frame"),
        ("4  Lift and fuse", seg_card, "mask -> Gaussian weights (VJP)\ncross-view voting, cleanup"),
        ("5  Final outputs", mosaic, "26 object splats (.ply)\n+ objects.json with 3D boxes"),
    ]
    save("pipeline_overview.png", pipeline_figure(cards), 2800)


if __name__ == "__main__":
    main()
