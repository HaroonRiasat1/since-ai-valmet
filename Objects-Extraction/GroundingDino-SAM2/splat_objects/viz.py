from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# BGR colors per class index
PALETTE_BGR = [
    (60, 200, 60),
    (240, 140, 30),
    (40, 40, 240),
    (0, 210, 255),
    (220, 60, 220),
    (255, 255, 0),
    (128, 128, 255),
]


def class_color_bgr(class_id: int) -> tuple[int, int, int]:
    return PALETTE_BGR[int(class_id) % len(PALETTE_BGR)]


def class_color_rgb01(class_id: int) -> np.ndarray:
    b, g, r = class_color_bgr(class_id)
    return np.array([r, g, b], dtype=np.float64) / 255.0


def draw_label(img_bgr: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.55) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(x, 0, max(0, img_bgr.shape[1] - tw - 4)))
    y = int(np.clip(y, th + 4, img_bgr.shape[0] - 1))
    cv2.rectangle(img_bgr, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    cv2.putText(img_bgr, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def overlay_detections(img_rgb: np.ndarray, dets: list[dict], alpha: float = 0.45) -> np.ndarray:
    """dets: dicts with class_id, label, score, box_xyxy, mask (H, W bool)."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    over = bgr.copy()
    for d in dets:
        over[d["mask"]] = class_color_bgr(d["class_id"])
    bgr = cv2.addWeighted(over, alpha, bgr, 1.0 - alpha, 0.0)
    for d in dets:
        c = class_color_bgr(d["class_id"])
        x0, y0, x1, y1 = (int(round(v)) for v in d["box_xyxy"])
        cv2.rectangle(bgr, (x0, y0), (x1, y1), c, 2)
        draw_label(bgr, f'{d["label"]} {d["score"]:.2f}', x0, y0, c)
    return bgr


def contact_sheets(
    tiles: list[tuple[str, np.ndarray]],
    out_dir: Path,
    prefix: str,
    cols: int = 6,
    rows: int = 5,
    thumb: tuple[int, int] = (320, 240),
) -> list[Path]:
    """tiles: (caption, BGR uint8 image). Writes paged grids, returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}_*.jpg"):
        old.unlink()
    per_page = cols * rows
    paths: list[Path] = []
    tw, th = thumb
    for page in range(0, max(1, len(tiles)), per_page):
        chunk = tiles[page : page + per_page]
        if not chunk:
            break
        canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
        for k, (cap, img) in enumerate(chunk):
            r, c = divmod(k, cols)
            t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            draw_label(t, cap, 2, 18, (255, 255, 255), 0.5)
            canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = t
        p = out_dir / f"{prefix}_{page // per_page:02d}.jpg"
        cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
    return paths
