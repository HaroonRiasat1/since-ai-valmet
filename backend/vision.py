"""Vision-model labelling and detection on rendered views. Every result is kept for auditing."""
import base64
import csv
import json
import os
import re
import threading
import time
from pathlib import Path

from backend.llm import Provider, chat

VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
CLASSES = ["CONTROL_VALVE", "PROCESS_PUMP", "ELECTRIC_MOTOR", "PRESSURE_TRANSMITTER", "VIBRATION_SENSOR",
           "OTHER_EQUIPMENT", "NOT_EQUIPMENT"]
CROP_FIELDS = ["file", "box_id", "label", "asset_class", "confidence", "model", "at", "visual_evidence"]
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")
JPEG_PREFIX = "data:image/jpeg;base64,"
LOCK = threading.Lock()

LABEL_PROMPT = """This is a render from a 3D Gaussian-splat scan of an industrial equipment room.
Identify the object inside the yellow rectangle (renders can be blurry; lower the confidence instead of guessing).
Answer with JSON only, no prose:
{"label": "<short name, e.g. 'gate valve with handwheel', 'centrifugal pump', 'electrical cabinet', 'ladder'>",
 "asset_class": "<one of: """ + ", ".join(CLASSES) + """>",
 "confidence": <0.0-1.0>,
 "visual_evidence": "<one sentence: what you see that supports this>"}
Any valve counts as CONTROL_VALVE (say 'manual' in the label if there is no actuator). Pipes, tanks, cabinets,
panels and boilers are OTHER_EQUIPMENT. Furniture, ladders, boxes, buckets, walls and clutter are NOT_EQUIPMENT."""

DETECT_PROMPT = """This is a render from a 3D Gaussian-splat scan of an industrial equipment room.
Detect every piece of plant equipment. Return JSON only, a list (empty if none):
[{"label": "<short name>", "asset_class": "<one of: """ + ", ".join(CLASSES[:-1]) + """>",
  "bbox_2d": [x1, y1, x2, y2], "confidence": <0.0-1.0>}]
bbox_2d is in 0-1000 coordinates relative to image width and height.
Any valve is CONTROL_VALVE (say 'manual' in the label if it has no actuator). Tanks, cabinets, panels,
boilers and compressors are OTHER_EQUIPMENT. Skip pipes, walls, furniture, ladders, carts, boxes and
clutter, and skip anything too blurry to identify."""


def _ask(provider: Provider, model: str | None, image: str, prompt: str, max_tokens: int):
    """The cloud model picker lists text models, so the cloud uses the configured vision model."""
    message = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": prompt}]}]
    result = chat(provider, message, model=VLM_MODEL if provider.needs_key else model, max_tokens=max_tokens)
    return result["message"]["content"], result["model"]


def _parse(text: str, pattern: str, empty):
    match = re.search(pattern, text, re.S)
    try:
        return json.loads(match.group(0)) if match else empty
    except json.JSONDecodeError:
        return empty


def clean_detections(raw) -> list[dict]:
    """Keep well-formed detections, clamped to the image and to known classes."""
    detections = []
    for item in raw if isinstance(raw, list) else []:
        box, cls = item.get("bbox_2d"), str(item.get("asset_class", "")).upper()
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box) and cls in CLASSES[:-1]:
            detections.append({"label": str(item.get("label", ""))[:80], "asset_class": cls,
                               "bbox_2d": [max(0, min(1000, v)) for v in box],
                               "confidence": round(min(1.0, max(0.0, float(item.get("confidence") or 0.5))), 2)})
    return detections


def detect(provider: Provider, data_dir: Path, request: dict) -> dict:
    """Detect equipment in one frame, and keep the frame and camera for lifting to 3D."""
    run, view, image = request.get("run", ""), request.get("view_id", ""), request.get("image", "")
    if not (SAFE_NAME.fullmatch(run) and SAFE_NAME.fullmatch(view)) or not image.startswith(JPEG_PREFIX):
        raise ValueError("need run, view_id and a JPEG data URL")
    if len(request.get("pose", [])) != 16 or len(request.get("proj", [])) != 16:
        raise ValueError("need 4x4 pose and proj")
    text, model = _ask(provider, request.get("model"), image, DETECT_PROMPT, 900)
    detections = clean_detections(_parse(text, r"\[.*\]", []))
    folder = data_dir / "vlm_views" / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{view}.jpg").write_bytes(base64.b64decode(image.split(",", 1)[1]))
    (folder / f"{view}.json").write_text(json.dumps(
        {"view_id": view, "pose": request["pose"], "proj": request["proj"], "width": request.get("width"), "height": request.get("height"),
         "detections": detections, "model": model, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    return {"detections": detections, "model": model}


def label(provider: Provider, data_dir: Path, box_id: str, image: str, model: str | None = None) -> dict:
    """Ask what an outlined object is, and store the answer with its crop."""
    if not SAFE_NAME.fullmatch(box_id or "") or not image.startswith(JPEG_PREFIX):
        raise ValueError("need box_id and a JPEG data URL")
    text, model = _ask(provider, model, image, LABEL_PROMPT, 250)
    answer = _parse(text, r"\{.*\}", {})
    cls = str(answer.get("asset_class", "")).upper()
    record = {"label": str(answer.get("label", "unknown"))[:80], "asset_class": cls if cls in CLASSES else "NOT_EQUIPMENT",
              "confidence": round(min(1.0, max(0.0, float(answer.get("confidence") or 0))), 2),
              "visual_evidence": str(answer.get("visual_evidence", ""))[:300], "model": model,
              "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": None if answer else text[:300]}
    slug = re.sub(r"[^a-z0-9]+", "-", record["label"].lower()).strip("-")[:40] or "unknown"
    record["image"] = f"{box_id}_{record['at'].replace(':', '').replace('-', '')}_{slug}.jpg"
    crops, labels_file = data_dir / "vlm_crops", data_dir / "vlm_labels.json"
    with LOCK:
        crops.mkdir(parents=True, exist_ok=True)
        (crops / record["image"]).write_bytes(base64.b64decode(image.split(",", 1)[1]))
        index = crops / "index.csv"
        new = not index.exists()
        with open(index, "a", newline="") as f:
            writer = csv.DictWriter(f, CROP_FIELDS, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow({**record, "file": record["image"], "box_id": box_id})
        labels = json.loads(labels_file.read_text()) if labels_file.exists() else {}
        labels[box_id] = record
        labels_file.write_text(json.dumps(labels, indent=1))
    return record
