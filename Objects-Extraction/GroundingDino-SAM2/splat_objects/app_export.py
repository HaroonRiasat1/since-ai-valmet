"""Map pipeline objects.json records into data/objects format for the Since-AI app."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

_SUFFIX = re.compile(r"_(\d+)$")


def app_object_id(record: dict[str, Any]) -> str:
    """Build a unique app id from class_key and the numeric suffix of name (e.g. Valves_13 -> valve_13)."""
    key = str(record["class_key"])
    name = str(record["name"])
    m = _SUFFIX.search(name)
    suffix = m.group(1) if m else "0"
    return f"{key}_{suffix}"


def app_class(record: dict[str, Any], cfg: dict[str, Any]) -> str:
    """The app's class text: the object's detection prompt in lower case."""
    return str(cfg["classes"][int(record["class_id"])]["prompt"]).strip().lower()


def _round4(v: float) -> float:
    return round(float(v), 4)


def app_box(record: dict[str, Any]) -> dict[str, list[float]]:
    lo = record["box"]["aabb_min"]
    hi = record["box"]["aabb_max"]
    return {"min": [_round4(lo[i]) for i in range(3)], "max": [_round4(hi[i]) for i in range(3)]}


def class_confidence(record: dict[str, Any], cfg: dict[str, Any]) -> str:
    ae = cfg["app_export"]
    views = int(record["n_detected_views"])
    ratio = float(record["detect_ratio"])
    h = ae["high"]
    if views >= int(h["min_detected_views"]) and ratio >= float(h["min_detect_ratio"]):
        return "high"
    m = ae["medium"]
    if views >= int(m["min_detected_views"]) and ratio >= float(m["min_detect_ratio"]):
        return "medium"
    return "low"


def _box_volume(lo: list[float], hi: list[float]) -> float:
    v = 1.0
    for i in range(3):
        d = hi[i] - lo[i]
        if d <= 0:
            return 0.0
        v *= d
    return v


def _intersection_volume(a0: list[float], a1: list[float], b0: list[float], b1: list[float]) -> float:
    v = 1.0
    for i in range(3):
        d = min(a1[i], b1[i]) - max(a0[i], b0[i])
        if d <= 0:
            return 0.0
        v *= d
    return v


def _same_class(ours: str, layer_class: str) -> bool:
    ours = ours.strip().lower()
    layer_class = layer_class.strip().lower()
    return ours in layer_class or layer_class in ours


def same_class_overlaps(
    record: dict[str, Any],
    cls_text: str,
    layer_objects: list[dict[str, Any]],
    min_share: float,
) -> list[tuple[str, str, str | None, float]]:
    """Layer hits where class matches and intersection covers >= min_share of our AABB volume."""
    a0, a1 = record["box"]["aabb_min"], record["box"]["aabb_max"]
    va = _box_volume(a0, a1)
    if va <= 0:
        return []
    hits: list[tuple[str, str, str | None, float]] = []
    for obj in layer_objects:
        if not _same_class(cls_text, str(obj.get("class", ""))):
            continue
        b0, b1 = obj["box"]["min"], obj["box"]["max"]
        share = _intersection_volume(a0, a1, b0, b1) / va
        if share >= min_share:
            tag = None
            if obj.get("asset") and obj["asset"].get("master"):
                tag = obj["asset"]["master"].get("tag")
            hits.append((obj["id"], str(obj["class"]), tag, share))
    hits.sort(key=lambda t: -t[3])
    return hits


def _overlap_note(hits: list[tuple[str, str, str | None, float]]) -> str:
    if not hits:
        return ""
    lid, lcls, tag, _ = hits[0]
    link = f", linked to {tag}" if tag else ""
    return f"Overlaps {lid} ({lcls}{link}) in the scan layer."


def evidence_note(record: dict[str, Any]) -> str:
    name = record["name"]
    views = record["n_detected_views"]
    vis = record.get("n_visible_views", views)
    conf = float(record["confidence"])
    splats = int(record["n_gaussians"])
    return f"Grounded-SAM-2 {name}: seen in {views} of {vis} views, score {conf:.2f}, {splats} splats."


def to_app_object(
    record: dict[str, Any],
    cfg: dict[str, Any],
    layer_objects: list[dict[str, Any]],
) -> dict[str, Any]:
    ae = cfg["app_export"]
    cls_text = app_class(record, cfg)
    note = evidence_note(record)
    hits = same_class_overlaps(record, cls_text, layer_objects, float(ae["same_class_share"]))
    extra = _overlap_note(hits)
    if extra:
        note = f"{note} {extra}"
    out: dict[str, Any] = {
        "id": app_object_id(record),
        "class": cls_text,
        "box": app_box(record),
        "class_confidence": class_confidence(record, cfg),
        "source": str(ae["source"]),
        "note": note,
    }
    tags = ae.get("tags") or {}
    tag = tags.get(record["name"])
    if tag:
        out["tag"] = str(tag).strip().upper()
    return out


def load_app_checker(app_root: Path) -> Any:
    path = (app_root / "pipeline" / "added_objects.py").resolve()
    spec = importlib.util.spec_from_file_location("added_objects", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load added_objects from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def known_tags_from_layer(layer: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    for obj in layer.get("objects", []):
        if obj.get("asset") and obj["asset"].get("master"):
            t = obj["asset"]["master"].get("tag")
            if t:
                tags.add(str(t))
    return tags
