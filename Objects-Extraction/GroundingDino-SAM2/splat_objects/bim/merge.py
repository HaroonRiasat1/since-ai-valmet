from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from splat_objects.app_export import _box_volume, _intersection_volume, app_class, app_object_id
from splat_objects.bim.coords import aabb_scan_to_local, asset_class_spec, class_spec, oriented_box_local, units_to_m
from splat_objects.scene import SceneFrame


def _share_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    va = _box_volume(a["lo_m"], a["hi_m"])
    vb = _box_volume(b["lo_m"], b["hi_m"])
    if va <= 0 or vb <= 0:
        return 0.0
    inter = _intersection_volume(a["lo_m"], a["hi_m"], b["lo_m"], b["hi_m"])
    return min(inter / va, inter / vb)


def _same_family(cfg: dict[str, Any], c1: str, c2: str) -> bool:
    f1 = class_spec(cfg, c1).get("family", "")
    f2 = class_spec(cfg, c2).get("family", "")
    return bool(f1) and f1 == f2 and f1 not in ("label", "loose")


def _layer_asset_objects(layer: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in layer.get("objects", []) if o.get("asset_id") or (o.get("asset") or {}).get("master")]


def _element_bbox_local(ob: dict[str, Any], lo_m: list[float], hi_m: list[float]) -> dict[str, Any]:
    return {
        "center_m": [(lo_m[i] + hi_m[i]) / 2 for i in range(3)],
        "size_m": [hi_m[i] - lo_m[i] for i in range(3)],
        "yaw_deg": ob.get("yaw_deg", 0.0),
        "corners_m": ob.get("corners_m", []),
        "bottom_height_m": lo_m[2],
        "top_height_m": hi_m[2],
    }


def merge_elements(cfg: dict[str, Any], repo: Path, scene_id: str, frame: SceneFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scale = units_to_m(cfg)
    bim = cfg["bim"]
    app_root = (repo / bim["app_root"]).resolve()
    objects_path = repo / "outputs" / scene_id / "objects" / "objects.json"
    layer = json.loads((app_root / bim["semantic_layer"]).read_text(encoding="utf-8"))
    layer_assets = _layer_asset_objects(layer)
    records = json.loads(objects_path.read_text(encoding="utf-8"))
    elements: list[dict[str, Any]] = []
    merges: list[dict[str, str]] = []
    min_share = float(cfg.get("app_export", {}).get("same_class_share", 0.5))

    for r in records:
        spec = class_spec(cfg, app_class(r, cfg))
        if spec.get("export") is False or spec.get("shape") == "label":
            continue
        ob = oriented_box_local(frame, r, scale)
        lo_m = [min(p[i] for p in ob["corners_m"]) for i in range(3)]
        hi_m = [max(p[i] for p in ob["corners_m"]) for i in range(3)]
        elements.append(
            {
                "id": app_object_id(r),
                "source": "grounded-sam-2",
                "class": app_class(r, cfg),
                "name": r["name"],
                "confidence": float(r["confidence"]),
                "detect_ratio": float(r["detect_ratio"]),
                "n_detected_views": int(r["n_detected_views"]),
                "n_gaussians": int(r.get("n_gaussians", 0)),
                "ply": r.get("ply"),
                "box_local": ob,
                "lo_m": lo_m,
                "hi_m": hi_m,
                "ifc_spec": spec,
                "tag": None,
                "unconfirmed_tag": None,
                "asset_id": None,
                "position_confirmed": False,
                "layer_object_id": None,
                "pipe_labels": [],
                "plant": {},
            }
        )

    used_layer: set[str] = set()
    for la in layer_assets:
        master = (la.get("asset") or {}).get("master") or {}
        tag = master.get("tag")
        asset_id = la.get("asset_id") or master.get("asset_id")
        cls = str(la.get("class", "")).lower()
        box = la["box"]
        loc = aabb_scan_to_local(frame, box["min"], box["max"], scale)
        la_vol = {"lo_m": loc["lo_m"], "hi_m": loc["hi_m"]}
        best_i, best_sh = -1, 0.0
        for i, el in enumerate(elements):
            if not _same_family(cfg, cls, el["class"]):
                continue
            sh = _share_overlap(la_vol, el)
            if sh >= min_share and sh > best_sh:
                best_sh, best_i = sh, i
        confirmed = la.get("mapping_method") == "class_match"
        if best_i >= 0:
            el = elements[best_i]
            merges.append({"gsam_id": el["id"], "layer_id": la["id"], "tag": tag})
            el["layer_object_id"] = la["id"]
            el["asset_id"] = asset_id
            el["position_confirmed"] = confirmed
            if confirmed:
                el["tag"] = tag
                ac = master.get("asset_class", "")
                if ac:
                    el["ifc_spec"] = asset_class_spec(cfg, ac)
            else:
                el["unconfirmed_tag"] = tag
            el["plant"] = _plant_slice(la)
            used_layer.add(la["id"])
        else:
            elements.append(
                {
                    "id": la["id"],
                    "source": "semantic_layer",
                    "class": cls,
                    "name": la["id"],
                    "box_local": _element_bbox_local({}, loc["lo_m"], loc["hi_m"]),
                    "lo_m": loc["lo_m"],
                    "hi_m": loc["hi_m"],
                    "ifc_spec": asset_class_spec(cfg, master.get("asset_class", "")) if master else class_spec(cfg, cls),
                    "tag": tag if confirmed else None,
                    "unconfirmed_tag": tag if not confirmed else None,
                    "asset_id": asset_id,
                    "layer_object_id": la["id"],
                    "position_confirmed": confirmed,
                    "pipe_labels": [],
                    "plant": _plant_slice(la),
                }
            )
            used_layer.add(la["id"])

    attach_m = float(bim.get("pipe_label_attach_m", 0.3))
    pipes = [e for e in elements if class_spec(cfg, e["class"]).get("family") == "pipe"]
    label_classes = {"green pipe label", "pipe label"}
    kept: list[dict[str, Any]] = []
    for el in elements:
        if el["class"] in label_classes:
            c = np.asarray(el["box_local"]["center_m"])
            best_d, best_p = 1e9, None
            for p in pipes:
                pc = np.asarray(p["box_local"]["center_m"])
                d = float(np.linalg.norm(c[:2] - pc[:2]))
                if d < best_d:
                    best_d, best_p = d, p
            if best_p is not None and best_d <= attach_m:
                best_p.setdefault("pipe_labels", []).append(el["id"])
            continue
        kept.append(el)
    elements = kept

    db_path = app_root / bim["plant_sqlite"]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    by_id = {el["id"]: el for el in elements}
    by_name = {str(el.get("name", "")).replace(" ", "_").lower(): el for el in elements}
    by_layer = {el["layer_object_id"]: el for el in elements if el.get("layer_object_id")}
    tag_cfg = {v: k.replace(" ", "_").lower() for k, v in (cfg.get("app_export") or {}).get("tags", {}).items()}

    def _resolve(object_id: str) -> dict[str, Any] | None:
        if object_id in by_id:
            return by_id[object_id]
        if object_id in by_layer:
            return by_layer[object_id]
        low = object_id.lower()
        if low in by_name:
            return by_name[low]
        return None

    for tag, object_id, matched in conn.execute("select tag, object_id, matched from asset_location"):
        el = _resolve(object_id)
        if el is None and tag in tag_cfg:
            el = _resolve(tag_cfg[tag]) or by_name.get(tag_cfg[tag])
        if el is None:
            raise RuntimeError(f"asset_location {tag} -> {object_id}: no BIM element")
        el["asset_tag"] = tag
        if matched:
            el["tag"] = tag
            el["unconfirmed_tag"] = None
            el["position_confirmed"] = True
        else:
            el["tag"] = None
            el["unconfirmed_tag"] = tag
            el["position_confirmed"] = False

    tag_hits: dict[str, int] = {}
    for el in elements:
        for t in (el.get("tag"), el.get("unconfirmed_tag")):
            if t:
                tag_hits[t] = tag_hits.get(t, 0) + 1
    bad = [f"{t}: {n}" for t, n in tag_hits.items() if n != 1]
    if bad:
        raise RuntimeError("each tag must map to one element: " + ", ".join(bad))

    return elements, {"merges": merges, "n_elements": len(elements), "scale": scale}


def _plant_slice(layer_obj: dict[str, Any]) -> dict[str, Any]:
    a = layer_obj.get("asset") or {}
    out: dict[str, Any] = {}
    if a.get("master"):
        out["master"] = a["master"]
    if a.get("state"):
        out["state"] = a["state"]
    return out
