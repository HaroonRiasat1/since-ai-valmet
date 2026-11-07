r"""Build the semantic layer: every extracted object joined to its Excel data and floor cells.

Run:    .venv\Scripts\python semantic.py [scan.ply]
Input:  extraction/<scan>/objects.json from extract.py, review_labels.json if a review was done,
        Since-AI/items_vlm.json (Haroon Riasat's vision-model detections) if present,
        plus the Excel builder pack
Output: extraction/<scan>/semantic_layer.json

Mapping is procedural, which the brief allows. Objects are matched to Excel assets of the same class,
worst condition first and most confident, largest object first, so faults land on clear objects. Leftover assets go
on leftover equipment. Anything left after that stays in the scene with no asset.
"""
import argparse
import datetime
import json
import os
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

import extract
import view_splat as vs

ROOT = vs.ROOT
PACK = Path(os.environ.get("BUILDER_PACK", ROOT / "data" / "builder_pack.xlsx"))
VIEW_STEP = 0.15            # grid cell width in scan units, 9 / 60 for this scan
DUPLICATE_OVERLAP = 0.5     # same-class boxes sharing this share of the smaller box are one object
CROSS_CLASS_OVERLAP = 0.4   # boxes of any class with this intersection over union are one object
WEAK_INSIDE_OVERLAP = 0.7   # an unknown or other box this far inside a labelled box is dropped
DENSE_CELL = 0.08           # cell size for finding an object's dense body, in scan units
DENSE_MIN_SPLATS = 2        # cells with fewer splats count as stray
# Smallest and largest allowed box side per class, in scan units. The room is about 1.75 units high.
# ponytail: set by eye for this scan, because the scale is not calibrated. Rescale for other scans.
SIZE_LIMITS = {"tank": (0.15, 1.3), "process pump": (0.05, 0.8), "electric motor": (0.05, 0.7),
               "control valve": (0.03, 0.5), "pressure transmitter": (0.02, 0.3), "vibration sensor": (0.02, 0.2),
               "electrical cabinet": (0.1, 1.5), "ladder": (0.1, 1.7), "cart": (0.2, 1.3), "cage": (0.15, 1.2),
               "pallet truck": (0.2, 1.2), "vacuum cleaner": (0.1, 0.8), "chair": (0.2, 1.0), "box": (0.08, 0.9),
               "crate": (0.1, 0.7), "bucket": (0.08, 0.5), "pipe": (0.03, 3.0)}
DEFAULT_LIMITS = 0.03, 2.2
MIN_PAINT_SPLATS = 200      # a tank box uses its paint colour only with at least this many splats
RANK = {"high": 3, "medium": 2, "low": 1}
VLM_ITEMS = ROOT / "items_vlm.json"
VLM_HIGH = 0.85             # vision-model confidence that counts as high, seen in 2 or more views
VLM_CLASSES = {"CONTROL_VALVE": "control valve", "PROCESS_PUMP": "process pump", "ELECTRIC_MOTOR": "electric motor",
               "PRESSURE_TRANSMITTER": "pressure transmitter", "VIBRATION_SENSOR": "vibration sensor"}
VLM_WORDS = (("tank", "tank"), ("vessel", "tank"), ("filter", "tank"), ("cabinet", "electrical cabinet"),
             ("panel", "electrical cabinet"), ("drive", "electrical cabinet"), ("ladder", "ladder"), ("pipe", "pipe"),
             ("cart", "cart"), ("chair", "chair"), ("bucket", "bucket"), ("crate", "crate"), ("box", "box"))
MEAN_WINDOW = 48            # telemetry rows in the last 24 hours, at one row every 30 minutes
SHIFT_COLUMNS = {"CONTROL_VALVE": ("position_setpoint_pct", "position_actual_pct", "pneumatic_supply_bar", "stiction_index"),
                 "PROCESS_PUMP": ("vibration_mm_s", "bearing_temp_c", "process_pressure_bar", "process_flow_pct"),
                 "ELECTRIC_MOTOR": ("vibration_mm_s", "bearing_temp_c", "motor_current_a", "speed_rpm"),
                 "PRESSURE_TRANSMITTER": ("process_pressure_bar", "signal_quality_pct", "sensor_bias"),
                 "VIBRATION_SENSOR": ("vibration_mm_s", "signal_quality_pct", "sensor_bias")}
VIEW_DISTANCE = 0.6, 0.95, 1.6  # closest, best and furthest viewing distance, in scan units
VIEW_CANDIDATES = 60        # cells tested for a clear view of each object
VIEW_LINE_RADIUS = 0.12     # how wide the sight line is, in scan units
WEAK_CLASSES = {"unknown", "other"}
FAULT_TARGET = "ASSET-017"  # P-005, the lowest health in the pack
EXCEL_CLASS = {"control valve": "CONTROL_VALVE", "process pump": "PROCESS_PUMP", "electric motor": "ELECTRIC_MOTOR",
               "pressure transmitter": "PRESSURE_TRANSMITTER", "vibration sensor": "VIBRATION_SENSOR"}
EQUIPMENT_FALLBACK = ("tank", "electrical cabinet", "other", "unknown")  # can carry leftover assets
SEVERITY = {"attention_required": 0, "degraded": 1, "watch": 2, "healthy": 3}
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def read_sheets(path, names):
    """Read named worksheets as lists of dicts, using only the standard library."""
    book = zipfile.ZipFile(path)
    shared = ["".join(t.text or "" for t in item.iter(NS + "t"))
              for item in ElementTree.fromstring(book.read("xl/sharedStrings.xml"))]
    workbook = ElementTree.fromstring(book.read("xl/workbook.xml"))
    targets = {r.get("Id"): r.get("Target") for r in ElementTree.fromstring(book.read("xl/_rels/workbook.xml.rels"))}

    def column(reference):
        number = 0
        for letter in re.match(r"[A-Z]+", reference).group():
            number = number * 26 + ord(letter) - 64
        return number - 1

    sheets = {}
    for sheet in workbook.find(NS + "sheets"):
        if sheet.get("name") not in names:
            continue
        target = targets[sheet.get(REL + "id")].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ElementTree.fromstring(book.read(target)).iter(NS + "row"):
            cells = {}
            for cell in row.findall(NS + "c"):
                value = cell.find(NS + "v")
                if value is not None:
                    cells[column(cell.get("r"))] = shared[int(value.text)] if cell.get("t") == "s" else number(value.text)
            if cells:
                rows.append([cells.get(i) for i in range(max(cells) + 1)])
        sheets[sheet.get("name")] = [dict(zip(rows[0], row)) for row in rows[1:]]
    return sheets


def number(text):
    """Numeric cell text as a number. Some cells hold plain text, which stays as it is."""
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


def excel_date(value):
    """Convert an Excel serial date to ISO text."""
    return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=float(value))).isoformat(timespec="minutes")


def clean(record, dates=()):
    """Drop empty and bookkeeping fields, and format date fields."""
    result = {k: v for k, v in record.items() if v is not None and k != "synthetic_record"}
    for key in dates:
        if key in result:
            result[key] = excel_date(result[key])
    return result


def telemetry_shifts(rows, asset_class):
    """How each key reading moved: the last 24 hours against the earlier days."""
    rows = sorted(rows, key=lambda r: r["timestamp"])
    shifts = []
    for column in SHIFT_COLUMNS[asset_class]:
        values = [float(r[column]) for r in rows if r.get(column) is not None]
        if len(values) > MEAN_WINDOW + 12:
            recent, before = np.mean(values[-MEAN_WINDOW:]), np.mean(values[:-MEAN_WINDOW])
            shifts.append({"metric": column, "last_24h": round(float(recent), 2), "before": round(float(before), 2),
                           "change_pct": round(float((recent - before) / (abs(before) or 1) * 100), 1)})
    if asset_class == "CONTROL_VALVE":
        gaps = [abs(float(r["position_setpoint_pct"]) - float(r["position_actual_pct"])) for r in rows
                if r.get("position_setpoint_pct") is not None and r.get("position_actual_pct") is not None]
        if len(gaps) > MEAN_WINDOW + 12:
            recent, before = np.mean(gaps[-MEAN_WINDOW:]), np.mean(gaps[:-MEAN_WINDOW])
            shifts.insert(0, {"metric": "position_error_pct", "last_24h": round(float(recent), 2), "before": round(float(before), 2),
                              "change_pct": round(float((recent - before) / (abs(before) or 1) * 100), 1)})
    return shifts


def build_assets(sheets):
    """Join every Excel table into one record per asset."""
    state = {r["asset_id"]: r for r in sheets["Current_State"]}
    docs = {r["doc_id"]: r for r in sheets["Knowledge_Docs"]}
    assets = {}
    for master in sheets["Asset_Master"]:
        asset_id = master["asset_id"]
        current = state[asset_id]
        telemetry = [r for r in sheets["Telemetry_Sample_3d"] if r["asset_id"] == asset_id]
        latest = max(telemetry, key=lambda r: r["timestamp"]) if telemetry else {}
        events = sorted((r for r in sheets["Alarms_Events"] if r["asset_id"] == asset_id),
                        key=lambda r: -r["timestamp"])[:5]
        work = sorted((r for r in sheets["Maintenance_History"] if r["asset_id"] == asset_id),
                      key=lambda r: -r["opened_at"])[:3]
        cases = [r for r in sheets["Historical_Service_Cases"]
                 if r["asset_class"] == master["asset_class"] and r["historical_issue_code"] == current["issue_code"]][:3]
        document = docs.get(current["knowledge_doc_id"], {})
        assets[asset_id] = {
            "master": clean(master, ("installation_date", "last_service_date")),
            "state": clean(current, ("last_updated",)),
            "latest_telemetry": clean({k: v for k, v in latest.items() if k not in ("asset_id", "tag", "asset_class", "area_id")},
                                      ("timestamp",)),
            "telemetry_shifts": telemetry_shifts(telemetry, master["asset_class"]),
            "data_quality": [clean(r) for r in sheets["Data_Quality"] if r["asset_id"] == asset_id],
            "recent_events": [clean(r, ("timestamp",)) for r in events],
            "recent_work_orders": [clean(r, ("opened_at",)) for r in work],
            "service_cases": [clean(r) for r in cases],
            "guidance": clean(document),
        }
    return assets


def map_assets(objects, assets):
    """Assign assets to objects. Returns {object id: (asset id, method)} and unplaced asset ids."""
    # Vision-model detections seen in one view were often wrong, so they cannot carry an asset.
    usable = [o for o in objects if not (o["id"].startswith("vlm_") and o["score"] == "low")]
    biggest = sorted(usable, key=lambda o: (-RANK.get(o.get("score"), 0), -o["splats"]))
    pending = sorted(assets, key=lambda a: (a != FAULT_TARGET, SEVERITY[assets[a]["state"]["condition_state"]], a))
    mapping, free = {}, list(biggest)
    for asset_id in list(pending):
        wanted = assets[asset_id]["master"]["asset_class"]
        match = next((o for o in free if EXCEL_CLASS.get(o["label"]) == wanted), None)
        if match:
            mapping[match["id"]] = (asset_id, "class_match")
            free.remove(match)
            pending.remove(asset_id)
    for asset_id in list(pending):
        match = next((o for o in free if o["label"] in EQUIPMENT_FALLBACK or o["label"] in EXCEL_CLASS), None)
        if match:
            mapping[match["id"]] = (asset_id, "procedural")
            free.remove(match)
            pending.remove(asset_id)
    return mapping, pending


def vlm_class(item):
    """Our class for a vision-model detection."""
    if item["asset_class"] in VLM_CLASSES:
        return VLM_CLASSES[item["asset_class"]]
    label = item["label"].lower()
    return next((name for word, name in VLM_WORDS if word in label), "other")


def load_vlm_items(path, v, folder):
    """Bring in vision-model detections, saving each one's splats so it works like our own objects."""
    if not path.exists():
        return []
    pos = np.stack([v[k].astype("f8") for k in "xyz"], axis=1)
    usable = v["opacity"] > 0
    objects = []
    for number, item in enumerate(json.loads(path.read_text()), 1):
        center, size = np.array(item["center"]), np.array(item["size"])
        low, high = center - size / 2, center + size / 2
        ids = np.flatnonzero(((pos >= low) & (pos <= high)).all(axis=1) & usable)
        if len(ids) < extract.MIN_SPLATS:
            continue
        name = f"vlm_{number:03d}"
        extract.save_ply(folder / "objects" / f"{name}.ply", v, ids)
        strong = item["confidence"] >= VLM_HIGH and item["views"] >= 2
        objects.append({"id": name, "label": vlm_class(item), "score": "high" if strong else "medium" if item["views"] >= 2 else "low",
                        "splats": len(ids), "views": item["views"], "box_min": low.round(4).tolist(), "box_max": high.round(4).tolist(),
                        "ply": f"objects/{name}.ply", "note": f"vision model: {item['label']}, {item['confidence']}, {item['views']} views"})
    return objects


def apply_review(objects, path):
    """Replace CLIP labels with reviewed ones, and remove objects the review rejected."""
    if not path.exists():
        return objects
    review = json.loads(path.read_text())
    kept = []
    for obj in objects:
        entry = review.get(obj["id"], {"label": "unknown", "confidence": "low", "note": "not reviewed"})
        if entry["label"] != "reject":
            kept.append({**obj, "label": entry["label"], "score": entry["confidence"], "note": entry.get("note", "")})
    return kept


def dense_body(points):
    """Mask of splats in the largest connected group of well-filled cells, dropping strays."""
    from scipy import ndimage

    cells = np.floor((points - points.min(0)) / DENSE_CELL).astype(int)
    counts = np.zeros(cells.max(0) + 1, dtype=int)
    np.add.at(counts, tuple(cells.T), 1)
    labels, found = ndimage.label(counts >= DENSE_MIN_SPLATS, structure=np.ones((3, 3, 3)))
    if found == 0:
        return np.ones(len(points), dtype=bool)
    point_labels = labels[tuple(cells.T)]
    return point_labels == np.bincount(point_labels[point_labels > 0]).argmax()


def densest_window(values, width):
    """Start of the width-wide stretch holding the most values."""
    ordered = np.sort(values)
    ends = np.searchsorted(ordered, ordered + width)
    return ordered[np.argmax(ends - np.arange(len(ordered)))]


def painted_splats(rgb):
    """Mask of the object's main paint, red or blue. Tanks are painted, attached pipes are not."""
    blue = rgb[:, 2] > rgb[:, 0] + 0.08
    red = rgb[:, 0] > np.maximum(rgb[:, 1], rgb[:, 2]) + 0.12
    paint = blue if blue.sum() > red.sum() else red
    return paint if paint.sum() >= MIN_PAINT_SPLATS else np.ones(len(rgb), dtype=bool)


def tighten_box(points, label, rgb=None):
    """Fit a box to an object's dense body within its class size. None if it is impossible."""
    if label == "tank" and rgb is not None:
        points = points[painted_splats(rgb)]
    points = points[dense_body(points)]
    smallest, largest = SIZE_LIMITS.get(label, DEFAULT_LIMITS)
    for axis in range(3):
        if np.ptp(points[:, axis]) > largest:
            start = densest_window(points[:, axis], largest)
            points = points[(points[:, axis] >= start) & (points[:, axis] <= start + largest)]
    low, high = np.percentile(points, [1, 99], axis=0)
    if len(points) < 30 or (high - low).min() < smallest:
        return None
    return low, high


def tighten_objects(objects, folder):
    """Replace each object's box with a tightened one, and remove impossible objects."""
    kept = []
    for obj in objects:
        v = vs.read_ply(folder / obj["ply"])
        rgb = 0.5 + vs.SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        box = tighten_box(np.stack([v[k].astype("f8") for k in "xyz"], axis=1), obj["label"], rgb)
        if box is not None:
            kept.append({**obj, "box_min": box[0].round(4).tolist(), "box_max": box[1].round(4).tolist()})
    return kept


def overlaps(a, b):
    """Intersection over union, and the share of the smaller box inside the other."""
    low_a, high_a, low_b, high_b = [np.array(x) for x in (a["box_min"], a["box_max"], b["box_min"], b["box_max"])]
    shared = np.prod(np.clip(np.minimum(high_a, high_b) - np.maximum(low_a, low_b), 0, None))
    volume_a, volume_b = np.prod(high_a - low_a), np.prod(high_b - low_b)
    return shared / (volume_a + volume_b - shared), shared / min(volume_a, volume_b)


def same_object(obj, other):
    """True when these two boxes describe one object."""
    union_share, inside_share = overlaps(obj, other)
    if obj["label"] == other["label"]:
        return inside_share > DUPLICATE_OVERLAP
    if obj["label"] in WEAK_CLASSES and inside_share > WEAK_INSIDE_OVERLAP:
        return True
    return union_share > CROSS_CLASS_OVERLAP


def drop_duplicates(objects):
    """Keep the strongest of overlapping objects: named, confident, then large."""
    order = sorted(objects, key=lambda o: (o["label"] not in WEAK_CLASSES, RANK.get(o["score"], 0), o["splats"]),
                   reverse=True)
    kept = []
    for obj in order:
        if not any(same_object(obj, other) for other in kept):
            kept.append(obj)
    return sorted(kept, key=lambda o: o["id"])


def viewing_cell(center, reach, sizes):
    """A free cell to look at an object from, in grid cells, with nothing blocking the view.

    center is the object's cell position and reach is its radius plus a margin, in cells. Blocked
    cells inside reach are the object itself, so they do not count against the view.
    """
    step_size = VIEW_STEP
    free = np.argwhere(sizes > 0)
    distance = np.linalg.norm(free - center, axis=1) * step_size
    near, best, far = VIEW_DISTANCE
    ring = (distance >= near) & (distance <= far)
    if not ring.any():
        return tuple(free[np.argmin(distance)])
    order = np.argsort(np.abs(distance[ring] - best) + 0.5 * (sizes[tuple(free[ring].T)] < 3))
    for cell in free[ring][order]:
        length = np.linalg.norm(cell - center)
        samples = np.linspace(0, 1, int(length * 3) + 2)
        line = cell + (center - cell) * samples[:, None]
        outside = np.linalg.norm(line - center, axis=1) > reach
        cells = np.round(line[outside]).astype(int)
        if not (sizes[tuple(cells.T)] == 0).any():
            return tuple(cell)
    return tuple(free[ring][order[0]])


def blockers(eye, target, points, radius):
    """Number of solid splats inside the sight line between the eye and the target."""
    line = target - eye
    length = np.linalg.norm(line)
    along = (points - eye) @ (line / length)
    between = (along > 0.1) & (along < length)
    sideways = np.linalg.norm((points[between] - eye) - np.outer(along[between], line / length), axis=1)
    return int((sideways < radius).sum())


def clear_view_cell(obj, sizes, solid, plane, up_sign, grid_x, grid_z, span):
    """The free cell, in grid indices, whose eye-level view of an object has the fewest splats in the way."""
    a, b, c = plane
    step = span / len(sizes)
    center = np.array(obj["box_center"])
    radius = max(obj["box_size"]) / 2
    free = np.argwhere(sizes > 0)
    x = grid_x - span / 2 + (free[:, 0] + 0.5) * step
    z = grid_z - span / 2 + (free[:, 1] + 0.5) * step
    distance = np.hypot(x - center[0], z - center[2])
    near, best, far = VIEW_DISTANCE
    ring = np.flatnonzero((distance >= max(near, radius + 0.2)) & (distance <= far))
    if not len(ring):
        return None
    ring = ring[np.argsort(np.abs(distance[ring] - best))][:VIEW_CANDIDATES]
    nearby = solid[np.linalg.norm(solid - center, axis=1) < far + 0.3]
    target_half = np.array(obj["box_size"]) / 2 * 0.8
    scores = []
    for index in ring:
        eye = np.array([x[index], a * x[index] + b * z[index] + c - up_sign * vs.EYE_HEIGHT, z[index]])
        # Stop short of the object, so its own splats do not count against the view.
        stop = center + (eye - center) / np.linalg.norm(eye - center) * np.linalg.norm(target_half)
        scores.append(blockers(eye, stop, nearby, VIEW_LINE_RADIUS))
    return tuple(free[ring[int(np.argmin(scores))]])


def floor_cells(obj, sizes, grid_x, grid_z, span, z_up):
    """Grid numbers under an object's box, and the free cell to view it from."""
    cells = len(sizes)
    step = span / cells
    x0, z0 = grid_x - span / 2, grid_z - span / 2
    ix = np.clip(((np.array([obj["box_min"][0], obj["box_max"][0]]) - x0) / step).astype(int), 0, cells - 1)
    iz = np.clip(((np.array([obj["box_min"][2], obj["box_max"][2]]) - z0) / step).astype(int), 0, cells - 1)

    def number_of(x, z):
        row = cells - 1 - z if z_up else z
        return int(row * cells + x + 1)

    under = [number_of(x, z) for x in range(ix[0], ix[1] + 1) for z in range(iz[0], iz[1] + 1)]
    center = np.array([ix.mean(), iz.mean()])
    reach = (max(ix[1] - ix[0], iz[1] - iz[0]) / 2 + 3) * step / VIEW_STEP
    return under, number_of(*viewing_cell(center, reach, sizes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", nargs="?", default=vs.PLY, type=Path)
    args = parser.parse_args()

    folder = ROOT / "extraction" / args.ply.stem
    scene = json.loads((folder / "objects.json").read_text())
    reviewed = apply_review(scene["objects"], folder / "review_labels.json")
    close_file, close_labels = folder / "objects_close.json", folder / "review_close_labels.json"
    if close_file.exists() and close_labels.exists():  # close finds join only after their review
        close = apply_review(json.loads(close_file.read_text())["objects"], close_labels)
        print(f"{len(close)} reviewed objects from the close pass")
        reviewed += close
    v, plane, band, up_sign, heights = extract.load_scan(args.ply)
    vlm = load_vlm_items(VLM_ITEMS, v, folder)
    print(f"{len(vlm)} objects from the vision-model detections")
    tightened = tighten_objects(reviewed + vlm, folder)
    objects = drop_duplicates(tightened)
    kept_vlm = sum(o["id"].startswith("vlm_") for o in objects)
    print(f"{len(scene['objects'])} extracted, {len(reviewed)} after review, {len(vlm)} vision-model, {len(tightened)} after size rules, "
          f"{len(objects)} after removing duplicates ({kept_vlm} vision-model)")
    assets = build_assets(read_sheets(PACK, {"Asset_Master", "Current_State", "Telemetry_Sample_3d", "Alarms_Events",
                                             "Maintenance_History", "Historical_Service_Cases", "Knowledge_Docs", "Data_Quality"}))
    mapping, unplaced = map_assets(objects, assets)

    sizes, grid_x, grid_z, span, _ = extract.free_grid(v, heights, band)
    solid_points = np.stack([v[k].astype("f8") for k in "xyz"], axis=1)[v["opacity"] >= extract.SOLID]
    records = []
    for obj in objects:
        under, approach = floor_cells(obj, sizes, grid_x, grid_z, span, up_sign > 0)
        low, high = np.array(obj["box_min"]), np.array(obj["box_max"])
        asset_id, method = mapping.get(obj["id"], (None, None))
        if asset_id:  # assets are walking targets, so they get a tested view
            cell = clear_view_cell({"box_center": (low + high) / 2, "box_size": high - low}, sizes, solid_points,
                                   plane, up_sign, grid_x, grid_z, span)
            if cell is not None:
                row = len(sizes) - 1 - cell[1] if up_sign > 0 else cell[1]
                approach = int(row * len(sizes) + cell[0] + 1)
        records.append({
            "id": obj["id"], "source": "vision model" if obj["id"].startswith("vlm_") else "extraction", "class": obj["label"], "class_confidence": obj["score"], "note": obj.get("note", ""),
            "seen_in_frames": obj["views"], "splats": obj["splats"], "ply": obj["ply"],
            "box": {"center": ((low + high) / 2).round(4).tolist(), "size": (high - low).round(4).tolist(),
                    "min": obj["box_min"], "max": obj["box_max"]},
            "floor_cells": under, "approach_cell": approach,
            "asset_id": asset_id, "mapping_method": method, "is_fault_target": asset_id == FAULT_TARGET,
            "asset": assets.get(asset_id),
        })

    counts = {}
    for record in records:
        counts[record["class"]] = counts.get(record["class"], 0) + 1
    target = next((r for r in records if r["is_fault_target"]), None)
    layer = {
        "scan": scene["scan"], "units": scene["units"], "floor_plane_y_ax_bz_c": scene["floor_plane_y_ax_bz_c"],
        "grid": {"cells_per_side": len(sizes), "cell_width": span / len(sizes), "center_x": grid_x, "center_z": grid_z},
        "summary": {"objects": len(records), "by_class": dict(sorted(counts.items(), key=lambda i: -i[1])),
                    "with_asset": sum(r["asset_id"] is not None for r in records),
                    "class_matched": sum(r["mapping_method"] == "class_match" for r in records),
                    "unplaced_assets": unplaced, "fault_target": target and target["id"]},
        "note": "All asset data is synthetic. Mapping to scene objects is procedural, not physically verified.",
        "objects": records,
    }
    (folder / "semantic_layer.json").write_text(json.dumps(layer, indent=1))
    print(json.dumps(layer["summary"], indent=1))
    if target:
        print(f"Fault target {target['asset']['master']['tag']} on {target['id']}, approach cell {target['approach_cell']}")


if __name__ == "__main__":
    # Self-check: dates convert, and the worst asset goes to the largest matching object first.
    assert excel_date(46303).startswith("2026-10-08")
    demo_assets = {FAULT_TARGET: {"master": {"asset_class": "PROCESS_PUMP"}, "state": {"condition_state": "healthy"}},
                   "A": {"master": {"asset_class": "PROCESS_PUMP"}, "state": {"condition_state": "healthy"}}}
    demo_objects = [{"id": "small", "label": "process pump", "splats": 1}, {"id": "big", "label": "process pump", "splats": 9}]
    assert map_assets(demo_objects, demo_assets)[0]["big"][0] == FAULT_TARGET
    # Self-check: a point on the sight line blocks it, and a point off to the side does not.
    wall = np.array([[0.5, 0.0, 0.0]])
    assert blockers(np.zeros(3), np.array([1.0, 0, 0]), wall, 0.1) == 1
    assert blockers(np.zeros(3), np.array([1.0, 0, 0]), wall + (0, 1, 0), 0.1) == 0
    # Self-check: the viewing cell is free, and it is on the clear side of a blocking wall.
    view_grid = np.ones((30, 30), dtype=int)
    view_grid[15:, :] = 1
    view_grid[8:10, :] = 0                    # a wall above the object
    view_cell = viewing_cell(np.array([15.0, 15.0]), 2.0, view_grid)
    assert view_grid[view_cell] and view_cell[0] > 10
    # Self-check: a single-view vision detection cannot take an asset.
    weak = [{"id": "vlm_001", "label": "process pump", "splats": 9, "score": "low"}]
    assert map_assets(weak, demo_assets)[0] == {}
    # Self-check: vision-model labels map onto our classes, and unknown labels stay "other".
    assert vlm_class({"asset_class": "PROCESS_PUMP", "label": "x"}) == "process pump"
    assert vlm_class({"asset_class": "OTHER_EQUIPMENT", "label": "white cabinet"}) == "electrical cabinet"
    assert vlm_class({"asset_class": "OTHER_EQUIPMENT", "label": "compressor"}) == "other"
    # Self-check: telemetry shifts compare the last 24 hours with the earlier rows.
    demo_rows = [{"timestamp": i, "vibration_mm_s": 1.0 if i < 60 else 3.0} for i in range(108)]
    assert telemetry_shifts(demo_rows, "PROCESS_PUMP")[0]["change_pct"] == 200.0
    # Self-check: a confident object takes the fault before a larger, less certain one.
    demo_objects[0]["score"] = "high"
    assert map_assets(demo_objects, demo_assets)[0]["small"][0] == FAULT_TARGET
    # Self-check: numbers parse, plain text survives, and an overlapping same-class box is dropped.
    assert number("4") == 4 and number("asset_id") == "asset_id"
    twin = {"id": "a", "label": "tank", "score": "high", "splats": 9, "box_min": [0, 0, 0], "box_max": [1, 1, 1]}
    apart = {**twin, "id": "c", "box_min": [5, 5, 5], "box_max": [6, 6, 6]}
    assert len(drop_duplicates([twin, {**twin, "id": "b", "splats": 1}, apart])) == 2
    # Self-check: overlapping boxes of different classes merge, and the confident one survives.
    survivors = drop_duplicates([twin, {**twin, "id": "b", "label": "box", "score": "low"}])
    assert [o["label"] for o in survivors] == ["tank"]
    # Self-check: a stray splat does not stretch a box, and an oversized pump is cut to its dense part.
    rng_check = np.random.default_rng(0)
    pump = rng_check.uniform(0, 0.3, (800, 3))
    pipe = np.column_stack([rng_check.uniform(0.1, 0.14, 300), rng_check.uniform(0.3, 1.6, 300),
                            rng_check.uniform(0.1, 0.14, 300)])
    stray = np.array([[3.0, 3.0, 3.0]])
    low_check, high_check = tighten_box(np.concatenate([pump, pipe, stray]), "process pump")
    assert high_check[0] < 1 and high_check[1] - low_check[1] <= 0.8
    # Self-check: a tank box follows its paint, even when beige pipe splats outnumber it.
    tank_points = np.concatenate([rng_check.uniform(0, 0.5, (400, 3)), rng_check.uniform(2, 3, (2000, 3))])
    tank_colors = np.concatenate([np.tile([0.2, 0.3, 0.7], (400, 1)), np.tile([0.8, 0.7, 0.55], (2000, 1))])
    tank_low, tank_high = tighten_box(tank_points, "tank", tank_colors)
    assert tank_high.max() < 1
    # Self-check: a box far too thin for its class is rejected.
    assert tighten_box(rng_check.uniform(0, 0.02, (500, 3)), "tank") is None
    # Self-check: rejected objects are removed and reviewed labels win.
    review_file = Path(__file__).with_name(".review_check.json")
    review_file.write_text(json.dumps({"a": {"label": "reject", "confidence": "high"},
                                       "b": {"label": "tank", "confidence": "high"}}))
    reviewed = apply_review([{"id": "a", "label": "x", "score": 0}, {"id": "b", "label": "x", "score": 0}], review_file)
    review_file.unlink()
    assert [o["label"] for o in reviewed] == ["tank"]
    main()
