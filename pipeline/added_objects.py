"""Objects added by people or other tools, merged into the spatial semantic layer when the scene is built.

Drop a .json file into data/objects/ (format: data_format/README.md), then check it:

    python pipeline/added_objects.py                 # every file in data/objects/
    python pipeline/added_objects.py my_file.json    # one file

A tool's complete output can also be written into data/semantic_layer.json itself:

    python pipeline/added_objects.py --into-layer data/objects/grounded_sam2_equipment_room.json
"""
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "data" / "objects"
LAYER_FILE = ROOT / "data" / "semantic_layer.json"
ID = re.compile(r"[A-Za-z0-9_-]{1,40}")
CONFIDENCE = ("high", "medium", "low")
KNOWN_KEYS = {"id", "class", "box", "tag", "source", "class_confidence", "note"}


class ObjectError(ValueError):
    """A problem in an added object. The message says which file and object, and how to fix it."""


def _vector(value, name, where):
    if not (isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise ObjectError(f"{where}: {name} must be three numbers [x, y, z]")
    return [float(v) for v in value]


def normalise(raw: dict, where: str) -> dict:
    """Check one added object and return it in the spatial layer's object format."""
    if not isinstance(raw, dict):
        raise ObjectError(f"{where}: each object must be a JSON object")
    unknown = set(raw) - KNOWN_KEYS
    if unknown:
        raise ObjectError(f"{where}: unknown field(s) {sorted(unknown)}. Allowed: {sorted(KNOWN_KEYS)}")
    oid = raw.get("id")
    if not isinstance(oid, str) or not ID.fullmatch(oid):
        raise ObjectError(f"{where}: id must be 1 to 40 letters, digits, _ or -")
    where = f"{where} ({oid})"
    cls = raw.get("class")
    if not isinstance(cls, str) or not cls.strip():
        raise ObjectError(f"{where}: class is required, for example \"fire extinguisher\"")
    box = raw.get("box")
    if not isinstance(box, dict):
        raise ObjectError(f"{where}: box is required, as {{center, size}} or {{min, max}}")
    if "center" in box and "size" in box:
        center, size = _vector(box["center"], "box.center", where), _vector(box["size"], "box.size", where)
        if min(size) <= 0:
            raise ObjectError(f"{where}: every box.size value must be above 0")
        low = [c - s / 2 for c, s in zip(center, size)]
        high = [c + s / 2 for c, s in zip(center, size)]
    elif "min" in box and "max" in box:
        low, high = _vector(box["min"], "box.min", where), _vector(box["max"], "box.max", where)
        if any(a >= b for a, b in zip(low, high)):
            raise ObjectError(f"{where}: box.min must be below box.max on every axis")
    else:
        raise ObjectError(f"{where}: box needs center and size, or min and max")
    confidence = raw.get("class_confidence", "medium")
    if confidence not in CONFIDENCE:
        raise ObjectError(f"{where}: class_confidence must be one of {list(CONFIDENCE)}")
    tag = raw.get("tag")
    if tag is not None and (not isinstance(tag, str) or not tag.strip()):
        raise ObjectError(f"{where}: tag must be an asset tag such as \"P-005\", or left out")
    rounded = lambda v: [round(x, 4) for x in v]  # noqa: E731
    return {
        "id": oid, "source": str(raw.get("source", "manual")), "class": cls.strip().lower(), "class_confidence": confidence,
        "note": str(raw.get("note", "")), "seen_in_frames": 0, "splats": 0, "ply": None,
        "box": {"center": rounded([(a + b) / 2 for a, b in zip(low, high)]), "size": rounded([b - a for a, b in zip(low, high)]),
                "min": rounded(low), "max": rounded(high)},
        "floor_cells": [], "approach_cell": None, "asset_id": None,
        "mapping_method": "manual" if tag else None, "is_fault_target": False,
        "asset": {"master": {"tag": tag.strip().upper()}} if tag else None,
    }


def read_file(path: Path) -> list[dict]:
    """A file holds one object, a list of objects, or {"objects": [...]}."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        raise ObjectError(f"{path.name}: not valid JSON, line {error.lineno}: {error.msg}") from error
    items = data.get("objects") if isinstance(data, dict) and "objects" in data else data
    items = items if isinstance(items, list) else [items]
    return [normalise(item, f"{path.name} #{n + 1}") for n, item in enumerate(items)]


def load(folder: Path = FOLDER) -> list[dict]:
    """Every added object, in file name order so builds repeat exactly."""
    return [o for path in sorted(Path(folder).glob("*.json")) for o in read_file(path)]


def merge(layer: dict, added: list[dict], known_tags) -> dict:
    """The layer with the added objects in it. Added objects replace every layer object from the same source, so a tool's
    new output replaces its old one. An added object with a tag takes that tag from the object that had it."""
    sources = {o["source"] for o in added}
    out = copy.deepcopy(layer)
    out["objects"] = [o for o in out["objects"] if o.get("source") not in sources]
    ids = {o["id"] for o in out["objects"]}
    claimed = {}
    for obj in added:
        if obj["id"] in ids:
            raise ObjectError(f"id {obj['id']} is already used. Pick another id")
        ids.add(obj["id"])
        tag = obj["asset"]["master"]["tag"] if obj["asset"] else None
        if tag:
            if tag not in known_tags:
                raise ObjectError(f"{obj['id']}: tag {tag} is not in the builder pack. Known tags: {', '.join(sorted(known_tags))}")
            if tag in claimed:
                raise ObjectError(f"{obj['id']}: tag {tag} is already given to {claimed[tag]}")
            claimed[tag] = obj["id"]
    for existing in out["objects"]:   # a tag moves to the added object, so one asset never sits in two places
        if existing.get("asset") and existing["asset"]["master"]["tag"] in claimed:
            existing.update(asset=None, asset_id=None, mapping_method=None, is_fault_target=False)
    out["objects"].extend(added)
    return out


def into_layer(paths, layer_path: Path = LAYER_FILE) -> dict:
    """Write a tool's objects into the spatial semantic layer file, replacing what that source put there before.

    Objects added by hand or linked to an asset stay in data/objects/, where the build merges them and a link can be undone.
    """
    added = [o for path in paths for o in read_file(Path(path))]
    for obj in added:
        if obj["source"] == "manual" or obj["asset"]:
            raise ObjectError(f"{obj['id']}: only unlinked objects from a named tool go into the layer. Keep it in data/objects/")
    raw = layer_path.read_bytes()
    layer = merge(json.loads(raw), added, set())
    objects = layer["objects"]
    layer.setdefault("summary", {}).update(objects=len(objects), by_class=dict(Counter(o["class"] for o in objects).most_common()),
                                           by_source=dict(Counter(o.get("source") for o in objects).most_common()))
    layer_path.write_text(json.dumps(layer, indent=1), encoding="utf8", newline="\r\n" if b"\r\n" in raw else "\n")
    return layer


def main(args):
    try:
        if args[:1] == ["--into-layer"]:
            if not args[1:]:
                sys.exit("Name the file: python pipeline/added_objects.py --into-layer data/objects/<file>.json")
            summary = into_layer(args[1:])["summary"]
            sources = ", ".join(f"{n} {source}" for source, n in summary["by_source"].items())
            print(f"{LAYER_FILE.relative_to(ROOT)} now holds {summary['objects']} objects: {sources}")
            return
        found = [o for p in args for o in read_file(Path(p))] if args else load()
    except ObjectError as error:
        sys.exit(f"Problem: {error}")
    linked = sum(1 for o in found if o["asset"])
    print(f"{len(found)} object(s) OK, {linked} linked to an asset. Tags are checked against the builder pack when the scene is built.")


if __name__ == "__main__":
    main(sys.argv[1:])
