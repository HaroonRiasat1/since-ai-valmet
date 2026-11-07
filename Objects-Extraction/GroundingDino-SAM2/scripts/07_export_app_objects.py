#!/usr/bin/env python3
"""Stage 7: write extracted objects as Since-AI data/objects JSON (data_format/README.md)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from splat_objects.app_export import (
    app_class,
    known_tags_from_layer,
    load_app_checker,
    same_class_overlaps,
    to_app_object,
)
from splat_objects.config import load_config, repo_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_REPO / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ae = cfg["app_export"]
    scene_id = cfg["scene_id"]

    app_root = (repo_root() / ae["app_root"]).resolve()
    out_rel = ae["file"]
    out_path = (app_root / out_rel).resolve()
    objects_json = repo_root() / "outputs" / scene_id / "objects" / "objects.json"
    layer_path = app_root / "data" / "semantic_layer.json"

    records = json.loads(objects_json.read_text(encoding="utf-8"))
    layer = json.loads(layer_path.read_text(encoding="utf-8"))
    layer_objs = [o for o in layer["objects"] if o.get("source") != ae["source"]]  # our earlier export may be in the layer

    raw_app = [to_app_object(r, cfg, layer_objs) for r in records]
    raw_app.sort(key=lambda o: o["id"])

    ao = load_app_checker(app_root)
    normalised = [ao.normalise(copy.deepcopy(o), f"export #{i + 1}") for i, o in enumerate(raw_app)]

    other_added: list[dict] = []
    folder = app_root / "data" / "objects"
    for path in sorted(folder.glob("*.json")):
        if path.resolve() == out_path:
            continue
        other_added.extend(ao.read_file(path))

    known_tags = known_tags_from_layer(layer)
    try:
        ao.merge(copy.deepcopy(layer), normalised + other_added, known_tags)
    except ao.ObjectError as err:
        sys.exit(f"Merge check failed: {err}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"objects": raw_app}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    reread = ao.read_file(out_path)
    assert len(reread) == len(raw_app)

    by_cls = Counter(o["class"] for o in raw_app)
    by_conf = Counter(o["class_confidence"] for o in raw_app)
    n_overlap = sum(1 for r in records if same_class_overlaps(r, app_class(r, cfg), layer_objs, float(ae["same_class_share"])))
    n_tag = sum(1 for o in raw_app if "tag" in o)

    print(f"Wrote {len(raw_app)} objects -> {out_path}")
    print(f"  by class: {dict(sorted(by_cls.items()))}")
    print(f"  class_confidence: {dict(sorted(by_conf.items()))}")
    print(f"  overlap notes (same-class >= {ae['same_class_share']}): {n_overlap}")
    print(f"  asset tags: {n_tag}")


if __name__ == "__main__":
    main()
