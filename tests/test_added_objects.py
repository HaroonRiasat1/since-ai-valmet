import json

import pytest

from backend.config import ROOT
from pipeline import added_objects as ao

EXAMPLES = sorted((ROOT / "data_format" / "examples").glob("0[123]_*.json"))
LAYER = {"objects": [
    {"id": "cls_1", "class": "electric motor", "asset": {"master": {"tag": "M-002"}}, "asset_id": "ASSET-020", "mapping_method": "procedural", "is_fault_target": False},
    {"id": "cls_2", "class": "tank", "asset": None, "asset_id": None, "mapping_method": None, "is_fault_target": False},
]}
TAGS = {"M-002", "P-005"}


def write(tmp_path, data, name="x.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_every_documented_example_is_valid():
    assert len(EXAMPLES) == 3
    found = [o for path in EXAMPLES for o in ao.read_file(path)]
    assert len(found) == 4 and sum(1 for o in found if o["asset"]) == 1


def test_center_and_size_become_a_full_box(tmp_path):
    (obj,) = ao.read_file(write(tmp_path, {"id": "a", "class": "Drum", "box": {"center": [1, 1, 1], "size": [0.2, 0.4, 0.2]}}))
    assert obj["box"]["min"] == [0.9, 0.8, 0.9] and obj["box"]["max"] == [1.1, 1.2, 1.1]
    assert obj["class"] == "drum" and obj["source"] == "manual" and obj["class_confidence"] == "medium" and obj["asset"] is None


def test_a_folder_loads_in_file_order(tmp_path):
    write(tmp_path, [{"id": "b", "class": "x", "box": {"min": [0, 0, 0], "max": [1, 1, 1]}}], "2.json")
    write(tmp_path, {"objects": [{"id": "a", "class": "x", "box": {"min": [0, 0, 0], "max": [1, 1, 1]}}]}, "1.json")
    assert [o["id"] for o in ao.load(tmp_path)] == ["a", "b"]


@pytest.mark.parametrize("bad, message", [
    ({"class": "x", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}, "id must be"),
    ({"id": "a b", "class": "x", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}, "id must be"),
    ({"id": "a", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}, "class is required"),
    ({"id": "a", "class": "x"}, "box is required"),
    ({"id": "a", "class": "x", "box": {"center": [0, 0], "size": [1, 1, 1]}}, "three numbers"),
    ({"id": "a", "class": "x", "box": {"center": [0, 0, 0], "size": [1, 0, 1]}}, "above 0"),
    ({"id": "a", "class": "x", "box": {"min": [1, 0, 0], "max": [0, 1, 1]}}, "below box.max"),
    ({"id": "a", "class": "x", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}, "class_confidence": "sure"}, "class_confidence"),
    ({"id": "a", "class": "x", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}, "colour": "red"}, "unknown field"),
])
def test_mistakes_get_a_clear_message(tmp_path, bad, message):
    with pytest.raises(ao.ObjectError, match=message):
        ao.read_file(write(tmp_path, bad))


def test_broken_json_names_the_line(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"id": "a",\n "class": }')
    with pytest.raises(ao.ObjectError, match="line 2"):
        ao.read_file(path)


def test_a_tag_moves_to_the_added_object():
    added = ao.normalise({"id": "motor", "class": "electric motor", "tag": "m-002", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}, "t")
    out = ao.merge(LAYER, [added], TAGS)
    holders = [o["id"] for o in out["objects"] if o.get("asset")]
    assert holders == ["motor"] and added["mapping_method"] == "manual"
    assert LAYER["objects"][0]["asset"] is not None   # the original layer is untouched


@pytest.mark.parametrize("objects, message", [
    ([{"id": "cls_2", "class": "x", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}], "already used"),
    ([{"id": "a", "class": "x", "tag": "P-999", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}], "not in the builder pack"),
    ([{"id": "a", "class": "x", "tag": "P-005", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}},
      {"id": "b", "class": "x", "tag": "P-005", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}], "already given"),
])
def test_merge_refuses_conflicts(objects, message):
    with pytest.raises(ao.ObjectError, match=message):
        ao.merge(LAYER, [ao.normalise(o, "t") for o in objects], TAGS)


def test_a_source_replaces_its_earlier_objects():
    layer = {"objects": LAYER["objects"] + [{"id": "t1", "class": "x", "source": "tool"}, {"id": "t2", "class": "x", "source": "tool"}]}
    added = [ao.normalise({"id": i, "class": "x", "source": "tool", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}, "t") for i in ("t1", "t3")]
    assert [o["id"] for o in ao.merge(layer, added, TAGS)["objects"]] == ["cls_1", "cls_2", "t1", "t3"]


def test_into_layer_writes_a_tools_objects_once(tmp_path):
    layer_path = write(tmp_path, {"summary": {"objects": 2}, "objects": [dict(o, source="extraction") for o in LAYER["objects"]]}, "layer.json")
    path = write(tmp_path, {"objects": [{"id": "v1", "class": "Valve", "source": "tool", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}}]})
    ao.into_layer([path], layer_path)
    ao.into_layer([path], layer_path)
    layer = json.loads(layer_path.read_text())
    assert [o["id"] for o in layer["objects"]] == ["cls_1", "cls_2", "v1"]
    assert layer["summary"] == {"objects": 3, "by_class": {"electric motor": 1, "tank": 1, "valve": 1}, "by_source": {"extraction": 2, "tool": 1}}


@pytest.mark.parametrize("extra", [{}, {"source": "tool", "tag": "M-002"}])
def test_into_layer_keeps_manual_and_linked_objects_out(tmp_path, extra):
    layer_path = write(tmp_path, {"objects": []}, "layer.json")
    path = write(tmp_path, {"id": "a", "class": "x", "box": {"center": [0, 0, 0], "size": [1, 1, 1]}, **extra})
    with pytest.raises(ao.ObjectError, match="data/objects"):
        ao.into_layer([path], layer_path)


def test_query_example_runs_on_the_data_layer():
    from backend.semantic import get_layer
    spec = json.loads((ROOT / "data_format" / "examples" / "04_query_asset_location.json").read_text())
    out = get_layer().run(spec)
    assert out["columns"][0] == "tag" and all(row[4] == 0 for row in out["rows"])
