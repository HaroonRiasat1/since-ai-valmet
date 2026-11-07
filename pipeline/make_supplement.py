"""Write data/synthetic/equipment_supplement.json for grounded-sam-2 equipment assets.

    python pipeline/make_supplement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supplement import SUPPLEMENT_PATH  # noqa: E402

DOCS = {
    "ELECTRICAL_CABINET": (
        "DOC-ELECTRICAL_CABINET-CORE",
        "Electrical cabinet condition context",
        "Use enclosure temperature, door seal condition, termination tightness and recent panel work together. "
        "Treat elevated internal temperature, moisture ingress or loose terminals as signals to inspect on site, not as proof of failure.",
    ),
    "HANDWHEEL_VALVE": (
        "DOC-HANDWHEEL_VALVE-CORE",
        "Handwheel valve inspection context",
        "Use stem packing condition, handwheel motion, visible leakage and line isolation status together. "
        "Stiff motion, weeping packing or an unverified line-up are reasons to verify manually before changing the valve position.",
    ),
    "VALVE_ACTUATOR": (
        "DOC-VALVE_ACTUATOR-CORE",
        "Valve actuator diagnostic context",
        "Use supply pressure, actuator travel, feedback consistency and linkage condition together. "
        "Low supply, sluggish travel or drifting feedback warrant pneumatic and mechanical checks, not immediate replacement.",
    ),
}

ASSET_ROWS = [
    dict(tag="EC-001", asset_id="ASSET-031", asset_class="ELECTRICAL_CABINET", object_id="electrical_cabinet_00",
         area="Utility Island", area_id="AREA-03", process="utilities", criticality="B - important", conn="MODBUS",
         health=94, state="healthy", code="NORMAL", issue="No significant deviation", urgency="routine", next_check="monitor", conf=0.86),
    dict(tag="EC-002", asset_id="ASSET-032", asset_class="ELECTRICAL_CABINET", object_id="electrical_cabinet_01",
         area="Stock Preparation", area_id="AREA-01", process="wet_process", criticality="B - important", conn="ETHERNET",
         health=78, state="watch", code="PANEL_OVERTEMP", issue="Enclosure running warm", urgency="medium",
         next_check="verify door seal and ventilation", conf=0.72),
    dict(tag="HV-001", asset_id="ASSET-033", asset_class="HANDWHEEL_VALVE", object_id="handwheel_valve_00",
         area="Chemical Preparation", area_id="AREA-02", process="chemical", criticality="A - production critical", conn="MANUAL",
         health=96, state="healthy", code="NORMAL", issue="No significant deviation", urgency="routine", next_check="monitor", conf=0.84),
    dict(tag="HV-002", asset_id="ASSET-034", asset_class="HANDWHEEL_VALVE", object_id="handwheel_valve_01",
         area="Stock Preparation", area_id="AREA-01", process="wet_process", criticality="B - important", conn="MANUAL",
         health=88, state="watch", code="STEM_PACKING_WEEP", issue="Minor packing weep noted", urgency="medium",
         next_check="verify stem packing on site", conf=0.76),
    dict(tag="HV-003", asset_id="ASSET-035", asset_class="HANDWHEEL_VALVE", object_id="handwheel_valve_02",
         area="Utility Island", area_id="AREA-03", process="utilities", criticality="B - important", conn="MANUAL",
         health=91, state="healthy", code="NORMAL", issue="No significant deviation", urgency="routine", next_check="monitor", conf=0.81),
    dict(tag="HV-004", asset_id="ASSET-036", asset_class="HANDWHEEL_VALVE", object_id="handwheel_valve_03",
         area="Chemical Preparation", area_id="AREA-02", process="chemical", criticality="A - production critical", conn="MANUAL",
         health=52, state="degraded", code="POSITION_UNVERIFIED", issue="Line-up not confirmed", urgency="high",
         next_check="verify valve position and line-up", conf=0.68),
    dict(tag="HV-005", asset_id="ASSET-037", asset_class="HANDWHEEL_VALVE", object_id="handwheel_valve_04",
         area="Dry End", area_id="AREA-04", process="dry_process", criticality="B - important", conn="MANUAL",
         health=85, state="watch", code="HANDWHEEL_STIFF", issue="Handwheel turning stiff", urgency="medium",
         next_check="inspect handwheel and stem", conf=0.74),
    dict(tag="HV-006", asset_id="ASSET-038", asset_class="HANDWHEEL_VALVE", object_id="handwheel_valve_05",
         area="Wet End", area_id="AREA-05", process="wet_end", criticality="B - important", conn="MANUAL",
         health=93, state="healthy", code="NORMAL", issue="No significant deviation", urgency="routine", next_check="monitor", conf=0.83),
    dict(tag="HV-007", asset_id="ASSET-039", asset_class="HANDWHEEL_VALVE", object_id="handwheel_valve_06",
         area="Wet End", area_id="AREA-05", process="wet_end", criticality="C - standard", conn="MANUAL",
         health=79, state="watch", code="STEM_PACKING_WEEP", issue="Packing trace on body", urgency="routine",
         next_check="verify packing during next round", conf=0.71),
    dict(tag="VA-001", asset_id="ASSET-040", asset_class="VALVE_ACTUATOR", object_id="valve_actuator_00",
         area="Chemical Preparation", area_id="AREA-02", process="chemical", criticality="A - production critical", conn="PNEUMATIC",
         health=58, state="degraded", code="SUPPLY_PRESSURE_LOW", issue="Pneumatic supply below target", urgency="high",
         next_check="verify air supply and actuator linkage", conf=0.66),
]

OBJECTS_FILE = HERE.parent / "data" / "objects" / "grounded_sam2_equipment_room.json"


def build() -> dict:
    master, state, docs = [], [], []
    for i, row in enumerate(ASSET_ROWS):
        tag, asset_id, cls = row["tag"], row["asset_id"], row["asset_class"]
        n = i + 31
        master.append({
            "asset_id": asset_id, "tag": tag, "asset_class": cls, "area_id": row["area_id"], "area_name": row["area"],
            "process_context": row["process"], "criticality": row["criticality"], "installation_date": "2022-06-15",
            "last_service_date": "2026-05-20", "service_interval_days": 180, "connectivity_profile": row["conn"],
            "scene_anchor_id": f"ANCHOR-{n:03d}", "qr_id": f"QR-SYN-{n:04d}", "nfc_id": f"NFC-SYN-{n:04d}",
            "data_classification": "SYNTHETIC", "synthetic_record": True,
        })
        doc_id = DOCS[cls][0]
        state.append({
            "asset_id": asset_id, "tag": tag, "asset_class": cls, "area_name": row["area"], "criticality": row["criticality"],
            "health_score": row["health"], "condition_state": row["state"], "issue_code": row["code"], "issue_summary": row["issue"],
            "urgency": row["urgency"], "recommended_next_check": row["next_check"], "model_confidence": row["conf"],
            "last_updated": "2026-10-08 07:40:00", "knowledge_doc_id": doc_id, "synthetic_record": True,
        })
    for cls, (doc_id, title, content) in DOCS.items():
        docs.append({"doc_id": doc_id, "title": title, "scope": cls, "content": content, "synthetic_record": True})
    return {
        "Asset_Master": master, "Current_State": state, "Knowledge_Docs": docs,
        "object_tags": {r["object_id"]: r["tag"] for r in ASSET_ROWS},
    }


def apply_tags_to_objects(path: Path = OBJECTS_FILE) -> int:
    """Set tag on grounded-sam-2 objects listed in object_tags."""
    payload = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf8"))
    tags = payload.get("object_tags") or {}
    data = json.loads(path.read_text(encoding="utf8"))
    n = 0
    for obj in data.get("objects", []):
        tag = tags.get(obj.get("id"))
        if tag:
            obj["tag"] = tag
            n += 1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    return n


def main() -> None:
    payload = build()
    SUPPLEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    print(f"Wrote {len(payload['Asset_Master'])} assets, {len(payload['Knowledge_Docs'])} docs -> {SUPPLEMENT_PATH}")
    if OBJECTS_FILE.is_file():
        print(f"Tagged {apply_tags_to_objects()} objects in {OBJECTS_FILE}")


if __name__ == "__main__":
    main()
