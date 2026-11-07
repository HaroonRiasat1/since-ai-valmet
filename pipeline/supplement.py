"""Synthetic plant records for grounded-sam-2 equipment, merged at build time (not in builder_pack.xlsx)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPPLEMENT_PATH = ROOT / "data" / "synthetic" / "equipment_supplement.json"
PACK_CLASSES = frozenset(
    {"CONTROL_VALVE", "PROCESS_PUMP", "ELECTRIC_MOTOR", "PRESSURE_TRANSMITTER", "VIBRATION_SENSOR"}
)


def load(path: Path = SUPPLEMENT_PATH) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def known_tags(sup: dict | None) -> set[str]:
    if not sup:
        return set()
    return {r["tag"] for r in sup.get("Asset_Master", []) if r.get("tag")}
