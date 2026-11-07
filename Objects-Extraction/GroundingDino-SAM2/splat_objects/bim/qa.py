from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_qa(
    out_dir: Path,
    elements: list[dict[str, Any]],
    structure: dict[str, Any],
    merge_stats: dict[str, Any],
    ifc_result: dict[str, Any],
) -> None:
    by_class = Counter((el.get("ifc_spec") or {}).get("ifc", "?") for el in elements)
    by_src = Counter(el["source"] for el in elements)
    tags_conf = [el.get("tag") for el in elements if el.get("tag")]
    tags_unconf = [el.get("unconfirmed_tag") for el in elements if el.get("unconfirmed_tag")]
    fits = sorted(
        [{"id": el["id"], "within_frac": (el.get("fit") or {}).get("within_frac", 0)} for el in elements if el.get("fit")],
        key=lambda x: x["within_frac"],
    )
    mesh_n = sum(1 for el in elements if (el.get("mesh") or {}).get("method") == "splat_surface")
    mesh_fb = sum(1 for el in elements if (el.get("mesh") or {}).get("method") == "box_fallback")
    payload = {
        "n_elements": len(elements),
        "mesh_splat_surface": mesh_n,
        "mesh_box_fallback": mesh_fb,
        "by_ifc_class": dict(by_class),
        "by_source": dict(by_src),
        "merges": merge_stats.get("merges", []),
        "tags_confirmed": len(tags_conf),
        "tags_unconfirmed": len(tags_unconf),
        "structure": {
            "walls": len(structure.get("walls", [])),
            "inferred_walls": sum(1 for w in structure.get("walls", []) if w.get("inferred")),
            "ceiling_height_m": structure.get("ceiling_height_m"),
            "floor_occupancy": structure.get("floor_occupancy"),
        },
        "fit_worst_10": fits[:10],
        "validation": ifc_result.get("validation"),
        "ids": ifc_result.get("ids"),
        "assumptions": structure.get("assumptions", {}),
        "scale": merge_stats.get("scale"),
    }
    (out_dir / "qa_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md = _markdown(payload)
    (out_dir / "qa_report.md").write_text(md, encoding="utf-8")


def _markdown(p: dict[str, Any]) -> str:
    lines = [
        "# BIM QA report",
        "",
        f"- Elements: **{p['n_elements']}**",
        f"- Asset tags confirmed: **{p['tags_confirmed']}**; unconfirmed: **{p['tags_unconfirmed']}**",
        f"- Walls: **{p['structure']['walls']}** ({p['structure']['inferred_walls']} inferred)",
        f"- Ceiling height (m): **{p['structure']['ceiling_height_m']}**",
        "",
        "## Validation",
        f"- IFC schema: `{json.dumps(p.get('validation'))}`",
        f"- IDS: `{json.dumps(p.get('ids'))}`",
        "",
        "## Worst fit (10)",
    ]
    for row in p.get("fit_worst_10", []):
        lines.append(f"- `{row['id']}`: within {row['within_frac']:.2%}")
    return "\n".join(lines) + "\n"
