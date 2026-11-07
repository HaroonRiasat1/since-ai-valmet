from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import ifcopenshell
import ifcopenshell.api
import numpy as np
from ifcopenshell.util.shape_builder import ShapeBuilder

from splat_objects.bim.coords import class_spec, stable_guid, units_to_m
from splat_objects.bim.meshing import load_mesh_npz
from splat_objects.scene import SceneFrame


def write_ifc(
    cfg: dict[str, Any],
    repo: Path,
    scene_id: str,
    frame: SceneFrame,
    elements: list[dict[str, Any]],
    structure: dict[str, Any],
    out_path: Path,
    guid_path: Path,
) -> dict[str, Any]:
    scale = units_to_m(cfg)
    bim = cfg["bim"]
    bim_dir = repo / "outputs" / scene_id / "bim"
    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name=scene_id)
    ifcopenshell.api.run("unit.assign_unit", model, length={"is_metric": True, "raw": "METERS"})
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Basement")
    storey.Elevation = 0.0
    space = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSpace", name="Equipment Room")
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[storey])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=storey, products=[space])

    body = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    ifcopenshell.api.run("context.add_context", model, context_type="Plan", target_view="PLAN_VIEW")
    body_ctx = ifcopenshell.api.run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=body,
    )
    builder = ShapeBuilder(model)

    guid_map: dict[str, str] = {}
    element_entities: dict[str, Any] = {}

    slab_t = float(structure.get("slab_thickness_m", bim.get("slab_thickness_m", 0.2)))
    poly = [tuple(p) for p in structure["room_polygon_m"]]
    slab = _add_slab(model, builder, body_ctx, storey, poly, slab_t, guid_map)
    ifcopenshell.api.run("spatial.assign_container", model, relating_structure=storey, products=[slab])

    ceiling_h = float(structure.get("ceiling_height_m", 2.5))
    for i, wall in enumerate(structure.get("walls", [])):
        w_ent = _add_wall(model, builder, body_ctx, storey, wall, ceiling_h, i, guid_map)
        ifcopenshell.api.run("spatial.assign_container", model, relating_structure=storey, products=[w_ent])

    for door in structure.get("doors", []):
        _add_door(model, builder, body_ctx, storey, door, guid_map, element_entities)

    plant = _load_plant(app_root := (repo / bim["app_root"]).resolve() / bim["plant_sqlite"])
    groups: dict[str, Any] = {}
    for name in list(bim.get("group_areas", {}).values()) + ["Loose items", "Unconfirmed asset positions"]:
        groups[name] = ifcopenshell.api.run("group.add_group", model, name=name)

    for el in elements:
        spec = el.get("ifc_spec") or class_spec(cfg, el["class"])
        ifc_cls = spec.get("ifc", "IfcBuildingElementProxy")
        key = el.get("tag") or el.get("unconfirmed_tag") or f"{el['class']}:{_centroid_key(el)}"
        gid = stable_guid(key)
        guid_map[key] = gid
        name = el.get("tag") or el.get("name") or el["id"]
        ent = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class=ifc_cls,
            name=str(name),
            predefined_type=spec.get("type"),
        )
        ent.GlobalId = gid
        if el.get("tag"):
            ent.Tag = el["tag"]
        _add_element_geom(model, builder, body_ctx, ent, el, bim_dir)
        ifcopenshell.api.run("spatial.assign_container", model, relating_structure=space, products=[ent])
        if spec.get("type"):
            try:
                ifcopenshell.api.run("type.assign_type", model, related_objects=[ent], relating_type=_type_object(model, ifc_cls, spec.get("type")))
            except (RuntimeError, TypeError):
                pass
        _pset(model, ent, "SinceAI_Extraction", {
            "Source": el["source"],
            "ElementId": el["id"],
            "Class": el["class"],
            "PositionConfirmed": bool(el.get("position_confirmed")),
            "FitWithinFrac": float((el.get("fit") or {}).get("within_frac", 0)),
            "FitMethod": (el.get("fit") or {}).get("method", ""),
            "MeshMethod": (el.get("mesh") or {}).get("method", ""),
        })
        if el.get("plant", {}).get("master"):
            _pset(model, ent, "Valmet_Asset", {k: str(v) for k, v in el["plant"]["master"].items() if v is not None})
        if el.get("asset_id") and el["asset_id"] in plant:
            _pset(model, ent, "Valmet_Condition", plant[el["asset_id"]])
        grp = spec.get("group") or bim.get("group_areas", {}).get((el.get("plant") or {}).get("master", {}).get("area_id"))
        if not el.get("position_confirmed") and el.get("unconfirmed_tag"):
            ifcopenshell.api.run("group.assign_group", model, group=groups["Unconfirmed asset positions"], products=[ent])
        elif grp and grp in groups:
            ifcopenshell.api.run("group.assign_group", model, group=groups[grp], products=[ent])
        elif spec.get("family") == "loose":
            ifcopenshell.api.run("group.assign_group", model, group=groups["Loose items"], products=[ent])
        element_entities[gid] = {"element": el, "entity": ent}

    _pset(model, site, "SinceAI_Frame", {
        "UnitsToM": scale,
        "SceneId": scene_id,
    })
    guid_path.write_text(json.dumps(guid_map, indent=2) + "\n", encoding="utf-8")
    model.write(str(out_path))
    val = _validate(out_path)
    ids = _ids_check(out_path, repo / "bim_requirements.ids")
    return {"guid_map": guid_map, "n_products": len(model.by_type("IfcProduct")), "validation": val, "ids": ids, "elements": element_entities}


def _centroid_key(el: dict[str, Any]) -> str:
    c = el["box_local"]["center_m"]
    return f"{round(c[0], 1)}:{round(c[1], 1)}:{round(c[2], 1)}"


def _add_slab(model, builder, ctx, storey, poly, thickness, guid_map):
    gid = stable_guid("slab:floor")
    guid_map["slab:floor"] = gid
    slab = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSlab", name="Floor slab", predefined_type="BASESLAB")
    slab.GlobalId = gid
    pts = [(float(x), float(y)) for x, y in poly]
    profile = builder.profile(builder.polyline(pts + [pts[0]]))
    body = builder.extrude(profile, thickness, extrusion_vector=(0.0, 0.0, -1.0))
    rep = builder.get_representation(ctx, [body])
    ifcopenshell.api.run("geometry.assign_representation", model, product=slab, representation=rep)
    _pset(model, slab, "SinceAI_Structure", {"ThicknessAssumed": True})
    return slab


def _add_wall(model, builder, ctx, storey, wall, height, index, guid_map):
    gid = stable_guid(f"wall:{index}")
    guid_map[f"wall:{index}"] = gid
    ent = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWall", name=f"Wall {index + 1}", predefined_type="STANDARD")
    ent.GlobalId = gid
    p0, p1 = wall["p0_m"], wall["p1_m"]
    thick = float(wall.get("thickness_m", 0.2))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = math.hypot(dx, dy) + 1e-9
    nx, ny = -dy / ln * thick, dx / ln * thick
    pts = [
        (p0[0], p0[1]),
        (p1[0], p1[1]),
        (p1[0] + nx, p1[1] + ny),
        (p0[0] + nx, p0[1] + ny),
    ]
    profile = builder.profile(builder.polyline(pts + [pts[0]]))
    body = builder.extrude(profile, height)
    rep = builder.get_representation(ctx, [body])
    ifcopenshell.api.run("geometry.assign_representation", model, product=ent, representation=rep)
    _pset(model, ent, "SinceAI_Structure", {
        "ThicknessAssumed": bool(wall.get("thickness_assumed", True)),
        "Inferred": bool(wall.get("inferred", False)),
    })
    return ent


def _add_door(model, builder, ctx, storey, door, guid_map, element_entities):
    gid = stable_guid(f"door:{door['element_id']}")
    guid_map[f"door:{door['element_id']}"] = gid
    ent = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcDoor", name=door["element_id"])
    ent.GlobalId = gid
    c = door["center_m"]
    w, h = door["width_m"], door["height_m"]
    profile = builder.profile(builder.rectangle(size=(w, h)))
    body = builder.extrude(profile, 0.05)
    rep = builder.get_representation(ctx, [body])
    ifcopenshell.api.run("geometry.assign_representation", model, product=ent, representation=rep)
    ifcopenshell.api.run("geometry.edit_object_placement", model, product=ent, matrix=[
        [1, 0, 0, c[0]],
        [0, 1, 0, c[1]],
        [0, 0, 1, c[2]],
        [0, 0, 0, 1],
    ])
    element_entities[gid] = {"door": door}


def _add_element_geom(model, builder, ctx, ent, el, bim_dir: Path):
    mesh_meta = el.get("mesh") or {}
    if mesh_meta.get("method") == "splat_surface" and mesh_meta.get("path"):
        loaded = load_mesh_npz(bim_dir, mesh_meta["path"])
        if loaded:
            verts, faces, colours = loaded
            body = builder.mesh(verts.tolist(), faces.tolist())
            rep = builder.get_representation(ctx, [body])
            ifcopenshell.api.run("geometry.assign_representation", model, product=ent, representation=rep)
            ifcopenshell.api.run("geometry.edit_object_placement", model, product=ent, matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            rgb = colours[:, :3].mean(axis=0) / 255.0
            _assign_colour(model, body, rgb)
            return
    _add_box_geom(model, builder, ctx, ent, el)


def _assign_colour(model, geom_item, rgb: np.ndarray) -> None:
    try:
        style = ifcopenshell.api.run("style.add_style", model)
        ifcopenshell.api.run(
            "style.add_surface_style",
            model,
            style=style,
            ifc_class="IfcSurfaceStyleShading",
            attributes={"SurfaceColour": {"Name": None, "Red": float(rgb[0]), "Green": float(rgb[1]), "Blue": float(rgb[2])}},
        )
        ifcopenshell.api.run("style.assign_item_style", model, item=geom_item, style=style)
    except Exception:
        pass


def _add_box_geom(model, builder, ctx, ent, el):
    box = el["box_local"]
    c, size = box["center_m"], box["size_m"]
    profile = builder.profile(builder.rectangle(size=(float(size[0]), float(size[1]))))
    body = builder.extrude(profile, float(size[2]))
    rep = builder.get_representation(ctx, [body])
    ifcopenshell.api.run("geometry.assign_representation", model, product=ent, representation=rep)
    yaw = math.radians(float(box.get("yaw_deg", 0.0)))
    m = [
        [math.cos(yaw), -math.sin(yaw), 0, c[0]],
        [math.sin(yaw), math.cos(yaw), 0, c[1]],
        [0, 0, 1, c[2] - float(size[2]) / 2],
        [0, 0, 0, 1],
    ]
    ifcopenshell.api.run("geometry.edit_object_placement", model, product=ent, matrix=m)


def _pset(model, ent, name: str, props: dict[str, Any]):
    pset = ifcopenshell.api.run("pset.add_pset", model, product=ent, name=name)
    ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=props)


def _type_object(model, ifc_cls: str, ptype: str):
    return ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcTypeProduct", name=f"{ifc_cls}:{ptype}", predefined_type=ptype)


def _load_plant(db: Path) -> dict[str, dict[str, str]]:
    if not db.is_file():
        return {}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, dict[str, str]] = {}
    try:
        for row in conn.execute("select asset_id, state, updated from asset_state"):
            out[str(row[0])] = {"State": str(row[1]), "Updated": str(row[2])[:10]}
    except sqlite3.Error:
        pass
    return out


def _validate(path: Path) -> dict[str, Any]:
    try:
        import ifcopenshell.validate

        issues = list(ifcopenshell.validate.validate(path))
        return {"ok": len(issues) == 0, "n_issues": len(issues), "sample": [str(i) for i in issues[:5]]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _ids_check(path: Path, ids_path: Path) -> dict[str, Any]:
    if not ids_path.is_file():
        return {"skipped": True}
    try:
        from ifctester import ids, reporter

        spec = ids.open(str(ids_path))
        ifc = ifcopenshell.open(str(path))
        rep = reporter.Console(spec.validate(ifc))
        return {"ok": True, "report": str(rep)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
