from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from splat_objects.bim.coords import class_spec, local_to_scan, stable_guid, units_to_m
from splat_objects.bim.meshing import load_mesh_npz
from splat_objects.scene import SceneFrame


def export_glb(
    cfg: dict[str, Any],
    frame: SceneFrame,
    elements: list[dict[str, Any]],
    structure: dict[str, Any],
    ifc_meta: dict[str, Any],
    glb_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    scale = units_to_m(cfg)
    bim_dir = glb_path.parent
    scene = trimesh.Scene()
    index: list[dict[str, Any]] = []
    cats = {
        "wall": (0.7, 0.7, 0.75, 0.35),
        "slab": (0.55, 0.55, 0.5, 0.4),
        "equipment": (0.2, 0.6, 0.9, 0.45),
        "door": (0.6, 0.4, 0.2, 0.5),
    }

    for i, wall in enumerate(structure.get("walls", [])):
        mesh = _wall_mesh(wall, structure.get("ceiling_height_m", 2.5))
        gid = stable_guid(f"wall:{i}")
        _add_mesh(scene, mesh, gid, frame, scale, cats["wall"])
        index.append({"global_id": gid, "category": "wall", "ifc_class": "IfcWall", "name": f"Wall {i + 1}"})

    for el in elements:
        spec = class_spec(cfg, el["class"])
        if spec.get("shape") == "door":
            continue
        key = el.get("tag") or el.get("unconfirmed_tag") or f"{el['class']}:{el['id']}"
        gid = ifc_meta.get("guid_map", {}).get(key) or stable_guid(key)
        mesh = _element_mesh(el, bim_dir)
        col = cats["equipment"]
        _add_mesh(scene, mesh, gid, frame, scale, col, tint=mesh.metadata.get("has_vertex_colour"))
        index.append(
            {
                "global_id": gid,
                "ifc_class": spec.get("ifc", "IfcBuildingElementProxy"),
                "name": el.get("tag") or el.get("name") or el["id"],
                "tag": el.get("tag"),
                "page_item_id": el["id"],
                "category": "equipment",
                "fit_within_frac": (el.get("fit") or {}).get("within_frac"),
                "position_confirmed": el.get("position_confirmed"),
            }
        )

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(glb_path))
    index_path.write_text(json.dumps({"scene": cfg["scene_id"], "elements": index}, indent=2) + "\n", encoding="utf-8")
    return {"n_meshes": len(index)}


def _add_mesh(
    scene: trimesh.Scene,
    mesh: trimesh.Trimesh,
    name: str,
    frame: SceneFrame,
    scale: float,
    rgba: tuple[float, float, float, float],
    *,
    tint: bool = False,
):
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v_scan = local_to_scan(frame, v, scale)
    mesh.vertices = v_scan
    if not tint or mesh.visual.vertex_colors is None or len(mesh.visual.vertex_colors) != len(v):
        mesh.visual.vertex_colors = np.column_stack(
            [
                np.tile(np.array([int(c * 255) for c in rgba[:3]], dtype=np.uint8), (len(v), 1)),
                np.full(len(v), int(rgba[3] * 255), dtype=np.uint8),
            ]
        )
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def _element_mesh(el: dict[str, Any], bim_dir: Path) -> trimesh.Trimesh:
    rel = (el.get("mesh") or {}).get("path")
    if rel and (el.get("mesh") or {}).get("method") == "splat_surface":
        loaded = load_mesh_npz(bim_dir, rel)
        if loaded:
            verts, faces, colours = loaded
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh.visual.vertex_colors = colours
            mesh.metadata["has_vertex_colour"] = True
            return mesh
    box = el["box_local"]
    c = np.asarray(box["center_m"])
    size = np.asarray(box["size_m"])
    yaw = np.radians(float(box.get("yaw_deg", 0.0)))
    t = trimesh.transformations.translation_matrix(c) @ trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    return trimesh.creation.box(extents=size, transform=t)


def _wall_mesh(wall: dict[str, Any], height: float) -> trimesh.Trimesh:
    p0, p1 = np.asarray(wall["p0_m"]), np.asarray(wall["p1_m"])
    thick = float(wall.get("thickness_m", 0.2))
    d = p1 - p0
    ln = float(np.linalg.norm(d))
    mid = 0.5 * (p0 + p1)
    yaw = np.arctan2(d[1], d[0])
    box = trimesh.creation.box(extents=[ln, thick, height])
    t = trimesh.transformations.translation_matrix([mid[0], mid[1], height / 2])
    r = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    box.apply_transform(t @ r)
    return box
