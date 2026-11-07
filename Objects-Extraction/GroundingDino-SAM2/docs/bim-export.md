# BIM export (stages 8–12)

Stages **8–12** turn the equipment-room splat extraction into **IFC4**, a **QA report**, and a **GLB overlay** in scan coordinates for the web viewer.

## Run

From `GroundingDino-SAM2/` with the venv active:

```bash
python run_pipeline.py --config config_all_objects.yaml --from-stage 8
```

Individual stages:

| Stage | Script | Output |
|------:|--------|--------|
| 8 | `08_merge_elements.py` | `outputs/<scene>/bim/elements.json` |
| 9 | `09_build_structure.py` | `structure.json`, `structure_topdown.png` |
| 10 | `10_fit_elements.py` | `elements_fitted.json`, `meshes/*.npz` (splat surface meshes) |
| 11 | `11_write_ifc.py` | `<scene>.ifc`, `guid_map.json` |
| 12 | `12_export_glb_qa.py` | `equipment_room.glb`, `index.json`, QA reports → `Since-AI/web/data/bim/` |

## Configuration

`config.yaml` → `bim:` block:

- `units_to_m` — scale from scan units to IFC metres (default `1.0` until calibrated).
- `class_map` / `asset_class_ifc` — IFC entity, type, shape builder, and merge families.
- Structure thresholds (wall height bands, voxel size, door yaw, fit tolerance).
- `mesh` — per-object surface reconstruction from Gaussian PLYs (voxel size, iso level, max faces, vertex colour).

Equipment from Grounded-SAM-2 is meshed from a density field built from each Gaussian's true anisotropic footprint (rotation and scale), then closed and, for solid classes (`mesh.solid_families`: cabinets, appliances, buckets, boxes, pumps...), hole-filled before marching cubes, Taubin smoothing and decimation to `max_faces`. Thin classes (chairs, ladders, pipes, carts) keep their open structure. Semantic-layer-only assets keep box fallbacks. The same mesh is written to **IFC** (`IfcPolygonalFaceSet`) and **GLB** (per-vertex colour from SH DC). Occluded backs may be open; very sparse objects stay fragmentary.

## Layout editor (drag objects, re-export)

Open `web/layout.html` (also the **Layout editor** button in the navigator). It loads `data/bim/equipment_room.glb` and lets you:

- click an object and drag it on the floor; **Q / E** rotate it; walls stay fixed;
- reset one or all objects, and load a saved layout;
- **Export layout.json** (per object: pivot, delta in scan coordinates, yaw).

Rebuild the IFC and GLB from a layout (stage 13, not part of the default run):

```bash
python scripts/13_apply_layout.py --config config_all_objects.yaml --layout layout.json
```

Output goes to `outputs/<scene>/layout/` (`<scene>_layout.ifc`, `equipment_room_layout.glb`, `index.json`). GlobalIds are keyed on tag or element id, so they stay stable when objects move.

## Viewer

The GLB is parented under `world` so it aligns with the splat and semantic boxes in scan coordinates.

## Validation

- Schema: `ifcopenshell.validate` on the written IFC.
- IDS: `bim_requirements.ids` checked with **ifctester** (property presence rules).
