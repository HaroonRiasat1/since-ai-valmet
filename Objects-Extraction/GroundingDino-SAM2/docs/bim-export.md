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

Equipment from Grounded-SAM-2 is meshed with marching cubes on a splat density field; semantic-layer-only assets keep box fallbacks. The same mesh is written to **IFC** (`IfcTriangulatedFaceSet`) and **GLB** (per-vertex colour from SH DC). Occluded backs may be open; thin pipes can be rough.

## Viewer

`Since-AI/web/index.html` exposes a **BIM** toggle when `web/data/bim/index.json` exists. The GLB is parented under `world` so it aligns with the splat and semantic boxes.

## Validation

- Schema: `ifcopenshell.validate` on the written IFC.
- IDS: `bim_requirements.ids` checked with **ifctester** (property presence rules).
