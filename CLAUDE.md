# Field-Service Navigator (Valmet x Since AI 2026 hackathon)

Demo for the Valmet "Industrial Metaverse" challenge: extract assets from a Gaussian Splatting
scan, map them to the Excel builder pack, walk an avatar on a collision-free path to a faulty
device, show nearby asset data, and give a grounded AI diagnosis on arrival.

Judging weights: asset extraction and 3D boxes 30%, navigation and proximity display 25%,
grounded AI diagnosis 20%, backend architecture 15%, demo clarity 10%.

## Files
- `navigator.py`: loads the .ply, finds the floor, builds a 5 cm top-down occupancy grid,
  A* path planning, writes `out/path.json`, `out/navgrid.npz`, `out/map.png`.
- `vlm_lift.py`: lifts "VLM detect" 2D boxes to 3D (splat depth inside each box) and fuses them
  across views into `items_vlm.json` (61 objects; items carry label, asset_class, confidence, views).
- `demo_items.py`: placeholder detector. Boxes obstacle blobs into `items.json`
  (40 boxes). Replace with real detection (SAM/CLIP etc.), keep the same format.
- `build_scene.py`: converts the .ply to chunked `.splat` (web/data/room_part*.glb),
  maps Excel assets to boxes, builds per-asset copilot evidence and the nav grid
  into `web/data/scene.json`.
- `web/index.html`: single-page viewer, three.js r180 + Spark 2.4 (Gaussian splats),
  plain JS, no build step. Libraries vendored in `web/lib/`.
- `web/server.py`: stdlib-only backend. Serves the page and `POST /api/copilot/query`,
  which calls Featherless (OpenAI-compatible) with the asset's evidence, and `POST /api/vlm/label`
  (image crop -> Qwen3-VL label/class/confidence; latest per box in `web/data/vlm_labels.json`, every
  capture kept as `web/data/vlm_crops/<box>_<time>_<label>.jpg`, listed in `vlm_crops/index.csv`).

## Run
    pip install numpy scipy matplotlib pandas openpyxl
    python demo_items.py Equipment_Room.ply items.json
    python build_scene.py Equipment_Room.ply <builder_pack>.xlsx items.json web/ [mapping.csv]
    # SKIP_SPLAT=1 skips rewriting the splat chunks (fast rebuild after mapping edits)
    cd web && export FEATHERLESS_API_KEY=... && python3 server.py   # http://localhost:8000
Optional: `export LLM_MODEL=<featherless model id>` (default Qwen/Qwen2.5-7B-Instruct),
`export VLM_MODEL=...` (default Qwen/Qwen3-VL-32B-Instruct).
VLM detection (preferred, ignores the blob boxes): click "VLM detect" in the viewer (10 standpoints x 6
headings, frames + cameras saved to `web/data/vlm_views/<run>/`), then
    .venv/bin/python vlm_lift.py Equipment_Room.ply web/data/vlm_views items_vlm.json   # 2D -> 3D, fused
    build_scene.py ... items_vlm.json web/
VLM labelling of existing boxes: "VLM scan" (shift-click rescans), then rerun build_scene.py. Python deps live in `.venv/` (`.venv/bin/python build_scene.py ...`).

## Coordinates and conventions
- Everything is in the raw .ply frame. Y is vertical and points DOWN; floor is y ≈ 1.167.
  Horizontal axes are x and z. In the viewer a `world` group rotated π about X converts
  to three.js y-up; boxes, path and splat all live inside it.
- Ceiling is only ~2.1 units above the floor, so units are not quite metres. Avatar is 0.75 tall.
- Planner clearance 0.25 (comfortable aisles); keyboard driving uses 0.1 so narrow gaps pass.
- `items.json` format: `[{"id","label","center":[x,y,z],"size":[sx,sy,sz]}]`.

## Excel mapping (builder pack)
- 30 assets (CV, P, M, PT, VS). `mapping.csv` (box_id,tag; `-` = force unmapped) overrides first,
  labelled manual, confidence 0.9. The rest are auto-mapped: each area_name gets its own spatial zone
  (capacitated k-means + assignment); cost = zone distance + footprint size class + VLM class evidence.
  VLM class match -> confidence 0.35-0.7; no VLM label -> procedural 0.35; VLM disagrees -> 0.2.
  Boxes the VLM calls NOT_EQUIPMENT (conf >= 0.6) and the 4 biggest blobs never get a tag. The
  assignment is padded so tags with no plausible box stay unplaced (printed), per DOC-3D-MAPPING.
- VLM finding on the placeholder boxes: ~17 of 40 are ladders/carts/chairs. With `items_vlm.json`
  instead, 26 of 30 tags sit on class-matching VLM objects; the 3 vibration sensors and M-002 land on
  OTHER_EQUIPMENT at 0.2 (sensors are too small for the VLM in splat renders).
- `mapping.csv` is written as a template (blank tag column, suggested_tag filled) only when missing.
- Scene_Anchors xyz are placeholders from before the scan existed: do not use them.
- Default faulty target: CV-003 (route starts at the room end farthest from it) (degraded, health 52, position error 3.4% to 6.4% in 24 h).
- Sheets used: Asset_Master, Current_State, Telemetry_Sample_3d, Alarms_Events,
  Maintenance_History, Historical_Service_Cases, Knowledge_Docs, Data_Quality.

## Viewer features
Target dropdown, click-to-set start, inspection round (all non-healthy assets, nearest next),
follow cam that rides the path, WASD/arrow driving with sliding collision, R = route from here,
proximity radius highlighting, XYZ overlay with teleport, rule-based copilot plus LLM block.
Click any box in the scene (or a Nearby card) for AI analysis: mapped assets open the copilot with
"Route here"; unmapped objects show their VLM evidence and an "Identify with VLM" button (current view).

## Rules
- Never put API keys in code, the page, or git. Keys only via environment variables.
- Copilot must cite record IDs, separate observation from hypothesis, state confidence,
  and recommend verification only (DOC-RAG-GUIDE, DOC-FIELD-SAFETY).
- Keep the viewer a single HTML file with no build step; test JS with `node --check`.
- After changing `build_scene.py`, rerun it so `web/data/scene.json` stays in sync.

## Next tasks (highest value first)
1. Tighter boxes: VLM-detect boxes come from depth bands, so they can bleed into the wall behind.
   SAM masks inside each VLM box (instead of the whole box) would sharpen them; more views per spot
   would raise the multi-view count (20 of 61 objects are single-view, kept only at conf >= 0.8).
2. Demo video (max 5 min), 5 to 8 slides, short technical note.
