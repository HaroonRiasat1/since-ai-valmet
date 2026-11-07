# Field-Service Navigator (Valmet x Since AI 2026 hackathon)

Demo for the Valmet "Industrial Metaverse" challenge: extract assets from a Gaussian Splatting
scan, map them to the Excel builder pack, walk an avatar on a collision-free path to a faulty
device, show nearby asset data, and give a grounded AI diagnosis on arrival.

Judging weights: asset extraction and 3D boxes 30%, navigation and proximity display 25%,
grounded AI diagnosis 20%, backend architecture 15%, demo clarity 10%.

## Files
- `backend/`: FastAPI app. `routing.py` (both grids), `agent.py` (intent routing, tools, grounding checks),
  `semantic.py` (data layer and spec compiler), `copilot.py` (diagnosis), `vision.py`, `llm.py` (providers), `scene.py`.
- `web/index.html`: single-page viewer, three.js r180 + Spark 2.4, plain JS, no build step. Libraries in `web/lib/`.
- `pipeline/`: offline builders. `navigator.py` (floor, occupancy grid, A*), `build_scene.py` (writes
  `web/data/scene.json` and splat chunks), `build_layer.py` (writes `data/semantic/layer.yaml` and `plant.sqlite`),
  `extract.py`, `semantic.py`, `refine_assets.py`, `export_assets.py`, `vlm_lift.py`, `view_splat.py`, `video_to_ply.py`.
- `data/semantic_layer.json`: spatial layer, what is where in the scan.
- `data/semantic/knowledge.yaml`: hand-written data knowledge. Edit this, never `layer.yaml`.
- `tests/`: pytest. AI providers are stubbed.

## Run
    uv sync --extra dev && cp .env.example .env
    uv run python -m backend --gpu    # --gpu opens a browser window on the NVIDIA GPU
    uv run pytest -q && uv run ruff check .
Rebuild data (needs the .ply and the builder pack, not in the repo):
    SKIP_SPLAT=1 python pipeline/build_scene.py "Equipment Room.ply" pack.xlsx data/semantic_layer.json web
    python pipeline/build_layer.py pack.xlsx

## Coordinates and conventions
- Everything is in the raw .ply frame. Y is vertical and points DOWN; floor is y ≈ 1.167.
  Horizontal axes are x and z. In the viewer a `world` group rotated π about X converts
  to three.js y-up; boxes, path and splat all live inside it.
- Ceiling is only ~2.1 units above the floor, so units are not quite metres. Avatar is 0.75 tall.
- Planner clearance 0.25 (comfortable aisles); keyboard driving uses 0.1 so narrow gaps pass.
- `items.json` format: `[{"id","label","center":[x,y,z],"size":[sx,sy,sz]}]`.

## Excel mapping (builder pack)
- 30 assets (CV, P, M, PT, VS). `data/mapping.csv` (box_id,tag; `-` = force unmapped) overrides first,
  labelled manual, confidence 0.9. The rest are auto-mapped: each area_name gets its own spatial zone
  (capacitated k-means + assignment); cost = zone distance + footprint size class + VLM class evidence.
  VLM class match -> confidence 0.35-0.7; no VLM label -> procedural 0.35; VLM disagrees -> 0.2.
  Boxes the VLM calls NOT_EQUIPMENT (conf >= 0.6) and the 4 biggest blobs never get a tag. The
  assignment is padded so tags with no plausible box stay unplaced (printed), per DOC-3D-MAPPING.
- VLM finding on the placeholder boxes: ~17 of 40 are ladders/carts/chairs. With `items_vlm.json`
  instead, 26 of 30 tags sit on class-matching VLM objects; the 3 vibration sensors and M-002 land on
  OTHER_EQUIPMENT at 0.2 (sensors are too small for the VLM in splat renders).
- `data/mapping.csv` is written as a template (blank tag column, suggested_tag filled) only when missing.
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
- After changing `build_scene.py`, rerun it so `web/data/scene.json` stays in sync. After changing the knowledge file, rerun `build_layer.py`.
- The model never writes SQL and never computes a path. It reads every request first; replies may only quote what tools returned. Rules are the fallback when there is no model.

## Next tasks (highest value first)
1. Test LM Studio with a real chat model that supports tools.
2. Demo video (max 5 min), 5 to 8 slides, short technical note.
