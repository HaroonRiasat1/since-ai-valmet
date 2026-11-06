# Splat navigator

Plans a collision-free walking path through a Gaussian Splatting `.ply` to a target item.

## Setup
    uv sync                          # creates .venv from pyproject.toml / uv.lock
    # or: pip install -r requirements.txt
Prefix the Python commands below with `uv run` (e.g. `uv run python build_scene.py ...`).
The scan (.ply) and the Excel builder pack are not in the repo; pass their paths on the command line.

## Run
    # 1. your detector output (or the demo one, which boxes obstacle blobs)
    python demo_items.py Equipment_Room.ply items.json

    # 2. plan a path to an item
    python navigator.py Equipment_Room.ply --items items.json --target OBJ-033 --out out
    # optional: --start X Y Z (world coords), --radius 0.25 (walker clearance in m)

Outputs in `out/`:
- `path.json`: `waypoints` (corner points) and `dense_path` (for smooth animation), world xyz
- `navgrid.npz`: walkable grid, reusable for other targets
- `map.png`: top-down debug map

## items.json format
    [{"id": "ASSET-003", "label": "pump", "center": [x, y, z], "size": [sx, sy, sz]}]
Same coordinate frame as the `.ply`. Replace `demo_items.py` with your real detections.

## How it works
1. Floor = densest opaque splat layer at the low end of the vertical axis (y, pointing down).
2. Top-down 5 cm grid: floor splats mark walkable evidence, splats 0.15 to 1.8 m above the floor mark obstacles.
3. Obstacles inflated by the walker radius; only the largest connected walkable area is kept.
4. A* (8-connected) with a small penalty near obstacles, so the path stays centred in aisles.
5. Line-of-sight shortcutting turns the grid path into a few straight waypoints.
6. The goal is the walkable cell about 0.4 m from the target's box, so the walker stops in front of it.

If the scene comes out sideways or upside down, change `UP` / `DOWN_SIGN` at the top of `navigator.py`.

## Web viewer (three.js + Spark)
    python build_scene.py Equipment_Room.ply builder_pack.xlsx items.json web/
    cd web
    export FEATHERLESS_API_KEY=...               # enables the AI copilot (Featherless)
    export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct    # optional: any model id from the provider
    python3 server.py                            # open http://localhost:8000

`server.py` (stdlib only) serves the page and exposes `POST /api/copilot/query`.
The key stays in the server's environment; the browser never sees it.
`web/` loads the bundled splat chunks in `web/data/`. Opened without them (e.g. the hosted copy),
it asks you to drop the .ply instead.
