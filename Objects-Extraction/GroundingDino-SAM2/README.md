# Equipment Room: object extraction with Grounded-SAM-2

Extracts separate Gaussian splats for Chair, Washing Machine, Valves, Ladders and Doors from `Equipment Room.ply`. The scan is not in git: put it in the repository root, as for the main app, or change `input_ply` in `config.yaml` (relative to this folder).

Pipeline:
1. `01_view_scene.py`: fits the floor plane and up axis, builds the floor plan and free-space map, renders a top-down view and in-room overview shots.
2. `02_render_views.py`: places cameras inside the room (floor underneath, clear of walls and equipment) at 1.6 m and 1.0 m, 8 directions each, ~90° horizontal FOV. Views are validated (holes, too close, blur) and saved for reuse.
3. `03_detect_segment.py`: Grounding DINO (`IDEA-Research/grounding-dino-base`) boxes, then SAM 2.1 Hiera-L masks (Grounded-SAM-2).
4. `04_lift_masks.py`: one backward pass through the mlx3d rasterizer gives each Gaussian's exact blending weight (alpha × transmittance) inside every mask, so occluded Gaussians get none.
5. `05_build_instances.py`: matches masks across views, takes a cross-view vote (with hysteresis growth for attached parts), filters out inconsistent detections, removes floor splats and floaters, resolves Gaussians claimed by several objects, and fits gravity-aligned boxes.
6. `06_export_objects.py`: one `.ply` per object (original splat rows), `objects.json`, colored and remainder scenes, previews.
7. `07_export_app_objects.py`: writes [`data/objects/grounded_sam2_equipment_room.json`](../../data/objects/grounded_sam2_equipment_room.json) in the Since-AI [`data_format`](../../data_format/README.md) so `build_scene.py` can merge the 26 objects into the twin.

Rendering uses [mlx3d](https://pypi.org/project/mlx3d/) (Metal) because gsplat needs CUDA. Grounding DINO and SAM 2 run on MPS.

## Setup

Needs an Apple Silicon Mac. `setup_env.sh` creates `.venv`, clones Grounded-SAM-2 into `third_party/` and downloads the SAM 2.1 checkpoint into `checkpoints/`. None of these are committed.

```bash
./setup_env.sh
source .venv/bin/activate
```

## Run

```bash
python run_pipeline.py --force        # full run, re-rendering the view set
python run_pipeline.py --from-stage 3 # reuse saved renders, redo detection onwards
python run_pipeline.py --from-stage 4 # retune class thresholds without re-detecting
python run_pipeline.py --from-stage 7 # refresh the app data/objects file after stage 6
```

After stage 7, from the Since-AI repo root:

```bash
python pipeline/added_objects.py data/objects/grounded_sam2_equipment_room.json
SKIP_SPLAT=1 python pipeline/build_scene.py "Equipment Room.ply" builder_pack.xlsx data/semantic_layer.json web
python pipeline/build_layer.py builder_pack.xlsx
```

See [data_format/README.md](../../data_format/README.md) for the object fields and coordinate rules.

## View in 3D

```bash
python view_objects.py                          # then open http://localhost:8080
python view_objects.py --select Washing_Machine_00 --display "Objects in class colors"
```

`view_objects.py` opens the splat scene in [viser](https://viser.studio), a browser-based viewer, with every extracted object's oriented 3D box and label. It reads the object splats and scene frame from `outputs/`, so run the pipeline first.

Navigation: click the 3D view once, then
- **W/S**: forward/back
- **A/D**: sideways
- **Q/E**: down/up
- **drag**: look around
- **right-drag**: pan
- **scroll**: step forward/back

In walk mode (default) turning happens in place and the camera stays between the floor and the ceiling pipes. Orbit mode circles the selected object instead. viser's arrow keys orbit the turning point, so ← turns right and ↑ tilts down; mouse drag is the natural way to look around.

The side panel has:
- **Display:** original scene, objects in class colors, objects only, or the scene without objects. Boxes and labels can be toggled.
- **Classes:** show or hide each class.
- **Nearby objects:** lists objects within a radius of the camera as you navigate. Optionally, only those are shown.
- **Selected object:** click inside a box or pick from the list to see confidence, supporting frames, size, height and file. "Fly to object" moves the camera to it.

Use `--max-splats 800000` on slower GPUs.

## Outputs

All of these are regenerated and ignored by git, except `outputs/equipment_room/objects/objects.json`, which is committed so the boxes are available without a rerun.

- `renders/equipment_room/`: reusable view set. It contains `images/`, `depth/` (uint16, metres × 1000), `alpha/`, `cameras.json` (K, OpenCV world-to-camera R/t, camera-to-world), `transforms.json` (nerfstudio), `candidates.json` (accepted and rejected views with reasons) and `contact_sheets/`.
- `outputs/equipment_room/scene_overview/`: `topdown.png`, `cameras_topdown.png` (camera map), `overview_*.png`, `scene_meta.json`.
- `outputs/equipment_room/detections/`: per-view JSON in Grounded-SAM-2 format (RLE masks), `annotated/`, `contact_sheets/`, `summary.json`.
- `outputs/equipment_room/objects/`:
  - `<Class>/<Class>_NN.ply`: one splat per object
  - `objects.json`: class, confidence, views, Gaussian count, oriented box corners, centroid
  - `scene_segmented_colored.ply`, `scene_remainder.ply`
  - `topdown_boxes.png`
  - `previews/`: each object shown in the scene next to its isolated render
- `../../data/objects/grounded_sam2_equipment_room.json` (stage 7): 26 objects as `id`, `class`, `box` (AABB min/max in the raw scan frame), `class_confidence`, `source`, and `note`. Boxes are axis-aligned; oriented boxes stay in `objects.json`. Where a same-class object in `data/semantic_layer.json` overlaps by at least 50% of our box volume, the note says which layer id it overlaps (no asset tags are guessed). Optional manual asset links: `app_export.tags` in `config.yaml`, e.g. `Valves_07: CV-006`.

## Tuning

- `classes[].box_threshold`: per-class detector score, applied in stage 4, so `--from-stage 4` is enough to retune.
- `classes[].min_gaussians`, `min_height_m`, `min_top_m`, `max_size_m`: physical sanity checks on final objects.
- `render.levels`, `render.fov_y_deg`, `render.validate`: camera heights, pitch, field of view and view rejection.
- `instances.core_vote_ratio` / `grow_vote_ratio` / `min_detect_ratio`: how strict 3D voting and cross-view consistency are.
- `app_export.high` / `app_export.medium`: view-count and detection-ratio thresholds for `class_confidence` in the app file.
- `app_export.same_class_share`: minimum overlap fraction for an overlap note against the scan layer.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

They need MLX and the scan, so the repo CI does not run them. It does lint this folder (`ruff check .` from the repo root).
