# Equipment Room: object extraction with Grounded-SAM-2

Extracts one Gaussian splat per object from `Equipment Room.ply`, for a list of text prompts. The scan is not in git: put it in the repository root, as for the main app, or change `input_ply` in `config.yaml` (relative to this folder).

Two configurations are included:
- `config.yaml`: the first run, 5 classes. Results in `outputs/equipment_room/`.
- `config_all_objects.yaml`: one class per prompt in [`objects_to_extract.json`](objects_to_extract.json) (43 prompts). It inherits everything else from `config.yaml`. Results in `outputs/equipment_room_all_objects/`. The method is described in [docs/README.md](docs/README.md).

Pipeline:
1. `01_view_scene.py`: fits the floor plane and up axis, builds the floor plan and free-space map, renders a top-down view and in-room overview shots.
2. `02_render_views.py`: places cameras inside the room (floor underneath, clear of walls and equipment) at 1.6 m and 1.0 m, 8 directions each, ~90° horizontal FOV. Views are validated (holes, too close, blur) and saved for reuse.
3. `03_detect_segment.py`: Grounding DINO (`IDEA-Research/grounding-dino-base`) boxes, then SAM 2.1 Hiera-L masks (Grounded-SAM-2). Long prompt lists are run in groups (`detection.prompts_per_pass`).
4. `04_lift_masks.py`: one backward pass through the mlx3d rasterizer gives each Gaussian's exact blending weight (alpha × transmittance) inside every mask, so occluded Gaussians get none.
5. `05_build_instances.py`: matches masks across views, takes a cross-view vote (with hysteresis growth for attached parts), filters out inconsistent detections, removes floor splats and floaters, merges one object found by two prompts, resolves Gaussians claimed by several objects, and fits gravity-aligned boxes.
6. `06_export_objects.py`: one `.ply` per object (original splat rows), `objects.json`, colored and remainder scenes, previews.
7. `07_export_app_objects.py`: writes [`data/objects/grounded_sam2_equipment_room.json`](../../data/objects/grounded_sam2_equipment_room.json) in the Since-AI [`data_format`](../../data_format/README.md), so `build_scene.py` can merge the objects into the twin. Both configurations write this file; the last export wins.

Rendering uses [mlx3d](https://pypi.org/project/mlx3d/) (Metal) because gsplat needs CUDA. Grounding DINO and SAM 2 run on MPS.

## Setup

Needs an Apple Silicon Mac. `setup_env.sh` creates `.venv`, clones Grounded-SAM-2 into `third_party/` and downloads the SAM 2.1 checkpoint into `checkpoints/`. None of these are committed.

```bash
./setup_env.sh
source .venv/bin/activate
```

## Run

```bash
python run_pipeline.py --config config_all_objects.yaml               # all prompts, every stage
python run_pipeline.py --config config_all_objects.yaml --from-stage 4 # retune class thresholds without re-detecting
python run_pipeline.py --config config_all_objects.yaml --from-stage 7 # refresh the app data/objects file only
python run_pipeline.py --force                                         # the 5-class configuration, re-rendering its view set
```

Each configuration renders its own view set the first time; `--force` renders it again.

After stage 7, from the Since-AI repo root:

```bash
python pipeline/added_objects.py data/objects/grounded_sam2_equipment_room.json
python pipeline/added_objects.py --into-layer data/objects/grounded_sam2_equipment_room.json
SKIP_SPLAT=1 python pipeline/build_scene.py "Equipment Room.ply" builder_pack.xlsx data/semantic_layer.json web
python pipeline/build_layer.py builder_pack.xlsx
```

See [data_format/README.md](../../data_format/README.md) for the object fields and coordinate rules.

## View in 3D

```bash
python view_objects.py --config config_all_objects.yaml   # then open http://localhost:8080
python view_objects.py --config config_all_objects.yaml --select <object name> --display "Objects in class colors"
```

Object names are in `outputs/<scene_id>/objects/objects.json`. Without `--config`, the viewer shows the 5-class run.

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

Per configuration, under its `scene_id`. All of these are regenerated and ignored by git, except `outputs/<scene_id>/objects/objects.json`, which is committed so the boxes are available without a rerun.

- `renders/<scene_id>/`: reusable view set. It contains `images/`, `depth/` (uint16, metres × 1000), `alpha/`, `cameras.json` (K, OpenCV world-to-camera R/t, camera-to-world), `transforms.json` (nerfstudio), `candidates.json` (accepted and rejected views with reasons) and `contact_sheets/`.
- `outputs/<scene_id>/scene_overview/`: `topdown.png`, `cameras_topdown.png` (camera map), `overview_*.png`, `scene_meta.json`.
- `outputs/<scene_id>/detections/`: per-view JSON in Grounded-SAM-2 format (RLE masks), `annotated/`, `contact_sheets/`, `summary.json`.
- `outputs/<scene_id>/objects/`:
  - `<Class>/<Class>_NN.ply`: one splat per object (names keep letters, digits and `_` only)
  - `objects.json`: class, confidence, views, Gaussian count, oriented box corners, centroid
  - `scene_segmented_colored.ply`, `scene_remainder.ply`
  - `topdown_boxes.png`
  - `previews/`: each object shown in the scene next to its isolated render
- `../../data/objects/grounded_sam2_equipment_room.json` (stage 7): the objects as `id`, `class` (the prompt in lower case), `box` (AABB min/max in the raw scan frame), `class_confidence`, `source` and `note`. Boxes are axis-aligned; oriented boxes stay in `objects.json`. Where a same-class object from another source in `data/semantic_layer.json` overlaps by at least 50% of our box volume, the note says which layer id it overlaps (no asset tags are guessed). `--into-layer` writes these objects into the layer itself, with source `grounded-sam-2`. Optional manual asset links: `app_export.tags` in the configuration, mapping an object name from `objects.json` to a builder-pack tag.

## Tuning

- `classes[].box_threshold`: per-class detector score, applied in stage 4, so `--from-stage 4` is enough to retune. In `config_all_objects.yaml`, `class_defaults` applies to every prompt and `class_overrides` (keyed by the prompt in lower case) changes single classes.
- `classes[].min_gaussians`, `min_height_m`, `min_top_m`, `max_size_m`, `floor_standing`: physical sanity checks and floor cleanup on final objects.
- `detection.prompts_per_pass`: how many prompts share one Grounding DINO pass. All prompts in one pass lower every score.
- `render.levels`, `render.fov_y_deg`, `render.validate`: camera heights, pitch, field of view and view rejection.
- `instances.core_vote_ratio` / `grow_vote_ratio` / `min_detect_ratio`: how strict 3D voting and cross-view consistency are.
- `instances.cross_class_dup_iou`: when two classes' objects share this share of their Gaussians (IoU), the more confident one keeps the object.
- `app_export.high` / `app_export.medium`: view-count and detection-ratio thresholds for `class_confidence` in the app file.
- `app_export.same_class_share`: minimum overlap fraction for an overlap note against the scan layer.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

They need MLX and the scan, so the repo CI does not run them. It does lint this folder (`ruff check .` from the repo root).
