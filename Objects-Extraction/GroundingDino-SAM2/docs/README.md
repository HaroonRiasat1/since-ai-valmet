# Methodology: Object Extraction from a 3D Gaussian Splatting Scene

This document describes how the object splats in `outputs/equipment_room_all_objects/objects/` were produced from the raw scene `Equipment Room.ply`. The classes are **43 text prompts**, one class per prompt, listed in [`objects_to_extract.json`](../objects_to_extract.json). [`config_all_objects.yaml`](../config_all_objects.yaml) holds the settings; it extends `config.yaml`, the configuration of an earlier run with 5 classes.

A Gaussian splat scene carries appearance but no semantics. The method therefore works in four steps:
1. Render the scene from many virtual cameras placed inside the room.
2. Detect and segment the prompted objects in those frames with Grounded-SAM-2 (Grounding DINO + SAM 2.1).
3. Lift every 2D mask back onto the exact Gaussians that produced its pixels.
4. Fuse the evidence from all frames into clean, separate 3D objects.

The result is **136 object splats in 19 of the 43 classes**. Each has a gravity-aligned 3D box and a record of the frames that support it.

![Pipeline overview](assets/pipeline_overview.png)

*Figure 1. Pipeline overview: raw scene, scene analysis, rendered frames, detections, lifting and fusion, final outputs.*

## Flow diagram

```mermaid
flowchart TD
    raw["Raw scene: Equipment Room.ply, 1,676,481 Gaussians"]
    s1["Stage 1: Scene analysis"]
    s2["Stage 2: Render frames"]
    s3["Stage 3: Detect and segment"]
    s4["Stage 4: Lift masks to Gaussians"]
    s5["Stage 5: Fuse into 3D instances"]
    s6["Stage 6: Export objects"]
    s7["Stage 7: App data/objects JSON"]
    a1(["Floor plane, up axis, floor plan, ceiling height, clearance map"])
    a2(["Rendered frames: 147 RGB, depth and alpha views with cameras.json"])
    a3(["Detections: 3,616 Grounding DINO boxes with SAM 2.1 masks, 43 prompts in 9 passes per frame"])
    a4(["Visibility per frame and Gaussian members per mask, 2,057 masks"])
    a5(["136 instances with votes, confidence and 3D boxes"])
    a6(["Final outputs: 136 object .ply files, objects.json, colored scene"])
    a7(["Since-AI: data/objects/grounded_sam2_equipment_room.json"])
    raw --> s1 --> s2 --> s3 --> s4 --> s5 --> s6 --> s7
    s1 -.-> a1
    s2 -.-> a2
    s3 -.-> a3
    s4 -.-> a4
    s5 -.-> a5
    s6 -.-> a6
    s7 -.-> a7
    a1 -.->|"camera-safe free space"| s2
    raw -.->|"same Gaussians"| s4
```

| Stage | Input | Output (artifact) |
| --- | --- | --- |
| 0. Raw scene | captured splat | `Equipment Room.ply` |
| 1. Scene analysis | raw scene | `outputs/equipment_room_all_objects/scene_overview/` |
| 2. Rendered frames | raw scene + free-space map | `renders/equipment_room_all_objects/` |
| 3. Detections | rendered frames + 43 prompts | `outputs/equipment_room_all_objects/detections/` |
| 4. Lifting | detections + cameras + raw scene | `outputs/equipment_room_all_objects/lifting/` |
| 5. 3D instances | lifted masks | `outputs/equipment_room_all_objects/instances/` |
| 6. Final outputs | instances + raw scene | `outputs/equipment_room_all_objects/objects/` |
| 7. App export | `objects.json` + scan layer | `Since-AI/data/objects/grounded_sam2_equipment_room.json` |

---

## Stage 0: Raw scene

The input is a binary little-endian 3D Gaussian Splatting PLY with **1,676,481 Gaussians**. Each Gaussian has:
- a position
- three log-scales
- a rotation quaternion
- an opacity logit
- spherical-harmonic color coefficients up to degree 3 (48 values)

The quaternion is stored in the order `rot_1, rot_2, rot_3, rot_0`, so every property is read by name, never by position.

The capture uses COLMAP-style axes, where −Y points up. The scene is a basement equipment room: about 9.0 × 8.0 m of floor, a dense overhead layer of building services at about 2.0 m, and equipment around the room's edges.

![Raw scene](assets/stage0_raw_scene.jpg)

*Figure 2. The raw splat rendered from inside the room (left) and from above with the ceiling removed (right).*

## Stage 1: Scene analysis (gravity frame and free space)

The goal of this stage is to know where the floor is, which way is up, and where a camera can stand. The analysis uses Gaussians with opacity ≥ 0.3.

1. **Floor plane.** Heights along the up hint (−Y) are histogrammed in 2 cm bins. The densest bin in the lowest 35% of the range is taken as the floor.
   - Gaussians within ±6 cm of that bin are fitted with RANSAC: 3,000 iterations, a 2 cm inlier threshold, and planes tilted more than 15° are rejected.
   - The fit is refined by SVD on the inliers.
   - The resulting up vector is (−0.0036, −1.0000, 0.0055), 0.4° from −Y. Height *h* is measured from this plane, so the floor is at *h* = 0.
2. **Ceiling height.** The densest 2 cm height bin in the upper 40% of the range gives the overhead services layer, at 2.03 m. The 99.5th percentile of all heights is 2.62 m.
3. **Horizontal axes.** Two axes, *a* ≈ x and *b* ≈ z, complete a right-handed gravity frame.
4. **Floor plan.**
   - Floor-level splats (|*h*| < 6 cm) are rasterized on a 5 cm grid.
   - The grid is closed with a 25 cm disk, reduced to its largest connected region, and its holes are filled (floor hidden under furniture).
   - This gives a 51.3 m² floor region. Eroding it by 35 cm gives the camera-safe *interior*, which keeps cameras away from the room's edges and out of the reconstruction's outer shell.
5. **Clearance map.** A 5 cm voxel grid marks a voxel occupied when it holds at least 2 opaque Gaussian centers. A Euclidean distance transform then gives, for any 3D point, its distance to the nearest geometry.

Overview renders are produced for inspection: a top-down view with every Gaussian above 1.55 m removed, and four establishing shots from inside the room.

![Scene analysis](assets/stage1_scene_analysis.jpg)

*Figure 3. Floor plan in the gravity frame with a 1 m grid (left). Floor region and camera-safe interior (right).*

## Stage 2: Rendered frames (virtual cameras)

The goal of this stage is a set of photo-like frames that show every object fully and from several sides.

- **Renderer.** [mlx3d](https://pypi.org/project/mlx3d/), a Metal translation of the reference tile-based 3DGS rasterizer. It was used because gsplat requires CUDA. Rendering uses view-dependent spherical-harmonic color (degree 3), Mip-Splatting-style antialiasing, and a 0.15 m near plane that clips floaters in front of the lens.
- **Camera model.** Pinhole, 1280 × 960 pixels, 74° vertical field of view (about 90° horizontal), OpenCV axes.
- **Positions.** Candidates lie on a 0.25 m grid over the interior, at two heights:
  - *Eye level:* 1.6 m above the floor, pitched 25° down, at least 0.40 m from any geometry. 111 valid cells.
  - *Low:* 1.0 m above the floor, pitched 12° down, at least 0.35 m from any geometry. 315 valid cells.

  Farthest-point sampling, started at the cell nearest the room center, picks 16 eye-level and 10 low positions spread over the whole room.
- **Directions.** Each position looks in 8 directions, 45° apart. With a 90° horizontal field of view, neighbouring frames overlap, so every object is seen from several positions and angles.
- **Validation.** Every candidate is rendered as RGB, alpha and expected depth. It is rejected when:
  - *holes:* more than 2% of pixels have alpha < 0.5
  - *too close:* more than 5% of pixels are nearer than 0.5 m
  - *shallow:* the median depth is below 1.0 m
  - *blurry:* the Laplacian variance at 512 × 384 is below 150 (views dominated by floaters)

  Of 208 candidates, **147 frames were accepted** (109 eye-level, 38 low). Rejections were 48 too close, 40 shallow and 4 blurry.
- **Saved for reuse.** For each frame:
  - RGB image
  - depth (16-bit, millimetres)
  - alpha
  - camera parameters in `cameras.json`: intrinsics K, world-to-camera R and t, camera-to-world
  - `transforms.json` (nerfstudio)
  - `candidates.json`: every rejected view, with its reason

![Camera map](assets/stage2_camera_map.jpg)

*Figure 4. Camera positions and viewing directions on the floor plan (green: 1.6 m, yellow: 1.0 m, red: rejected).*

![Rendered frames](assets/stage2_rendered_frames.jpg)

*Figure 5. A page of the 147 validated rendered frames.*

## Stage 3: Detections (Grounded-SAM-2)

The goal of this stage is 2D boxes and pixel masks for the 43 prompts in every frame.

- **Detector.** Grounding DINO base (`IDEA-Research/grounding-dino-base`), run through Hugging Face `transformers` as in Grounded-SAM-2's Hugging Face demo. Each prompt is one class.
- **Prompt groups.** All 43 prompts fit in one detector pass (125 of 256 text tokens), but one long prompt lowers the scores of every class. In a full run with a single pass, 21 prompts produced no candidate at all. The prompts are therefore split, in list order, into 9 groups of up to 5 (`detection.prompts_per_pass`), and each frame gets one detector pass per group. With groups, 10 prompts produce no candidate.
- **Class scoring from token spans.** Within a group, the tokens of each prompt are located in the tokenized text. For each of the 900 detector queries, a class's score is the maximum sigmoid logit over that prompt's tokens. The box gets the highest-scoring class of its group. This avoids decoding a phrase from tokens, which can merge two prompts into one class.
- **Box filtering.** The boxes of all groups are pooled, then:
  - Boxes scoring at least 0.22 are kept as candidates.
  - A box must cover between 0.04% and 60% of the image.
  - Non-maximum suppression is applied per class (IoU 0.5).
  - When boxes of different classes overlap with IoU ≥ 0.85, only the more confident one is kept. This also resolves two groups finding the same object.
- **Segmentation.** SAM 2.1 Hiera-L, from the Grounded-SAM-2 repository, is prompted with all boxes of a frame at once and returns one mask per box. A mask is kept when SAM's predicted quality is at least 0.5 and it covers at least 200 pixels.
- **Per-class thresholds.** These are applied in stage 4:
  - Every prompt starts at 0.27 (`class_defaults`).
  - The five prompts tuned in the 5-class run keep their thresholds (0.24 to 0.35). These were chosen there by viewing crops of the detections sorted by score.
  - Three prompts that name variants of one tuned prompt score like it, and share its threshold of 0.24.
  - Crops of the most frequent prompts were reviewed the same way. Their high scores come from the objects they name or from look-alikes that no threshold separates, so no other threshold was changed.

The detector produced **3,616 candidates** from 33 of the 43 prompts. **2,057 masks** pass the class thresholds.

![Detections](assets/stage3_detections.jpg)

*Figure 6. Detections that pass the class thresholds in the two frames with the most of them: boxes and SAM 2.1 masks, colored by class.*

![Detection sheet](assets/stage3_detection_sheet.jpg)

*Figure 7. Frames spread over the view set, with their detections colored by class.*

## Stage 4: Lifting 2D masks to Gaussians (visibility-aware attribution)

The goal of this stage is to know which Gaussians actually produced the pixels of each mask, accounting for occlusion.

The renderer composites any per-Gaussian feature $f_i$ front to back:

$$F(p) = \sum_i f_i\,\alpha_i(p)\,T_i(p), \qquad T_i(p) = \prod_{j<i}\bigl(1-\alpha_j(p)\bigr)$$

Here $\alpha_i(p)$ is Gaussian *i*'s opacity at pixel *p*, and $T_i(p)$ is the transmittance (the share of light not yet blocked by Gaussians in front).

$F$ is linear in $f$. So a single vector–Jacobian product through the rasterizer, with the masks $M_k$ as the cotangent, returns for every Gaussian *i* and mask *k*:

$$w_{ik} = \sum_p M_k(p)\,\alpha_i(p)\,T_i(p)$$

That is exactly how much Gaussian *i* contributes inside mask *k*. Channel 0 uses an all-ones mask, so $w_{i0}$ is the Gaussian's total visible contribution in pixel units.

Occluded Gaussians have $T_i \approx 0$ and Gaussians outside the frame cover no pixels, so both get no weight. No depth test or center projection is involved. The identity $\sum_i w_{i0} = \sum_p \alpha(p)$ holds to float precision, and unit tests check it.

- **Visible.** A Gaussian is visible in a frame when $w_{i0} \ge 0.5$ pixels.
- **Mask member.** A Gaussian belongs to mask *k* when at least half of its visible contribution falls inside the mask ($w_{ik} \ge 0.5\,w_{i0}$) and its largest axis is ≤ 0.5 m.
- **Mask erosion.** Masks are eroded by 2 px first, so boundary splats from the background are not pulled in.
- **Cost.** All 1,676,481 Gaussians are used at full resolution, with up to 30 mask channels per pass. The 2,057 masks of all 147 frames took 101 s.
- **Stored per frame:**
  - a visibility bitmask
  - log-quantized $w_{i0}$ (about 4% resolution), used later to measure an object's visible area
  - the member list of every mask

![Lifting](assets/stage4_lifting.jpg)

*Figure 8. (a) SAM 2.1 mask in one frame. (b) The Gaussians lifted from that single mask, rendered in the same frame. (c) The same Gaussians from the opposite side: a single mask also picks up stray background splats. (d) The object after fusing 47 frames in stage 5.*

## Stage 5: Fusing masks into 3D instances

The goal of this stage is to turn many noisy per-frame masks into one clean set of Gaussians per physical object. Steps 1 to 7 run per class.

1. **Proposals.** Every lifted mask with at least 30 member Gaussians is a proposal: 2,054 proposals in 30 classes.
2. **Visibility-aware affinity.** For proposal *A* (frame *a*) and proposal *B* (frame *b*), let $V_b$ be the Gaussians visible in frame *b*. Then

   $$r_{A\to B} = \frac{|A\cap B|}{|A\cap V_b|}$$

   is the share of *A*'s Gaussians visible in *b* that fall inside *B*. The affinity is $\min(r_{A\to B}, r_{B\to A})$. It is only defined when both co-visible counts are at least 20. A pair that cannot see each other's Gaussians counts as unknown, not as different.
3. **Clustering.** Clusters are merged by average linkage over the known pairs while the average affinity is at least 0.30. Two proposals from the same frame are never merged, because they are distinct detections. This gives 428 clusters.
4. **Cross-view vote with hysteresis.** For each Gaussian *g* in a cluster seen in frames $\mathcal{V}$:

   $$\rho_g = \frac{\#\{v\in\mathcal{V} : g \text{ inside the cluster's mask in } v\}}{\#\{v\in\mathcal{V} : g \text{ visible in } v\}}$$

   Gaussians with $\rho_g \ge 0.5$ (and in at least 2 frames) form the core. Gaussians with $\rho_g \ge 0.2$ join only by region growing: they must lie within 2× the core's median point spacing (1.5 to 5 cm) of the core. This recovers thin or attached parts that SAM sometimes leaves out. Background seen through gaps is not connected and stays out.
5. **Consistency.** An instance needs at least 2 detected frames and a **detection ratio** of at least 0.25. The ratio is detected frames divided by frames in which the object is clearly visible, meaning it covers at least 2,500 px (measured with $w_{i0}$) and at least 30% of its Gaussians are visible. This rejects objects that a prompt matches only occasionally. Here 170 single-frame clusters and 77 inconsistent ones were dropped.
6. **Cleanup.**
   - For floor-standing classes (14 of the 43), flat floor splats are removed: within 3 cm of the floor, with their thinnest axis within 25° of vertical.
   - Oversized splats (larger than half the object's diagonal) are removed.
   - DBSCAN runs with eps = 3× the median 8-neighbour distance (2 to 10 cm). It keeps the main component plus any component that is at least 20% of its size, within 10 cm of it, or (for floor-standing classes) directly under its footprint, such as a base.
   - Statistical outliers are removed: 8-neighbour mean distance above mean + 3σ.
7. **Duplicates.** Instances of the same class are merged when they share at least 30% of the smaller one's Gaussians, or when one box is at least 70% inside the other.
8. **Cross-class duplicates.** Two prompts can name the same object, for example a general and a more specific name. When two instances of different classes have a Gaussian IoU of at least 0.5 (`instances.cross_class_dup_iou`), the more confident one keeps the object and takes the other's Gaussians, frames and masks. IoU, unlike containment, keeps a small part separate from the large object it sits on. There were 6 such merges.
9. **Exclusive ownership.** A Gaussian still claimed by several instances, in any class, goes to the one with the highest $\rho_g \times$ confidence. Confidence is the mean detection score of the instance's masks.
10. **Physical checks.** A class can set a minimum number of splats, a minimum height, a minimum top height (not lying on the floor) and a maximum side length. The five tuned prompts keep the checks of the 5-class run; the others need at least 80 splats. No object failed them in this run.
11. **3D box.** Each object gets a gravity-aligned oriented box: the minimum-area rectangle of its floor-plane footprint (1st to 99th percentile of points) extended over its height range. A world axis-aligned box and the centroid are also recorded.

![Instances on the floor plan](assets/stage5_instances_topdown.jpg)

*Figure 9. Final instances on the floor plan (footprints of the oriented boxes, colored by class).*

![Segmentation](assets/stage5_segmentation.jpg)

*Figure 10. Rendered frames (left) and the same frames rendered from the segmented scene, with each extracted object painted in its class color (right).*

## Stage 6: Final outputs

The goal of this stage is one standalone splat per object, plus a machine-readable manifest.

- **Object splats.** Each object is written to `objects/<Class>/<Class>_NN.ply`, with names reduced to letters, digits and `_` because prompts may contain other characters. These files contain the original binary rows of its Gaussians with the original header and property layout, so they are lossless subsets that open in any 3DGS viewer.
- **`objects.json`.** For each object it records:
  - class and confidence
  - detection ratio
  - number of supporting frames, and their ids
  - Gaussian count and centroid
  - oriented box: corners, size, yaw, bottom and top height
  - axis-aligned box
  - the best frame
- **Scene files.**
  - `scene_segmented_colored.ply`: the full scene with object Gaussians painted by class.
  - `scene_remainder.ply`: the scene minus all objects.
  - The objects (260,912 Gaussians) and the remainder (1,415,569) add up exactly to the original 1,676,481.
- **Previews.** Each object is shown in its best frame with its 3D box, next to the isolated splat rendered from the same camera.

| Measure | Value |
| --- | --- |
| Prompts (classes) | 43 |
| Prompts with at least one detection candidate | 33 |
| Classes with at least one object | 19 |
| Objects | 136 |
| Objects per class, where present | 1 to 22, median 5 |
| Supporting frames per object | 2 to 47, median 7 |
| Gaussians per object | 96 to 15,886, median 852 |
| Gaussians in all objects | 260,912 |

![Object splats](assets/stage6_object_splats.jpg)

*Figure 11. Six well-supported objects from different classes, rendered in isolation.*

![Previews](assets/stage6_previews.jpg)

*Figure 12. Six more objects in their best frame with their 3D box (left of each pair) and the isolated splat (right of each pair).*

---

## Stage 7: App `data/objects` export

Stage 6 writes rich pipeline metadata in `objects.json` (oriented boxes, view lists, splat counts). The Since-AI twin consumes a separate, stricter format documented in [`data_format/README.md`](../../../data_format/README.md). Stage 7 maps each of the 136 records into that format and writes `data/objects/grounded_sam2_equipment_room.json`, replacing the export of the 5-class run, so the app shows this run's objects after its next rebuild. It validates the file with the app's `pipeline/added_objects.py` (`normalise` and `merge` against `data/semantic_layer.json`).

| Pipeline field | App field |
| --- | --- |
| prompt | `class` (lower case) |
| `class_key` + `name` suffix | `id` (`<class key>_NN`) |
| `box.aabb_min` / `aabb_max` | `box.min` / `box.max` (4 decimal places) |
| supporting frames + `detect_ratio` | `class_confidence`: 18 high, 46 medium, 72 low |
| fixed | `source`: `grounded-sam-2` |
| evidence + overlap | `note` |

`class_confidence` is high with at least 20 supporting frames and a detection ratio of at least 0.75, medium with at least 5 frames and 0.5, and low otherwise. An overlap note is added when an object of the same class from another source in the scan layer covers at least 50% of the object's axis-aligned box; 30 objects have one. Asset tags are never inferred; set `app_export.tags` in the configuration only for tags verified on site.

To load the objects into the page, from the Since-AI repo root run `python pipeline/added_objects.py`, then rebuild with `build_scene.py` and `build_layer.py` as in the data_format guide.

---

## Design choices

- **Render, then lift.** 2D foundation models detect and segment far better than any 3D method can classify raw splats. Rendering from cameras placed inside the room yields clean, well-exposed frames with known cameras, whatever the original capture trajectory was.
- **Exact attribution instead of center projection.** Projecting Gaussian centers with a depth test fails for large or semi-transparent splats and at occlusion edges. The vector–Jacobian product gives each Gaussian's true contribution to each mask, transmittance included, at the cost of one backward pass per frame.
- **Voting only where visible.** A Gaussian is judged only in frames where it is actually visible. Parts that are occluded or outside the frame are therefore never counted against the object.
- **Hysteresis instead of one threshold.** A single strict threshold drops parts that SAM masks inconsistently. A single loose threshold lets in background. A strict core with connected growth keeps attached parts and still rejects background.
- **Prompt groups instead of one long prompt.** Splitting 43 prompts into 9 passes costs 9 detector runs per frame but keeps the scores at the level a short prompt gives, which is what the thresholds were tuned for.
- **One object, one class.** Overlapping prompts are allowed. Each box keeps one class in 2D, and each object keeps one class in 3D.

## Limitations

- **Open-vocabulary classes.** The detector names each object with the closest prompt. Prompts that pair a color or material with an object type also match other objects of that color or material, so some objects carry the class of a look-alike. 10 of the 43 prompts produced no detection candidate, and 24 produced no object.
- **Overlapping prompts.** Prompts that name the same kind of object at different levels compete. A merged object keeps the more confident class, which is not always the most specific one.
- **Small parts.** Small parts score low and are often seen in only a few frames, so most of the 72 low-confidence objects are small.
- **Single-frame objects.** An object detected in only one frame is discarded (170 clusters here). Objects standing close together can merge or split.
- **Capture quality.** Object quality is limited by the original capture: surfaces never seen by the capture cameras render blurry and lift poorly.

## Software and runtime

The pipeline ran on an Apple M2 Pro (32 GB) without CUDA, using:
- mlx 0.32.4 and mlx3d 0.3.2 (rendering and vector–Jacobian products on Metal)
- PyTorch 2.15 on MPS
- `transformers` 5.22.2 (Grounding DINO base)
- SAM 2.1 Hiera-L from the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) repository
- SciPy and scikit-learn for clustering

All thresholds are in `config.yaml` and `config_all_objects.yaml`.

| Stage | Time |
| --- | --- |
| 1. Scene analysis | 5.3 s |
| 2. Rendered frames (208 candidates) | 62.3 s |
| 3. Detections (147 frames, 9 detector passes each) | about 1,100 s |
| 4. Lifting | 114.5 s |
| 5. 3D instances | 19.4 s |
| 6. Final outputs | 9.0 s |
| 7. App export | 0.1 s |
| **Total** | **about 1,310 s (22 min)** |

The figures in this document are generated from the pipeline outputs by `python docs/build_figures.py --config config_all_objects.yaml`, which draws them without class names.
