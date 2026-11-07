r"""Extract objects from a Gaussian splat scan into a semantic layer.

Run:    .venv\Scripts\python extract.py [scan.ply]
        .venv\Scripts\python extract.py [scan.ply] --close    second pass for small devices
Output: extraction/<scan>/
          frames/               rendered views, from view_splat.py --capture
          proposals.npz         object candidates traced back to the splats
          objects/obj_001.ply   one .ply per object
          objects_close.json    small devices from the close pass, with objects/cls_001.ply and so on
          objects.json          objects with labels and 3D boxes
          review_01.jpg ...     large labelled crops, for checking the result by eye

Steps: choose camera stops on free floor, render them, find candidates with Grounding DINO, outline
them with SAM 2.1, trace pixels back to splats, merge the same object across views, then name each
object with CLIP from objects.txt. Each step is cached, so a rerun resumes where it stopped.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import view_splat as vs

ROOT = vs.ROOT
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL = "facebook/sam2.1-hiera-tiny"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Tuning knobs. Heights and distances are in scan units, which are not calibrated.
STOP_SPACING = 5                  # camera stops every this many grid cells
STOP_MIN_SPRITE = 3               # stops only on cells with this much clearance
AISLE_REACH = 3                   # objects this many cells away on both sides make an aisle
AISLE_SPACING = 3                 # narrow aisle stops every this many grid cells
STOP_VIEWS = (0.55, -0.12), (1.1, -0.45)  # (height above floor, pitch in radians)
STOP_YAWS = 8                     # views per stop
DETECT_THRESHOLD = 0.25
MAX_MASK_SHARE = 0.25             # ignore outlines covering more of the frame than this
SOLID = np.log(0.3 / 0.7)         # opacity logit, splats below this are not traced
MIN_SPLATS = 40                   # a candidate needs this many splats
DEPTH_MARGIN = 0.5                # keep splats this far behind the object's front surface
VOXEL = 0.06                      # size used to find the connected part of a candidate
MAX_EXTENT = 3.0                  # reject candidates larger than this, usually walls
MERGE_OVERLAP = 0.5               # candidates sharing this share of splats are the same object
MIN_VIEWS = 2                     # an object must be seen in this many frames
LABEL_CROPS = 4                   # crops per object used for naming
MIN_LABEL_SCORE = 0.3             # below this, the object is unknown
REJECT_LABELS = {"wall", "floor", "ceiling"}

# Close pass: cameras right beside the equipment, looking for small devices only.
CLOSE_SPACING = 4                 # close stops on cells where (ix + iz) divides by this
CLOSE_REACH = 2                   # a close stop has blocked cells within this many cells
CLOSE_VIEWS = (0.4, -0.1), (0.8, 0.0), (1.05, 0.3)  # (height above floor, pitch in radians)
CLOSE_TURNS = -0.6, 0.0, 0.6      # radians either side of facing the equipment
CLOSE_PROMPTS = ["valve", "blue valve", "valve actuator", "pressure gauge", "water pump",
                 "circulation pump", "electric motor", "small sensor box"]
CLOSE_LIMITS = 20, 0.25, 0.8      # minimum splats, depth margin, maximum extent


def read_objects(path):
    """Return (class, detector prompt) rows from objects.txt."""
    rows = [[c.strip() for c in line.split("|")] for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    return [(row[0], row[1]) for row in rows]


def load_scan(path):
    """Read a scan and fit its floor. Returns splats, plane, up sign, heights and solid mask."""
    v = vs.read_ply(path)
    v = vs.level_scan(v[vs.valid_mask(v)])
    plane, band, to_viewer = vs.fit_floor(v)
    up_sign = 1.0 if to_viewer is vs.Y_DOWN else -1.0
    a, b, c = plane
    x, y, z = [v[k].astype("f8") for k in "xyz"]
    heights = up_sign * (a * x + b * z + c - y)
    return v, plane, band, up_sign, heights


def free_grid(v, heights, band):
    """Rebuild the same blocked grid as the viewer. Returns sizes and its placement."""
    pos = np.stack([v[k].astype("f4") for k in "xyz"], axis=1)
    visible = v["opacity"] >= np.log(vs.MIN_OPACITY / (1 - vs.MIN_OPACITY))
    low, high = np.percentile(pos[visible], [1, 99], axis=0)
    span = float((high - low)[[0, 2]].max())
    grid_x, grid_z = float((low[0] + high[0]) / 2), float((low[2] + high[2]) / 2)
    wall_height = float(np.percentile(heights[visible], 99))
    solid = v["opacity"] >= SOLID
    clearance = vs.object_clearance(heights[solid & visible], band)
    in_object = solid & visible & (heights > clearance) & (heights < vs.CUT_FRACTION * wall_height)
    blocked = vs.blocked_cells(pos[in_object, 0], pos[in_object, 2], grid_x, grid_z, span, vs.GRID_CELLS)
    return vs.max_sprite_sizes(vs.clean_cells(blocked)), grid_x, grid_z, span, wall_height


def in_aisle(sizes, ix, iz):
    """True when objects sit on opposite sides, like red yellow yellow red, in a row or column."""
    for dx, dz in ((1, 0), (0, 1)):
        hits = 0
        for sign in (1, -1):
            for reach in range(1, AISLE_REACH + 1):
                x, z = ix + sign * reach * dx, iz + sign * reach * dz
                if 0 <= x < sizes.shape[0] and 0 <= z < sizes.shape[1] and sizes[x, z] == 0:
                    hits += 1
                    break
        if hits == 2:
            return True
    return False


def camera_stops(sizes, grid_x, grid_z, span):
    """Open floor stops plus narrow aisle stops, each looking all the way around at two heights.

    Looking all the way around means a stop in an aisle sees the objects in front of and behind it.
    """
    step = span / len(sizes)
    stops = []
    for ix, iz in np.argwhere(sizes > 0):
        open_stop = sizes[ix, iz] >= STOP_MIN_SPRITE and ix % STOP_SPACING == 0 and iz % STOP_SPACING == 0
        aisle_stop = (sizes[ix, iz] < STOP_MIN_SPRITE and ix % AISLE_SPACING == 0 and iz % AISLE_SPACING == 0
                      and in_aisle(sizes, ix, iz))
        if open_stop or aisle_stop:
            x = grid_x - span / 2 + (ix + 0.5) * step
            z = grid_z - span / 2 + (iz + 0.5) * step
            for height, pitch in STOP_VIEWS:
                for turn in range(STOP_YAWS):
                    stops.append([float(x), height, float(z), 2 * np.pi * turn / STOP_YAWS, pitch])
    return stops


def close_stops(sizes, grid_x, grid_z, span, y_down):
    """Stops beside blocked cells, each facing the equipment next to it from several heights."""
    step = span / len(sizes)
    blocked = sizes == 0
    stops = []
    for ix, iz in np.argwhere(sizes > 0):
        if (ix + iz) % CLOSE_SPACING:
            continue
        x0, x1 = max(ix - CLOSE_REACH, 0), ix + CLOSE_REACH + 1
        z0, z1 = max(iz - CLOSE_REACH, 0), iz + CLOSE_REACH + 1
        near = np.argwhere(blocked[x0:x1, z0:z1])
        if not len(near):
            continue
        toward = (near + (x0, z0) - (ix, iz)).mean(0)
        if not toward.any():
            continue
        # The viewer's yaw looks along (sin yaw, -cos yaw), and y-down scans flip z.
        yaw = float(np.arctan2(toward[0], toward[1] if y_down else -toward[1]))
        x = grid_x - span / 2 + (ix + 0.5) * step
        z = grid_z - span / 2 + (iz + 0.5) * step
        for height, pitch in CLOSE_VIEWS:
            for turn in CLOSE_TURNS:
                stops.append([float(x), height, float(z), yaw + turn, pitch])
    return stops


def project(points, view, focal, size):
    """Pixel x, pixel y and depth for scene points. Depth is positive in front of the camera."""
    camera = points @ view[:3, :3].T + view[:3, 3]
    depth = -camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        px = size[0] / 2 + focal * camera[:, 0] / depth
        py = size[1] / 2 - focal * camera[:, 1] / depth
    return px, py, depth


def connected_part(points):
    """Indices of the largest connected group of points on a voxel grid."""
    from scipy import ndimage

    cells = np.floor((points - points.min(0)) / VOXEL).astype(int)
    grid = np.zeros(cells.max(0) + 1, dtype=bool)
    grid[tuple(cells.T)] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if count == 1:
        return np.arange(len(points))
    point_labels = labels[tuple(cells.T)]
    return np.flatnonzero(point_labels == np.bincount(point_labels).argmax())


def trace_mask(mask, px, py, depth, candidates, points, limits):
    """Splats behind a mask, keeping the connected group at the object's front surface."""
    min_splats, depth_margin, max_extent = limits
    h, w = mask.shape
    inside = (depth > 0.05) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    ids = np.flatnonzero(inside)
    ids = ids[mask[py[ids].astype(int), px[ids].astype(int)]]
    if len(ids) < min_splats:
        return None
    front = np.percentile(depth[ids], 5)
    ids = ids[depth[ids] < front + depth_margin]
    if len(ids) < min_splats:
        return None
    ids = ids[connected_part(points[ids])]
    if len(ids) < min_splats or (np.ptp(points[ids], axis=0)).max() > max_extent:
        return None
    return candidates[ids]


def find_proposals(frames_dir, meta, v, prompts, keep_mask, out, limits=(MIN_SPLATS, DEPTH_MARGIN, MAX_EXTENT)):
    """Detect and outline objects in every frame, and trace each outline to splat ids."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(device).half().eval()
    sam_processor = Sam2Processor.from_pretrained(SAM_MODEL)
    sam = Sam2Model.from_pretrained(SAM_MODEL).to(device).eval()

    candidates = np.flatnonzero(keep_mask)
    points = np.stack([v[k].astype("f4") for k in "xyz"], axis=1)[candidates]
    size = meta["size"]
    focal = size[1] / 2 / np.tan(meta["fov_y"] / 2)
    id_lists, records = [], []
    started = time.time()
    for index, frame in enumerate(meta["frames"]):
        image = Image.open(frames_dir / frame["file"]).convert("RGB")
        with torch.inference_mode(), torch.autocast(device, dtype=torch.float16):
            inputs = dino_processor(images=image, text=[prompts], return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].half()
            result = dino_processor.post_process_grounded_object_detection(
                dino(**inputs), inputs.input_ids, threshold=DETECT_THRESHOLD, text_threshold=DETECT_THRESHOLD,
                target_sizes=[image.size[::-1]])[0]
            boxes = result["boxes"].float()
            if len(boxes):
                sam_inputs = sam_processor(images=image, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
                masks = sam(**sam_inputs, multimask_output=False).pred_masks[0, :, 0]
                masks = F.interpolate(masks[None].float(), size=size[::-1], mode="bilinear")[0] > 0
                # Shrink each outline so edge pixels do not catch the wall behind.
                masks = -F.max_pool2d(-masks[None].float(), 7, stride=1, padding=3)[0] > 0.5
                masks = masks.cpu().numpy()
        if len(boxes):
            px, py, depth = project(points, np.array(frame["view"]), focal, size)
            for mask, box, score in zip(masks, boxes.cpu().numpy(), result["scores"].float().cpu().numpy()):
                if mask.mean() > MAX_MASK_SHARE or mask.sum() < 100:
                    continue
                ids = trace_mask(mask, px, py, depth, candidates, points, limits)
                if ids is not None:
                    id_lists.append(ids)
                    records.append((index, *box.tolist(), float(score), float(mask.sum())))
        if index % 25 == 0:
            print(f"frame {index + 1}/{len(meta['frames'])}, {len(id_lists)} candidates, "
                  f"{time.time() - started:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    lengths = np.array([len(i) for i in id_lists])
    np.savez_compressed(out, ids=np.concatenate(id_lists), lengths=lengths, records=np.array(records))
    del dino, sam
    torch.cuda.empty_cache()


def merge_proposals(id_lists, records, points):
    """Merge candidates that share splats. Largest first, so parts join their whole."""
    order = np.argsort([-len(ids) for ids in id_lists])
    objects = []  # each: ids array, set of frames, record indices, box
    for index in order:
        ids = id_lists[index]
        low, high = points[ids].min(0), points[ids].max(0)
        match = None
        for obj in objects:
            if (low > obj["high"]).any() or (high < obj["low"]).any():
                continue
            shared = len(np.intersect1d(ids, obj["ids"], assume_unique=True))
            if shared / min(len(ids), len(obj["ids"])) >= MERGE_OVERLAP:
                match = obj
                break
        if match is None:
            objects.append({"ids": ids, "low": low, "high": high, "frames": {int(records[index, 0])}, "records": [index]})
        else:
            match["frames"].add(int(records[index, 0]))
            match["records"].append(index)
            # Grow only slowly, so one bad candidate cannot join two neighbouring objects.
            if len(np.intersect1d(ids, match["ids"], assume_unique=True)) / len(ids) >= 0.8:
                match["ids"] = np.union1d(match["ids"], ids)
                match["low"] = np.minimum(match["low"], low)
                match["high"] = np.maximum(match["high"], high)
    return [obj for obj in objects if len(obj["frames"]) >= MIN_VIEWS]


def name_objects(objects, records, frames_dir, meta, classes):
    """Choose a class for each object by CLIP voting over its clearest crops."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    names = classes + sorted(REJECT_LABELS)
    text = processor(text=[f"a photo of a {name} in an industrial room" for name in names],
                     return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text).pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    for obj in objects:
        best = sorted(obj["records"], key=lambda i: -records[i, 6])[:LABEL_CROPS]
        crops = []
        for index in best:
            frame = meta["frames"][int(records[index, 0])]
            x0, y0, x1, y1 = records[index, 1:5]
            margin = 0.1 * max(x1 - x0, y1 - y0)
            image = Image.open(frames_dir / frame["file"]).convert("RGB")
            crops.append(image.crop((int(x0 - margin), int(y0 - margin), int(x1 + margin), int(y1 + margin))))
        pixels = processor(images=crops, return_tensors="pt").to(device)
        with torch.inference_mode():
            features = model.get_image_features(**pixels).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
            scores = (100 * features @ text_features.T).softmax(dim=-1).mean(0).cpu().numpy()
        top = int(scores.argmax())
        obj["label"] = names[top] if scores[top] >= MIN_LABEL_SCORE else "unknown"
        obj["score"] = round(float(scores[top]), 3)
        obj["crops"] = crops


def save_ply(path, v, ids):
    """Write these splats as a standard .ply."""
    subset = v[np.sort(ids)]
    types = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "u2": "ushort", "i2": "short",
             "u4": "uint", "i4": "int"}
    properties = "".join(f"property {types[subset.dtype[name].str[1:]]} {name}\n" for name in subset.dtype.names)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(subset)}\n{properties}end_header\n".encode())
        subset.tofile(f)


def review_pages(objects, folder, prefix="review", columns=5, rows=4, cell=(300, 280)):
    """Pages of large crops. Each cell shows an object's two clearest views, with its box marked."""
    per_page = columns * rows
    for page, start in enumerate(range(0, len(objects), per_page), 1):
        sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), (20, 22, 26))
        draw = ImageDraw.Draw(sheet)
        for index, obj in enumerate(objects[start:start + per_page]):
            x, y = index % columns * cell[0], index // columns * cell[1]
            draw.text((x + 4, y + 3), f"{obj['id']}  CLIP: {obj['label']}", fill=(255, 230, 120))
            for slot, crop in enumerate(obj["crops"][:2]):
                copy = crop.copy()
                copy.thumbnail(((cell[0] - 12) // 2, cell[1] - 26))
                sheet.paste(copy, (x + 4 + slot * (cell[0] // 2), y + 20))
        sheet.save(folder / f"{prefix}_{page:02d}.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", nargs="?", default=vs.PLY, type=Path)
    parser.add_argument("--classes", default=Path(__file__).parent / "objects.txt", type=Path)
    parser.add_argument("--close", action="store_true")
    args = parser.parse_args()

    out = ROOT / "extraction" / args.ply.stem
    frames_dir = out / ("frames_close" if args.close else "frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = read_objects(args.classes)
    prompts = sorted({prompt for _, prompt in rows})
    classes = sorted({name for name, _ in rows})

    v, plane, band, up_sign, heights = load_scan(args.ply)
    sizes, grid_x, grid_z, span, wall_height = free_grid(v, heights, band)

    if not (frames_dir / "frames.json").exists():
        stops = (close_stops(sizes, grid_x, grid_z, span, up_sign > 0) if args.close
                 else camera_stops(sizes, grid_x, grid_z, span))
        (frames_dir / "stops.json").write_text(json.dumps(stops))
        print(f"{len(stops)} camera stops", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "view_splat.py"), str(args.ply), f"--capture={frames_dir / 'stops.json'}"],
                       check=True)
    meta = json.loads((frames_dir / "frames.json").read_text())

    # Trace only solid splats between the floor and the ceiling, inside the scanned room.
    pos = np.stack([v[k].astype("f8") for k in "xyz"], axis=1)
    room = (np.abs(pos[:, 0] - grid_x) < span / 2) & (np.abs(pos[:, 2] - grid_z) < span / 2)
    keep = (v["opacity"] >= SOLID) & (heights > vs.object_clearance(heights[v["opacity"] >= SOLID], band)) & (heights < wall_height) & room
    print(f"{keep.sum():,} splats can belong to objects", flush=True)

    proposals = out / ("proposals_close.npz" if args.close else "proposals.npz")
    if not proposals.exists():
        if args.close:
            find_proposals(frames_dir, meta, v, CLOSE_PROMPTS, keep, proposals, CLOSE_LIMITS)
        else:
            find_proposals(frames_dir, meta, v, prompts, keep, proposals)
    data = np.load(proposals)
    id_lists = np.split(data["ids"], np.cumsum(data["lengths"])[:-1])
    records = data["records"]
    print(f"{len(id_lists)} candidates", flush=True)

    points = pos.astype("f4")
    objects = merge_proposals(id_lists, records, points)
    print(f"{len(objects)} objects after merging across views", flush=True)

    name_objects(objects, records, frames_dir, meta, classes)
    kept = [obj for obj in objects if obj["label"] not in REJECT_LABELS]
    print(f"{len(kept)} objects after removing walls, floors and ceilings", flush=True)

    (out / "objects").mkdir(exist_ok=True)
    results = []
    for number, obj in enumerate(sorted(kept, key=lambda o: -len(o["ids"])), 1):
        obj["id"] = f"{'cls' if args.close else 'obj'}_{number:03d}"
        low, high = np.percentile(points[obj["ids"]], [2, 98], axis=0)
        save_ply(out / "objects" / f"{obj['id']}.ply", v, obj["ids"])
        results.append({"id": obj["id"], "label": obj["label"], "score": obj["score"], "splats": len(obj["ids"]),
                        "views": len(obj["frames"]), "box_min": low.round(4).tolist(), "box_max": high.round(4).tolist(),
                        "ply": f"objects/{obj['id']}.ply"})
    review_pages(sorted(kept, key=lambda o: o["id"]), out, "review_close" if args.close else "review")
    (out / ("objects_close.json" if args.close else "objects.json")).write_text(json.dumps({
        "scan": args.ply.name, "units": "scan units, not calibrated", "floor_plane_y_ax_bz_c": list(map(float, plane)),
        "objects": results}, indent=1))
    print(f"Wrote {len(results)} objects to {out}", flush=True)


if __name__ == "__main__":
    # Self-check: a point straight ahead of a camera at the origin lands mid-frame at its depth.
    px_, py_, depth_ = project(np.array([[0.0, 0.0, -2.0]]), np.eye(4), 100, (200, 100))
    assert (px_[0], py_[0], depth_[0]) == (100, 50, 2)
    # Self-check: red yellow yellow red is an aisle, and a wide open row is not.
    assert in_aisle(np.array([[0, 1, 1, 0]]), 0, 1)
    assert not in_aisle(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]]), 0, 4)
    # Self-check: a close stop faces its blocked cells, along +x and on a y-down scan.
    test_sizes = np.ones((6, 3), dtype=int)
    test_sizes[5, :] = 0
    test_stops = [s for s in close_stops(test_sizes, 0, 0, 6, True) if s[0] == 1.5 and s[3 + 0] and s[1] == 0.4]
    assert any(abs(s[3] - np.pi / 2) < 0.7 for s in test_stops)
    main()
