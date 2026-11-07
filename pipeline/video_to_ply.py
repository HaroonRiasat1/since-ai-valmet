r"""Turn a video into a Gaussian splat .ply that view_splat.py can open.

Run:    .venv\Scripts\python video_to_ply.py "my room.mp4"
Output: <video name>.ply next to the video, and a <video name>_work folder of frames and cameras

Steps: take frames with ffmpeg, find the camera positions with COLMAP, then train the splats with gsplat.
Needs ffmpeg on PATH. A GPU is strongly recommended for training, and COLMAP here runs on the CPU.
Good video: walk slowly around the room, keep it sharp, and let each view overlap the last.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image

SH_C0 = 0.28209479177387814
LEARNING_RATES = {"means": 2e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}


def video_seconds(video):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                            capture_output=True, text=True, check=True)
    return float(result.stdout)


def extract_frames(video, folder, count, width):
    """Save evenly spaced frames. Skips the work if they already exist."""
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob("*.jpg")):
        return
    rate = count / video_seconds(video)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={rate},scale={width}:-2", "-q:v", "2",
                    str(folder / "frame_%05d.jpg")], check=True)


def find_cameras(images, work):
    """Run COLMAP on the frames. Returns its largest reconstruction."""
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    if any(sparse.glob("*/images.bin")):
        return max((pycolmap.Reconstruction(p.parent) for p in sparse.glob("*/images.bin")), key=lambda r: r.num_reg_images())
    database = work / "database.db"
    database.unlink(missing_ok=True)
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE)
    pycolmap.match_sequential(database)
    maps = pycolmap.incremental_mapping(database, images, sparse)
    if not maps:
        sys.exit("COLMAP could not place the cameras. Try a slower, sharper video with more overlap.")
    return max(maps.values(), key=lambda r: r.num_reg_images())


def load_views(reconstruction, images, device):
    """Return (image tensor, 4x4 world-to-camera, 3x3 intrinsics) for each placed frame."""
    views = []
    for image in reconstruction.images.values():
        camera = reconstruction.cameras[image.camera_id]
        pose = image.cam_from_world
        pose = pose() if callable(pose) else pose
        w2c = np.eye(4)
        w2c[:3, :4] = pose.matrix()
        pixels = np.asarray(Image.open(images / image.name).convert("RGB"), dtype="f4") / 255
        views.append((torch.from_numpy(pixels).to(device), torch.tensor(w2c, dtype=torch.float32, device=device),
                      torch.tensor(camera.calibration_matrix(), dtype=torch.float32, device=device)))
    return views


def scene_points(reconstruction):
    points = reconstruction.points3D.values()
    return (np.array([p.xyz for p in points], dtype="f4"), np.array([p.color for p in points], dtype="f4") / 255)


def make_splats(xyz, rgb, device):
    """Start with one small round splat on each COLMAP point."""
    gaps = np.linalg.norm(xyz[None, :8] - xyz[:: max(len(xyz) // 2000, 1), None], axis=2)
    size = max(float(np.median(np.sort(gaps, axis=1)[:, 1])), 1e-3)
    count = len(xyz)
    quats = torch.zeros(count, 4)
    quats[:, 0] = 1
    data = {"means": torch.from_numpy(xyz), "scales": torch.full((count, 3), float(np.log(size))), "quats": quats,
            "opacities": torch.logit(torch.full((count,), 0.1)), "sh0": torch.from_numpy((rgb - 0.5) / SH_C0)}
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in data.items()})


def render(splats, w2c, intrinsics, width, height, device):
    from gsplat.rendering import rasterization  # compiles CUDA code the first time, so it needs the GPU

    colors = torch.clamp(0.5 + SH_C0 * splats["sh0"], 0, 1)
    image, _, info = rasterization(splats["means"], F.normalize(splats["quats"], dim=-1), torch.exp(splats["scales"]),
                                   torch.sigmoid(splats["opacities"]), colors, w2c[None], intrinsics[None],
                                   width, height, packed=device == "cpu")
    return image[0], info


def train(splats, views, steps, scene_size, device):
    from gsplat.strategy import DefaultStrategy

    optimizers = {k: torch.optim.Adam([splats[k]], lr=rate * (scene_size if k == "means" else 1))
                  for k, rate in LEARNING_RATES.items()}
    strategy = DefaultStrategy(verbose=False, refine_stop_iter=int(steps * 0.8))
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_size)
    generator = torch.Generator().manual_seed(0)
    for step in range(steps):
        target, w2c, intrinsics = views[int(torch.randint(len(views), (1,), generator=generator))]
        height, width = target.shape[:2]
        image, info = render(splats, w2c, intrinsics, width, height, device)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss = F.l1_loss(image, target)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=device == "cpu")
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            print(f"step {step + 1}/{steps}, loss {loss.item():.4f}, splats {len(splats['means']):,}", flush=True)


def save_ply(path, splats):
    """Write the standard 3DGS fields, in the layout view_splat.py and other viewers read."""
    fields = {"x": splats["means"][:, 0], "y": splats["means"][:, 1], "z": splats["means"][:, 2]}
    quats = F.normalize(splats["quats"], dim=-1)
    for index in range(3):
        fields[f"scale_{index}"] = splats["scales"][:, index]
        fields[f"f_dc_{index}"] = splats["sh0"][:, index]
    for index in range(4):
        fields[f"rot_{index}"] = quats[:, index]
    fields["opacity"] = splats["opacities"]
    columns = {k: v.detach().cpu().numpy().astype("<f4") for k, v in fields.items()}
    records = np.zeros(len(columns["x"]), dtype=[(k, "<f4") for k in columns])
    for key, values in columns.items():
        records[key] = values
    properties = "".join(f"property float {k}\n" for k in columns)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(records)}\n{properties}end_header\n".encode())
        records.tofile(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=200, help="frames taken from the video")
    parser.add_argument("--width", type=int, default=960, help="frame width in pixels")
    parser.add_argument("--steps", type=int, default=7000, help="training steps")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No GPU found. Training on the CPU will be very slow.", flush=True)

    work = args.video.with_name(args.video.stem + "_work")
    images = work / "images"
    extract_frames(args.video, images, args.frames, args.width)
    print(f"{len(list(images.glob('*.jpg')))} frames", flush=True)
    reconstruction = find_cameras(images, work)
    xyz, rgb = scene_points(reconstruction)
    print(f"{reconstruction.num_reg_images()} cameras placed, {len(xyz):,} points", flush=True)

    views = load_views(reconstruction, images, device)
    scene_size = float(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1).mean())
    splats = make_splats(xyz, rgb, device)
    train(splats, views, args.steps, scene_size, device)
    output = args.video.with_suffix(".ply")
    save_ply(output, splats)
    print(f"Saved {output} with {len(splats['means']):,} splats", flush=True)


if __name__ == "__main__":
    # Self-check: a saved splat reads back with the same position.
    demo = make_splats(np.array([[1, 2, 3], [4, 5, 6]], dtype="f4"), np.full((2, 3), 0.5, dtype="f4"), "cpu")
    demo_path = Path(__file__).with_name(".ply_check.ply")
    save_ply(demo_path, demo)
    header_end = demo_path.read_bytes().index(b"end_header\n") + len("end_header\n")
    assert np.frombuffer(demo_path.read_bytes()[header_end:header_end + 4], "<f4")[0] == 1
    demo_path.unlink()
    main()
