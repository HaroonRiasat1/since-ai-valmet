#!/usr/bin/env python3
"""Browse the Gaussian-splat scene with the extracted objects' 3D boxes and details (viser).

Open the printed URL in a browser and click the 3D view once so it receives key presses.
Walk mode (default): W/S forward/back, A/D sideways, Q/E down/up, arrow keys or drag to look around.
Orbit mode circles the selected object instead.
Click inside a box (or pick an object in the panel) to see its details; "Fly to object" moves the camera to it.
"""
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import viser

from splat_objects.config import load_config, repo_root, resolve_repo_path
from splat_objects.ply_io import PlyGaussian, read_binary_gaussian_ply
from splat_objects.scene import SceneFrame
from splat_objects.viz import class_color_rgb01

SH_C0 = 0.28209479177387814
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
DISPLAY_MODES = ["Original scene", "Objects in class colors", "Objects only", "Scene without objects"]
NAV_WALK, NAV_ORBIT = "Walk: turn in place", "Orbit: circle the selected object"
MIN_PICK_HALF_EXTENT = 0.08  # small objects (valves) get a larger click target
EYE_HEIGHT_M = 1.6
MIN_WALK_HEIGHT_M = 0.3
ORBIT_DISTANCE_M = 2.5  # pivot distance when orbiting with nothing selected
KEY_HELP = (
    "Click the 3D view first; keys are ignored while a panel control has focus.\n\n"
    "- **W / S**: forward / back · **A / D**: left / right · **Q / E**: down / up (2 m/s)\n"
    "- **Drag**: look around · **right-drag**: pan · **scroll**: step forward / back\n"
    "- **Arrow keys** orbit the turning point, so ← turns right and ↑ tilts down\n\n"
    "Walk mode keeps the camera between the floor and the ceiling pipes."
)


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def splat_arrays(ply: PlyGaussian, max_splats: int = 0) -> dict[str, np.ndarray]:
    """Convert 3DGS PLY rows to the centers/covariances/colors/opacities viser expects."""
    d = ply.raw_rows
    opacity = 1.0 / (1.0 + np.exp(-d["opacity"].astype(np.float32)))
    scales = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], axis=1).astype(np.float32))
    if 0 < max_splats < len(d):
        importance = opacity * scales.max(axis=1) ** 2
        keep = np.sort(np.argpartition(-importance, max_splats)[:max_splats])
        d, opacity, scales = d[keep], opacity[keep], scales[keep]
    q = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], axis=1).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    w, x, y, z = q.T
    R = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )
    cov = np.einsum("nij,nj,nkj->nik", R, scales**2, R).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], axis=1), 0.0, 1.0)
    return {
        "centers": np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float32),
        "covariances": cov,
        "rgbs": rgb.astype(np.float32),
        "opacities": opacity[:, None].astype(np.float32),
    }


@dataclass
class SceneObject:
    record: dict
    color: tuple[int, int, int]
    corners: np.ndarray
    center: np.ndarray
    dims: np.ndarray
    axes: np.ndarray
    handles: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.record["name"]

    def edges(self) -> np.ndarray:
        return np.stack([[self.corners[i], self.corners[j]] for i, j in BOX_EDGES]).astype(np.float32)

    def ray_hit(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        """Distance along the ray to this object's (slightly inflated) box, or None."""
        o = self.axes.T @ (origin - self.center)
        d = self.axes.T @ direction
        half = np.maximum(self.dims / 2, MIN_PICK_HALF_EXTENT)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / d, (half - o) / d
        t_near = np.nanmax(np.minimum(t1, t2))
        t_far = np.nanmin(np.maximum(t1, t2))
        if t_far < max(t_near, 0.0):
            return None
        return max(t_near, 0.0)


def make_object(r: dict) -> SceneObject:
    """Box pose from the 4 bottom then 4 top corners stored in objects.json."""
    corners = np.asarray(r["box"]["corners"], dtype=np.float64)
    u, v, w = corners[1] - corners[0], corners[3] - corners[0], corners[4] - corners[0]
    dims = np.array([np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(w)])
    axes = np.stack([u / (dims[0] + 1e-12), v / (dims[1] + 1e-12), w / (dims[2] + 1e-12)], axis=1)
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1
    uu, _, vt = np.linalg.svd(axes)
    rgb = tuple(int(round(255 * c)) for c in class_color_rgb01(r["class_id"]))
    return SceneObject(r, rgb, corners, corners.mean(axis=0), dims, uu @ vt)


def details_markdown(r: dict) -> str:
    b = r["box"]
    size = " × ".join(f"{s:.2f}" for s in b["size_m"])
    c = ", ".join(f"{v:.2f}" for v in r["centroid"])
    return (
        f"**{r['name']}** ({r['label']})\n\n"
        f"- Confidence: **{r['confidence']:.2f}**\n"
        f"- Detected in **{r['n_detected_views']}** of {r['n_visible_views']} frames where it is clearly visible "
        f"(ratio {r['detect_ratio']:.2f})\n"
        f"- Gaussians: **{r['n_gaussians']:,}**\n"
        f"- Box size: {size} m, yaw {b['yaw_deg']:.0f}°\n"
        f"- Height above floor: {b['bottom_height_m']:.2f} to {b['top_height_m']:.2f} m\n"
        f"- Centroid: ({c})\n"
        f"- File: `objects/{r['ply']}`"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=repo_root() / "config.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-splats", type=int, default=0, help="cap on scene splats for slower GPUs (0 = all)")
    ap.add_argument("--display", choices=DISPLAY_MODES, default=DISPLAY_MODES[0])
    ap.add_argument("--select", default=None, help="object name to select and fly to when a browser connects")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scene_id = cfg["scene_id"]
    out = repo_root() / "outputs" / scene_id
    objects_dir = out / "objects"
    records = json.loads((objects_dir / "objects.json").read_text(encoding="utf-8"))
    frame = SceneFrame.load(out / "scene_overview" / "scene_frame.npz")
    sources = {
        DISPLAY_MODES[0]: resolve_repo_path(cfg, cfg["input_ply"]),
        DISPLAY_MODES[1]: objects_dir / "scene_segmented_colored.ply",
        DISPLAY_MODES[3]: objects_dir / "scene_remainder.ply",
    }
    if args.select and args.select not in {r["name"] for r in records}:
        ap.error(f"unknown object {args.select!r}; see objects.json for names")

    server = viser.ViserServer(host=args.host, port=args.port, label=f"{scene_id}: extracted objects")
    server.scene.set_up_direction(tuple(float(v) for v in frame.up))

    # Start where the room is visible: an interior corner at eye height, looking at the room center.
    lo, hi = frame.floor_extent()
    interior = frame.interior_mask(float(cfg["scene"]["wall_margin_m"]))
    ii, jj = np.nonzero(interior)
    ab = frame.grid_origin + (np.stack([ii, jj], axis=1) + 0.5) * frame.cell
    corner = ab[np.argmin(np.linalg.norm(ab - lo, axis=1))]
    room_center = frame.world(np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.4]))
    eye_height = min(EYE_HEIGHT_M, frame.ceiling_h - 0.25)
    max_walk_height = frame.ceiling_h - 0.1
    start_eye = frame.world(np.array([corner[0], corner[1], eye_height]))
    start_dir = unit(room_center - start_eye)
    # viser's arrow keys and drag rotate around the camera's look-at point, so in walk mode that
    # pivot is kept just in front of the camera to make them turn the view in place.
    start_radius = 0.3
    server.initial_camera.position = tuple(float(v) for v in start_eye)
    server.initial_camera.look_at = tuple(float(v) for v in start_eye + start_radius * start_dir)
    server.initial_camera.up = tuple(float(v) for v in frame.up)
    server.initial_camera.fov = float(np.radians(75.0))

    objects = [make_object(r) for r in records]
    by_name = {o.name: o for o in objects}
    for obj in objects:
        base = f"/objects/{obj.record['label'].replace(' ', '_')}/{obj.name}"
        obj.handles["box"] = server.scene.add_line_segments(
            f"{base}/box", obj.edges(), np.array(obj.color, np.uint8), thickness=3.0, thickness_units="screen"
        )
        obj.handles["label"] = server.scene.add_label(
            f"{base}/label",
            f"{obj.name}  {obj.record['confidence']:.2f}",
            position=tuple(float(v) for v in obj.corners[4:].mean(axis=0) + 0.06 * frame.up),
            anchor="bottom-center",
            font_screen_scale=0.8,
        )
        obj.handles["splats"] = server.scene.add_gaussian_splats(
            f"{base}/splats", **splat_arrays(read_binary_gaussian_ply(objects_dir / obj.record["ply"])), visible=False
        )

    # ---- GUI ----
    counts: dict[int, int] = {}
    for r in records:
        counts[r["class_id"]] = counts.get(r["class_id"], 0) + 1
    server.gui.add_markdown(
        f"### {scene_id.replace('_', ' ').title()}\n"
        f"{len(records)} extracted objects. Click inside a box or pick an object below."
    )
    with server.gui.add_folder("Navigation"):
        server.gui.add_markdown(KEY_HELP)
        nav_mode = server.gui.add_dropdown("Mode", [NAV_WALK, NAV_ORBIT], initial_value=NAV_WALK)
        turn_radius = server.gui.add_slider(
            "Turn radius (m)",
            0.2,
            3.0,
            0.05,
            start_radius,
            hint="Walk mode: distance of the turning pivot in front of the camera. Smaller turns more in place.",
        )
        eye_level = server.gui.add_button("Eye height, level view")
        reset_view = server.gui.add_button("Reset view")
    with server.gui.add_folder("Display"):
        display = server.gui.add_dropdown("Scene", DISPLAY_MODES, initial_value=args.display)
        show_boxes = server.gui.add_checkbox("Boxes", True)
        show_labels = server.gui.add_checkbox("Labels", True)
    class_on: dict[int, viser.GuiCheckboxHandle] = {}
    with server.gui.add_folder("Classes"):
        for c_id, c in enumerate(cfg["classes"]):
            if counts.get(c_id):
                class_on[c_id] = server.gui.add_checkbox(f"{c['display']} ({counts[c_id]})", True)
    with server.gui.add_folder("Nearby objects"):
        radius = server.gui.add_slider("Radius (m)", 0.5, 8.0, 0.25, 3.0)
        only_near = server.gui.add_checkbox("Only show nearby", False)
        nearby_md = server.gui.add_markdown("Move the camera to list objects around it.")
    with server.gui.add_folder("Selected object"):
        selected = server.gui.add_dropdown("Object", ["(none)"] + [o.name for o in objects], initial_value="(none)")
        fly = server.gui.add_button("Fly to object")
        details_md = server.gui.add_markdown("No object selected.")

    splat_nodes: dict[str, viser.GaussianSplatHandle] = {}
    lock = threading.Lock()
    highlight: list = []
    camera_pos = {"value": np.array(server.initial_camera.position, dtype=np.float64)}

    def visible(obj: SceneObject) -> bool:
        if not class_on[obj.record["class_id"]].value:
            return False
        return not only_near.value or float(np.linalg.norm(obj.center - camera_pos["value"])) <= radius.value

    def refresh_visibility() -> None:
        for obj in objects:
            on = visible(obj)
            obj.handles["box"].visible = on and show_boxes.value
            obj.handles["label"].visible = on and show_labels.value
            obj.handles["splats"].visible = on and display.value == DISPLAY_MODES[2]

    def show_mode(mode: str) -> None:
        with lock:
            if mode in sources and mode not in splat_nodes:
                print(f"Loading '{mode}' from {sources[mode].name} ...", flush=True)
                arrays = splat_arrays(read_binary_gaussian_ply(sources[mode]), args.max_splats)
                splat_nodes[mode] = server.scene.add_gaussian_splats(f"/splats/{mode}", **arrays)
            for name, node in splat_nodes.items():
                node.visible = name == mode
        refresh_visibility()

    def refresh_nearby() -> None:
        dists = sorted(
            (float(np.linalg.norm(o.center - camera_pos["value"])), o.name, o)
            for o in objects
            if class_on[o.record["class_id"]].value
        )
        rows = [f"- **{o.name}** ({o.record['label']}): {d:.1f} m" for d, _, o in dists if d <= radius.value]
        nearby_md.content = "\n".join(rows) if rows else f"No objects within {radius.value:.1f} m."

    def select(name: str) -> None:
        for h in highlight:
            h.remove()
        highlight.clear()
        if name not in by_name:
            details_md.content = "No object selected."
            return
        obj = by_name[name]
        r = obj.record
        highlight.append(
            server.scene.add_line_segments(
                "/selected/box", obj.edges(), np.array([255, 255, 255], np.uint8), thickness=6.0, thickness_units="screen"
            )
        )
        size = "×".join(f"{s:.2f}" for s in r["box"]["size_m"])
        highlight.append(
            server.scene.add_label(
                "/selected/label",
                f"{r['label']} · conf {r['confidence']:.2f} · {r['n_detected_views']} views · {size} m",
                position=tuple(float(v) for v in obj.corners[4:].mean(axis=0) + 0.22 * frame.up),
                anchor="bottom-center",
                font_screen_scale=1.0,
            )
        )
        details_md.content = details_markdown(r)
        if selected.value != name:
            selected.value = name

    def set_view(client: viser.ClientHandle, eye: np.ndarray, target: np.ndarray) -> None:
        """Place the camera at `eye` looking at `target`; in walk mode the pivot is moved close."""
        if nav_mode.value == NAV_WALK:
            target = eye + turn_radius.value * unit(target - eye)
        with client.atomic():
            client.camera.position = tuple(float(v) for v in eye)
            client.camera.look_at = tuple(float(v) for v in target)

    def recenter_pivot(client: viser.ClientHandle) -> None:
        """Walk mode: keep the pivot `turn radius` ahead on the current view ray (the view does not change).

        WASD and turning keep the pivot distance, so this only acts after scrolling or a fly-to.
        """
        pos = np.asarray(client.camera.position, dtype=np.float64)
        ray = np.asarray(client.camera.look_at, dtype=np.float64) - pos
        dist = float(np.linalg.norm(ray))
        if dist > 1e-6 and abs(dist - turn_radius.value) > max(0.02, 0.1 * turn_radius.value):
            client.camera.look_at = tuple(float(v) for v in pos + ray / dist * turn_radius.value)

    def keep_in_room(client: viser.ClientHandle) -> None:
        """Walk mode: stop Q/E or scrolling from taking the camera through the floor or ceiling."""
        pos = np.asarray(client.camera.position, dtype=np.float64)
        h = float(frame.heights(pos[None])[0])
        clamped = min(max(h, MIN_WALK_HEIGHT_M), max_walk_height)
        if abs(clamped - h) > 1e-3:
            client.camera.position = tuple(float(v) for v in pos + (clamped - h) * frame.up)

    def fly_to(client: viser.ClientHandle, name: str) -> None:
        if name not in by_name:
            return
        obj = by_name[name]
        away = room_center - obj.center
        away -= (away @ frame.up) * frame.up
        if np.linalg.norm(away) < 1e-3:
            away = frame.e1.copy()
        away /= np.linalg.norm(away)
        eye = obj.center + float(np.clip(2.2 * obj.dims.max(), 1.2, 4.0)) * away
        eye_h = float(np.clip(frame.heights(obj.center[None])[0] + 0.5, 0.8, 1.9))
        eye = eye + (eye_h - frame.heights(eye[None])[0]) * frame.up
        set_view(client, eye, obj.center)

    @nav_mode.on_update
    def _(_) -> None:
        for client in server.get_clients().values():
            pos = np.asarray(client.camera.position, dtype=np.float64)
            ahead = unit(np.asarray(client.camera.look_at, dtype=np.float64) - pos)
            if nav_mode.value == NAV_ORBIT and selected.value in by_name:
                client.camera.look_at = tuple(float(v) for v in by_name[selected.value].center)
            elif nav_mode.value == NAV_ORBIT:
                client.camera.look_at = tuple(float(v) for v in pos + ORBIT_DISTANCE_M * ahead)
            else:
                client.camera.look_at = tuple(float(v) for v in pos + turn_radius.value * ahead)

    @turn_radius.on_update
    def _(_) -> None:
        if nav_mode.value == NAV_WALK:
            for client in server.get_clients().values():
                recenter_pivot(client)

    @eye_level.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            return
        pos = np.asarray(client.camera.position, dtype=np.float64)
        ahead = np.asarray(client.camera.look_at, dtype=np.float64) - pos
        eye = pos + (eye_height - frame.heights(pos[None])[0]) * frame.up
        if nav_mode.value == NAV_ORBIT and selected.value in by_name:
            set_view(client, eye, by_name[selected.value].center)
            return
        level = ahead - (ahead @ frame.up) * frame.up
        if np.linalg.norm(level) < 1e-6:
            level = frame.e1.copy()
        set_view(client, eye, eye + unit(level))

    @reset_view.on_click
    def _(event: viser.GuiEvent) -> None:
        if event.client is None:
            return
        if nav_mode.value != NAV_WALK:
            nav_mode.value = NAV_WALK
        set_view(event.client, start_eye, start_eye + start_dir)

    @display.on_update
    def _(_) -> None:
        show_mode(display.value)

    for handle in [show_boxes, show_labels, only_near, radius, *class_on.values()]:
        handle.on_update(lambda _: (refresh_visibility(), refresh_nearby()))

    @selected.on_update
    def _(_) -> None:
        select(selected.value)

    @fly.on_click
    def _(event: viser.GuiEvent) -> None:
        if event.client is not None:
            fly_to(event.client, selected.value)

    @server.scene.on_click()
    def _(event: viser.SceneClickEvent) -> None:
        origin = np.asarray(event.ray_origin, dtype=np.float64)
        direction = np.asarray(event.ray_direction, dtype=np.float64)
        hits = [(t, o.name) for o in objects if visible(o) and (t := o.ray_hit(origin, direction)) is not None]
        if hits:
            select(min(hits)[1])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            camera_pos["value"] = np.asarray(cam.position, dtype=np.float64)
            if nav_mode.value == NAV_WALK:
                keep_in_room(client)
                recenter_pivot(client)
            refresh_nearby()
            if only_near.value:
                refresh_visibility()

        if args.select:
            select(args.select)
            fly_to(client, args.select)

    show_mode(args.display)
    refresh_nearby()
    host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"Viewer ready: http://{host}:{server.get_port()}", flush=True)
    server.sleep_forever()


if __name__ == "__main__":
    main()
