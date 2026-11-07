r"""Native Gaussian splat viewer. Works on any 3DGS .ply and finds its floor automatically.

Run:      .venv\Scripts\python view_splat.py [scan.ply]
Controls: left-drag look, WASD move, Q/E down/up, Shift fast, wheel speed, F floor highlight,
          R reset, Esc quit
Top view: T toggle, WASD pan, wheel zoom, [ and ] lower or raise the roof cut.
          Floor is yellow, everything else red. F switches both colours off.
Walking:  the sprite walks the path from launch, seen from its eyes. Space pause, V sprite view
          on or off, R restart. Press T to watch the sprite from above.
3D view:  a live 2D map sits in the top-right corner: yellow free, red blocked, purple path,
          green start, cyan end, blue sprite, white arrow your view. M shows or hides it.
Boxes:    O shows or hides boxes and panels from the semantic layer. By default only objects whose
          Excel asset matches their type are shown. --all-boxes shows every object.
Nearby:   going close to an asset fills its box with its condition colour, and the nearest one's
          full data appears. This follows the sprite while it walks, or your camera otherwise.
Arrival:  the sprite walks to the faulty device by default, turns to face it, and shows its data.
Any view: G floor grid, B red cells that contain an object, H raise them into 3D blocks (not in top view),
          N cell numbers, P purple path
          Under each free cell's number is the widest square sprite that fits there, in boxes.
          Numbers run from 1 at the top-left of the top view, row by row. Zoom in to read them.
Options:  --view=x,y,z,yaw,pitch   start camera (viewer coordinates, degrees)
          --shot=out.png           save one frame and exit
          --top                    start in the top view
          --zoom=2.5               start top view zoom
          --from 3093 --to 911     walk between two cell numbers (--start and --end also work)
          --all-boxes              show all semantic layer boxes, not only matched assets
          --no-semantic            skip the semantic layer: no boxes, panels or fault target
          --object scan.ply        view one extracted object, with the camera outside it
          --flip                   turn the scan over, if it opens upside down
          --no-sprite              no sprite or path, just the grid and numbers on a free camera
          --capture=stops.json     render clean frames from a list of camera stops, then exit
          --walk=0.5               with --shot, how far along the path the sprite is, 0 to 1
          --sprite=1               sprite width in boxes, odd number
Output:   <scan>_cells.csv         cell number, row, col, scene x and z, free, max sprite width
          <scan>_path.csv          path cells in walking order
"""
import argparse
import csv
import heapq
import json
import os
import sys
import textwrap
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import glfw
import moderngl
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLY = Path(os.environ.get("SCAN_PLY", ROOT / "data" / "Equipment Room.ply"))
SH_C0 = 0.28209479177387814  # converts f_dc spherical harmonics to RGB
MIN_OPACITY = 0.02           # drop near-invisible splats
LEVEL_MIN_DEGREES = 3.0      # scans tilted by less than this are left alone
MIN_FLAT_SPLATS = 500        # fewer flat splats than this cannot show the room's direction
TEX_WIDTH = 4096
FOV_Y = np.radians(60)
START_YAW = 90  # degrees, faces the pipe wall

# Scan coordinates to y-up viewer coordinates. Scans differ, so fit_floor picks one.
Y_DOWN = np.diag([1.0, -1.0, -1.0])
Y_UP = np.eye(3)
CUT_FRACTION = 0.55  # top view hides everything above this share of the wall height
GRID_CELLS = 60      # floor grid size, cells per side
LABEL_MIN_ZOOM = 2.0 # top view hides cell numbers below this zoom, they would be unreadable
BLOCK_SHADES = (1.0, 0.3, 0.25), (1.0, 0.12, 0.1), (0.8, 0.06, 0.05)  # tops, x walls, z walls
BLOCK_ALPHA = 0.4    # 0 = invisible, 1 = solid
WALK_SPEED = 0.7     # scan units per second
LOOK_AHEAD = 0.6     # the sprite looks at the path this far ahead, in scan units
EYE_HEIGHT = 0.9     # eye height above the floor, in scan units. Scale is not calibrated.
SPRITE_HEIGHT = 1.05 # sprite height above the floor, in scan units
NEAR_RADIUS = 1.2    # objects closer than this to the sprite or camera light up, in scan units
NEAR_STICKY = 0.15   # the panel only switches object when another is this much closer
NEAR_LIMIT = 6       # most objects lit at once
CONDITION_COLORS = {"attention_required": (1.0, 0.12, 0.1), "degraded": (1.0, 0.55, 0.1),
                    "watch": (1.0, 0.9, 0.2), "healthy": (0.25, 0.95, 0.45)}
NO_ASSET_COLOR = 0.8, 0.8, 0.85
TURN_SECONDS = 1.2   # time the sprite takes to turn to the target on arrival
MAP_CELL_PIXELS = 4  # 2D map size on screen, in pixels per cell
MIN_CELL_SPLATS = 4  # a cell turns red with at least this many solid splats above the floor
CLEARANCE = 2.0      # objects start at most this many floor bands above the floor
FLOOR_EDGE_SHARE = 0.2   # the floor layer ends where splat density falls below this share of its peak
EDGE_MARGIN = 2.4    # objects start this many times the floor layer's height above the floor
OBJECT_TINT = 0.8    # top view: 0 = natural colour, 1 = solid red
FLOOR_TINT = 0.65    # 0 = natural colour, 1 = solid yellow

VERTEX_SHADER = f"""#version 330
in vec2 corner;
in uint id;
uniform sampler2D data;
uniform mat4 view;
uniform vec2 res;
uniform float focal;
uniform float floor_tint;
uniform float object_tint;
uniform float clearance;
uniform vec3 floor_plane;
uniform float up_sign;
uniform float cut;
out vec2 q;
out vec4 c;

vec4 texel(int i) {{ return texelFetch(data, ivec2(i % {TEX_WIDTH}, i / {TEX_WIDTH}), 0); }}

void main() {{
    int i = 4 * int(id);
    vec4 p = texel(i);         // xyz, alpha
    float h = up_sign * (floor_plane.x * p.x + floor_plane.y * p.z + floor_plane.z - p.y);
    if (h > cut) {{ gl_Position = vec4(0., 0., 2., 1.); return; }}  // above the roof cut
    vec4 t1 = texel(i + 1);    // sigma, floor weight
    vec3 s = t1.xyz;
    vec3 rgb = texel(i + 2).xyz, yellow = vec3(1., .85, .05), red = vec3(.9, .05, .05);
    if (object_tint > 0.) {{  // top view: split floor and objects by height above the floor
        float object = smoothstep(.5 * clearance, clearance, h);
        rgb = mix(rgb, mix(yellow, red, object), mix(floor_tint, object_tint, object));
    }} else {{
        rgb = mix(rgb, yellow, t1.w * floor_tint);
    }}
    vec4 r = texel(i + 3);     // quaternion wxyz
    float w = r.x, x = r.y, y = r.z, z = r.w;
    mat3 R = mat3(1. - 2.*(y*y + z*z), 2.*(x*y + w*z), 2.*(x*z - w*y),
                  2.*(x*y - w*z), 1. - 2.*(x*x + z*z), 2.*(y*z + w*x),
                  2.*(x*z + w*y), 2.*(y*z - w*x), 1. - 2.*(x*x + y*y));

    vec3 v = (view * vec4(p.xyz, 1.)).xyz;
    if (v.z > -.03) {{ gl_Position = vec4(0., 0., 2., 1.); return; }}

    // Project the 3D ellipsoid to a 2D screen ellipse: J * Sigma * J^T.
    mat3 M = mat3(view) * R;
    M[0] *= s.x; M[1] *= s.y; M[2] *= s.z;
    float k = -focal / v.z, l = focal / (v.z * v.z);
    vec3 ju = vec3(k, 0., l * v.x), jv = vec3(0., k, l * v.y);
    vec3 u = vec3(dot(ju, M[0]), dot(ju, M[1]), dot(ju, M[2]));
    vec3 e = vec3(dot(jv, M[0]), dot(jv, M[1]), dot(jv, M[2]));
    float a = dot(u, u), b = dot(u, e), d = dot(e, e);
    float mid = .5 * (a + d), rad = length(vec2(.5 * (a - d), b));
    float l1 = mid + rad, l2 = max(mid - rad, 0.);
    vec2 dir = abs(b) < 1e-6 ? (a >= d ? vec2(1., 0.) : vec2(0., 1.)) : normalize(vec2(b, l1 - a));
    vec2 major = min(sqrt(2. * l1), 1024.) * dir;
    vec2 minor = min(sqrt(2. * l2), 1024.) * vec2(dir.y, -dir.x);

    vec2 px = k * v.xy + corner.x * major + corner.y * minor;
    gl_Position = vec4(px / (.5 * res), 0., 1.);
    q = corner;
    c = vec4(rgb, p.a);
}}
"""

MAP_VERTEX_SHADER = """#version 330
in vec2 corner;
uniform vec4 rect;  // left, bottom, width, height in clip space
out vec2 uv;
void main() {
    uv = vec2(corner.x, 1. - corner.y);  // the first image row is the top of the map
    gl_Position = vec4(rect.xy + corner * rect.zw, 0., 1.);
}
"""

MAP_FRAGMENT_SHADER = """#version 330
in vec2 uv;
uniform sampler2D image;
out vec4 o;
void main() { o = texture(image, uv); }
"""

MARK_VERTEX_SHADER = """#version 330
in vec2 position;
void main() { gl_Position = vec4(position, 0., 1.); }
"""

MARK_FRAGMENT_SHADER = """#version 330
uniform vec4 color;
out vec4 o;
void main() { o = color; }
"""

GRID_VERTEX_SHADER = """#version 330
in vec3 position;
uniform mat4 view;
uniform vec2 res;
uniform float focal;
uniform vec3 offset;
void main() {
    vec3 v = (view * vec4(position + offset, 1.)).xyz;
    float w = -v.z;  // homogeneous output, so lines passing behind the camera clip correctly
    gl_Position = vec4(focal * v.xy / (.5 * res), w - .06, w);
}
"""

GRID_FRAGMENT_SHADER = """#version 330
uniform vec4 color;
out vec4 o;
void main() { o = vec4(color.rgb * color.a, color.a); }
"""

FRAGMENT_SHADER = """#version 330
in vec2 q;
in vec4 c;
out vec4 o;
void main() {
    float a = -dot(q, q);
    if (a < -4.) discard;
    a = exp(a) * c.a;
    o = vec4(c.rgb * a, a);
}
"""


PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "char": "i1", "ushort": "<u2",
             "short": "<i2", "uint": "<u4", "int": "<i4"}
REQUIRED = ("x y z scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3 "
            "f_dc_0 f_dc_1 f_dc_2 opacity").split()


def grid_lines(plane, center_x, center_z, size, cells):
    """Scene-space line endpoints for a square grid lying on the floor plane."""
    a, b, c = plane
    edge = np.linspace(-size / 2, size / 2, cells + 1)
    x0, x1, z0, z1 = center_x - size / 2, center_x + size / 2, center_z - size / 2, center_z + size / 2
    ends = []
    for offset in edge:
        x, z = center_x + offset, center_z + offset
        ends += [(x, z0), (x, z1), (x0, z), (x1, z)]
    ends = np.array(ends, dtype="f8")
    return np.column_stack([ends[:, 0], a * ends[:, 0] + b * ends[:, 1] + c, ends[:, 1]]).astype("f4")


def blocked_cells(x, z, center_x, center_z, size, cells):
    """Boolean [cells, cells] grid: True where at least MIN_CELL_SPLATS points land in a cell."""
    ix = np.floor((x - center_x + size / 2) / size * cells).astype(int)
    iz = np.floor((z - center_z + size / 2) / size * cells).astype(int)
    inside = (ix >= 0) & (ix < cells) & (iz >= 0) & (iz < cells)
    counts = np.bincount(ix[inside] * cells + iz[inside], minlength=cells * cells)
    return counts.reshape(cells, cells) >= MIN_CELL_SPLATS


def clean_cells(blocked):
    """Remove noise: fill yellow cells between reds, then keep only the largest free area."""
    result = blocked.copy()
    while True:
        between = np.zeros_like(result)
        between[1:-1, :] = result[:-2, :] & result[2:, :]
        between[:, 1:-1] |= result[:, :-2] & result[:, 2:]
        if not (between & ~result).any():
            break
        result |= between

    unvisited = ~result
    largest = np.zeros_like(result)
    while unvisited.any():
        start = tuple(int(i) for i in np.argwhere(unvisited)[0])
        region, queue = {start}, deque([start])
        unvisited[start] = False
        while queue:
            x, z = queue.popleft()
            for cell in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
                if 0 <= cell[0] < result.shape[0] and 0 <= cell[1] < result.shape[1] and unvisited[cell]:
                    unvisited[cell] = False
                    region.add(cell)
                    queue.append(cell)
        if len(region) > largest.sum():
            largest[:] = False
            largest[tuple(np.array(list(region)).T)] = True
    return ~largest


DIGITS = ("111101101101111", "010110010010111", "111001111100111", "111001111001111",
          "101101111001001", "111100111001111", "111100111101111", "111001010010010",
          "111101111101111", "111101111001111")  # 3 x 5 pixel font


def cell_grid_index(row, col, cells, z_points_up_screen):
    """Grid indices (ix, iz) of a cell, where row 0 is the top row of the top view."""
    return col, cells - 1 - row if z_points_up_screen else row


def max_sprite_sizes(blocked):
    """Widest odd square sprite, in boxes, that fits centred on each cell. 0 on blocked cells."""
    sizes = np.zeros(blocked.shape, dtype=int)
    fits, radius = ~blocked, 0
    while fits.any():
        sizes[fits] = 2 * radius + 1
        padded = np.pad(fits, 1)  # the grid edge counts as blocked
        fits = fits.copy()
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                fits &= padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dz:padded.shape[1] - 1 + dz]
        radius += 1
    return sizes


def neighbours(cell, usable):
    """The up to 4 usable cells sharing an edge with cell."""
    x, z = cell
    for other in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
        if 0 <= other[0] < usable.shape[0] and 0 <= other[1] < usable.shape[1] and usable[other]:
            yield other


def dijkstra_path(sizes, sprite, start, end):
    """Cheapest route from start to end on the cells this sprite fits on, or None.

    Each step costs 1, plus more the tighter the squeeze, so a wider sprite avoids narrow gaps.
    """
    usable = sizes >= sprite
    best, parent, heap, order = {start: 0.0}, {start: None}, [(0.0, 0, start)], 0
    while heap:
        cost, _, cell = heapq.heappop(heap)
        if cell == end:
            path = [end]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        if cost > best[cell]:
            continue
        for other in neighbours(cell, usable):
            total = cost + 1 + sprite / min(sizes[cell], sizes[other])
            if total < best.get(other, np.inf):
                best[other], parent[other] = total, cell
                order += 1
                heapq.heappush(heap, (total, order, other))
    return None


def shortest_steps(usable, start, end):
    """Fewest steps between two cells, ignoring weights, for comparison."""
    steps, queue = {start: 0}, deque([start])
    while queue:
        cell = queue.popleft()
        if cell == end:
            return steps[cell]
        for other in neighbours(cell, usable):
            if other not in steps:
                steps[other] = steps[cell] + 1
                queue.append(other)
    return None


def smooth_route(points, spacing=0.03, window=0.45):
    """Resample a corner-heavy path evenly, then round its corners. Ends stay in place."""
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    even = np.arange(0, lengths[-1] + spacing, spacing).clip(max=lengths[-1])
    line = np.column_stack([np.interp(even, lengths, points[:, i]) for i in range(3)])
    half = max(int(window / spacing / 2), 1)
    # Point-mirrored ends average back to the exact end points.
    padded = np.pad(line, ((half, half), (0, 0)), mode="reflect", reflect_type="odd")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    smooth = np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)])
    return smooth, np.r_[0, np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]


def route_point(route, lengths, distance):
    """Position along a route at a distance from its start, clamped to the ends."""
    return np.array([np.interp(distance, lengths, route[:, i]) for i in range(3)])


def sprite_triangles(radius, height, up_sign, sides=16):
    """A simple person shape at the origin. Returns body triangles and head triangles."""
    def prism(r, low, high):
        angles = np.linspace(0, 2 * np.pi, sides + 1)
        rim = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        triangles = []
        for (x1, z1), (x2, z2) in zip(rim[:-1], rim[1:]):
            triangles += [(x1, low, z1), (x2, low, z2), (x1, high, z1),
                          (x2, low, z2), (x2, high, z2), (x1, high, z1),
                          (0, high, 0), (x1, high, z1), (x2, high, z2)]
        points = np.array(triangles, dtype="f4")
        points[:, 1] *= -up_sign  # scan y may point down
        return points

    return prism(radius, 0, 0.74 * height), prism(0.62 * radius, 0.78 * height, height)


def map_image(blocked, path_mask, start, end, scale, z_points_up_screen):
    """RGBA image of the grid, laid out like the top view, with dark lines between cells."""
    colors = np.full(blocked.shape + (3,), (235, 205, 70), dtype=float)
    colors[blocked] = 190, 40, 35
    colors[path_mask] = 150, 40, 235
    if start is not None:
        colors[start] = 60, 220, 90
        colors[end] = 40, 220, 235
    image = colors.transpose(1, 0, 2)  # [iz, ix], so rows run along z
    image = image[::-1] if z_points_up_screen else image
    image = image.repeat(scale, axis=0).repeat(scale, axis=1)
    image[::scale, :] *= 0.72
    image[:, ::scale] *= 0.72
    return np.dstack([image, np.full(image.shape[:2], 255.0)]).astype("u1")


def box_lines(low, high):
    """The 12 edges of a box as 24 line endpoints."""
    corners = np.array([[(low, high)[i >> b & 1][b] for b in range(3)] for i in range(8)], dtype="f4")
    pairs = [(i, i | 1 << b) for i in range(8) for b in range(3) if not i >> b & 1]
    return corners[np.array(pairs).ravel()]


def asset_panel(obj, nearby=(), width=430):
    """RGBA image with everything known about an object, for the information display."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default(size)

    white, soft, dim, blue = (255, 255, 255), (225, 225, 225), (165, 165, 165), (140, 210, 255)
    asset = obj["asset"]
    if asset is None:
        lines = [(obj["class"].title(), 21, white), (f"Object: {obj['id']}", 14, dim),
                 ("No Excel asset mapped to this object", 14, soft)]
    else:
        master, state = asset["master"], asset["state"]
        color = tuple(int(c * 255) for c in CONDITION_COLORS.get(state["condition_state"], NO_ASSET_COLOR))
        lines = [(f"{master['tag']}  {master['asset_class'].replace('_', ' ').title()}", 21, white),
                 (f"{state['condition_state'].replace('_', ' ').upper()}  |  Health {state['health_score']}/100", 16, color),
                 (f"Issue: {state['issue_summary']}", 15, soft),
                 (f"Urgency: {state['urgency']}  |  {master['criticality']}", 14, soft),
                 (f"Next check: {state['recommended_next_check']}", 15, blue),
                 ("Latest readings", 14, dim)]
        units = {"process_pressure_bar": ("Pressure", "bar"), "process_flow_pct": ("Flow", "%"),
                 "position_setpoint_pct": ("Position setpoint", "%"), "position_actual_pct": ("Position actual", "%"),
                 "pneumatic_supply_bar": ("Air supply", "bar"), "stiction_index": ("Stiction index", ""),
                 "vibration_mm_s": ("Vibration", "mm/s"), "bearing_temp_c": ("Bearing temp", "C"),
                 "motor_current_a": ("Motor current", "A"), "speed_rpm": ("Speed", "rpm"),
                 "signal_quality_pct": ("Signal quality", "%")}
        readings = [f"{name}: {asset['latest_telemetry'][key]:.1f} {unit}".rstrip()
                    for key, (name, unit) in units.items() if key in asset["latest_telemetry"]]
        for start in range(0, len(readings), 2):
            lines.append(("   " + "   |   ".join(readings[start:start + 2]), 13, soft))
        if asset["recent_events"]:
            lines.append(("Recent events", 14, dim))
            for event in asset["recent_events"][:3]:
                lines.append((f"   {event['event_type']} ({event['severity']}), {event['status']}", 13, soft))
        if asset["recent_work_orders"]:
            work = asset["recent_work_orders"][0]
            lines.append((f"Last work order: {work['action']}, {work['result']}", 13, soft))
        lines.append((f"Last service: {master['last_service_date'][:10]}  |  Link: {master['connectivity_profile']}", 13, soft))
        if asset["guidance"].get("content"):
            lines.append(("Guidance", 14, dim))
            lines += [("   " + row, 13, soft) for row in textwrap.wrap(asset["guidance"]["content"], 60)[:3]]
        lines.append((f"Area: {master['area_name']}  |  Object: {obj['id']}", 13, dim))
    if nearby:
        lines.append((f"Also nearby: {', '.join(nearby)}", 13, blue))
    lines.append(("Synthetic data, mapped procedurally", 12, dim))

    height = 18 + sum(size + 7 for _, size, _ in lines)
    image = Image.new("RGBA", (width, height), (12, 14, 18, 255))
    draw = ImageDraw.Draw(image)
    edge = NO_ASSET_COLOR if asset is None else CONDITION_COLORS.get(asset["state"]["condition_state"], NO_ASSET_COLOR)
    draw.rectangle((0, 0, width - 1, height - 1), outline=tuple(int(c * 255) for c in edge) + (255,), width=2)
    y = 10
    for text, size, color in lines:
        draw.text((14, y), text, font=font(size), fill=color)
        y += size + 7
    return image


def box_lines(low, high):
    """The 12 edges of a box as 24 line endpoints."""
    corners = np.array([[(low, high)[i >> b & 1][b] for b in range(3)] for i in range(8)], dtype="f4")
    pairs = [(i, i | 1 << b) for i in range(8) for b in range(3) if not i >> b & 1]
    return corners[np.array(pairs).ravel()]


def box_faces(low, high):
    """The 6 faces of a box as 36 triangle corners."""
    corners = np.array([[(low, high)[i >> b & 1][b] for b in range(3)] for i in range(8)], dtype="f4")
    quads = (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)
    return corners[np.array([(a, b, c, a, c, d) for a, b, c, d in quads]).ravel()]


def near_objects(point, lows, highs, radius):
    """Indices of boxes within radius of a point on the floor, nearest first, with their distances."""
    if not len(lows):
        return np.zeros(0, dtype=int), np.zeros(0)
    xz = np.array([point[0], point[2]])
    gaps = np.maximum(np.maximum(lows[:, [0, 2]] - xz, xz - highs[:, [0, 2]]), 0)
    distance = np.linalg.norm(gaps, axis=1)
    order = np.argsort(distance)
    order = order[distance[order] <= radius][:NEAR_LIMIT]
    return order, distance[order]


def cell_labels(plane, center_x, center_z, size, cells, z_points_up_screen, sizes=None):
    """Triangles drawing each cell's number, and max sprite width below it, on the floor plane."""
    a, b, c = plane
    step, pixel = size / cells, size / cells * 0.045
    screen_up = 1.0 if z_points_up_screen else -1.0
    left, bottom = [], []
    for row in range(cells):
        for col in range(cells):
            ix, iz = cell_grid_index(row, col, cells, z_points_up_screen)
            lines = [(str(row * cells + col + 1), 2.5 if sizes is None else 5.75)]
            if sizes is not None and sizes[ix, iz]:
                lines.append((str(sizes[ix, iz]), -0.75))
            for text, height in lines:
                x = center_x - size / 2 + (ix + 0.5) * step - (len(text) * 4 - 1) * pixel / 2
                z = center_z - size / 2 + (iz + 0.5) * step + screen_up * height * pixel
                for position, digit in enumerate(text):
                    for index, lit in enumerate(DIGITS[int(digit)]):
                        if lit == "1":
                            left.append(x + (position * 4 + index % 3) * pixel)
                            bottom.append(z - screen_up * (index // 3 + 1) * pixel)
    corners = np.array([(0, 0), (1, 0), (0, 1), (1, 0), (1, 1), (0, 1)]) * pixel
    px = np.array(left)[:, None] + corners[:, 0]
    pz = np.array(bottom)[:, None] + corners[:, 1]
    return np.stack([px, a * px + b * pz + c, pz], axis=2).astype("f4").reshape(-1, 3)


def write_cells(path, sizes, center_x, center_z, size, z_points_up_screen):
    """Save the lookup from cell number to grid and scene position, free flag and sprite width."""
    cells = len(sizes)
    step = size / cells
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["cell", "row", "col", "x", "z", "free", "max_sprite"])
        for row in range(cells):
            for col in range(cells):
                ix, iz = cell_grid_index(row, col, cells, z_points_up_screen)
                x = center_x - size / 2 + (ix + 0.5) * step
                z = center_z - size / 2 + (iz + 0.5) * step
                out.writerow([row * cells + col + 1, row, col, round(float(x), 4), round(float(z), 4),
                              int(sizes[ix, iz] > 0), int(sizes[ix, iz])])


def cell_triangles(blocked, plane, center_x, center_z, size):
    """Two floor-plane triangles for every blocked cell."""
    a, b, c = plane
    step = size / len(blocked)
    corners = np.array([(0, 0), (1, 0), (0, 1), (1, 0), (1, 1), (0, 1)])
    cells = np.argwhere(blocked)[:, None, :] + corners  # shape: cells, 6 corners, (ix, iz)
    x = center_x - size / 2 + cells[:, :, 0] * step
    z = center_z - size / 2 + cells[:, :, 1] * step
    return np.stack([x, a * x + b * z + c, z], axis=2).astype("f4").reshape(-1, 3)


def block_triangles(blocked, plane, center_x, center_z, size, height, up_sign):
    """Raise blocked cells into blocks. Returns triangles and the size of each shaded group.

    Groups are tops, x-facing walls and z-facing walls. Walls between two blocked cells are skipped.
    """
    a, b, c = plane
    step = size / len(blocked)
    x0, z0 = center_x - size / 2, center_z - size / 2
    top = cell_triangles(blocked, plane, center_x, center_z, size)
    top[:, 1] -= up_sign * height

    def walls(x1, z1, x2, z2):
        """One vertical quad per edge, from the floor up to height."""
        corners = [(x1, z1, 0), (x2, z2, 0), (x1, z1, 1), (x2, z2, 0), (x2, z2, 1), (x1, z1, 1)]
        faces = []
        for x, z, raised in corners:
            faces.append(np.stack([x, a * x + b * z + c - up_sign * height * raised, z], axis=1))
        return np.stack(faces, axis=1).reshape(-1, 3).astype("f4")

    padded = np.pad(blocked, 1)
    high_x = np.argwhere(blocked & ~padded[2:, 1:-1])
    low_x = np.argwhere(blocked & ~padded[:-2, 1:-1])
    high_z = np.argwhere(blocked & ~padded[1:-1, 2:])
    low_z = np.argwhere(blocked & ~padded[1:-1, :-2])
    x_edges = np.concatenate([high_x + (1, 0), low_x])
    z_edges = np.concatenate([high_z + (0, 1), low_z])
    x_walls = walls(x0 + x_edges[:, 0] * step, z0 + x_edges[:, 1] * step,
                    x0 + x_edges[:, 0] * step, z0 + (x_edges[:, 1] + 1) * step)
    z_walls = walls(x0 + z_edges[:, 0] * step, z0 + z_edges[:, 1] * step,
                    x0 + (z_edges[:, 0] + 1) * step, z0 + z_edges[:, 1] * step)
    return np.concatenate([top, x_walls, z_walls]), (len(top), len(x_walls), len(z_walls))


def read_ply(path):
    """Read a binary 3DGS .ply into a structured array, using names from its header."""
    with open(path, "rb") as f:
        fields, count = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
            elif parts[0] == "property":
                if parts[1] not in PLY_TYPES:
                    sys.exit(f"Unexpected property type: {line}")
                fields.append((parts[2], PLY_TYPES[parts[1]]))
            elif line.startswith("format") and "binary_little_endian" not in line:
                sys.exit(f"Only binary little-endian .ply is supported: {line}")
        v = np.fromfile(f, dtype=fields, count=count)
    missing = set(REQUIRED) - set(v.dtype.names)
    if missing:
        sys.exit(f"Not a Gaussian splat .ply, missing: {sorted(missing)}")
    return v


def valid_mask(v):
    """True for records that are real Gaussians (rejects NaN or out-of-range values)."""
    with np.errstate(invalid="ignore", over="ignore"):
        pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype("f8")
        log_scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype("f8")
        quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype("f8")
        length = np.sqrt((quat ** 2).sum(1))
        ok = (length > 1e-3) & (length < 1e3)  # many exporters store unnormalised quaternions
        ok &= np.abs(pos).max(1) < 1000
        ok &= np.abs(log_scale).max(1) < 30
    return ok


def surface_normals(v, limit=300000):
    """Directions that flat, solid splats face, with their areas and positions."""
    spread = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype("f8"))
    order = np.sort(spread, axis=1)
    flat = (v["opacity"] > 0) & (order[:, 0] < 0.3 * order[:, 1]) & (order[:, 1] < 0.1)
    index = np.flatnonzero(flat)
    index = index[np.random.default_rng(0).permutation(len(index))[:limit]]
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype("f8")[index]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = quat.T
    rotation = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)
    normals = rotation[np.arange(len(index)), :, spread[index].argmin(axis=1)]
    position = np.stack([v[k].astype("f8") for k in "xyz"], axis=1)[index]
    return normals, order[index, 1] * order[index, 2], position


def find_up(v):
    """The room's vertical line: the direction with the most flat surface area. None if unclear.

    Floors and ceilings share one direction, while walls split across two, so it wins in a room.
    ponytail: tested on two rooms. A long, narrow corridor could favour a wall, and --flip fixes
    only an upside-down result.
    """
    normals, area, _ = surface_normals(v)
    if len(normals) < MIN_FLAT_SPLATS:
        return None
    picks = np.random.default_rng(1).choice(len(normals), 1500, p=area / area.sum())
    scores = [area[np.abs(normals @ normal) > 0.97].sum() for normal in normals[picks]]
    best = normals[picks[int(np.argmax(scores))]]
    near = np.abs(normals @ best) > 0.97
    up = (normals[near] * np.sign(normals[near] @ best)[:, None] * area[near, None]).sum(axis=0)
    return up / np.linalg.norm(up)


def rotation_between(a, b):
    """Smallest rotation matrix taking direction a onto direction b."""
    axis, cosine = np.cross(a, b), float(a @ b)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def rotate_splats(v, rotation):
    """Return a rotated copy of the splats, turning positions and orientations together."""
    v = v.copy()
    position = np.stack([v[k].astype("f8") for k in "xyz"], axis=1) @ rotation.T
    v["x"], v["y"], v["z"] = position.T
    # Quaternion of the rotation, w first.
    m = rotation
    w0 = np.sqrt(max(1 + m[0, 0] + m[1, 1] + m[2, 2], 1e-12)) / 2
    x0, y0, z0 = (m[2, 1] - m[1, 2]) / (4 * w0), (m[0, 2] - m[2, 0]) / (4 * w0), (m[1, 0] - m[0, 1]) / (4 * w0)
    w1, x1, y1, z1 = [v[f"rot_{i}"].astype("f8") for i in range(4)]
    v["rot_0"] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    v["rot_1"] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    v["rot_2"] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    v["rot_3"] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    return v


def level_scan(v, flip=False, report=True):
    """Rotate a tilted scan so its vertical line is the Y axis. Nearly level scans stay untouched."""
    up = find_up(v)
    if up is None:
        if report:
            print("Could not find the room's vertical direction, so the scan is left as it is.")
        return v
    target = np.array([0.0, 1.0 if up[1] >= 0 else -1.0, 0.0])
    tilt = float(np.degrees(np.arccos(np.clip(up @ target, -1, 1))))
    if flip:
        v = rotate_splats(v, np.diag([1.0, -1.0, -1.0]))
        up = up * (1, -1, -1)
        target = -target
    if tilt < LEVEL_MIN_DEGREES:
        return v
    if report:
        print(f"Levelled the scan by {tilt:.1f} degrees")
    v = rotate_splats(v, rotation_between(up, target))
    # Surface directions are approximate, so finish on the floor plane itself.
    (a, b, _), _, _ = fit_floor(v, report=False)
    floor_up = np.array([-a, 1.0, -b]) / np.hypot(np.hypot(a, b), 1.0)
    return rotate_splats(v, rotation_between(floor_up, np.array([0.0, 1.0, 0.0])))


def fit_floor(v, report=True):
    """Fit y = a*x + b*z + c to the floor. Returns (a, b, c), highlight band, viewer rotation.

    The floor is the sharpest layer of opaque splats near the top or bottom of the scan.
    ponytail: assumes y is the vertical axis and the floor is near level. Add an --up option
    or a search over plane normals if a scan arrives lying on its side.
    """
    solid = v[v["opacity"] >= np.log(0.3 / 0.7)]
    x, y, z = [solid[k].astype("f8") for k in "xyz"]
    x0, x1 = np.percentile(x, [5, 95])
    z0, z1 = np.percentile(z, [5, 95])
    keep = (x > x0) & (x < x1) & (z > z0) & (z < z1)  # stay inside the room, away from the outer shell
    x, y, z = x[keep], y[keep], z[keep]

    low, high = np.percentile(y, [1, 99])
    width = (high - low) / 150
    counts, edges = np.histogram(y, bins=150, range=(low, high))
    counts[45:105] = 0  # only the outer 30% at each end can be floor
    peak = edges[counts.argmax()] + width / 2
    y_down = peak > (low + high) / 2

    near = np.abs(y - peak) < 5 * width
    points, heights = np.column_stack([x[near], z[near], np.ones(near.sum())]), y[near]
    for _ in range(8):
        plane = np.linalg.lstsq(points, heights, rcond=None)[0]
        error = np.abs(heights - points @ plane)
        inliers = error < max(2 * np.median(error), 1e-6)
        if inliers.sum() < 3:  # too few points left to refine, so keep this fit
            inliers = np.ones(len(error), dtype=bool)
            break
        points, heights = points[inliers], heights[inliers]
    band = max(9 * np.percentile(error[inliers], 90), width / 2)
    tilt = np.degrees(np.arctan(np.hypot(plane[0], plane[1])))
    if report:
        print(f"Floor: y = {plane[0]:.4f}x {plane[1]:+.4f}z {plane[2]:+.4f} | tilt {tilt:.2f} deg | "
              f"band {band:.3f} | {len(heights):,} splats | scan is y-{'down' if y_down else 'up'}")
    if report and (tilt > 10 or len(heights) < 500):
        print("Warning: this floor fit looks unreliable. Check the yellow area.")
    return tuple(plane), band, Y_DOWN if y_down else Y_UP


def object_clearance(heights, band):
    """Height above the floor where objects start, from where this scan's floor layer ends.

    heights are solid splats, measured from the floor. The floor is a sharp peak of them, so its top
    edge is where the density falls well below that peak.
    """
    step = band / 10
    counts, edges = np.histogram(heights, bins=np.arange(-20 * step, 60 * step, step))
    peak = int(counts.argmax())
    below = np.flatnonzero(counts[peak:] < FLOOR_EDGE_SHARE * counts[peak])
    if not len(below):
        return CLEARANCE * band
    return min(CLEARANCE * band, EDGE_MARGIN * max(float(edges[peak + below[0]]), step))


def floor_weight(v, plane, band):
    """1 on the floor plane, fading to 0 at band away from it."""
    a, b, c = plane
    return np.clip(1 - np.abs(v["y"] - (a * v["x"] + b * v["z"] + c)) / band, 0, 1)


def to_texels(v, plane, band):
    """Pack each splat into 4 RGBA float texels: xyz+alpha, sigma+floor weight, rgb, wxyz."""
    n = len(v)
    d = np.zeros((n, 16), dtype="f4")
    d[:, 0], d[:, 1], d[:, 2] = v["x"], v["y"], v["z"]
    d[:, 3] = 1 / (1 + np.exp(-v["opacity"]))  # stored as logit
    d[:, 4:7] = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1))  # stored as log
    d[:, 7] = floor_weight(v, plane, band)
    d[:, 8:11] = np.clip(0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1), 0, 1)
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype("f8")
    d[:, 12:16] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    height = int(np.ceil(4 * n / TEX_WIDTH))
    padded = np.zeros((height, TEX_WIDTH, 4), dtype="f4")
    padded.reshape(-1, 4)[: 4 * n] = d.reshape(-1, 4)
    return padded, height


class Camera:
    def __init__(self, position, to_viewer, yaw=0.0, pitch=0.0):
        self.to_viewer = to_viewer
        self.home = (np.array(position, dtype="f8"), yaw, pitch)
        self.reset()

    def reset(self):
        position, self.yaw, self.pitch = self.home
        self.position = position.copy()

    def axes(self):
        cp, sp, cy, sy = np.cos(self.pitch), np.sin(self.pitch), np.cos(self.yaw), np.sin(self.yaw)
        forward = np.array([cp * sy, sp, -cp * cy])
        right = np.array([cy, 0.0, sy])
        return forward, right, np.cross(right, forward)

    def view(self):
        """Scene-to-camera matrix."""
        forward, right, up = self.axes()
        m = np.eye(4)
        m[:3, :3] = np.stack([right, up, -forward]) @ self.to_viewer
        m[:3, 3] = -right @ self.position, -up @ self.position, forward @ self.position
        return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", nargs="?", default=PLY, type=Path)
    parser.add_argument("--view")
    parser.add_argument("--shot")
    parser.add_argument("--top", action="store_true")
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--from", "--start", dest="start", type=int, default=1488)
    parser.add_argument("--to", "--end", dest="end", type=int)
    parser.add_argument("--all-boxes", action="store_true")
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--walk", type=float, default=0.0)
    parser.add_argument("--no-sprite", action="store_true")
    parser.add_argument("--object", type=Path)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--sprite", type=int, default=1)
    args = parser.parse_args()
    if args.object:
        args.ply, args.no_sprite = args.object, True
    if args.capture:
        args.no_sprite = True
    layer_file = ROOT / "extraction" / args.ply.stem / "semantic_layer.json"
    use_layer = layer_file.exists() and not args.no_semantic
    layer_objects = json.loads(layer_file.read_text())["objects"] if use_layer else []
    fault = next((o for o in layer_objects if o["is_fault_target"]), None)
    if args.no_sprite:
        args.start = args.end = None
    elif args.end is None:
        args.end = fault["approach_cell"] if fault else 3093
    focus = fault if fault and args.end == fault["approach_cell"] else None

    v = read_ply(args.ply)
    ok = valid_mask(v)
    print(f"{len(v):,} records in file, {ok.sum():,} valid ({ok.mean():.1%})")
    if ok.mean() < 0.99:
        print("Warning: this file has invalid records and may be damaged.")
    v = v[ok]
    v = v[v["opacity"] >= np.log(MIN_OPACITY / (1 - MIN_OPACITY))]  # opacity is a logit
    v = level_scan(v, args.flip)
    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype("f4")
    low, mid, high = np.percentile(pos, [1, 50, 99], axis=0)
    print(f"{len(v):,} splats shown | p1 {low.round(2)} | median {mid.round(2)} | p99 {high.round(2)}")

    plane, band, to_viewer = fit_floor(v, report=not args.object)
    if args.object:
        to_viewer = Y_DOWN  # extracted objects keep the coordinates of the scan they came from
    if args.view:
        x, y, z, yaw, pitch = map(float, args.view.split(","))
        cam = Camera([x, y, z], to_viewer, np.radians(yaw), np.radians(pitch))
    else:
        cam = Camera(to_viewer @ mid, to_viewer, yaw=np.radians(START_YAW))  # outside the room big background splats block the view
    speed = float((high - low).max()) / 6

    if not glfw.init():
        sys.exit("Could not start GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, not args.shot)
    window = glfw.create_window(1280, 720, args.ply.stem, None, None)
    if not window:
        sys.exit("Could not open an OpenGL 3.3 window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    print("GPU:", ctx.info["GL_RENDERER"])
    ctx.enable(moderngl.BLEND)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA

    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    texels, height = to_texels(v, plane, band)
    texture = ctx.texture((TEX_WIDTH, height), 4, texels.tobytes(), dtype="f4")
    texture.filter = moderngl.NEAREST, moderngl.NEAREST
    texture.use(0)
    prog["data"] = 0
    quad = ctx.buffer(np.array([-2, -2, 2, -2, -2, 2, 2, 2], dtype="f4").tobytes())

    def sort_far_to_near(axis):
        order = np.argsort(pos @ axis.astype("f4")).astype("u4")
        on_floor = is_floor[order]
        return order[on_floor].tobytes(), order[~on_floor].tobytes()

    def write_order(parts):
        floor_ids.write(parts[0])
        object_ids.write(parts[1])

    show_floor = [True]
    show_grid = [True]
    show_cells = [True]
    show_labels = [True]
    show_blocks = [True]
    show_path = [True]
    show_map = [True]
    show_boxes = [True]
    moved = [True]

    # Top view: a distant, narrow camera, so walls stay nearly vertical like a floor plan.
    up_sign = 1.0 if to_viewer is Y_DOWN else -1.0
    a, b, c = plane
    heights = up_sign * (a * pos[:, 0] + b * pos[:, 2] + c - pos[:, 1])
    wall_height = float(np.percentile(heights, 99))

    # Splats have no depth, so layer instead: floor, then grid, then everything above the floor.
    # ponytail: assumes the camera is above the floor. Going below it shows the grid on top.
    clearance = object_clearance(heights[v["opacity"] >= np.log(0.3 / 0.7)], band)
    print(f"Objects start {clearance:.3f} above the floor")
    is_floor = heights <= clearance
    floor_ids = ctx.buffer(reserve=max(int(is_floor.sum()), 1) * 4, dynamic=True)
    object_ids = ctx.buffer(reserve=max(int((~is_floor).sum()), 1) * 4, dynamic=True)
    floor_vao = ctx.vertex_array(prog, [(quad, "2f", "corner"), (floor_ids, "1u /i", "id")])
    object_vao = ctx.vertex_array(prog, [(quad, "2f", "corner"), (object_ids, "1u /i", "id")])
    span = float((high - low)[[0, 2]].max())
    altitude = 6 * span
    overhead = to_viewer @ ((low + high) / 2) + np.array([0, altitude, 0])
    grid_prog = ctx.program(vertex_shader=GRID_VERTEX_SHADER, fragment_shader=GRID_FRAGMENT_SHADER)
    grid_x, grid_z = (low[0] + high[0]) / 2, (low[2] + high[2]) / 2
    grid = grid_lines(plane, grid_x, grid_z, span, GRID_CELLS)
    grid_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(grid.tobytes()), "3f", "position")])
    print(f"Grid: {GRID_CELLS} x {GRID_CELLS}, each cell {span / GRID_CELLS:.3f} scan units wide")
    solid = v["opacity"] >= np.log(0.3 / 0.7)
    in_object = solid & (heights > clearance) & (heights < CUT_FRACTION * wall_height)
    blocked = blocked_cells(pos[in_object, 0], pos[in_object, 2], grid_x, grid_z, span, GRID_CELLS)
    raw_count = int(blocked.sum())
    blocked = clean_cells(blocked)
    cells = cell_triangles(blocked, plane, grid_x, grid_z, span)
    cell_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(cells.tobytes() or bytes(36)), "3f", "position")])
    blocks, block_groups = block_triangles(blocked, plane, grid_x, grid_z, span,
                                           CUT_FRACTION * wall_height, up_sign)
    block_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(blocks.tobytes() or bytes(36)), "3f", "position")])
    z_up_screen = to_viewer is Y_DOWN  # the top view's "up" is +z for y-down scans, -z for y-up
    sizes = max_sprite_sizes(blocked)
    labels = cell_labels(plane, grid_x, grid_z, span, GRID_CELLS, z_up_screen, sizes)
    label_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(labels.tobytes()), "3f", "position")])
    cells_csv = args.ply.with_name(args.ply.stem + "_cells.csv")
    if not args.object:
        write_cells(cells_csv, sizes, grid_x, grid_z, span, z_up_screen)
        print(f"Cell table: {cells_csv.name}")

    def to_cell(number):
        row, col = (number - 1) // GRID_CELLS, (number - 1) % GRID_CELLS
        return cell_grid_index(row, col, GRID_CELLS, z_up_screen)

    def to_number(cell):
        row = GRID_CELLS - 1 - cell[1] if z_up_screen else cell[1]
        return row * GRID_CELLS + cell[0] + 1

    path_mask = np.zeros(blocked.shape, dtype=bool)
    route, route_lengths = None, None
    if args.sprite % 2 == 0:
        sys.exit("--sprite must be odd, so the sprite stays centred on a cell.")
    for number in (args.start, args.end):
        if number is not None and not 1 <= number <= GRID_CELLS * GRID_CELLS:
            sys.exit(f"Cell {number} does not exist. Cells run from 1 to {GRID_CELLS * GRID_CELLS}.")
    start, end = (None, None) if args.no_sprite else (to_cell(args.start), to_cell(args.end))
    if start is None:
        pass  # no route requested
    elif sizes[start] < args.sprite or sizes[end] < args.sprite:
        print(f"No path: a {args.sprite}-box sprite needs both cells to fit it. Cell {args.start} fits "
              f"{sizes[start]} and cell {args.end} fits {sizes[end]}, where 0 means blocked.")
    else:
        path = dijkstra_path(sizes, args.sprite, start, end)
        if path is None:
            print(f"No path: a {args.sprite}-box sprite cannot get from {args.start} to {args.end}.")
        else:
            fewest = shortest_steps(sizes >= args.sprite, start, end)
            print(f"Dijkstra: sprite {args.sprite} box wide | path {len(path) - 1} steps, "
                  f"fewest possible {fewest}")
            path_mask[tuple(np.array(path).T)] = True
            steps = np.array(path, dtype="f8")
            x = grid_x - span / 2 + (steps[:, 0] + 0.5) * span / GRID_CELLS
            z = grid_z - span / 2 + (steps[:, 1] + 0.5) * span / GRID_CELLS
            route, route_lengths = smooth_route(np.column_stack([x, a * x + b * z + c, z]))
            cell_of = np.floor((route[:, [0, 2]] - (grid_x - span / 2, grid_z - span / 2))
                               / (span / GRID_CELLS)).astype(int)
            off_floor = int((sizes[cell_of[:, 0], cell_of[:, 1]] < args.sprite).sum())
            print(f"Walk: {route_lengths[-1]:.2f} scan units, {route_lengths[-1] / WALK_SPEED:.1f} s, "
                  f"{off_floor} of {len(route)} route points outside free cells")
            path_csv = args.ply.with_name(args.ply.stem + "_path.csv")
            with open(path_csv, "w", newline="") as f:
                out = csv.writer(f)
                out.writerow(["step", "cell", "x", "z"])
                for step, cell in enumerate(path):
                    x = grid_x - span / 2 + (cell[0] + 0.5) * span / GRID_CELLS
                    z = grid_z - span / 2 + (cell[1] + 0.5) * span / GRID_CELLS
                    out.writerow([step, to_number(cell), round(float(x), 4), round(float(z), 4)])
            print(f"Path table: {path_csv.name}")
    path_cells = cell_triangles(path_mask, plane, grid_x, grid_z, span)
    path_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(path_cells.tobytes() or bytes(36)), "3f", "position")])
    body, head = sprite_triangles(max(args.sprite, 1) * span / GRID_CELLS * 0.45, SPRITE_HEIGHT, up_sign)
    body_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(body.tobytes()), "3f", "position")])
    head_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(head.tobytes()), "3f", "position")])
    walk = {"pov": route is not None, "paused": False, "s": 0.0, "turn": 0.0}

    shown = [o for o in layer_objects if args.all_boxes or o["mapping_method"] == "class_match"]
    box_lows = np.array([o["box"]["min"] for o in shown], dtype="f8").reshape(-1, 3)
    box_highs = np.array([o["box"]["max"] for o in shown], dtype="f8").reshape(-1, 3)
    edge_points = np.concatenate([box_lines(o["box"]["min"], o["box"]["max"]) for o in shown]) if shown else np.zeros((1, 3), "f4")
    face_points = np.concatenate([box_faces(o["box"]["min"], o["box"]["max"]) for o in shown]) if shown else np.zeros((1, 3), "f4")
    box_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(edge_points.astype("f4").tobytes()), "3f", "position")])
    face_vao = ctx.vertex_array(grid_prog, [(ctx.buffer(face_points.astype("f4").tobytes()), "3f", "position")])
    if shown:
        print(f"Boxes: {len(shown)} objects from the semantic layer")
    near = {"ids": np.zeros(0, dtype=int), "current": None, "key": None, "texture": None, "size": (0, 0)}

    def object_color(obj):
        return CONDITION_COLORS.get(obj["asset"]["state"]["condition_state"], NO_ASSET_COLOR) if obj["asset"] else NO_ASSET_COLOR

    def update_near():
        """Find what is close to the sprite or camera, and rebuild the panel if that changed."""
        if route is not None:
            point = route_point(route, route_lengths, walk["s"])
        else:
            point = to_viewer.T @ cam.position
        order, distance = near_objects(point, box_lows, box_highs, NEAR_RADIUS)
        near["ids"] = order
        arrived_at_focus = focus in shown and walk["turn"] > 0.6
        if arrived_at_focus:
            current = shown.index(focus)
        elif not len(order):
            current = None
        elif near["current"] in order and distance[list(order).index(near["current"])] < distance[0] + NEAR_STICKY:
            current = near["current"]  # stay on this object, so the panel does not flicker
        else:
            current = int(order[0])
        near["current"] = current
        key = None if current is None else (current, tuple(int(i) for i in order))
        if key == near["key"]:
            return
        near["key"] = key
        if near["texture"]:
            near["texture"].release()
            near["texture"] = None
        if current is not None:
            names = [(shown[i]["asset"]["master"]["tag"] if shown[i]["asset"] else shown[i]["class"])
                     for i in order if i != current]
            image = asset_panel(shown[current], names)
            near["texture"] = ctx.texture(image.size, 4, image.tobytes())
            near["size"] = image.size

    # 2D map overlay, pixel-exact so the cell lines stay crisp.
    image = map_image(blocked, path_mask, start, end, MAP_CELL_PIXELS, z_up_screen)
    map_size = GRID_CELLS * MAP_CELL_PIXELS
    map_texture = ctx.texture((map_size, map_size), 4, image.tobytes())
    map_texture.filter = moderngl.NEAREST, moderngl.NEAREST
    map_texture.use(1)
    map_prog = ctx.program(vertex_shader=MAP_VERTEX_SHADER, fragment_shader=MAP_FRAGMENT_SHADER)
    map_prog["image"] = 1
    unit_quad = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype="f4").tobytes())
    map_vao = ctx.vertex_array(map_prog, [(unit_quad, "2f", "corner")])
    mark_prog = ctx.program(vertex_shader=MARK_VERTEX_SHADER, fragment_shader=MARK_FRAGMENT_SHADER)
    mark_buffer = ctx.buffer(reserve=6 * 8, dynamic=True)
    mark_vao = ctx.vertex_array(mark_prog, [(mark_buffer, "2f", "position")])

    def map_pixel(point, left, top):
        """Screen pixel on the map for a scene position."""
        across = (point[0] - grid_x + span / 2) / span
        down = (point[2] - grid_z + span / 2) / span
        down = 1 - down if z_up_screen else down
        return np.array([left + np.clip(across, 0, 1) * map_size, top + np.clip(down, 0, 1) * map_size])

    def draw_mark(pixels, color, size):
        """Fill screen-pixel triangles in one colour."""
        clip = np.asarray(pixels, dtype="f4") / size * (2, -2) + (-1, 1)
        mark_buffer.write(clip.astype("f4").tobytes())
        mark_prog["color"].value = color
        mark_vao.render(moderngl.TRIANGLES, vertices=len(clip))

    def draw_map(size):
        left, top = size[0] - map_size - 12, 12
        border = 3
        draw_mark(np.array([(0, 0), (1, 0), (0, 1), (1, 0), (1, 1), (0, 1)]) * (map_size + 2 * border)
                  + (left - border, top - border), (0.05, 0.05, 0.06, 1.0), size)
        map_prog["rect"].value = (2 * left / size[0] - 1, 1 - 2 * (top + map_size) / size[1],
                                  2 * map_size / size[0], 2 * map_size / size[1])
        map_vao.render(moderngl.TRIANGLE_STRIP)
        if route is not None:
            spot = map_pixel(route_point(route, route_lengths, walk["s"]), left, top)
            draw_mark(np.array([(-5, -5), (5, -5), (-5, 5), (5, -5), (5, 5), (-5, 5)]) + spot,
                      (0.05, 0.45, 1.0, 1.0), size)
        tip = map_pixel(to_viewer.T @ cam.position, left, top)
        facing = to_viewer.T @ cam.axes()[0]
        heading = np.array([facing[0], -facing[2] if z_up_screen else facing[2]])
        heading = heading / max(np.linalg.norm(heading), 1e-6)
        side = np.array([-heading[1], heading[0]])
        draw_mark([tip + heading * 8, tip - heading * 4 + side * 5, tip - heading * 4 - side * 5],
                  (1.0, 1.0, 1.0, 1.0), size)
    if args.no_sprite:
        show_blocks[0] = False  # raised blocks would hide the numbers
    if args.object:  # just the object, seen from a distance that fits it in the frame
        show_labels[0] = show_grid[0] = show_floor[0] = show_cells[0] = show_map[0] = show_boxes[0] = False
        center = to_viewer @ pos.mean(0)
        reach = float(np.ptp(pos, axis=0).max())
        if not args.view:
            cam.home = center + np.array([0.0, 0.0, 1.05 * reach + 0.1]), 0.0, 0.0
            cam.reset()
        speed = reach / 2
    if route is not None:
        walk["s"] = float(np.clip(args.walk, 0, 1)) * route_lengths[-1]
        walk["turn"] = 1.0 if args.walk >= 1 else 0.0
        if not args.top:  # a clean view of the room from the sprite's eyes
            show_labels[0] = show_grid[0] = show_blocks[0] = show_floor[0] = False

    def sprite_view():
        """Put the camera at the sprite's eyes, looking along the path ahead."""
        here = to_viewer @ route_point(route, route_lengths, walk["s"])
        ahead = to_viewer @ route_point(route, route_lengths, walk["s"] + LOOK_AHEAD)
        if walk["s"] + LOOK_AHEAD > route_lengths[-1]:  # near the end, keep the last heading
            here_back = to_viewer @ route_point(route, route_lengths, route_lengths[-1] - LOOK_AHEAD)
            ahead = to_viewer @ route[-1]
            heading = ahead - here_back
        else:
            heading = ahead - here
        cam.position = here + np.array([0, EYE_HEIGHT, 0])
        cam.yaw, cam.pitch = float(np.arctan2(heading[0], -heading[2])), -0.12
        if focus and walk["turn"] > 0:  # on arrival, turn smoothly to face the target
            to_target = to_viewer @ np.array(focus["box"]["center"]) - cam.position
            yaw = np.arctan2(to_target[0], -to_target[2])
            pitch = np.arctan2(to_target[1], np.hypot(to_target[0], to_target[2]))
            blend = walk["turn"] * walk["turn"] * (3 - 2 * walk["turn"])
            cam.yaw += float((yaw - cam.yaw + np.pi) % (2 * np.pi) - np.pi) * blend
            cam.pitch += float(pitch - cam.pitch) * blend
    print(f"Red cells: {int(blocked.sum())} of {blocked.size} ({blocked.mean():.1%}), "
          f"{int(blocked.sum()) - raw_count} added by cleanup")
    top = {"on": False, "zoom": args.zoom, "cut": CUT_FRACTION * wall_height, "saved": None}

    def set_top(on):
        if on:
            top["saved"] = cam.position.copy(), cam.yaw, cam.pitch
            cam.position, cam.yaw, cam.pitch = overhead.copy(), 0.0, -np.pi / 2
        elif top["saved"]:
            cam.position, cam.yaw, cam.pitch = top["saved"]
        top["on"] = on
        moved[0] = True

    def draw(framebuffer, size):
        framebuffer.use()
        framebuffer.clear(0.08, 0.09, 0.1, 1.0)
        prog["view"].write(view.T.astype("f4").tobytes())
        prog["res"].value = size
        if top["on"]:
            focal = size[1] * altitude / (1.1 * span) * top["zoom"]
        else:
            focal = size[1] / 2 / np.tan(FOV_Y / 2)
        prog["focal"].value = focal
        prog["floor_plane"].value = plane
        prog["up_sign"].value = up_sign
        prog["cut"].value = top["cut"] if top["on"] else 1e9
        prog["object_tint"].value = OBJECT_TINT if top["on"] and show_floor[0] else 0.0
        prog["clearance"].value = clearance
        prog["floor_tint"].value = FLOOR_TINT if show_floor[0] else 0.0
        grid_prog["offset"].value = 0.0, 0.0, 0.0
        floor_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=int(is_floor.sum()))
        grid_prog["view"].write(view.T.astype("f4").tobytes())
        grid_prog["res"].value = size
        grid_prog["focal"].value = focal
        if show_cells[0]:
            grid_prog["color"].value = 0.95, 0.05, 0.05, 0.55
            cell_vao.render(moderngl.TRIANGLES)
        if show_path[0] and path_mask.any():
            grid_prog["color"].value = 0.6, 0.1, 0.95, 0.85
            path_vao.render(moderngl.TRIANGLES)
        if show_grid[0]:
            grid_prog["color"].value = 0.75, 0.75, 0.75, 0.6
            grid_vao.render(moderngl.LINES)
        if show_labels[0] and (not top["on"] or top["zoom"] >= LABEL_MIN_ZOOM):
            grid_prog["color"].value = 0.0, 0.0, 0.0, 0.9
            label_vao.render(moderngl.TRIANGLES)
        object_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=int((~is_floor).sum()))
        if show_cells[0] and show_blocks[0] and not top["on"]:
            # Depth-only pass (transparent colour), so only the nearest block face tints each pixel.
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.depth_func = "<"
            grid_prog["color"].value = 0.0, 0.0, 0.0, 0.0
            block_vao.render(moderngl.TRIANGLES)
            ctx.depth_func = "<="
            first = 0
            for shade, count in zip(BLOCK_SHADES, block_groups):
                grid_prog["color"].value = *shade, BLOCK_ALPHA
                block_vao.render(moderngl.TRIANGLES, vertices=count, first=first)
                first += count
            ctx.disable(moderngl.DEPTH_TEST)
        if route is not None and (top["on"] or not walk["pov"]):
            grid_prog["offset"].value = tuple(route_point(route, route_lengths, walk["s"]))
            grid_prog["color"].value = 0.05, 0.45, 1.0, 1.0
            body_vao.render(moderngl.TRIANGLES)
            grid_prog["color"].value = 1.0, 0.8, 0.6, 1.0
            head_vao.render(moderngl.TRIANGLES)
        if show_boxes[0] and shown:
            grid_prog["offset"].value = 0.0, 0.0, 0.0  # the sprite moved this
            grid_prog["color"].value = 0.85, 0.85, 0.85, 0.3
            box_vao.render(moderngl.LINES)
            for index in near["ids"]:
                color = object_color(shown[index])
                strength = 0.3 if index == near["current"] else 0.14
                grid_prog["color"].value = *color, strength
                face_vao.render(moderngl.TRIANGLES, vertices=36, first=int(index) * 36)
                grid_prog["color"].value = *color, 1.0
                box_vao.render(moderngl.LINES, vertices=24, first=int(index) * 24)
        if show_map[0] and not top["on"]:
            draw_map(size)
        if show_boxes[0] and near["texture"]:
            near["texture"].use(2)
            map_prog["image"] = 2
            width, height = near["size"]
            map_prog["rect"].value = (2 * 12 / size[0] - 1, 1 - 2 * (12 + height) / size[1],
                                      2 * width / size[0], 2 * height / size[1])
            map_vao.render(moderngl.TRIANGLE_STRIP)
            map_prog["image"] = 1

    if route is not None:
        sprite_view()
    if args.capture:
        from PIL import Image  # only needed for captures

        # Stops are [x, y, z, yaw, pitch] in scene x, height above floor, scene z, radians.
        stops = json.loads(args.capture.read_text())
        show_labels[0] = show_grid[0] = show_blocks[0] = show_floor[0] = show_cells[0] = show_map[0] = False
        show_boxes[0] = False  # frames feed the detector, so they must show only the real scene
        folder = args.capture.parent
        size = 1280, 720
        fbo = ctx.simple_framebuffer(size, components=4)
        frames = []
        for index, (x, height, z, yaw, pitch) in enumerate(stops):
            scene = np.array([x, a * x + b * z + c - up_sign * height, z])
            cam.position, cam.yaw, cam.pitch = to_viewer @ scene, yaw, pitch
            view = cam.view()
            write_order(sort_far_to_near(view[2, :3]))
            draw(fbo, size)
            Image.frombytes("RGBA", size, fbo.read(components=4)).transpose(
                Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB").save(folder / f"frame_{index:04d}.jpg", quality=92)
            frames.append({"file": f"frame_{index:04d}.jpg", "view": view.tolist()})
        (folder / "frames.json").write_text(json.dumps({"size": size, "fov_y": float(FOV_Y), "frames": frames}))
        print(f"Captured {len(frames)} frames in {folder}")
        glfw.terminate()
        return
    if args.top:
        set_top(True)
    view = cam.view()
    update_near()
    if args.shot:
        from PIL import Image  # only needed for --shot

        write_order(sort_far_to_near(view[2, :3]))
        size = 1280, 720
        fbo = ctx.simple_framebuffer(size, components=4)
        draw(fbo, size)
        Image.frombytes("RGBA", size, fbo.read(components=4)).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB").save(args.shot)
        print("Saved", args.shot)
        glfw.terminate()
        return

    # Input
    drag = {"on": False, "x": 0.0, "y": 0.0}

    def on_button(_, button, action, __):
        if button == glfw.MOUSE_BUTTON_LEFT:
            drag["on"] = action == glfw.PRESS
            drag["x"], drag["y"] = glfw.get_cursor_pos(window)

    def on_cursor(_, x, y):
        if drag["on"] and not top["on"] and not walk["pov"]:
            cam.yaw += (x - drag["x"]) * 0.004
            cam.pitch = float(np.clip(cam.pitch - (y - drag["y"]) * 0.004, -1.5, 1.5))
            drag["x"], drag["y"] = x, y
            moved[0] = True

    def on_scroll(_, __, dy):
        nonlocal speed
        if top["on"]:
            top["zoom"] *= 1.15 if dy > 0 else 0.87
        else:
            speed *= 1.15 if dy > 0 else 0.87

    glfw.set_mouse_button_callback(window, on_button)
    glfw.set_cursor_pos_callback(window, on_cursor)

    def on_key(_, key, __, action, ___):
        if key == glfw.KEY_F and action == glfw.PRESS:
            show_floor[0] = not show_floor[0]
        elif key == glfw.KEY_G and action == glfw.PRESS:
            show_grid[0] = not show_grid[0]
        elif key == glfw.KEY_N and action == glfw.PRESS:
            show_labels[0] = not show_labels[0]
        elif key == glfw.KEY_P and action == glfw.PRESS:
            show_path[0] = not show_path[0]
        elif key == glfw.KEY_H and action == glfw.PRESS:
            show_blocks[0] = not show_blocks[0]
        elif key == glfw.KEY_B and action == glfw.PRESS:
            show_cells[0] = not show_cells[0]
        elif key == glfw.KEY_O and action == glfw.PRESS:
            show_boxes[0] = not show_boxes[0]
        elif key == glfw.KEY_M and action == glfw.PRESS:
            show_map[0] = not show_map[0]
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            walk["paused"] = not walk["paused"]
        elif key == glfw.KEY_V and action == glfw.PRESS and route is not None:
            walk["pov"] = not walk["pov"]
        elif key == glfw.KEY_T and action == glfw.PRESS:
            set_top(not top["on"])
        elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET) and action != glfw.RELEASE:
            step = 0.03 * wall_height * (1 if key == glfw.KEY_RIGHT_BRACKET else -1)
            top["cut"] = float(np.clip(top["cut"] + step, 0.02, wall_height))

    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    # Splats must be drawn far to near. Sorting runs off the render thread to keep movement smooth.
    pool = ThreadPoolExecutor(max_workers=1)
    write_order(sort_far_to_near(view[2, :3]))
    pending = None
    last = time.perf_counter()
    frames, title_time = 0, last

    while not glfw.window_should_close(window):
        glfw.poll_events()
        now = time.perf_counter()
        dt, last = min(now - last, 0.1), now

        pressed = lambda key: glfw.get_key(window, key) == glfw.PRESS
        if pressed(glfw.KEY_ESCAPE):
            break
        if route is not None and not walk["paused"] and walk["s"] < route_lengths[-1]:
            walk["s"] = min(walk["s"] + WALK_SPEED * dt, route_lengths[-1])
            if walk["s"] == route_lengths[-1]:
                print(f"Sprite arrived at cell {args.end}")
        elif route is not None and focus and walk["s"] >= route_lengths[-1] and not walk["paused"]:
            walk["turn"] = min(walk["turn"] + dt / TURN_SECONDS, 1.0)
        if pressed(glfw.KEY_R) and route is not None:
            walk["s"], walk["turn"] = 0.0, 0.0
        elif pressed(glfw.KEY_R):
            if top["on"]:
                cam.position, top["zoom"] = overhead.copy(), 1.0
            else:
                cam.reset()
            moved[0] = True
        forward, right, up = cam.axes()
        if top["on"]:
            direction = (up * (pressed(glfw.KEY_W) - pressed(glfw.KEY_S))
                         + right * (pressed(glfw.KEY_D) - pressed(glfw.KEY_A)))
        else:
            direction = (forward * (pressed(glfw.KEY_W) - pressed(glfw.KEY_S))
                         + right * (pressed(glfw.KEY_D) - pressed(glfw.KEY_A))
                         + np.array([0, 1, 0]) * (pressed(glfw.KEY_E) - pressed(glfw.KEY_Q)))
        if route is not None and walk["pov"] and not top["on"]:
            sprite_view()
            moved[0] = True
        elif direction.any():
            boost = 3 if pressed(glfw.KEY_LEFT_SHIFT) else 1
            cam.position += direction * speed * boost * dt
            moved[0] = True

        view = cam.view()
        update_near()
        if pending is not None and pending.done():
            write_order(pending.result())
            pending = None
        if moved[0] and pending is None:
            pending = pool.submit(sort_far_to_near, view[2, :3].copy())
            moved[0] = False

        draw(ctx.screen, glfw.get_framebuffer_size(window))
        glfw.swap_buffers(window)

        frames += 1
        if now - title_time >= 1:
            glfw.set_window_title(window, f"{args.ply.stem} | {len(v):,} splats | {frames / (now - title_time):.0f} fps")
            frames, title_time = 0, now

    pool.shutdown(wait=False, cancel_futures=True)
    glfw.terminate()


if __name__ == "__main__":
    # Self-check: a good record passes, a corrupt one is rejected without crashing.
    names = "x y z scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
    sample = np.zeros(2, dtype=[(k, "<f4") for k in names])
    sample["rot_0"] = 2  # unnormalised is still valid
    sample["x"][1] = np.nan
    assert valid_mask(sample).tolist() == [True, False]
    # Self-check: a cell turns red at MIN_CELL_SPLATS points and not before.
    test_x = np.array([0.1] * MIN_CELL_SPLATS + [-0.9] * (MIN_CELL_SPLATS - 1) + [99.0])
    test_cells = blocked_cells(test_x, np.full(len(test_x), 0.1), 0, 0, 2, 2)
    assert test_cells.tolist() == [[False, False], [False, True]]
    assert len(cell_triangles(test_cells, (0, 0, 1), 0, 0, 2)) == 6
    # Self-check: row 0 of the top view is the far z edge on y-down scans, the near edge on y-up.
    assert cell_grid_index(0, 3, 60, True) == (3, 59) and cell_grid_index(0, 3, 60, False) == (3, 0)
    # Self-check: "1" lights 8 pixels, so one cell numbered 1 draws 8 quads.
    assert len(cell_labels((0, 0, 0), 0, 0, 1, 1, True)) == 8 * 6
    # Self-check: only boxes in range count, nearest first, measured to the box edge.
    box_low, box_high = np.array([[1.0, 0, 0], [3.0, 0, 0]]), np.array([[2.0, 1, 1], [4.0, 1, 1]])
    assert near_objects((0, 0, 0.5), box_low, box_high, 1.5)[0].tolist() == [0]
    assert near_objects((2.5, 0, 0.5), box_low, box_high, 1.5)[0].tolist() == [0, 1]
    assert len(box_faces((0, 0, 0), (1, 1, 1))) == 36
    # Self-check: the panel lists a plain object with no asset without failing.
    assert asset_panel({"class": "ladder", "id": "obj_1", "asset": None}).size[0] == 430
    # Self-check: a box has 12 edges, each one unit long on one axis.
    edges = box_lines((0, 0, 0), (1, 1, 1))
    assert len(edges) == 24 and np.allclose(np.abs(edges[::2] - edges[1::2]).sum(1), 1)
    # Self-check: map rows match the top view: the far z edge is on top for y-down scans.
    corner_cell = np.zeros((2, 2), bool)
    corner_cell[0, 1] = True  # ix 0, iz 1
    assert map_image(corner_cell, np.zeros((2, 2), bool), None, None, 1, True)[0, 0, 0] != \
        map_image(corner_cell, np.zeros((2, 2), bool), None, None, 1, False)[0, 0, 0]
    # Self-check: smoothing keeps the ends, rounds a corner, and route lookups clamp at the ends.
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype="f8")
    rounded, distance = smooth_route(corner)
    assert np.allclose(rounded[0], corner[0]) and np.allclose(rounded[-1], corner[2])
    assert distance[-1] < 2 and np.linalg.norm(rounded - corner[1], axis=1).min() > 0.02
    assert np.allclose(route_point(rounded, distance, 99), rounded[-1])
    # Self-check: the sprite mesh stands on the floor and reaches its full height.
    sprite_body, sprite_head = sprite_triangles(0.1, 1.0, 1.0)
    assert sprite_body[:, 1].max() == 0 and sprite_head[:, 1].min() == -1
    # Self-check: sprite sizes grow with open space, and the grid edge counts as blocked.
    open_room = np.zeros((5, 5), bool)
    assert max_sprite_sizes(open_room)[2, 2] == 5 and max_sprite_sizes(open_room)[0, 0] == 1
    open_room[2, 2] = True
    assert max_sprite_sizes(open_room)[2, 2] == 0 and max_sprite_sizes(open_room)[1, 1] == 1
    # Self-check: Dijkstra joins the ends through the only gap, and matches the fewest steps here.
    walled = np.zeros((3, 3), bool)
    walled[1, 0:2] = True
    wall_sizes = max_sprite_sizes(walled)
    route = dijkstra_path(wall_sizes, 1, (0, 0), (2, 0))
    assert route[0] == (0, 0) and route[-1] == (2, 0) and (1, 2) in route
    assert shortest_steps(wall_sizes >= 1, (0, 0), (2, 0)) == len(route) - 1 == 6
    assert dijkstra_path(np.array([[1, 0, 1]]), 1, (0, 0), (0, 2)) is None
    # Self-check: a sprite wider than every cell has nowhere to stand.
    assert not (wall_sizes >= 3).any()
    # Self-check: one lone cell becomes a top and 4 walls, and two touching cells share no wall.
    one = np.zeros((3, 3), bool)
    one[1, 1] = True
    assert block_triangles(one, (0, 0, 0), 0, 0, 3, 1, 1)[1] == (6, 12, 12)
    one[1, 2] = True
    assert block_triangles(one, (0, 0, 0), 0, 0, 3, 1, 1)[1] == (12, 24, 12)
    # Self-check: a gap between reds fills, a sealed pocket fills, open floor and corners stay free.
    room = np.zeros((9, 9), bool)
    room[4, 0:3] = room[4, 4:6] = True           # wall with a one-cell gap at (4, 3)
    room[0:3, 6] = room[2, 7:] = room[0, 6] = True  # seals a pocket in the top-right corner
    cleaned = clean_cells(room)
    assert cleaned[4, 3] and cleaned[0, 8] and not cleaned[8, 0] and not cleaned[3, 5]
    # Self-check: a 20 x 20 grid has 21 lines each way, and its ends sit on the floor plane.
    lines = grid_lines((0.1, 0.2, 1.0), 0, 0, 2, 20)
    assert len(lines) == 2 * 2 * 21 and np.allclose(lines[:, 1], 0.1 * lines[:, 0] + 0.2 * lines[:, 2] + 1)
    # Self-check: rotating splats turns their surface directions the same way as their positions.
    splat_names = "x y z scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3 opacity".split()
    flat_splats = np.zeros(800, dtype=[(k, "<f4") for k in splat_names])
    flat_splats["x"], flat_splats["z"] = np.random.default_rng(1).uniform(-1, 1, (2, 800))
    flat_splats["scale_0"], flat_splats["scale_2"], flat_splats["scale_1"] = np.log(0.05), np.log(0.05), np.log(0.005)
    flat_splats["rot_0"], flat_splats["opacity"] = 1, 5
    assert np.allclose(np.abs(find_up(flat_splats)), (0, 1, 0))
    angle = np.radians(25)
    tilt_matrix = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    tilted = rotate_splats(flat_splats, tilt_matrix)
    assert np.allclose(np.abs(find_up(tilted)), np.abs(tilt_matrix @ (0, 1, 0)), atol=1e-3)
    assert np.allclose(np.abs(find_up(level_scan(tilted, report=False))), (0, 1, 0), atol=1e-3)
    # Self-check: the object limit sits just above a thin floor layer, even with objects higher up.
    floor_layer = np.random.default_rng(2).normal(0, 0.01, 20000)
    objects_above = np.random.default_rng(3).uniform(0.05, 1, 2000)
    assert 0.02 < object_clearance(np.concatenate([floor_layer, objects_above]), 0.1) < 0.09
    # Self-check: fit_floor recovers a tilted floor below a cloud of objects.
    rng = np.random.default_rng(0)
    cloud = np.zeros(40000, dtype=[(k, "<f4") for k in ("x", "y", "z", "opacity")])
    cloud["opacity"] = 5
    cloud["x"], cloud["z"] = rng.uniform(-3, 3, (2, len(cloud)))
    cloud["y"] = rng.uniform(-1, 1, len(cloud))
    cloud["y"][:20000] = 0.01 * cloud["x"][:20000] - 0.02 * cloud["z"][:20000] + 1.5
    fitted, _, rotation = fit_floor(cloud, report=False)
    assert np.allclose(fitted, (0.01, -0.02, 1.5), atol=1e-3) and rotation is Y_DOWN
    assert floor_weight(cloud[:1], fitted, 0.05).item() > 0.9
    main()
