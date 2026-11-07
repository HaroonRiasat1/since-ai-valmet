"""
Splat navigator: builds a walkable occupancy map from a 3D Gaussian Splatting .ply
and plans a collision-free path to a target item.

Pipeline
  1. load splats (positions, opacity)
  2. find the floor height along the vertical axis
  3. rasterise to a top-down grid: floor cells = walkable evidence, splats at
     body height = obstacles
  4. inflate obstacles by the walker's radius (so the path keeps clearance)
  5. A* from start to the free cell closest to the target item's box
  6. simplify the path (line-of-sight shortcuts) and export it in world coords

Usage
  python navigator.py scene.ply --items items.json --target ASSET-003 --out out/
items.json: [{"id": "ASSET-003", "label": "pump", "center": [x,y,z], "size": [sx,sy,sz]}, ...]
(world coordinates, same frame as the .ply)
"""
import argparse, heapq, json, os
import numpy as np
from scipy import ndimage

UP = 1            # index of vertical axis in the .ply (y here)
DOWN_SIGN = 1     # +1 if larger y means lower (COLMAP / most 3DGS exports)


def load_splats(path):
    with open(path, "rb") as f:
        h = b""
        while not h.endswith(b"end_header\n"):
            h += f.readline()
    lines = h.decode().splitlines()
    props = [l.split()[-1] for l in lines if l.startswith("property")]
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    d = np.memmap(path, dtype="<f4", mode="r", offset=len(h), shape=(n, len(props)))
    xyz = np.array(d[:, :3], dtype=np.float32)
    op = 1 / (1 + np.exp(-np.clip(np.array(d[:, props.index("opacity")]), -30, 30)))
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz).max(1) < 1e3)
    return xyz[ok], op[ok]


def height_above_floor(xyz, op):
    """Floor = densest opaque layer at the low end of the vertical axis."""
    v = xyz[op > 0.5, UP] * DOWN_SIGN          # larger = lower
    lo, hi = np.percentile(v, [1, 99.5])
    hist, edges = np.histogram(v, bins=200, range=(lo, hi))
    low_half = slice(len(hist) // 2, None)       # floor is in the lower half
    i = np.argmax(hist[low_half]) + len(hist) // 2
    floor = (edges[i] + edges[i + 1]) / 2
    return floor - xyz[:, UP] * DOWN_SIGN, floor


def build_grid(xyz, op, h, cell=0.05, body=(0.15, 1.8), floor_band=0.08,
               radius=0.25, min_hits=2):
    horiz = [a for a in range(3) if a != UP]
    xy = xyz[:, horiz]
    origin = np.percentile(xy, 0.5, axis=0) - 0.2
    extent = np.percentile(xy, 99.5, axis=0) + 0.2
    shape = np.ceil((extent - origin) / cell).astype(int)
    ij = np.floor((xy - origin) / cell).astype(int)
    inb = (ij >= 0).all(1) & (ij < shape).all(1)

    def count(mask):
        g = np.zeros(shape, np.int32)
        m = mask & inb
        np.add.at(g, (ij[m, 0], ij[m, 1]), 1)
        return g

    floor_hits = count((np.abs(h) < floor_band) & (op > 0.2))
    obst_hits = count((h > body[0]) & (h < body[1]) & (op > 0.4))

    floor = ndimage.binary_closing(floor_hits > 0, iterations=2)   # fill small holes
    obstacle = ndimage.binary_opening(obst_hits >= min_hits, iterations=1)  # drop specks
    free = floor & ~obstacle
    # keep the largest connected walkable region (removes floating floor bits)
    lab, n = ndimage.label(free)
    if n:
        free = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    # clearance: distance (in m) from every cell to the nearest non-free cell
    clearance = ndimage.distance_transform_edt(free) * cell
    walk = clearance > radius
    lab, n = ndimage.label(walk, structure=np.ones((3, 3)))   # one connected walkable area
    if n:
        walk = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return dict(origin=origin, cell=cell, horiz=horiz, floor=floor,
                obstacle=obstacle, free=free, walk=walk, clearance=clearance)


def astar(walk, clearance, start, goal):
    """8-connected A*; small penalty near obstacles keeps paths centred in aisles."""
    H, W = walk.shape
    nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    pen = 1 + 0.6 / (clearance + 0.1)
    g = {start: 0.0}; came = {}
    pq = [(0.0, start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        for di, dj, c in nbrs:
            nx = (cur[0] + di, cur[1] + dj)
            if not (0 <= nx[0] < H and 0 <= nx[1] < W) or not walk[nx]:
                continue
            ng = g[cur] + c * pen[nx]
            if ng < g.get(nx, 1e18):
                g[nx] = ng; came[nx] = cur
                heapq.heappush(pq, (ng + np.hypot(goal[0] - nx[0], goal[1] - nx[1]), nx))
    return None


def line_free(walk, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    ii = np.linspace(a[0], b[0], n).round().astype(int)
    jj = np.linspace(a[1], b[1], n).round().astype(int)
    return walk[ii, jj].all()


def simplify(walk, path):
    out = [path[0]]; k = 0
    while k < len(path) - 1:
        j = len(path) - 1
        while j > k + 1 and not line_free(walk, path[k], path[j]):
            j -= 1
        out.append(path[j]); k = j
    return out


def to_cell(G, p_world):
    p = np.asarray(p_world)[G["horiz"]]
    return tuple(np.floor((p - G["origin"]) / G["cell"]).astype(int))


def to_world(G, ij, floor, eye=0.0):
    w = np.zeros(3)
    w[G["horiz"]] = G["origin"] + (np.asarray(ij) + 0.5) * G["cell"]
    w[UP] = (floor - eye) * DOWN_SIGN
    return w.tolist()


def goal_near_item(G, item, standoff=0.4):
    """Free walkable cell closest to the item's footprint, within reach."""
    c = np.asarray(item["center"])[G["horiz"]]
    half = np.asarray(item.get("size", [0.5] * 3))[G["horiz"]] / 2
    H, W = G["walk"].shape
    ii, jj = np.nonzero(G["walk"])
    pts = G["origin"] + (np.stack([ii, jj], 1) + 0.5) * G["cell"]
    # distance from cell to the item's 2D box
    dvec = np.maximum(np.abs(pts - c) - half, 0)
    dist = np.hypot(dvec[:, 0], dvec[:, 1])
    k = np.argmin(np.abs(dist - standoff) + (dist < 0.05) * 10)
    return (ii[k], jj[k]), float(dist[k])


def render(G, path, items, target_id, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    img = np.ones(G["free"].shape + (3,)) * 0.15            # unknown = dark
    img[G["floor"]] = (0.55, 0.55, 0.55)                     # floor seen
    img[G["free"] & ~G["walk"]] = (0.95, 0.8, 0.45)          # too close to obstacles
    img[G["walk"]] = (0.85, 0.95, 0.85)                      # walkable
    img[G["obstacle"]] = (0.75, 0.2, 0.2)                    # obstacles
    o, c = G["origin"], G["cell"]
    ext = [o[1], o[1] + img.shape[1] * c, o[0] + img.shape[0] * c, o[0]]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(img, extent=ext, interpolation="nearest")
    for it in items:
        cc = np.asarray(it["center"])[G["horiz"]]; s = np.asarray(it["size"])[G["horiz"]]
        tgt = it["id"] == target_id
        ax.add_patch(Rectangle((cc[1] - s[1] / 2, cc[0] - s[0] / 2), s[1], s[0], fill=False,
                               ec="magenta" if tgt else "royalblue", lw=2.5 if tgt else 1.2))
        ax.text(cc[1], cc[0], it.get("label", it["id"]), fontsize=7, ha="center",
                color="magenta" if tgt else "navy")
    if path:
        P = o + (np.asarray(path) + 0.5) * c
        ax.plot(P[:, 1], P[:, 0], "-", c="black", lw=2.5)
        ax.plot(P[0, 1], P[0, 0], "o", c="lime", ms=12, mec="black")
        ax.plot(P[-1, 1], P[-1, 0], "*", c="magenta", ms=18, mec="black")
    ax.set_xlabel(f"axis {G['horiz'][1]} (m)"); ax.set_ylabel(f"axis {G['horiz'][0]} (m)")
    ax.set_title("Walkable map: green = walkable, red = obstacle, amber = too tight, dark = unseen")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply"); ap.add_argument("--items"); ap.add_argument("--target")
    ap.add_argument("--start", type=float, nargs=3, help="world xyz start")
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--out", default="out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    xyz, op = load_splats(a.ply)
    h, floor = height_above_floor(xyz, op)
    G = build_grid(xyz, op, h, radius=a.radius)
    print(f"floor at {UP=} value {floor * DOWN_SIGN:.3f}; grid {G['walk'].shape}, "
          f"walkable {G['walk'].mean():.1%}")

    items = json.load(open(a.items)) if a.items else []
    target = next(i for i in items if i["id"] == a.target)
    goal, standoff = goal_near_item(G, target)

    if a.start:
        start = to_cell(G, a.start)
    else:   # default: walkable cell farthest (by walking) from the goal side
        ii, jj = np.nonzero(G["walk"])
        k = np.argmax(np.hypot(ii - goal[0], jj - goal[1]))
        start = (ii[k], jj[k])
    if not G["walk"][start]:
        ii, jj = np.nonzero(G["walk"])
        k = np.argmin(np.hypot(ii - start[0], jj - start[1])); start = (ii[k], jj[k])

    raw = astar(G["walk"], G["clearance"], start, goal)
    if raw is None:
        raise SystemExit("no path found: target unreachable with this radius")
    path = simplify(G["walk"], raw)
    length = sum(np.hypot(*(np.subtract(b, a) * G["cell"])) for a, b in zip(path, path[1:]))

    result = dict(target=a.target, standoff_m=round(standoff, 2), length_m=round(length, 2),
                  waypoints=[to_world(G, p, floor) for p in path],
                  dense_path=[to_world(G, p, floor) for p in raw[::4]])
    json.dump(result, open(os.path.join(a.out, "path.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "navgrid.npz"), walk=G["walk"], obstacle=G["obstacle"],
                        origin=G["origin"], cell=G["cell"], floor=floor)
    render(G, path, items, a.target, os.path.join(a.out, "map.png"))
    print(f"path: {len(path)} waypoints, {length:.1f} m -> {a.out}/path.json, map.png")


if __name__ == "__main__":
    main()
