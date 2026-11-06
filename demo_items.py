"""Stand-in for your detector: turns obstacle blobs into item boxes (items.json)."""
import json, sys, numpy as np
from scipy import ndimage
sys.path.insert(0, '.')
import navigator as nv
xyz, op = nv.load_splats(sys.argv[1])
h, floor = nv.height_above_floor(xyz, op)
G = nv.build_grid(xyz, op, h)
lab, n = ndimage.label(G['obstacle'])
items = []
sel = (h > 0.15) & (h < 1.8) & (op > 0.4)
hz = xyz[sel][:, G['horiz']]; hh = h[sel]
cells = np.floor((hz - G['origin']) / G['cell']).astype(int)
inb = (cells >= 0).all(1) & (cells < lab.shape).all(1)
cl = np.zeros(len(cells), int); cl[inb] = lab[cells[inb, 0], cells[inb, 1]]
for k, sl in enumerate(ndimage.find_objects(lab), 1):
    m = cl == k
    if m.sum() < 300: continue
    p = hz[m]; lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
    top = np.percentile(hh[m], 98)
    c = np.zeros(3); s = np.zeros(3)
    c[G['horiz']] = (lo + hi) / 2; s[G['horiz']] = hi - lo
    c[nv.UP] = (floor - top / 2) * nv.DOWN_SIGN; s[nv.UP] = top
    items.append(dict(id=f"OBJ-{len(items)+1:03d}", label=f"obj{len(items)+1}", center=c.round(3).tolist(), size=s.round(3).tolist()))
json.dump(items, open(sys.argv[2], 'w'), indent=1)
print(len(items), 'items')
