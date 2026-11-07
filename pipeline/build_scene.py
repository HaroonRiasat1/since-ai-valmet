"""
Builds everything the three.js viewer needs from the .ply + Excel builder pack:
  web/data/room.splat.partN   splat converted to compact .splat format (32 B/splat), chunked
  web/data/scene.json         item boxes + mapped Excel metadata, nav path, copilot diagnosis

python pipeline/build_scene.py Equipment_Room.ply builder_pack.xlsx data/semantic_layer.json web/
python build_scene.py Equipment_Room.ply builder_pack.xlsx data/semantic_layer.json web/

The second form uses the semantic layer's objects and asset mapping as they are, and adds the 60 x 60
cell grid. TARGET_TAG picks the faulty asset (default CV-003).
"""
import base64, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navigator as nv

ply, xlsx, items_path, web = sys.argv[1:5]
os.makedirs(f"{web}/data", exist_ok=True)

# ---------------------------------------------------------------- splat -> .splat
with open(ply, "rb") as f:
    h = b""
    while not h.endswith(b"end_header\n"):
        h += f.readline()
L = h.decode().splitlines()
props = [l.split()[-1] for l in L if l.startswith("property")]
n = int(next(l for l in L if l.startswith("element vertex")).split()[-1])
d = np.memmap(ply, dtype="<f4", mode="r", offset=len(h), shape=(n, len(props)))
P = {k: np.array(d[:, props.index(k)]) for k in
     ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
      "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]}
alpha = 1 / (1 + np.exp(-np.clip(P["opacity"], -30, 30)))
keep = (alpha > 0.03) & np.isfinite(P["x"])
SH_C0 = 0.28209479177387814
buf = np.zeros(keep.sum(), dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
buf["pos"] = np.stack([P["x"], P["y"], P["z"]], 1)[keep]
buf["scale"] = np.exp(np.stack([P["scale_0"], P["scale_1"], P["scale_2"]], 1)[keep])
rgb = np.clip((0.5 + SH_C0 * np.stack([P["f_dc_0"], P["f_dc_1"], P["f_dc_2"]], 1)[keep]) * 255, 0, 255)
buf["rgba"] = np.concatenate([rgb, alpha[keep, None] * 255], 1).round().astype(np.uint8)
q = np.stack([P["rot_0"], P["rot_1"], P["rot_2"], P["rot_3"]], 1)[keep]
q /= np.linalg.norm(q, axis=1, keepdims=True)
buf["rot"] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
order = np.argsort(-(buf["scale"].prod(1) * alpha[keep]))      # big/opaque first
raw = buf[order].tobytes()
CH = 18 * 1024 * 1024 // 32 * 32
parts = [raw[i:i + CH] for i in range(0, len(raw), CH)]
for i, p in enumerate(parts if not os.environ.get("SKIP_SPLAT") else []):
    open(f"{web}/data/room_part{i}.glb", "wb").write(p)
print(f"splats kept {keep.sum()} of {n}, {len(raw)/1e6:.1f} MB in {len(parts)} parts")

# ---------------------------------------------------------------- nav grid
xyz, op = nv.load_splats(ply)
hgt, floor = nv.height_above_floor(xyz, op)
G = nv.build_grid(xyz, op, hgt, radius=0.25)
items = json.load(open(items_path))
SEMANTIC = isinstance(items, dict)               # a semantic layer rather than a plain item list
TARGET_TAG = os.environ.get("TARGET_TAG", "CV-003")
LAYER_CONF = {"high": 0.7, "medium": 0.55, "low": 0.4}


def semantic_items(layer):
    """Turn semantic layer objects into this builder's item format."""
    out = []
    for o in layer["objects"]:
        low, high = np.array(o["box"]["min"]), np.array(o["box"]["max"])
        vision = o["source"] == "vision model"
        out.append(dict(id=o["id"], label=o["class"], center=((low + high) / 2).round(3).tolist(),
                        size=(high - low).round(3).tolist(), tag=o["asset"]["master"]["tag"] if o["asset"] else None,
                        _how="semantic", _method=o["mapping_method"], _class=o["class"], _source=o["source"],
                        _class_conf=o["class_confidence"], _vlm=dict(label=o["class"], asset_class=o["class"].upper().replace(" ", "_"),
                        confidence=LAYER_CONF.get(o["class_confidence"], 0.4), visual_evidence=o.get("note", ""), model="vlm_lift") if vision else None))
    return out


if SEMANTIC:   # objects dropped into data/objects/ join the extracted ones (format: data_format/README.md)
    import added_objects
    layer = added_objects.merge(items, added_objects.load(), set(pd.read_excel(xlsx, sheet_name="Asset_Master")["tag"]))
    items = semantic_items(layer)

# ---------------------------------------------------------------- Excel
xl = {s: pd.read_excel(xlsx, s) for s in
      ["Asset_Master", "Current_State", "Telemetry_Sample_3d", "Alarms_Events",
       "Maintenance_History", "Historical_Service_Cases", "Knowledge_Docs", "Data_Quality"]}
cur = xl["Current_State"].set_index("tag")
master = xl["Asset_Master"].set_index("tag")
tel = xl["Telemetry_Sample_3d"].sort_values("timestamp")
KEY = {
    "CONTROL_VALVE": ["position_setpoint_pct", "position_actual_pct", "pneumatic_supply_bar", "stiction_index"],
    "PROCESS_PUMP": ["vibration_mm_s", "bearing_temp_c", "process_pressure_bar", "process_flow_pct"],
    "ELECTRIC_MOTOR": ["vibration_mm_s", "bearing_temp_c", "motor_current_a", "speed_rpm"],
    "PRESSURE_TRANSMITTER": ["process_pressure_bar", "signal_quality_pct", "sensor_bias"],
    "VIBRATION_SENSOR": ["vibration_mm_s", "signal_quality_pct", "sensor_bias"],
}


def telemetry(tag, cls):
    t = tel[tel.tag == tag]
    out = {}
    for c in KEY[cls]:
        s = t[c].dropna()
        if len(s):
            out[c] = dict(last=round(float(s.iloc[-1]), 2),
                          mean_24h=round(float(s.iloc[-48:].mean()), 2),
                          mean_prev=round(float(s.iloc[:-48].mean()), 2) if len(s) > 48 else None)
    return out


# ---------------------------------------------------------------- mapping: mapping.csv first, then area-zoned auto
# No identity is certain without real detection (DOC-3D-MAPPING). The auto mapping keeps the Excel
# structure plausible: each area_name gets its own spatial zone of the room, and inside a zone big
# footprints take pumps/motors, medium valves, small transmitters/sensors. Rows in mapping.csv
# (box_id,tag; tag "-" = force unmapped) win over the auto mapping and are labelled manual.
from itertools import permutations
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

MAP_CSV = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(items_path) or ".", "mapping.csv")
SIZE_Q = {"PROCESS_PUMP": 0.85, "ELECTRIC_MOTOR": 0.7, "CONTROL_VALVE": 0.45,   # target footprint quantile
          "PRESSURE_TRANSMITTER": 0.15, "VIBRATION_SENSOR": 0.1}
byid = {it["id"]: it for it in items}
for it in items:
    s = np.asarray(it["size"]); it["_foot"] = float(s[G["horiz"]].prod()); it["_vol"] = it["_foot"] * float(s[nv.UP])

# VLM labels from the viewer's "VLM scan" (web/data/vlm_labels.json): class evidence per box
VLM_PATH = f"{web}/data/vlm_labels.json"
vlm = json.load(open(VLM_PATH)) if os.path.exists(VLM_PATH) else {}
for it in items if not SEMANTIC else []:      # items found by vlm_lift.py carry their own VLM class
    it["_vlm"] = vlm.get(it["id"]) or (dict(label=it["label"], asset_class=it["asset_class"], confidence=it["confidence"],
        visual_evidence=f"VLM detection fused from {it['views']} rendered views", model="vlm_lift")
        if it.get("source") == "vlm" else None)
n_vlm = sum(1 for it in items if it["_vlm"])
print(f"VLM labels: {n_vlm} of {len(items)} items" if n_vlm else "VLM labels: none")


def vlm_cost(it, cls):
    """Negative when the VLM saw this class, positive when it saw something else. Valves and pumps are
    easy to recognise, so calling one a cabinet/tank is strong evidence against; motors and small
    sensors can plausibly read as a cabinet or a pipe in a blurry render, so that costs little."""
    v = it["_vlm"]
    if not v:
        return 0.0
    if v["asset_class"] == cls:
        return -1.2 * v["confidence"]
    if v["asset_class"] == "OTHER_EQUIPMENT":
        return (1.2 if cls in ("CONTROL_VALVE", "PROCESS_PUMP") else 0.3) * v["confidence"]
    return (3.0 if v["asset_class"] == "NOT_EQUIPMENT" else 1.5) * v["confidence"]


UNPLACED = 0.6       # cost of leaving a box empty / a tag unplaced: pairs costing more than 2x this are dropped


manual = {}
if os.path.exists(MAP_CSV) and not SEMANTIC:
    for r in pd.read_csv(MAP_CSV, dtype=str).fillna("").to_dict("records"):
        bid, t = r.get("box_id", "").strip(), r.get("tag", "").strip()
        if not t:
            continue
        if bid not in byid:
            print(f"mapping.csv: unknown box {bid}, ignored")
        elif t.lower() in ("-", "none"):
            manual[bid] = None
        elif t not in cur.index:
            print(f"mapping.csv: unknown tag {t} for {bid}, ignored")
        elif t in manual.values():
            print(f"mapping.csv: {t} already mapped, {bid} ignored")
        else:
            manual[bid] = t
for bid, t in manual.items():
    byid[bid]["tag"], byid[bid]["_how"] = t, "manual"

walls = set() if any(i.get("source") == "vlm" for i in items) else \
    {i["id"] for i in sorted(items, key=lambda i: -i["_vol"])[:4]}         # blob detector: wall / shelving runs
not_equip = {i["id"] for i in items if i["_vlm"] and i["_vlm"]["asset_class"] == "NOT_EQUIPMENT"
             and i["_vlm"]["confidence"] >= 0.6}            # ladders, carts, chairs: never carry a tag
cand = [] if SEMANTIC else [i for i in items if i["id"] not in walls | not_equip and i["id"] not in manual]
todo = [t for t in cur.index if t not in manual.values()]
if cand and todo:
    xy = np.array([np.asarray(i["center"])[G["horiz"]] for i in cand])
    q = np.argsort(np.argsort([i["_foot"] for i in cand])) / max(len(cand) - 1, 1)
    area_of = [cur.loc[t, "area_name"] for t in todo]
    areas = sorted(set(area_of), key=lambda a: -area_of.count(a))
    size_cost = np.abs(q[:, None] - np.array([SIZE_Q[cur.loc[t, "asset_class"]] for t in todo])[None])
    diag = float(np.linalg.norm(np.ptp(xy, 0)))
    vlm_c = np.array([[vlm_cost(i, cur.loc[t, "asset_class"]) for t in todo] for i in cand])
    seeds, _ = kmeans2(xy, len(areas), seed=0, minit="++")
    prev = np.inf
    for _ in range(20):                       # capacitated k-means: zones follow their assigned boxes
        best = None
        for perm in permutations(range(len(areas))):
            zc = {a: seeds[perm[k]] for k, a in enumerate(areas)}
            C = np.stack([np.linalg.norm(xy - zc[a], axis=1) for a in area_of], 1) / diag + 0.35 * size_cost + vlm_c
            nb, nt = C.shape                    # pad so a box may stay empty and a tag may stay unplaced
            P = np.block([[C, np.full((nb, nb), UNPLACED)], [np.full((nt, nt), UNPLACED), np.zeros((nt, nb))]])
            r, c = linear_sum_assignment(P)
            real = (r < nb) & (c < nt)
            if best is None or P[r, c].sum() < best[0]:
                best = (P[r, c].sum(), r[real], c[real])
        cost, r, c = best
        seeds = np.array([xy[r[[area_of[j] == a for j in c]]].mean(0) if a in {area_of[j] for j in c} else seeds[k]
                          for k, a in enumerate(areas)])
        if cost >= prev - 1e-9:
            break
        prev = cost
    for i, j in zip(r, c):
        cand[i]["tag"], cand[i]["_how"], cand[i]["_zone"] = todo[j], "auto", area_of[j]
    placed = {todo[j] for j in c}
    if len(placed) < len(todo):                 # better unplaced than pinned to a ladder (DOC-3D-MAPPING)
        print(f"{len(todo) - len(placed)} tags not located in the scan: {', '.join(t for t in todo if t not in placed)}")

if not SEMANTIC and not os.path.exists(MAP_CSV):   # template for hand corrections; tag column left blank
    pd.DataFrame([dict(box_id=i["id"], tag="", suggested_tag=i.get("tag") or "",
                       vlm_label=i["_vlm"]["label"] if i["_vlm"] else "",
                       asset_class=cur.loc[i["tag"], "asset_class"] if i.get("tag") else "",
                       area=cur.loc[i["tag"], "area_name"] if i.get("tag") else "",
                       center=" ".join(f"{v:.2f}" for v in i["center"]),
                       size=" ".join(f"{v:.2f}" for v in i["size"])) for i in items]).to_csv(MAP_CSV, index=False)
    print(f"wrote {MAP_CSV}: fill the tag column to override the auto mapping")

# route target: TARGET_TAG (CV-003 is degraded, high urgency, A-critical), else the first mapped valve
target_box = next((i for i in items if i.get("tag") == TARGET_TAG), None) or \
             next(i for i in items if (i.get("tag") or "").startswith("CV"))
ii, jj = np.nonzero(G["walk"])
goal = nv.goal_near_item(G, target_box)[0]
paths = [nv.astar(G["walk"], G["clearance"], (ii[k], jj[k]), goal)      # start at the far end of the room
         for k in (np.argmin(ii), np.argmax(ii), np.argmin(jj), np.argmax(jj))]
raw_path = max((p for p in paths if p), key=len, default=None)
assert raw_path, f"no walkable path to {target_box['id']}"

def mapping_of(it, cls):
    if it["_how"] == "semantic" and it["_method"] == "manual":
        return dict(mapping="manual (data/objects)", mapping_conf=0.9, matched=True)
    if it["_how"] == "semantic":
        matched = it["_method"] == "class_match"
        how = f"{it['_source']}: {it['_class']} matches {cls.lower().replace('_', ' ')}" if matched else \
            f"procedural, on a {it['_class']}"
        return dict(mapping=how, mapping_conf=LAYER_CONF.get(it["_class_conf"], 0.4) if matched else 0.2, matched=matched)
    if it["_how"] == "manual":
        return dict(mapping="manual (mapping.csv)", mapping_conf=0.9)
    v = it["_vlm"]
    if v and v["asset_class"] == cls:            # VLM agrees on class; identity within the class is still a guess
        return dict(mapping=f"VLM class match '{v['label']}' + area zone {it['_zone']}",
                    mapping_conf=round(min(0.7, 0.35 + 0.35 * v["confidence"]), 2))
    return dict(mapping=f"procedural (area zone {it['_zone']} + size class"
                        + (f"; VLM saw '{v['label']}'" if v else "") + ")", mapping_conf=0.35 if not v else 0.2)


out_items = []
for it in items:
    rec = dict(id=it["id"], center=it["center"], size=it["size"], source=it.get("_source", "detector"))
    tag = it.get("tag")
    if tag:
        c, m = cur.loc[tag], master.loc[tag]
        rec.update(tag=tag, asset_id=c["asset_id"], asset_class=c["asset_class"],
                   area=c["area_name"], criticality=c["criticality"],
                   health=int(c["health_score"]), state=c["condition_state"],
                   issue_code=c["issue_code"], issue=c["issue_summary"], urgency=c["urgency"],
                   next_check=c["recommended_next_check"], model_conf=float(c["model_confidence"]),
                   last_service=str(m["last_service_date"])[:10], connectivity=m["connectivity_profile"],
                   doc=c["knowledge_doc_id"], telemetry=telemetry(tag, c["asset_class"]),
                   **mapping_of(it, c["asset_class"]))
    else:
        rec.update(tag=None, label=it["_vlm"]["label"] if it["_vlm"] else it.get("_class") or "unmapped object", mapping="none", mapping_conf=0.0)
    if it["_vlm"]:
        rec["vlm"] = {k: it["_vlm"][k] for k in ("label", "asset_class", "confidence", "visual_evidence", "model")}
    out_items.append(rec)

# ---------------------------------------------------------------- copilot evidence, every asset
docs = xl["Knowledge_Docs"].set_index("doc_id")
TREND = {"CONTROL_VALVE": ("position_setpoint_pct", "position_actual_pct"),
         "PROCESS_PUMP": ("vibration_mm_s", "bearing_temp_c"),
         "ELECTRIC_MOTOR": ("vibration_mm_s", "bearing_temp_c"),
         "PRESSURE_TRANSMITTER": ("process_pressure_bar", "signal_quality_pct"),
         "VIBRATION_SENSOR": ("vibration_mm_s", "signal_quality_pct")}


def evidence_for(tg):
    c = cur.loc[tg]; cls = c["asset_class"]
    ev = xl["Alarms_Events"]; ev = ev[ev.tag == tg].sort_values("timestamp", ascending=False)
    wo = xl["Maintenance_History"]; wo = wo[wo.tag == tg].sort_values("opened_at", ascending=False)
    case = xl["Historical_Service_Cases"]
    case = case[(case.asset_class == cls) & (case.historical_issue_code == c["issue_code"])]
    dq = xl["Data_Quality"]; dq = dq[dq.tag == tg]
    t = tel[tel.tag == tg]
    shifts = []
    for k in KEY[cls]:
        s_ = t[k].dropna()
        if len(s_) > 60:
            a, b = float(s_.iloc[-48:].mean()), float(s_.iloc[:-48].mean())
            shifts.append(dict(metric=k, last24h=round(a, 2), before=round(b, 2),
                               change_pct=round((a - b) / (abs(b) or 1) * 100, 1)))
    if cls == "CONTROL_VALVE":
        dev = (t.position_setpoint_pct - t.position_actual_pct).abs()
        shifts.insert(0, dict(metric="position_error_pct", last24h=round(float(dev.iloc[-48:].mean()), 2),
                              before=round(float(dev.iloc[:-48].mean()), 2),
                              change_pct=round((dev.iloc[-48:].mean() / dev.iloc[:-48].mean() - 1) * 100, 1)))
    a_, b_ = TREND[cls]
    return dict(
        tag=tg, asset_id=c["asset_id"],
        state={k: (v.item() if hasattr(v, "item") else v) for k, v in c.items() if k != "last_updated"},
        shifts=shifts,
        trend=dict(a=a_, b=b_, A=t[a_].iloc[::2].round(2).fillna(0).tolist(), B=t[b_].iloc[::2].round(2).fillna(0).tolist()),
        events=ev.head(5)[["event_id", "timestamp", "event_type", "severity", "event_code", "status"]].astype(str).to_dict("records"),
        work_orders=wo.head(3)[["work_order_id", "opened_at", "action", "result"]].astype(str).to_dict("records"),
        cases=case[["case_id", "historical_interpretation", "recommended_next_check"]].head(2).to_dict("records"),
        doc=c["knowledge_doc_id"], doc_text=docs.loc[c["knowledge_doc_id"], "content"],
        data_quality=dq[["issue_id", "issue_type", "severity", "recommended_handling"]].to_dict("records"),
    )


evidence = {tg: evidence_for(tg) for tg in cur.index}

# ---------------------------------------------------------------- nav grid for in-browser A*
walk_u8 = G["walk"].astype(np.uint8)
clear_u8 = np.clip(G["clearance"] / G["cell"], 0, 255).astype(np.uint8)
navgrid = dict(origin=G["origin"].tolist(), cell=G["cell"], horiz=G["horiz"], shape=list(walk_u8.shape),
               walk=base64.b64encode(walk_u8.tobytes()).decode(),
               clear=base64.b64encode(clear_u8.tobytes()).decode())


def cell_grid():
    """The numbered 60 x 60 grid from the extraction pipeline: free width per cell, 0 when blocked."""
    import extract
    v, _, band, up_sign, heights = extract.load_scan(ply)
    sizes, grid_x, grid_z, span, _ = extract.free_grid(v, heights, band)
    return dict(cells=len(sizes), step=span / len(sizes), center_x=grid_x, center_z=grid_z, z_up=bool(up_sign > 0),
                sizes=base64.b64encode(sizes.astype(np.uint8).tobytes()).decode())


scene = dict(
    frame=dict(up_axis=nv.UP, down_sign=nv.DOWN_SIGN, floor=float(floor)),
    splat=dict(parts=[f"data/room_part{i}.glb" for i in range(len(parts))], bytes=len(raw)),
    items=out_items, target_id=target_box["id"],
    path=[nv.to_world(G, p, floor) for p in raw_path[::2]] + [nv.to_world(G, raw_path[-1], floor)],
    copilot=evidence, navgrid=navgrid,
)
if SEMANTIC:
    scene["cellgrid"] = cell_grid()
json.dump(scene, open(f"{web}/data/scene.json", "w"), default=str)
print(f"items {len(out_items)} (mapped {sum(1 for i in out_items if i['tag'])}), "
      f"target {target_box['id']} -> {target_box.get('tag')}, path {len(raw_path) * G['cell']:.1f} m")
