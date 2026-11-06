"""
Local backend for the navigator: serves the web page and proxies the AI copilot to an
Featherless (OpenAI-compatible API). The key stays on this machine (environment
variable), never in the browser.

    export FEATHERLESS_API_KEY=...                  # enables the AI copilot
    export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct       # optional, any Featherless model id
    python3 server.py                               # open http://localhost:8000

POST /api/copilot/query  {"tag": "CV-003", "question": "optional follow-up", "nearby": ["P-003", ...]}
  -> {"answer": "<markdown>", "model": "...", "evidence_ids": [...]}
POST /api/vlm/label  {"box_id": "OBJ-015", "image": "data:image/jpeg;base64,..."}
  -> {"label", "asset_class", "confidence", "visual_evidence", "model"}; merged into data/vlm_labels.json,
     every crop kept as data/vlm_crops/<box_id>_<time>_<label>.jpg and logged in data/vlm_crops/index.csv.
     VLM_MODEL overrides the default Qwen3-VL model.
POST /api/vlm/detect {"run", "view_id", "image", "pose", "proj", "width", "height"}
  -> {"detections": [{"label", "asset_class", "bbox_2d" (0-1000), "confidence"}], "model"}; the frame and
     its camera are saved to data/vlm_views/<run>/<view_id>.jpg/.json for vlm_lift.py to lift into 3D.
No dependencies beyond the Python standard library.
"""
import base64, csv, json, os, re, sys, threading, time, urllib.request, urllib.error
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE = json.load(open(os.path.join(ROOT, "data", "scene.json")))
ITEMS = {i["tag"]: i for i in SCENE["items"] if i.get("tag")}
PROVIDER = "featherless"
KEY = os.environ.get("FEATHERLESS_API_KEY", "")
BASE = "https://api.featherless.ai/v1"
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
LABELS = os.path.join(ROOT, "data", "vlm_labels.json")
CROPS = os.path.join(ROOT, "data", "vlm_crops")
CROP_FIELDS = ["file", "box_id", "label", "asset_class", "confidence", "model", "at", "visual_evidence"]
LABELS_LOCK = threading.Lock()
VLM_CLASSES = ["CONTROL_VALVE", "PROCESS_PUMP", "ELECTRIC_MOTOR", "PRESSURE_TRANSMITTER", "VIBRATION_SENSOR",
               "OTHER_EQUIPMENT", "NOT_EQUIPMENT"]
VLM_PROMPT = """This is a render from a 3D Gaussian-splat scan of an industrial equipment room.
Identify the object inside the yellow rectangle (renders can be blurry; lower the confidence instead of guessing).
Answer with JSON only, no prose:
{"label": "<short name, e.g. 'gate valve with handwheel', 'centrifugal pump', 'electrical cabinet', 'ladder'>",
 "asset_class": "<one of: """ + ", ".join(VLM_CLASSES) + """>",
 "confidence": <0.0-1.0>,
 "visual_evidence": "<one sentence: what you see that supports this>"}
Any valve counts as CONTROL_VALVE (say 'manual' in the label if there is no actuator). Pipes, tanks, cabinets,
panels and boilers are OTHER_EQUIPMENT. Furniture, ladders, boxes, buckets, walls and clutter are NOT_EQUIPMENT."""

SYSTEM = """You are a field-service copilot for an industrial plant digital twin (Valmet x Since AI demo).
A technician has walked to an asset. Diagnose it using ONLY the evidence JSON provided.

Rules (from DOC-RAG-GUIDE and DOC-FIELD-SAFETY in the data pack):
- Name the asset. Cite evidence IDs inline in backticks (event IDs, work order IDs, case IDs, doc IDs, sheet names).
- State a confidence (low / medium / high) and say what lowers it (data-quality issues, the 3D mapping method and confidence in the evidence).
- Clearly separate OBSERVATION (what the data shows) from HYPOTHESIS (what might cause it).
- Recommend verification and escalation only. Never instruct autonomous physical action; remind about permits / lockout-tagout.
- If the evidence does not support an answer, say so. Do not invent values, IDs or procedures.

Format (markdown, concise, max ~220 words):
**Observation** (bullets) / **Hypothesis** (bullets) / **Next safe steps** (numbered) / **Confidence** (one line)."""


def evidence(tag, nearby):
    ev = dict(SCENE["copilot"][tag])
    it = ITEMS.get(tag, {})
    ev["latest_telemetry"] = it.get("telemetry")
    ev["mapping"] = {"method": it.get("mapping"), "confidence": it.get("mapping_conf")}
    ev.pop("trend", None)                       # raw series are long; shifts summarise them
    ev["nearby_assets"] = [
        {k: ITEMS[t].get(k) for k in ("tag", "asset_class", "state", "health", "issue_code")}
        for t in nearby if t in ITEMS and t != tag][:6]
    return ev


def ids_in(ev):
    out = [ev.get("asset_id"), ev.get("doc")]
    out += [e["event_id"] for e in ev.get("events", [])]
    out += [w["work_order_id"] for w in ev.get("work_orders", [])]
    out += [c["case_id"] for c in ev.get("cases", [])]
    out += [q["issue_id"] for q in ev.get("data_quality", [])]
    return [i for i in out if i]


def ask_llm(messages, model=MODEL, max_tokens=700):
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    text = data["choices"][0]["message"]["content"] or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # reasoning models
    return text, data.get("model", model)


DETECT_PROMPT = """This is a render from a 3D Gaussian-splat scan of an industrial equipment room.
Detect every piece of plant equipment. Return JSON only, a list (empty if none):
[{"label": "<short name>", "asset_class": "<one of: """ + ", ".join(VLM_CLASSES[:-1]) + """>",
  "bbox_2d": [x1, y1, x2, y2], "confidence": <0.0-1.0>}]
bbox_2d is in 0-1000 coordinates relative to image width and height.
Any valve is CONTROL_VALVE (say 'manual' in the label if it has no actuator). Tanks, cabinets, panels,
boilers and compressors are OTHER_EQUIPMENT. Skip pipes, walls, furniture, ladders, carts, boxes and
clutter, and skip anything too blurry to identify."""


def vlm_detect(req):
    """Detect equipment in one rendered frame; keep frame + camera so detections can be lifted to 3D."""
    run, vid, image = req.get("run", ""), req.get("view_id", ""), req.get("image", "")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,40}", x) for x in (run, vid)) or not image.startswith("data:image/jpeg;base64,"):
        raise ValueError("need run, view_id and a JPEG data URL")
    if len(req.get("pose", [])) != 16 or len(req.get("proj", [])) != 16:
        raise ValueError("need 4x4 pose and proj")
    text, model = ask_llm([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": DETECT_PROMPT}]}],
        model=VLM_MODEL, max_tokens=900)
    m = re.search(r"\[.*\]", text, re.S)
    try:
        raw = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        raw = []
    dets = []
    for d in raw if isinstance(raw, list) else []:
        b, cls = d.get("bbox_2d"), str(d.get("asset_class", "")).upper()
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)) or cls not in VLM_CLASSES[:-1]:
            continue
        dets.append(dict(label=str(d.get("label", ""))[:80], asset_class=cls, bbox_2d=[max(0, min(1000, v)) for v in b],
                         confidence=round(min(1.0, max(0.0, float(d.get("confidence") or 0.5))), 2)))
    folder = os.path.join(ROOT, "data", "vlm_views", run)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{vid}.jpg"), "wb") as f:
        f.write(base64.b64decode(image.split(",", 1)[1]))
    json.dump(dict(view_id=vid, pose=req["pose"], proj=req["proj"], width=req.get("width"), height=req.get("height"),
                   detections=dets, model=model, at=time.strftime("%Y-%m-%dT%H:%M:%S")),
              open(os.path.join(folder, f"{vid}.json"), "w"))
    return dict(detections=dets, model=model)


def vlm_label(box_id, image):
    """Ask the VLM what the outlined object is; store the answer and the crop for auditing."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", box_id or "") or not image.startswith("data:image/jpeg;base64,"):
        raise ValueError("need box_id and a JPEG data URL")
    text, model = ask_llm([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": VLM_PROMPT}]}],
        model=VLM_MODEL, max_tokens=250)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    cls = str(out.get("asset_class", "")).upper()
    rec = dict(label=str(out.get("label", "unknown"))[:80], asset_class=cls if cls in VLM_CLASSES else "NOT_EQUIPMENT",
               confidence=round(min(1.0, max(0.0, float(out.get("confidence") or 0))), 2),
               visual_evidence=str(out.get("visual_evidence", ""))[:300], model=model,
               at=time.strftime("%Y-%m-%dT%H:%M:%S"), raw=None if out else text[:300])
    slug = re.sub(r"[^a-z0-9]+", "-", rec["label"].lower()).strip("-")[:40] or "unknown"
    rec["image"] = f"{box_id}_{rec['at'].replace(':', '').replace('-', '')}_{slug}.jpg"
    with LABELS_LOCK:                           # every capture is kept; index.csv lists them all
        os.makedirs(CROPS, exist_ok=True)
        with open(os.path.join(CROPS, rec["image"]), "wb") as f:
            f.write(base64.b64decode(image.split(",", 1)[1]))
        index = os.path.join(CROPS, "index.csv")
        new = not os.path.exists(index)
        with open(index, "a", newline="") as f:
            w = csv.DictWriter(f, CROP_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(dict(rec, file=rec["image"], box_id=box_id))
        labels = json.load(open(LABELS)) if os.path.exists(LABELS) else {}
        labels[box_id] = rec
        json.dump(labels, open(LABELS, "w"), indent=1)
    return rec


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        if self.path == "/api/health":
            return self._json(200, {"ok": True, "llm": bool(KEY), "model": f"{PROVIDER}: {MODEL}", "vlm": VLM_MODEL})
        return super().do_GET()

    def do_POST(self):
        if self.path in ("/api/vlm/label", "/api/vlm/detect"):
            try:
                if not KEY:
                    return self._json(503, {"error": "FEATHERLESS_API_KEY is not set on the server"})
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/vlm/detect":
                    return self._json(200, vlm_detect(req))
                return self._json(200, vlm_label(req.get("box_id"), req.get("image", "")))
            except urllib.error.HTTPError as e:
                return self._json(502, {"error": f"{PROVIDER} {e.code}: {e.read().decode()[:300]}"})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if self.path != "/api/copilot/query":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            tag = req.get("tag")
            if tag not in SCENE["copilot"]:
                return self._json(400, {"error": f"unknown asset {tag}"})
            if not KEY:
                return self._json(503, {"error": "FEATHERLESS_API_KEY is not set on the server"})
            ev = evidence(tag, req.get("nearby", []))
            q = (req.get("question") or "").strip()[:500]
            user = (f"Evidence JSON for {tag}:\n```json\n{json.dumps(ev, default=str)}\n```\n\n"
                    + (f"Technician's question: {q}" if q else "The technician has just arrived. Give the diagnosis."))
            answer, model = ask_llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            return self._json(200, {"answer": answer, "model": model, "evidence_ids": ids_in(ev)})
        except urllib.error.HTTPError as e:
            return self._json(502, {"error": f"{PROVIDER} {e.code}: {e.read().decode()[:300]}"})
        except Exception as e:
            return self._json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "/api/" in msg or "error" in fmt.lower():
            sys.stderr.write(msg + "\n")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Navigator on http://localhost:{port}  (LLM: {f'on, {PROVIDER} {MODEL}' if KEY else 'off, set FEATHERLESS_API_KEY'})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
