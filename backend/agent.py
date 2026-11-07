"""Chat agent for any OpenAI-compatible model.

Clear navigation requests are resolved by code, because a small model tested on the real scene invented
tags and ignored "nearest". The model handles everything else through checked tools, and code computes every path.
"""
import json
import math
import re
from dataclasses import asdict

from backend import copilot
from backend.llm import Provider, ProviderError, chat
from backend.routing import RouteError, nearest_next
from backend.semantic import SpecError, fit, get_layer

MAX_TURNS = 6
WALK_CHECKS = 8
MODEL_ROWS = 15
DIAGNOSE_WORDS = re.compile(r"\b(why|wrong|diagnos\w*|problem|fail\w*|fault\w*|issue|status|condition|happening)\b", re.I)
TAG = re.compile(r"\b(CV|PT|VS|P|M)-?(\d{3})\b", re.I)
NAV_WORDS = re.compile(r"\b(go|goes|take|walk|route|navigate|bring|lead|head|show me the way|get me)\b", re.I)
ROUND_WORDS = re.compile(r"\b(round|inspect (all|every)|all (the )?(faulty|faults|unhealthy|problems))\b", re.I)
WORST_WORDS = re.compile(r"\b(worst|faulty|fault|broken|failing|degraded|problem|unhealthy|lowest health)\b", re.I)
CLASS_WORDS = {"valve": "CONTROL_VALVE", "pump": "PROCESS_PUMP", "motor": "ELECTRIC_MOTOR",
               "transmitter": "PRESSURE_TRANSMITTER", "sensor": "VIBRATION_SENSOR"}

SYSTEM = """You are the field-service assistant inside an industrial digital twin. The technician is walking in a scanned equipment room.
Use tools for every fact about assets. Never invent tags, distances, numbers or readings.

Answering data questions:
- Questions about counts, lists, trends, history or comparisons use run_report if a report fits, otherwise run_query.
- Use the entity and column names given in the layer hints. If unsure, call describe.
- The app shows the result table itself. Do not retype it: state the finding in one or two sentences, using numbers from the rows.
- If a tool returns an error, read it, fix the spec and try again.

Choosing a destination:
- An exact tag means that asset.
- Use only tags returned by search_assets, copied exactly.
- "nearest" or no qualifier means the first row of search_assets.
- "worst" or "faulty" means the first row of search_assets with order=health. Say which rule you used.
- Prefer assets with matched=true. If you pick matched=false, say its position is unconfirmed.
- Only route_to and inspection_round move the technician. Call them only when asked to go somewhere.

Diagnosing:
- Call get_evidence and use only that evidence. Cite the IDs from its allowed_citations in backticks.
- Separate observation from hypothesis, state a confidence, and recommend verification and escalation only.
- Remind about permits and lockout-tagout. Never instruct autonomous physical action.

Reply in at most 120 words."""

NAV_TOOLS = {"search_assets", "route_to", "inspection_round"}
DATA_TOOLS = {"search_layer", "describe", "run_query", "run_report"}
NOT_FOUND = "I could not answer that from the plant data, so I am not guessing. Try naming an asset, a signal or a period."

TOOLS = [
    {"type": "function", "function": {"name": "search_assets", "description": "List assets with health and state, nearest walk first or lowest health first.", "parameters": {
        "type": "object", "properties": {
            "asset_class": {"type": "string", "enum": sorted(CLASS_WORDS.values())},
            "unhealthy_only": {"type": "boolean"},
            "area": {"type": "string", "description": "Plant area, exactly as shown in the area field of a row"},
            "order": {"type": "string", "enum": ["walk", "health"]}}}}},
    {"type": "function", "function": {"name": "route_to", "description": "Compute a collision-free path from the technician to an asset and start it.", "parameters": {
        "type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}}},
    {"type": "function", "function": {"name": "inspection_round", "description": "Route through every non-healthy asset, nearest next.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "search_layer", "description": "Find entities, columns, reports and glossary terms for a business phrase.", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "describe", "description": "Describe an entity, entity.column, report or glossary term.", "parameters": {
        "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "run_query", "description": "Run a query spec on the plant data. There is no SQL: name an entity and its columns.", "parameters": {
        "type": "object", "required": ["entity"], "properties": {
            "entity": {"type": "string"},
            "measures": {"type": "array", "description": "Measure names, or 'count'. Use {column, agg} to change the aggregate.", "items": {}},
            "dimensions": {"type": "array", "items": {"type": "string"}, "description": "Columns to group by or list, including '<join>.<column>'."},
            "filters": {"type": "array", "items": {"type": "object", "required": ["field", "op", "value"], "properties": {
                "field": {"type": "string"},
                "op": {"type": "string", "enum": ["=", "!=", ">", "<", ">=", "<=", "like", "in", "between"]},
                "value": {"description": "Text, number, or a list for in and between."}}}},
            "order_by": {"type": "array", "items": {"type": "object", "properties": {"field": {"type": "string"}, "dir": {"type": "string", "enum": ["asc", "desc"]}}}},
            "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "run_report", "description": "Run a verified report by name.", "parameters": {
        "type": "object", "properties": {"name": {"type": "string"}, "params": {"type": "object"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "get_evidence", "description": "Telemetry, events, work orders, cases and data-quality records for one asset.", "parameters": {
        "type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}}},
]


class Turn:
    """The tools for one request. Everything the model asks for is checked here."""

    def __init__(self, scene, grid, position, allowed=NAV_TOOLS | DATA_TOOLS):
        self.scene, self.grid, self.position, self.allowed = scene, grid, position, allowed
        self.action = None
        self.records = []
        self.table = None
        self.layer = get_layer()

    def walk(self, item):
        try:
            return round(self.grid.route(self.position, item).length, 2)
        except RouteError:
            return None

    def search_assets(self, asset_class=None, unhealthy_only=False, order="walk", area=None):
        found = [a for a in self.scene.assets() if (not asset_class or a["asset_class"] == asset_class)
                 and (not unhealthy_only or a["state"] != "healthy") and (not area or a.get("area") == area)]
        # ponytail: walking distance only for the closest by straight line, since each route costs up to 150 ms on the fine grid
        found.sort(key=(lambda a: a["health"]) if order == "health" else (lambda a: math.dist(a["center"], self.position)))
        rows = [{"tag": a["tag"], "asset_class": a["asset_class"], "area": a.get("area"), "state": a["state"], "health": a["health"], "issue": a.get("issue"),
                 "matched": a.get("matched", False), "walk": self.walk(a)} for a in found[:WALK_CHECKS]]
        rows = [r for r in rows if r["walk"] is not None]
        return rows if order == "health" else sorted(rows, key=lambda r: r["walk"])

    def route_to(self, tag):
        route = self.grid.route(self.position, self.scene.item(tag))
        self.action = {"type": "route", "target": tag, **asdict(route)}
        return {"target": tag, "length": round(route.length, 2)}

    def inspection_round(self):
        todo = [a for a in self.scene.assets() if a["state"] != "healthy"]
        legs = [{"target": item["tag"], **asdict(leg)} for item, leg in nearest_next(self.grid, self.position, todo)]
        self.action = {"type": "round", "legs": legs}
        return {"stops": [leg["target"] for leg in legs]}

    def get_evidence(self, tag):
        if tag not in self.scene.copilot:
            raise RouteError(f"unknown asset {tag}")
        record = copilot.evidence(self.scene, tag, [])
        self.records.append(record)
        return {**record, "allowed_citations": copilot.evidence_ids(record)}

    def search_layer(self, query):
        return self.layer.search(query)

    def describe(self, name):
        return self.layer.describe(name)

    def _table(self, out):
        """The page shows every row. The model gets enough of them to state the finding."""
        self.table = out
        return fit({"columns": out["columns"], "rows": out["rows"][:MODEL_ROWS], "row_count": out["row_count"]})

    def run_query(self, **spec):
        return self._table(self.layer.run(spec))

    def run_report(self, name, params=None):
        return self._table(self.layer.report(name, params))

    def hints(self, message):
        """What the layer knows about this question, found in code so a small model starts in the right place."""
        hits = self.layer.search(message)
        entity = next((h["name"] for h in hits if h["kind"] == "entity"), None)
        names = [f"{h['kind']} {h['name']}" for h in hits]
        return json.dumps({"layer_hits": names, "best_entity": self.layer.describe(entity) if entity else None})

    def call(self, name, arguments):
        """Run one tool call. Failures go back to the model so it can correct itself."""
        if name not in self.allowed:   # a question can never move the technician
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, name)(**json.loads(arguments or "{}"))
        except (RouteError, SpecError, TypeError, ValueError) as error:
            return {"error": str(error)}


def rule_based(turn: Turn, message: str) -> str | None:
    """Understand a navigation request without a model. Returns None if it is not one."""
    if ROUND_WORDS.search(message):
        stops = turn.inspection_round()["stops"]
        return f"Inspection round of {len(stops)} stops: {', '.join(stops)}."
    tag = TAG.search(message)
    if tag:
        wanted = f"{tag.group(1).upper()}-{tag.group(2)}"
        if wanted not in turn.scene.by_tag:
            return f"There is no asset {wanted}." if NAV_WORDS.search(message) else None
        if NAV_WORDS.search(message):
            return f"Routing to {wanted}, {turn.route_to(wanted)['length']:.1f} m."
        return None
    if not NAV_WORDS.search(message):
        return None
    lowered = message.lower()
    asset_class = next((cls for word, cls in CLASS_WORDS.items() if word in lowered), None)
    worst = bool(WORST_WORDS.search(message))
    area = next((a for a in {i.get("area") for i in turn.scene.assets()} if a and a.lower() in lowered), None)
    if not asset_class and not worst and not area:
        return None
    rows = turn.search_assets(asset_class, unhealthy_only=worst, order="health" if worst else "walk", area=area)
    choices = [r for r in rows if r["matched"]] or rows
    if not choices:
        return "No reachable asset fits that request."
    pick = choices[0]
    length = turn.route_to(pick["tag"])["length"]
    rule = f"lowest health ({pick['health']})" if worst else "nearest walk"
    note = "" if pick["matched"] else " Its position is unconfirmed."
    return f"Routing to {pick['tag']}, {length:.1f} m. Chosen by {rule}.{note}"


def text_tool_call(content: str) -> dict | None:
    """Small models sometimes write a tool call as JSON text instead of making one."""
    found = re.search(r"\{.*\}", content or "", re.S)
    try:
        call = json.loads(found.group(0)) if found else None
    except ValueError:
        return None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    return {"id": "text", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call.get("arguments", call.get("parameters", {})))}}


def table_summary(table: dict) -> str:
    if not table["rows"]:
        return "No records match that."
    first = ", ".join(f"{c} {v}" for c, v in zip(table["columns"], table["rows"][0]))
    return f"{table['row_count']} rows. First: {first}."


def grounded(reply: str, turn: Turn, used: list[str]) -> str:
    """A data answer must come from the rows. Anything else is replaced, because models can invent figures."""
    if turn.table is None:
        return reply if reply and "describe" in used else NOT_FOUND
    cells = {str(v) for row in turn.table["rows"] for v in row} | {str(turn.table["row_count"])}
    numbers = {float(n) for n in re.findall(r"(?<![\w-])\d+(?:\.\d+)?", reply)}
    known = {float(c) for c in cells if re.fullmatch(r"-?\d+(?:\.\d+)?", c)}
    tags = {m.group(0).upper() for m in TAG.finditer(reply)}
    if not turn.table["rows"] or not reply or numbers - known or tags - cells:
        return table_summary(turn.table)
    return reply


def named_asset(scene, message: str) -> str | None:
    tag = TAG.search(message)
    wanted = f"{tag.group(1).upper()}-{tag.group(2)}" if tag else None
    return wanted if wanted in scene.copilot else None


def result(turn: Turn, reply: str, mode: str, model: str | None, used: list[str]) -> dict:
    ids = [i for record in turn.records for i in copilot.evidence_ids(record)]
    return {"reply": reply, "mode": mode, "model": model, "tools": used, "action": turn.action, "table": turn.table,
            "evidence_ids": ids, "unverified_citations": copilot.unknown_citations(reply, ids)}


def respond(scene, grid, position, message: str, provider: Provider | None = None, model: str | None = None, history: list[dict] | None = None) -> dict:
    turn = Turn(scene, grid, position)
    planned = rule_based(turn, message)
    if planned:
        return result(turn, planned, "rules", None, [])
    term = turn.layer.term(message)
    if term:
        return result(turn, f"**{term['term']}**: {term['meaning']}", "rules", None, [])
    if provider is None or not provider.configured:
        return offline(turn, "No AI provider is ready.")
    tag = named_asset(scene, message)
    if tag and DIAGNOSE_WORDS.search(message):   # this path cites the asset's records
        try:
            found = copilot.diagnose(scene, provider, tag, message, model=model)
        except ProviderError as error:
            return offline(turn, str(error))
        turn.records.append(copilot.evidence(scene, tag, []))
        return result(turn, found["answer"], "agent", found["model"], ["diagnose"])
    navigating = bool(NAV_WORDS.search(message))
    turn.allowed = NAV_TOOLS if navigating else DATA_TOOLS
    tools = [t for t in TOOLS if t["function"]["name"] in turn.allowed]
    messages = [{"role": "system", "content": f"{SYSTEM}\n\nLayer hints for this question: {turn.hints(message)}"}, *(history or []),
                {"role": "user", "content": f"{message}\n\n(Technician position: {[round(v, 2) for v in position]})"}]
    used, seen = [], set()
    try:
        for _ in range(MAX_TURNS):
            answer = chat(provider, messages, model=model, tools=tools)
            reply, model = answer["message"], answer["model"]
            written = text_tool_call(reply["content"]) if not reply.get("tool_calls") else None
            calls = reply.get("tool_calls") or ([written] if written else [])
            fresh = [c for c in calls if (c["function"]["name"], c["function"].get("arguments")) not in seen]
            if not fresh:   # no new call: the model is done, or going round in circles
                reply = {**reply, "content": "" if calls else reply["content"]}
                break
            calls = fresh
            reply["tool_calls"] = calls
            messages.append(reply)
            for call in calls:
                used.append(call["function"]["name"])
                seen.add((call["function"]["name"], call["function"].get("arguments")))
                output = turn.call(call["function"]["name"], call["function"].get("arguments"))
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(output, default=str)})
    except ProviderError as error:
        return offline(turn, str(error))
    if navigating:
        return result(turn, reply["content"] or "I could not finish that request.", "agent", model, used)
    return result(turn, grounded(reply["content"], turn, used), "agent", model, used)


def offline(turn: Turn, why: str) -> dict:
    return result(turn, f"{why} Without a model I can only navigate, for example: go to CV-003, go to the nearest pump, go to the worst valve.", "rules", None, [])
