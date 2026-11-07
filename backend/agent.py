"""Chat agent for any OpenAI-compatible model.

The model reads every request first and chooses its tools: navigation, the data semantic layer and the
grounded diagnosis. Code computes every path, checks every name, and replaces any figure no tool returned.
Rules take over only when there is no model, the provider fails, or a trip request ends without a route.
"""
import json
import logging
import math
import re
from dataclasses import asdict

from backend import copilot
from backend.llm import Provider, ProviderError, chat
from backend.routing import RouteError, nearest_next
from backend.semantic import SpecError, fit, get_layer

log = logging.getLogger(__name__)
MAX_TURNS = 8
WALK_CHECKS = 8
MODEL_ROWS = 15
TAG = re.compile(r"\b(CV|PT|VS|P|M)-?(\d{3})\b", re.I)
NAV_WORDS = re.compile(r"\b(go|goes|take|walk|route|navigate|bring|lead|head|show me the way|get me)\b", re.I)
ROUND_WORDS = re.compile(r"\b(round|inspect (all|every)|all (the )?(faulty|faults|unhealthy|problems))\b", re.I)
WORST_WORDS = re.compile(r"\b(worst|faulty|fault|broken|failing|degraded|problem|unhealthy|lowest health)\b", re.I)
NUMBER = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?(?![\w-])")
CLASS_MATCH = (
    ("handwheel", "HANDWHEEL_VALVE"),
    ("actuator", "VALVE_ACTUATOR"),
    ("cabinet", "ELECTRICAL_CABINET"),
    ("valve", "CONTROL_VALVE"),
    ("pump", "PROCESS_PUMP"),
    ("motor", "ELECTRIC_MOTOR"),
    ("transmitter", "PRESSURE_TRANSMITTER"),
    ("sensor", "VIBRATION_SENSOR"),
)
CLASS_WORDS = dict(CLASS_MATCH)
ASSET_CLASS_ENUM = sorted({cls for _, cls in CLASS_MATCH})

SYSTEM = """You are the field-service assistant inside an industrial digital twin. The technician is walking in a scanned equipment room.
First work out what the technician wants, then use tools. Never invent tags, distances, numbers or readings.

Going somewhere:
- An exact tag: route_to that tag.
- A type, "nearest", "worst" or an area: search_assets, then route_to a tag from its rows. Nearest is the first row
  with order=walk, worst the first row with order=health. Prefer matched=true and say if a position is unconfirmed.
- A destination described by plant data, such as the highest vibration, the most downtime or open alerts:
  run_query or run_report to find the tag, then route_to it.
- Every non-healthy asset: inspection_round.
- route_to and inspection_round are only for requests to go somewhere.

Diagnosing:
- When asked what is wrong with one asset, why it fails or what to check, call diagnose with its tag. Its answer is
  shown to the technician as it is, so do not repeat it.

Data questions:
- Counts, lists, trends, history and comparisons use run_report if a report fits, otherwise run_query.
- Use the entity and column names in the layer hints, or call describe. Meanings of terms come from describe.
- The app shows the result table. Do not retype it: give the finding in one or two sentences with numbers from the rows.
- If a tool returns an error, read it, fix the call and try again.

Reply in at most 80 words, and say which rule or data chose a destination."""

NAV_TOOLS = {"search_assets", "route_to", "inspection_round"}
DATA_TOOLS = {"search_layer", "describe", "run_query", "run_report", "diagnose"}
NOT_FOUND = "I could not answer that from the plant data, so I am not guessing. Try naming an asset, a signal or a period."

TOOLS = [
    {"type": "function", "function": {"name": "search_assets", "description": "List assets with health and state, nearest walk first or lowest health first.", "parameters": {
        "type": "object", "properties": {
            "asset_class": {"type": "string", "enum": ASSET_CLASS_ENUM},
            "unhealthy_only": {"type": "boolean"},
            "area": {"type": "string", "description": "Plant area, exactly as shown in the area field of a row"},
            "order": {"type": "string", "enum": ["walk", "health"]}}}}},
    {"type": "function", "function": {"name": "route_to", "description": "Compute a collision-free path from the technician to an asset and start walking.", "parameters": {
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
    {"type": "function", "function": {"name": "diagnose", "description": "Grounded diagnosis of one asset from its records, with citations.", "parameters": {
        "type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}}},
]


class Turn:
    """The tools for one request. Everything the model asks for is checked here."""

    def __init__(self, scene, grid, position, provider=None, model=None, allowed=NAV_TOOLS | DATA_TOOLS):
        self.scene, self.grid, self.position, self.allowed = scene, grid, position, allowed
        self.provider, self.model = provider, model
        self.action = self.table = self.diagnosis = None
        self.records = []
        self.facts = []   # every tool output, which is all a reply may quote
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
        item = self.scene.item(tag)
        route = self.grid.route(self.position, item)
        self.action = {"type": "route", "target": item.get("tag") or item["id"], **asdict(route)}
        return {"target": self.action["target"], "length": round(route.length, 2)}

    def inspection_round(self):
        todo = [a for a in self.scene.assets() if a["state"] != "healthy"]
        legs = [{"target": item["tag"], **asdict(leg)} for item, leg in nearest_next(self.grid, self.position, todo)]
        self.action = {"type": "round", "legs": legs}
        return {"stops": [leg["target"] for leg in legs]}

    def diagnose(self, tag):
        tag = tag.upper()
        if tag not in self.scene.copilot:
            raise RouteError(f"unknown asset {tag}")
        found = copilot.diagnose(self.scene, self.provider, tag, model=self.model)
        self.records.append(copilot.evidence(self.scene, tag, []))
        self.diagnosis = found["answer"]
        return {"shown_to_technician": True}

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
        """What the layer knows about this question, found in code so the model starts in the right place."""
        hits = self.layer.search(message)
        entity = next((h["name"] for h in hits if h["kind"] == "entity"), None)
        return json.dumps({"layer_hits": [f"{h['kind']} {h['name']}" for h in hits], "best_entity": self.layer.describe(entity) if entity else None})

    def call(self, name, arguments):
        """Run one tool call. Failures go back to the model so it can correct itself."""
        if name not in self.allowed:   # a question can never move the technician
            output = {"error": f"unknown tool {name}"}
        else:
            try:
                output = getattr(self, name)(**json.loads(arguments or "{}"))
            except (RouteError, SpecError, TypeError, ValueError) as error:
                output = {"error": str(error)}
        self.facts.append(json.dumps(self.table if name in ("run_query", "run_report") and self.table else output, default=str))
        return output


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
    asset_class = next((cls for word, cls in CLASS_MATCH if word in lowered), None)
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


def fallback_reply(turn: Turn) -> str:
    """What code can say from the tool results alone."""
    if turn.action and turn.action["type"] == "route":
        return f"Routing to {turn.action['target']}, {turn.action['length']:.1f} m."
    if turn.action:
        return f"Inspection round of {len(turn.action['legs'])} stops."
    if turn.table is not None:
        if not turn.table["rows"]:
            return "No records match that."
        first = ", ".join(f"{c} {v}" for c, v in zip(turn.table["columns"], turn.table["rows"][0]))
        return f"{turn.table['row_count']} rows. First: {first}."
    return NOT_FOUND


def grounded(reply: str, turn: Turn, message: str) -> str:
    """Every figure and tag in a reply must come from a tool result. Anything else is replaced, because models invent figures."""
    if not reply or not turn.facts or (turn.table is not None and not turn.table["rows"] and turn.action is None):
        return fallback_reply(turn)
    facts = " ".join(turn.facts) + " " + message
    known = [float(n) for n in NUMBER.findall(facts)]
    tags = {m.group(0).upper() for m in TAG.finditer(facts)}

    def quoted(number: str) -> bool:   # "6.0 m" may quote a length of 5.98
        places = len(number.split(".")[1]) if "." in number else 0
        return any(round(k, places) == float(number) for k in known)

    if not all(quoted(n) for n in NUMBER.findall(reply)) or any(m.group(0).upper() not in tags for m in TAG.finditer(reply)):
        return fallback_reply(turn)
    return reply


def result(turn: Turn, reply: str, mode: str, model: str | None, used: list[str]) -> dict:
    ids = [i for record in turn.records for i in copilot.evidence_ids(record)]
    log.info("agent mode=%s model=%s tools=%s rows=%s", mode, model, used, turn.table["row_count"] if turn.table else None)
    return {"reply": reply, "mode": mode, "model": model, "tools": used, "action": turn.action, "table": turn.table,
            "evidence_ids": ids, "unverified_citations": copilot.unknown_citations(reply, ids)}


def rules(turn: Turn, message: str, why: str) -> dict:
    """Without a model: navigation and glossary still work."""
    planned = rule_based(turn, message)
    if planned:
        return result(turn, planned, "rules", None, [])
    term = turn.layer.term(message)
    if term:
        return result(turn, f"**{term['term']}**: {term['meaning']}", "rules", None, [])
    return result(turn, f"{why} Without a model I can only navigate, for example: go to CV-003, go to the nearest pump, go to the worst valve.", "rules", None, [])


def respond(scene, grid, position, message: str, provider: Provider | None = None, model: str | None = None, history: list[dict] | None = None) -> dict:
    trip = bool(NAV_WORDS.search(message) or ROUND_WORDS.search(message))
    turn = Turn(scene, grid, position, provider, model, DATA_TOOLS | (NAV_TOOLS if trip else set()))
    if provider is None or not provider.configured:
        return rules(turn, message, "No AI provider is ready.")
    tools = [t for t in TOOLS if t["function"]["name"] in turn.allowed]
    messages = [{"role": "system", "content": f"{SYSTEM}\n\nLayer hints for this question: {turn.hints(message)}"}, *(history or []),
                {"role": "user", "content": f"{message}\n\n(Technician position: {[round(v, 2) for v in position]})"}]
    used, seen, reply = [], set(), {"content": ""}
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
            reply["tool_calls"] = fresh
            messages.append(reply)
            for call in fresh:
                used.append(call["function"]["name"])
                seen.add((call["function"]["name"], call["function"].get("arguments")))
                output = turn.call(call["function"]["name"], call["function"].get("arguments"))
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(output, default=str)})
            if turn.diagnosis:   # the diagnosis is the answer, with its own citations
                return result(turn, turn.diagnosis, "agent", model, used)
    except ProviderError as error:
        if turn.action or turn.table is not None:
            return result(turn, fallback_reply(turn), "agent", model, used)
        return rules(turn, message, str(error))
    if trip and turn.action is None:   # the model did not route, so the rules try before giving up
        planned = rule_based(turn, message)
        if planned:
            return result(turn, planned, "rules", model, used)
    return result(turn, grounded(reply.get("content") or "", turn, message), "agent", model, used)
