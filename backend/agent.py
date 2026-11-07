"""Chat agent for any OpenAI-compatible model. The model picks the destination, code computes every path."""
import json
import math
import re
from dataclasses import asdict

from backend import copilot
from backend.llm import Provider, ProviderError, chat
from backend.routing import RouteError, nearest_next

MAX_TURNS = 6
WALK_CHECKS = 8
TAG = re.compile(r"\b(CV|PT|VS|P|M)-?(\d{3})\b", re.I)
NAV_WORDS = re.compile(r"\b(go|goes|take|walk|route|navigate|bring|lead|head|show me the way|get me)\b", re.I)
ROUND_WORDS = re.compile(r"\b(round|inspect (all|every)|all (the )?(faulty|faults|unhealthy|problems))\b", re.I)
WORST_WORDS = re.compile(r"\b(worst|faulty|fault|broken|failing|degraded|problem|unhealthy|lowest health)\b", re.I)
CLASS_WORDS = {"valve": "CONTROL_VALVE", "pump": "PROCESS_PUMP", "motor": "ELECTRIC_MOTOR",
               "transmitter": "PRESSURE_TRANSMITTER", "sensor": "VIBRATION_SENSOR"}

SYSTEM = """You are the field-service assistant inside an industrial digital twin. The technician is walking in a scanned equipment room.
Use tools for every fact about assets. Never invent tags, distances or readings.

Choosing a destination:
- An exact tag means that asset.
- "nearest" or no qualifier means the shortest walking distance from search_assets.
- "worst" or "faulty" means the lowest health. Say which rule you used.
- Prefer assets with matched=true. If you pick matched=false, say its position is unconfirmed.
- Only route_to and inspection_round move the technician. Call them only when asked to go somewhere.

Diagnosing:
- Call get_evidence and use only that evidence. Cite record IDs in backticks.
- Separate observation from hypothesis, state a confidence, and recommend verification and escalation only.
- Remind about permits and lockout-tagout. Never instruct autonomous physical action.

Reply in at most 120 words."""

TOOLS = [
    {"type": "function", "function": {"name": "search_assets", "description": "List assets, nearest walk first, with health and state.", "parameters": {
        "type": "object", "properties": {
            "asset_class": {"type": "string", "enum": sorted(CLASS_WORDS.values())},
            "unhealthy_only": {"type": "boolean"}}}}},
    {"type": "function", "function": {"name": "route_to", "description": "Compute a collision-free path from the technician to an asset and start it.", "parameters": {
        "type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}}},
    {"type": "function", "function": {"name": "inspection_round", "description": "Route through every non-healthy asset, nearest next.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_evidence", "description": "Telemetry, events, work orders, cases and data-quality records for one asset.", "parameters": {
        "type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}}},
]


class Turn:
    """The tools for one request. Everything the model asks for is checked here."""

    def __init__(self, scene, grid, position):
        self.scene, self.grid, self.position = scene, grid, position
        self.action = None
        self.records = []

    def walk(self, item):
        try:
            return round(self.grid.route(self.position, item).length, 2)
        except RouteError:
            return None

    def search_assets(self, asset_class=None, unhealthy_only=False):
        found = [a for a in self.scene.assets() if (not asset_class or a["asset_class"] == asset_class)
                 and (not unhealthy_only or a["state"] != "healthy")]
        # ponytail: walking distance only for the closest by straight line, since each route costs up to 150 ms on the fine grid
        found.sort(key=lambda a: math.dist(a["center"], self.position))
        rows = [{"tag": a["tag"], "asset_class": a["asset_class"], "state": a["state"], "health": a["health"], "issue": a.get("issue"),
                 "matched": a.get("matched", False), "walk": self.walk(a)} for a in found[:WALK_CHECKS]]
        return sorted([r for r in rows if r["walk"] is not None], key=lambda r: r["walk"])

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
        return record

    def call(self, name, arguments):
        """Run one tool call. Failures go back to the model so it can correct itself."""
        if name not in {t["function"]["name"] for t in TOOLS}:
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, name)(**json.loads(arguments or "{}"))
        except (RouteError, TypeError, ValueError) as error:
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
    if not asset_class and not worst:
        return None
    rows = turn.search_assets(asset_class, unhealthy_only=worst)
    choices = [r for r in rows if r["matched"]] or rows
    if not choices:
        return "No reachable asset fits that request."
    pick = min(choices, key=lambda r: r["health"]) if worst else choices[0]
    length = turn.route_to(pick["tag"])["length"]
    rule = f"lowest health ({pick['health']})" if worst else "nearest walk"
    note = "" if pick["matched"] else " Its position is unconfirmed."
    return f"Routing to {pick['tag']}, {length:.1f} m. Chosen by {rule}.{note}"


def result(turn: Turn, reply: str, mode: str, model: str | None, used: list[str]) -> dict:
    ids = [i for record in turn.records for i in copilot.evidence_ids(record)]
    return {"reply": reply, "mode": mode, "model": model, "tools": used, "action": turn.action,
            "evidence_ids": ids, "unverified_citations": copilot.unknown_citations(reply, ids)}


def respond(scene, grid, position, message: str, provider: Provider | None = None, model: str | None = None, history: list[dict] | None = None) -> dict:
    turn = Turn(scene, grid, position)
    if provider is None or not provider.configured:
        return offline(turn, message, "No AI provider is ready.")
    messages = [{"role": "system", "content": SYSTEM}, *(history or []),
                {"role": "user", "content": f"{message}\n\n(Technician position: {[round(v, 2) for v in position]})"}]
    used = []
    try:
        for _ in range(MAX_TURNS):
            answer = chat(provider, messages, model=model, tools=TOOLS)
            reply, model = answer["message"], answer["model"]
            calls = reply.get("tool_calls") or []
            if not calls:
                break
            messages.append(reply)
            for call in calls:
                used.append(call["function"]["name"])
                output = turn.call(call["function"]["name"], call["function"].get("arguments"))
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(output, default=str)})
    except ProviderError as error:
        return offline(turn, message, str(error))
    if not used:   # models without tool support answer in text, so navigation still has to work
        planned = rule_based(turn, message)
        if planned:
            return result(turn, planned, "rules", model, [])
    return result(turn, reply["content"] or "I could not finish that request.", "agent", model, used)


def offline(turn: Turn, message: str, why: str) -> dict:
    planned = rule_based(turn, message)
    if planned:
        return result(turn, planned, "rules", None, [])
    return result(turn, f"{why} Without a model I can only navigate, for example: go to CV-003, go to the nearest pump, go to the worst valve.", "rules", None, [])
