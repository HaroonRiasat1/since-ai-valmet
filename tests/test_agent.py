import json

import pytest

from backend import agent
from backend.llm import Provider, ProviderError

START = [0.1, 1.0, 0.1]
PROVIDER = Provider("stub", "Stub", "http://stub/v1", default_model="m")


def tool_call(tool, **arguments):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": tool, "arguments": json.dumps(arguments)}}]}, "model": "m"}


def text(content):
    return {"message": {"role": "assistant", "content": content}, "model": "m"}


def script(monkeypatch, *replies):
    queue = list(replies)
    seen = []

    def fake(_, messages, **kwargs):
        seen.append({"messages": list(messages), "tools": [t["function"]["name"] for t in kwargs.get("tools") or []]})
        return queue.pop(0)

    monkeypatch.setattr(agent, "chat", fake)
    return seen


def ask(scene, message, provider=PROVIDER):
    return agent.respond(scene, scene.grid("cells"), START, message, provider)


def last_tool_output(seen):
    return seen[-1]["messages"][-1]["content"]


# ---------- the model reads every request first

def test_clear_request_still_goes_to_the_model(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("route_to", tag="CV-001"), text("Heading to CV-001."))
    out = ask(scene, "go to CV-001")
    assert seen and out["mode"] == "agent" and out["action"]["target"] == "CV-001"


def test_model_picks_the_destination_and_code_computes_the_path(scene, monkeypatch):
    script(monkeypatch, tool_call("route_to", tag="P-001"), text("Heading to P-001."))
    out = ask(scene, "take me to whatever is leaking")
    assert out["tools"] == ["route_to"] and out["action"]["length"] > 0
    assert out["action"]["points"][0][0] == pytest.approx(START[0], abs=0.25)   # START is at the wall: the sprite stands where it fits


def test_data_can_choose_a_destination(scene, monkeypatch):
    spec = dict(entity="asset_state", dimensions=["tag", "health_score"], order_by=[{"field": "health_score", "dir": "asc"}], limit=1)
    seen = script(monkeypatch, tool_call("run_query", **spec), tool_call("route_to", tag="P-001"), text("P-005 has the lowest health, 44."))
    out = ask(scene, "take me to the asset with the lowest health score")
    assert out["tools"] == ["run_query", "route_to"] and out["table"]["rows"][0][0] == "P-005"
    assert {"run_query", "route_to", "diagnose"} <= set(seen[0]["tools"])


def test_trip_tools_are_offered_only_for_trips(scene, monkeypatch):
    seen = script(monkeypatch, text("Hello."))
    ask(scene, "what should we fix first?")
    assert "route_to" not in seen[0]["tools"] and "run_report" in seen[0]["tools"]


def test_a_question_cannot_move_the_technician(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("route_to", tag="P-001"), text("Done."))
    out = ask(scene, "what should we fix first?")
    assert out["action"] is None and "unknown tool" in last_tool_output(seen)


def test_bad_tag_goes_back_to_the_model(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("route_to", tag="CV-999"), text("That asset does not exist."))
    out = ask(scene, "take me to the leaking one")
    assert "unknown asset" in last_tool_output(seen) and out["action"] is None


def test_trip_without_a_route_falls_back_to_rules(scene, monkeypatch):
    script(monkeypatch, text("Sure, walk north."))
    out = ask(scene, "go to the valve")
    assert out["mode"] == "rules" and out["action"]["target"] == "CV-001"


def test_diagnosis_is_returned_with_its_citations(scene, monkeypatch):
    scene.copilot["CV-001"] = {"asset_id": "ASSET-001"}
    script(monkeypatch, tool_call("diagnose", tag="CV-001"))
    monkeypatch.setattr(agent.copilot, "chat", lambda *_, **__: text("Drift, see `ASSET-001`."))
    out = ask(scene, "why is CV-001 failing?")
    assert out["reply"] == "Drift, see `ASSET-001`." and out["evidence_ids"] == ["ASSET-001"] and not out["unverified_citations"]


def test_invented_citation_in_a_diagnosis_is_flagged(scene, monkeypatch):
    scene.copilot["CV-001"] = {"asset_id": "ASSET-001"}
    script(monkeypatch, tool_call("diagnose", tag="CV-001"))
    monkeypatch.setattr(agent.copilot, "chat", lambda *_, **__: text("See `EVT-99999`."))
    assert ask(scene, "why did CV-001 fail?")["unverified_citations"] == ["EVT-99999"]


# ---------- answers must come from the tools

def test_data_question_runs_a_spec_and_returns_the_table(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("run_query", entity="event", measures=["count"], filters=[{"field": "status", "op": "=", "value": "open"}]), text("There are open events."))
    out = ask(scene, "how many open alerts are there?")
    assert out["tools"] == ["run_query"] and out["table"]["rows"][0][0] > 0
    assert "layer_hits" in seen[0]["messages"][0]["content"]


def test_invented_figure_is_replaced_by_the_rows(scene, monkeypatch):
    script(monkeypatch, tool_call("run_report", name="health_by_class"), text("PROCESS_PUMP_007 had 45 hours of downtime on P-999."))
    out = ask(scene, "which class is healthiest?")
    assert "45" not in out["reply"] and "P-999" not in out["reply"] and out["reply"].startswith("8 rows")


def test_answer_using_the_rows_is_kept(scene, monkeypatch):
    script(monkeypatch, tool_call("run_report", name="health_by_class"), text("Transmitters lead at 96 on 3 assets."))
    assert ask(scene, "which class is healthiest?")["reply"] == "Transmitters lead at 96 on 3 assets."


def test_rounded_route_length_is_accepted(scene, monkeypatch):
    script(monkeypatch, tool_call("route_to", tag="P-001"), text("Heading to P-001."))
    length = f"{ask(scene, 'go to P-001')['action']['length']:.1f}"
    script(monkeypatch, tool_call("route_to", tag="P-001"), text(f"Heading to P-001, {length} m."))
    assert ask(scene, "go to P-001")["reply"] == f"Heading to P-001, {length} m."


def test_invented_distance_is_replaced(scene, monkeypatch):
    script(monkeypatch, tool_call("route_to", tag="P-001"), text("P-001 is 99 m away."))
    assert ask(scene, "go to P-001")["reply"].startswith("Routing to P-001")


def test_answer_with_no_rows_says_so(scene, monkeypatch):
    spec = dict(entity="event", dimensions=["tag"], filters=[{"field": "event_id", "op": "=", "value": "none"}])
    script(monkeypatch, tool_call("run_query", **spec), text("Lots of events."))
    assert ask(scene, "list those events")["reply"] == "No records match that."


def test_no_tool_means_no_answer(scene, monkeypatch):
    script(monkeypatch, text("There are 12 open alerts."))
    assert ask(scene, "hello")["reply"] == agent.NOT_FOUND


def test_bad_spec_goes_back_to_the_model(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("run_query", entity="event", dimensions=["nope"]), text("Could not do that."))
    ask(scene, "list the alerts")
    assert "has no column" in seen[1]["messages"][-1]["content"]


def test_tool_call_written_as_text_still_runs(scene, monkeypatch):
    written = json.dumps({"name": "run_report", "arguments": {"name": "priority_list"}})
    script(monkeypatch, text(written), text("Here is the list."))
    out = ask(scene, "what should we fix first?")
    assert out["tools"] == ["run_report"] and out["table"]["rows"]


def test_repeated_calls_stop_the_loop(scene, monkeypatch):
    seen = script(monkeypatch, *[tool_call("describe", name="event")] * 6)
    ask(scene, "tell me about events")
    assert len(seen) == 2


def test_unknown_tool_is_refused(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("delete_everything"), text("Sorry."))
    ask(scene, "hello")
    assert "unknown tool" in seen[1]["messages"][-1]["content"]


# ---------- without a model, the rules still navigate

def test_provider_failure_falls_back_to_rules(scene, monkeypatch):
    def down(*_, **__):
        raise ProviderError("offline")

    monkeypatch.setattr(agent, "chat", down)
    out = ask(scene, "go to the worst asset")
    assert out["mode"] == "rules" and out["action"]["target"] == "CV-001" and "lowest health" in out["reply"]


def test_provider_failure_on_a_question_is_explained(scene, monkeypatch):
    def down(*_, **__):
        raise ProviderError("offline")

    monkeypatch.setattr(agent, "chat", down)
    assert "only navigate" in ask(scene, "why is it vibrating?")["reply"]


def test_no_provider_still_navigates_and_explains(scene):
    assert ask(scene, "go to CV-001", None)["action"]["target"] == "CV-001"
    assert "only navigate" in ask(scene, "why is it vibrating?", None)["reply"]


def test_glossary_works_without_a_model(scene):
    assert "friction" in ask(scene, "what does stiction mean?", None)["reply"]


def test_area_narrows_a_rule(scene):
    assert ask(scene, "take me to the one in the Utility Island", None)["action"]["target"] == "P-001"


def test_unknown_tag_in_a_rule_moves_nobody(scene):
    assert ask(scene, "go to CV-999", None)["action"] is None


def test_rules_ignore_questions(scene):
    assert agent.rule_based(agent.Turn(scene, scene.grid("cells"), START), "what does CV-001 do?") is None


def test_search_sorts_by_walking_distance(scene):
    rows = agent.Turn(scene, scene.grid("cells"), START).search_assets()
    assert [r["tag"] for r in rows] == ["CV-001", "P-001"]
    assert rows[0]["walk"] <= rows[1]["walk"]


def test_worst_looks_past_the_nearest_assets(scene, monkeypatch):
    monkeypatch.setattr(agent, "WALK_CHECKS", 1)
    assert agent.Turn(scene, scene.grid("cells"), [1.7, 1.0, 0.3]).search_assets(order="health")[0]["tag"] == "CV-001"


def test_any_tag_family_is_recognised():
    """New tag families (handwheel valves, cabinets, actuators...) must route without a code change."""
    for tag in ("HV-005", "EC-001", "VA-001", "CV-003", "P-002"):
        assert agent.TAG.search(f"go to {tag} please").group(0).upper() == tag
    assert agent.TAG.search("EVT-00012 and WO-00031") is None   # record ids are not tags
