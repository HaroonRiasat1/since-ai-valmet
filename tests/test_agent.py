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

    def fake(_, messages, **__):
        seen.append(list(messages))
        return queue.pop(0)

    monkeypatch.setattr(agent, "chat", fake)
    return seen


def ask(scene, message, provider=PROVIDER):
    return agent.respond(scene, scene.grid("fine"), START, message, provider)


def test_model_picks_the_destination_and_code_computes_the_path(scene, monkeypatch):
    script(monkeypatch, tool_call("route_to", tag="P-001"), text("Heading to P-001."))
    out = ask(scene, "take me to whatever is leaking")
    assert out["mode"] == "agent" and out["tools"] == ["route_to"]
    assert out["action"]["target"] == "P-001" and out["action"]["length"] > 0
    assert out["action"]["points"][0][0] == pytest.approx(START[0], abs=0.1)


def test_bad_tag_goes_back_to_the_model_and_moves_nobody(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("route_to", tag="CV-999"), text("That asset does not exist."))
    out = ask(scene, "take me to the leaking one")
    assert out["action"] is None
    assert "unknown asset" in seen[1][-1]["content"]


def test_unknown_tool_is_refused(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("delete_everything"), text("Sorry."))
    assert ask(scene, "hello")["reply"] == agent.NOT_FOUND
    assert "unknown tool" in seen[1][-1]["content"]


def test_clear_requests_never_reach_the_model(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("route_to", tag="P-001"))
    out = ask(scene, "go to the valve")
    assert out["mode"] == "rules" and out["action"]["target"] == "CV-001" and not seen


def test_unknown_tag_in_a_clear_request_moves_nobody(scene):
    assert ask(scene, "go to CV-999")["action"] is None


def test_provider_failure_falls_back_to_rules(scene, monkeypatch):
    def down(*_, **__):
        raise ProviderError("offline")

    monkeypatch.setattr(agent, "chat", down)
    out = ask(scene, "go to the worst asset")
    assert out["mode"] == "rules" and out["action"]["target"] == "CV-001" and "lowest health" in out["reply"]


def test_worst_looks_past_the_nearest_assets(scene, monkeypatch):
    monkeypatch.setattr(agent, "WALK_CHECKS", 1)
    assert agent.Turn(scene, scene.grid("fine"), [1.7, 1.0, 0.3]).search_assets(order="health")[0]["tag"] == "CV-001"


def test_provider_failure_on_a_question_is_explained(scene, monkeypatch):
    def down(*_, **__):
        raise ProviderError("offline")

    monkeypatch.setattr(agent, "chat", down)
    assert "only navigate" in ask(scene, "why is it vibrating?")["reply"]


def test_no_provider_explains_what_still_works(scene):
    out = ask(scene, "why is it vibrating?", None)
    assert out["action"] is None and "only navigate" in out["reply"]


def test_search_sorts_by_walking_distance(scene):
    rows = agent.Turn(scene, scene.grid("fine"), START).search_assets()
    assert [r["tag"] for r in rows] == ["CV-001", "P-001"]
    assert rows[0]["walk"] <= rows[1]["walk"]


def test_rules_ignore_questions(scene):
    assert agent.rule_based(agent.Turn(scene, scene.grid("fine"), START), "what does CV-001 do?") is None


def test_invented_citation_is_flagged(scene, monkeypatch):
    scene.copilot["CV-001"] = {"asset_id": "ASSET-001"}
    monkeypatch.setattr(agent.copilot, "chat", lambda *_, **__: text("See `EVT-99999`."))
    assert ask(scene, "why did CV-001 fail?")["unverified_citations"] == ["EVT-99999"]


def test_question_about_one_asset_uses_the_grounded_diagnosis(scene, monkeypatch):
    scene.copilot["CV-001"] = {"asset_id": "ASSET-001"}
    monkeypatch.setattr(agent.copilot, "chat", lambda *_, **__: text("Drift, see `ASSET-001`."))
    out = ask(scene, "what is wrong with CV-001?")
    assert out["tools"] == ["diagnose"] and out["evidence_ids"] == ["ASSET-001"] and not out["unverified_citations"]


def test_area_narrows_a_clear_request(scene):
    assert ask(scene, "take me to the one in the Utility Island")["action"]["target"] == "P-001"


def test_data_question_runs_a_spec_and_returns_the_table(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("run_query", entity="event", measures=["count"], filters=[{"field": "status", "op": "=", "value": "open"}]), text("There are open events."))
    out = ask(scene, "how many open alerts are there?")
    assert out["tools"] == ["run_query"] and out["table"]["rows"][0][0] > 0
    assert "layer_hits" in seen[0][0]["content"]


def test_bad_spec_goes_back_to_the_model(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("run_query", entity="event", dimensions=["nope"]), text("Could not do that."))
    assert ask(scene, "list the alerts")["reply"] == agent.NOT_FOUND
    assert "has no column" in seen[1][-1]["content"]


def test_tool_call_written_as_text_still_runs(scene, monkeypatch):
    written = json.dumps({"name": "run_report", "arguments": {"name": "priority_list"}})
    script(monkeypatch, text(written), text("Here is the list."))
    out = ask(scene, "what should we fix first?")
    assert out["tools"] == ["run_report"] and out["table"]["rows"]


def test_a_question_cannot_move_the_technician(scene, monkeypatch):
    seen = script(monkeypatch, tool_call("route_to", tag="P-001"), text("Done."))
    out = ask(scene, "what should we fix first?")
    assert out["action"] is None and "unknown tool" in seen[1][-1]["content"]


def test_invented_figure_is_replaced_by_the_rows(scene, monkeypatch):
    script(monkeypatch, tool_call("run_report", name="health_by_class"), text("PROCESS_PUMP_007 had 45 hours of downtime on P-999."))
    out = ask(scene, "which class is healthiest?")
    assert "45" not in out["reply"] and "P-999" not in out["reply"] and out["reply"].startswith("5 rows")


def test_answer_using_the_rows_is_kept(scene, monkeypatch):
    script(monkeypatch, tool_call("run_report", name="health_by_class"), text("Transmitters lead at 96 on 3 assets."))
    assert ask(scene, "which class is healthiest?")["reply"] == "Transmitters lead at 96 on 3 assets."


def test_answer_with_no_rows_says_so(scene, monkeypatch):
    spec = dict(entity="event", dimensions=["tag"], filters=[{"field": "event_id", "op": "=", "value": "none"}])
    script(monkeypatch, tool_call("run_query", **spec), text("Lots of events."))
    assert ask(scene, "list those events")["reply"] == "No records match that."


def test_repeated_calls_stop_the_loop(scene, monkeypatch):
    seen = script(monkeypatch, *[tool_call("describe", name="event")] * 6)
    ask(scene, "tell me about events")
    assert len(seen) == 2


def test_glossary_question_needs_no_model(scene):
    assert "friction" in ask(scene, "what does stiction mean?", None)["reply"]
