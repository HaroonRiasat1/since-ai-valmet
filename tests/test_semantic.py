import pytest

from backend.config import ROOT
from backend.semantic import Layer, SpecError, fit


@pytest.fixture(scope="module")
def layer():
    return Layer(ROOT / "data" / "semantic")


def test_every_declared_column_exists_in_the_data(layer):
    for entity in layer.entities.values():
        needs = [{"field": "tag", "op": "=", "value": "P-005"}] if entity.get("required_filters") else []
        out = layer.run({"entity": entity["name"], "dimensions": list(entity["cols"]), "filters": needs, "limit": 1})
        assert out["columns"] == list(entity["cols"])


def test_row_counts_match_the_pack(layer):
    expected = {"asset": 30, "asset_state": 30, "telemetry": 4320, "event": 1000, "work_order": 300, "service_case": 50,
                "knowledge_doc": 8, "data_quality_issue": 120, "asset_location": 30}
    for name, count in expected.items():
        spec = {"entity": name, "measures": ["count"]}
        if name == "telemetry":
            spec["filters"] = [{"field": "asset_class", "op": "in", "value": ["CONTROL_VALVE", "PROCESS_PUMP", "ELECTRIC_MOTOR", "PRESSURE_TRANSMITTER", "VIBRATION_SENSOR"]}]
        assert layer.run(spec)["rows"] == [[count]]


def test_every_join_is_proven(layer):
    joins = [j for e in layer.entities.values() for j in e.get("joins", [])]
    assert joins and all(j["verified"]["match_pct"] == 100.0 for j in joins)


def test_every_report_runs(layer):
    for name, recipe in layer.recipes.items():
        params = {"tag": "CV-003"} if recipe["parameters"] else {}
        assert layer.report(name, params)["rows"], name


def test_a_joined_filter_returns_the_expected_asset(layer):
    out = layer.run({"entity": "event", "measures": ["count"], "dimensions": ["tag"],
                     "filters": [{"field": "state.health_score", "op": "<", "value": 45}], "order_by": [{"field": "count", "dir": "desc"}]})
    assert {row[0] for row in out["rows"]} == {"P-005"}


def test_search_finds_the_right_entity(layer):
    assert layer.search("which pump has the highest vibration")[0]["name"] == "telemetry"
    assert any(h["name"] == "event" for h in layer.search("open alerts"))


def test_telemetry_requires_an_asset_or_class(layer):
    with pytest.raises(SpecError, match="Choose an asset"):
        layer.run({"entity": "telemetry", "measures": ["vibration_mm_s"]})


def test_unknown_column_suggests_the_real_one(layer):
    with pytest.raises(SpecError, match="health_score"):
        layer.run({"entity": "asset_state", "dimensions": ["health_scor"]})


def test_value_case_is_corrected_and_unknown_values_are_refused(layer):
    high = layer.run({"entity": "event", "measures": ["count"], "filters": [{"field": "severity", "op": "=", "value": "HIGH"}]})
    assert high["rows"][0][0] > 0
    with pytest.raises(SpecError, match="Values"):
        layer.run({"entity": "event", "measures": ["count"], "filters": [{"field": "severity", "op": "=", "value": "awful"}]})


@pytest.mark.parametrize("spec", [
    {"entity": 'asset"; DROP TABLE asset; --', "measures": ["count"]},
    {"entity": "asset", "dimensions": ['tag" FROM asset; --']},
    {"entity": "asset", "measures": [{"column": "service_interval_days", "agg": "sum(1)); --"}]},
    {"entity": "asset", "measures": ["count"], "order_by": [{"field": "tag; DROP TABLE asset"}]},
])
def test_injected_names_never_reach_sql(layer, spec):
    with pytest.raises(SpecError):
        layer.compile(spec)


def test_values_are_bound_not_inlined(layer):
    attack = "x' OR '1'='1"
    sql, params = layer.compile({"entity": "work_order", "measures": ["count"], "filters": [{"field": "work_order_id", "op": "=", "value": attack}]})
    assert attack not in sql and params == [attack]
    assert layer.run({"entity": "work_order", "measures": ["count"], "filters": [{"field": "work_order_id", "op": "=", "value": attack}]})["rows"] == [[0]]


def test_limit_is_capped(layer):
    assert layer.compile({"entity": "event", "dimensions": ["event_id"], "limit": 99999})[0].endswith("LIMIT 200")


def test_results_stay_small():
    big = {"columns": ["a"], "rows": [["x" * 100]] * 500}
    assert len(str(fit(big))) < 7000


def test_tag_in_an_id_filter_is_refused_with_the_real_values(layer):
    with pytest.raises(SpecError, match="ASSET-001"):
        layer.run({"entity": "work_order", "measures": ["count"], "filters": [{"field": "asset_id", "op": "=", "value": "P-005"}]})


def test_synonyms_resolve_to_columns(layer):
    out = layer.run({"entity": "work_order", "measures": ["downtime"], "dimensions": ["asset"], "order_by": [{"field": "downtime_h", "dir": "desc"}], "limit": 1})
    assert out["columns"] == ["asset", "sum_downtime_h"] and len(out["rows"]) == 1
