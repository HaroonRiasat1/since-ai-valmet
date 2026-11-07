import pytest
from fastapi.testclient import TestClient

from backend import app as application


@pytest.fixture
def client(scene, monkeypatch):
    monkeypatch.setattr(application, "get_scene", lambda: scene)
    return TestClient(application.app)


def test_health_lists_both_grids(client):
    assert client.get("/api/health").json()["grids"] == ["fine", "cells"]


def test_route_to_a_tag_on_each_grid(client):
    for grid in ("fine", "cells"):
        body = client.post("/api/route", json={"grid": grid, "position": [0.2, 1, 0.2], "target": "P-001"}).json()
        assert body["grid"] == grid and body["target"] == "P-001" and len(body["points"]) >= 2


def test_route_between_cell_numbers(client):
    body = client.post("/api/route", json={"grid": "cells", "start_cell": 91, "target_cell": 100}).json()
    assert body["end_cell"] == 100


def test_blocked_cell_gives_a_clear_error(client):
    response = client.post("/api/route", json={"grid": "cells", "start_cell": 96, "target": "P-001"})
    assert response.status_code == 422 and "blocked" in response.json()["error"]


def test_round_visits_unhealthy_assets_only(client):
    legs = client.post("/api/route/round", json={"grid": "fine"}).json()["legs"]
    assert [leg["target"] for leg in legs] == ["CV-001"]


def test_route_needs_a_target(client):
    assert client.post("/api/route", json={"grid": "fine"}).status_code == 422


def test_remove_unmapped_object(client, scene):
    scene.data["items"].append({"id": "obj_c", "center": [0.5, 0, 0.5], "size": [0.1, 0.1, 0.1]})
    scene.by_id["obj_c"] = scene.data["items"][-1]
    body = client.post("/api/objects/obj_c/remove").json()
    assert body["ok"] and body["id"] == "obj_c"
    assert "obj_c" not in scene.by_id
    assert client.get("/api/health").json()["objects"] == len(scene.items)


def test_remove_mapped_asset(client, scene):
    body = client.post("/api/objects/P-001/remove").json()
    assert body["tag"] == "P-001"
    assert "P-001" not in scene.by_tag
    tags = {a["tag"] for a in client.get("/api/assets").json()}
    assert "P-001" not in tags


def test_remove_unknown_is_404(client):
    assert client.post("/api/objects/no-such-id/remove").status_code == 404


def test_route_to_removed_object_fails(client, scene):
    client.post("/api/objects/P-001/remove")
    r = client.post("/api/route", json={"grid": "fine", "position": [0.2, 1, 0.2], "target": "P-001"})
    assert r.status_code == 422 and "unknown" in r.json()["error"]
