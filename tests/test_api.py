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
