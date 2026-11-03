from __future__ import annotations

from fastapi.testclient import TestClient

from geostat.api import create_app


def test_api_health():
    client = TestClient(create_app())
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "version" in data


def test_api_scenes():
    client = TestClient(create_app())
    res = client.get("/api/scenes")
    assert res.status_code == 200
    data = res.json()
    assert "scenes" in data
    assert isinstance(data["scenes"], list)


def test_api_scene_not_found():
    client = TestClient(create_app())
    res = client.get("/api/scenes/nonexistent_scene_123")
    assert res.status_code == 404
    assert res.json()["detail"] == "scene not found"


def test_api_analyses_known_scene():
    client = TestClient(create_app())
    res = client.get("/api/analyses/Bolivia_103757")
    assert res.status_code == 200
    data = res.json()
    assert data["scene_id"] == "Bolivia_103757"
    assert "layers" in data


def test_frontend_served():
    client = TestClient(create_app())
    res = client.get("/")
    assert res.status_code == 200
    assert "GeoStat" in res.text
