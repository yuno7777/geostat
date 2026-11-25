from __future__ import annotations

from contextlib import closing

from geostat import api, db


def test_api_health(api_client):
    res = api_client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "version" in data


def test_api_scenes(api_client):
    res = api_client.get("/api/scenes")
    assert res.status_code == 200
    data = res.json()
    assert "scenes" in data
    assert isinstance(data["scenes"], list)


def test_api_scene_not_found(api_client):
    res = api_client.get("/api/scenes/nonexistent_scene_123")
    assert res.status_code == 404
    assert res.json()["detail"] == "scene not found"


def test_api_analyses_known_scene(api_client):
    with closing(db.connect(api.DB_PATH)) as conn:
        db.upsert_study_area(conn, "test", "Test study area")
        db.upsert_scene(conn, "Bolivia_103757", "test")
    res = api_client.get("/api/analyses/Bolivia_103757")
    assert res.status_code == 200
    data = res.json()
    assert data["scene_id"] == "Bolivia_103757"
    assert "layers" in data


def test_frontend_served(api_client):
    res = api_client.get("/")
    assert res.status_code == 200
    assert "GeoStat" in res.text
