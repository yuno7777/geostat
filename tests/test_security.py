from __future__ import annotations

from contextlib import closing

from geostat import api, db


def test_files_path_traversal_blocked(api_client):
    # Upward traversal
    res = api_client.get("/files/run-123/..%2f..%2fetc%2fpasswd")
    assert res.status_code in (400, 404)

    res = api_client.get("/files/run-123/..\\secret.txt")
    assert res.status_code in (400, 404)

    # Hidden files with leading dot
    res = api_client.get("/files/run-123/.env")
    assert res.status_code == 400


def test_files_unregistered_file_blocked(api_client):
    # Assuming analyze run exists
    res = api_client.get("/files/analyze-20261103T091830Z/unregistered.txt")
    assert res.status_code in (403, 404)


def test_registered_file_is_confined_to_storage(api_client, tmp_path):
    from geostat import geoio

    inside = api.ARTIFACTS_DIR / "inside.json"
    outside = tmp_path / "outside.json"
    geoio.save_json(inside, {})
    geoio.save_json(outside, {})
    with closing(db.connect(api.DB_PATH)) as conn:
        with db.begin_run(conn, "test", console=False) as run:
            run.add_artifact("inside", inside)
            run.add_artifact("outside", outside)
    assert api_client.get(f"/files/{run.run_id}/inside.json").status_code == 200
    assert api_client.get(f"/files/{run.run_id}/outside.json").status_code == 403
