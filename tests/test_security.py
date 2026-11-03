from __future__ import annotations

from fastapi.testclient import TestClient

from geostat.api import create_app


def test_files_path_traversal_blocked():
    client = TestClient(create_app())
    # Upward traversal
    res = client.get("/files/run-123/..%2f..%2fetc%2fpasswd")
    assert res.status_code in (400, 404)

    res = client.get("/files/run-123/..\\secret.txt")
    assert res.status_code in (400, 404)

    # Hidden files with leading dot
    res = client.get("/files/run-123/.env")
    assert res.status_code == 400


def test_files_unregistered_file_blocked():
    client = TestClient(create_app())
    # Assuming analyze run exists
    res = client.get("/files/analyze-20261103T091830Z/unregistered.txt")
    assert res.status_code in (403, 404)
