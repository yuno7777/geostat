import json

from geostat import settings
from geostat.cli import main


def test_doctor_json_fresh_workstation_is_non_destructive(tmp_path, monkeypatch, capsys):
    data = tmp_path / "new data"
    artifacts = tmp_path / "new artifacts"
    monkeypatch.setattr(settings, "DATA_DIR", data)
    monkeypatch.setattr(settings, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(settings, "DB_PATH", data / "geostat.sqlite3")

    assert main(["doctor", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"]
    assert {c["name"] for c in result["checks"]} == {
        "runtime",
        "gdal_cog",
        "data_storage",
        "artifact_storage",
        "registry",
        "package_data",
    }
    assert all(c["ok"] for c in result["checks"])
    assert not list(tmp_path.iterdir())


def test_doctor_reports_invalid_storage_and_registry(tmp_path, monkeypatch, capsys):
    file = tmp_path / "file"
    file.write_text("not SQLite", encoding="utf-8")
    monkeypatch.setattr(settings, "DATA_DIR", file / "child")
    monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "DB_PATH", file)

    assert main(["doctor", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"]
    assert {c["name"] for c in result["checks"] if not c["ok"]} == {"data_storage", "registry"}
    assert file.read_text(encoding="utf-8") == "not SQLite"
