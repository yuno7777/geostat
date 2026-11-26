from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import Future
from contextlib import closing

import rasterio
import torch

from geostat import batch, db, settings
from geostat.unet import SmallUNet


def test_batch_cli_resumes_repairs_and_retries(tmp_path, synthetic_raster_pair):
    data = tmp_path / "scene store"
    artifacts = tmp_path / "output store"
    raw = data / "raw" / settings.DATASET_NAME / settings.DATASET_VERSION
    (raw / "S1Hand").mkdir(parents=True)
    (raw / "LabelHand").mkdir(parents=True)
    for scene in ("Alpha_1", "Beta_2"):
        shutil.copyfile(synthetic_raster_pair[0], raw / "S1Hand" / f"{scene}_S1Hand.tif")
        shutil.copyfile(synthetic_raster_pair[1], raw / "LabelHand" / f"{scene}_LabelHand.tif")

    checkpoint = tmp_path / "fixed weights.pth"
    channels = (2, 4, 8, 16)
    model = SmallUNet(channels=channels)
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.constant_(model.head.bias, 1)
    torch.save({"channels": channels, "state_dict": model.state_dict()}, checkpoint)

    db_path = data / "geostat.sqlite3"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        db.upsert_study_area(conn, "test", "Test")
        for scene in ("Alpha_1", "Beta_2", "Missing_3"):
            db.upsert_scene(conn, scene, "test")
        with db.begin_run(conn, "train", config={"threshold": 0.5}, console=False) as train:
            db.upsert_model(
                conn,
                "batch-test",
                "unet",
                "{}",
                checkpoint_path=str(checkpoint),
                created_by_run=train.run_id,
            )

    summary = tmp_path / "summary.json"
    env = {
        **os.environ,
        "GEOSTAT_DATA_DIR": str(data),
        "GEOSTAT_ARTIFACTS_DIR": str(artifacts),
        "PYTHONIOENCODING": "utf-8",
    }

    def invoke(*args):
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "geostat.cli",
                "analyze-batch",
                "--run",
                train.run_id,
                "--tile-size",
                "64",
                "--summary",
                str(summary),
                *args,
            ],
            env=env,
            capture_output=True,
            encoding="utf-8",
            timeout=120,
        )
        assert summary.is_file(), process.stdout + process.stderr
        return process, {
            item["scene"]: item
            for item in json.loads(summary.read_text(encoding="utf-8"))["scenes"]
        }

    first, initial = invoke("--all", "--jobs", "2", "--retries", "1")
    assert first.returncode == 1, first.stdout + first.stderr
    assert [item["status"] for item in initial.values()] == ["completed", "completed", "failed"]
    assert initial["Missing_3"]["attempts"] == 2

    second, reused = invoke("--scene", "Beta_2", "--scene", "Alpha_1", "--scene", "Beta_2")
    assert second.returncode == 0, second.stdout + second.stderr
    assert list(reused) == ["Beta_2", "Alpha_1"]
    assert all(
        s["status"] == "reused" and s["run_id"] == initial[k]["run_id"] for k, s in reused.items()
    )

    alpha = raw / "S1Hand" / "Alpha_1_S1Hand.tif"
    with rasterio.open(alpha, "r+") as raster:
        pixel = raster.read(1)
        pixel[-1, -1] = -12.5
        raster.write(pixel, 1)
    alpha_old = initial["Alpha_1"]["run_id"]
    beta_old = initial["Beta_2"]["run_id"]
    (artifacts / "runs" / beta_old / "flood_Beta_2.geojson").unlink()
    third, updated = invoke("--scene", "Alpha_1", "--scene", "Beta_2", "--jobs", "2")
    assert third.returncode == 0, third.stdout + third.stderr
    assert [item["status"] for item in updated.values()] == ["completed", "completed"]
    assert updated["Alpha_1"]["run_id"] != alpha_old
    assert updated["Beta_2"]["run_id"] != beta_old
    with closing(db.connect(db_path)) as conn:
        for scene in ("Alpha_1", "Beta_2"):
            row = db.latest_run_kind(conn, "analyze", scene_id=scene)
            assert row["id"] == updated[scene]["run_id"]
            assert row["model_id"] == "batch-test"


def test_batch_interrupt_drains_active_scenes_and_leaves_rest_pending(monkeypatch, tmp_path):
    submitted = []

    class InlineExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, scene, *args):
            submitted.append(scene)
            future = Future()
            future.set_result(batch.SceneResult(scene, "completed", 1, "first-run"))
            return future

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(batch, "wait", interrupt)
    model = batch.BatchModel("model", tmp_path / "weights", "digest", 0.5, 64)
    results, cancelled = batch.analyze_batch(["A_1", "B_2"], model, jobs=1)
    assert cancelled
    assert submitted == ["A_1"]
    assert [r.status for r in results] == ["completed", "pending"]
