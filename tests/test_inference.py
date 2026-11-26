from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from shapely.geometry import shape

from geostat import geoio
from geostat.inference import infer_scene_prob_map
from geostat.unet import SmallUNet


def test_prob_map_stitching(tmp_path: Path, monkeypatch):
    # Setup synthetic scene in monkeypatched RAW_DIR
    import geostat.inference as inf
    import geostat.settings as settings

    monkeypatch.setattr(settings, "RAW_DIR", tmp_path)
    monkeypatch.setattr(inf, "RAW_DIR", tmp_path)
    base = tmp_path / settings.DATASET_NAME / settings.DATASET_VERSION / "S1Hand"
    base.mkdir(parents=True, exist_ok=True)
    s1_path = base / "scene_test_S1Hand.tif"

    # Create 512x512 raster
    transform = rasterio.transform.from_origin(0, 0, 0.001, 0.001)
    arr = np.random.uniform(-25, -10, (2, 512, 512)).astype(np.float32)
    with rasterio.open(
        s1_path,
        "w",
        driver="GTiff",
        height=512,
        width=512,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(arr)

    model = SmallUNet(in_ch=2, out_ch=1, channels=(8, 16, 32, 64))
    prob, valid, stats = infer_scene_prob_map(model, "scene_test", tile_size=256)

    assert prob.shape == (512, 512)
    assert valid.shape == (512, 512)
    assert stats["tiles"] == 4
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_border_tiles_and_cog_nodata(tmp_path, monkeypatch):
    from contextlib import closing

    from pyproj import CRS

    from geostat import db, inference, settings
    from geostat.unet import SmallUNet

    raws = tmp_path / "raw"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(inference, "RAW_DIR", raws)
    monkeypatch.setattr(settings, "runs_dir", lambda: artifacts)
    base = raws / settings.DATASET_NAME / settings.DATASET_VERSION / "S1Hand"
    base.mkdir(parents=True)
    src = base / "edge_S1Hand.tif"
    samples = np.full((2, 19, 11), -13, dtype=np.float32)
    samples[:, 0, 0] = -9999
    transform = rasterio.Affine(50, 0, 1000, 0, -50, 4000)
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        height=19,
        width=11,
        count=2,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999,
    ) as dest:
        dest.write(samples)
    model = SmallUNet(channels=(2, 4, 8, 16))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    torch.nn.init.constant_(model.head.bias, 1.0)
    weights = tmp_path / "weights.pth"
    torch.save({"channels": [2, 4, 8, 16], "state_dict": model.state_dict()}, weights)

    with closing(db.connect(tmp_path / "registry.sqlite3")) as conn:
        db.init_db(conn)
        with db.begin_run(conn, "analyze", console=False) as run:
            result = inference.export_scene_flood(
                conn,
                run,
                "edge",
                weights,
                tile_size=9,
                make_label_comparison=False,
            )
    assert result["stats"]["tiles"] == 6
    assert result["n_valid_px"] == result["n_water_pred_px"] == 208
    assert result["nan_masked_prob_is_nan_outside_valid"]
    with rasterio.open(result["prob_cog"]) as ds:
        assert CRS.from_user_input(ds.crs) == CRS.from_epsg(3857)
        assert ds.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        probs = ds.read(1, masked=True)
        assert probs.mask.sum() == 1
        assert probs.mask[0, 0]
    with rasterio.open(result["mask_cog"]) as ds:
        assert ds.read(1, masked=True).mask.sum() == 1
    flood = geoio.load_json(Path(result["geojson"]))["features"]
    assert 0 < shape(flood[0]["geometry"]).bounds[0] < 1


def test_failed_polygon_export_leaves_no_published_products(tmp_path, monkeypatch):
    from contextlib import closing

    import pytest

    from geostat import db, inference, settings

    raws = tmp_path / "raw"
    root = tmp_path / "staged outputs"
    monkeypatch.setattr(inference, "RAW_DIR", raws)
    monkeypatch.setattr(settings, "runs_dir", lambda: root)
    scene = raws / settings.DATASET_NAME / settings.DATASET_VERSION / "S1Hand"
    scene.mkdir(parents=True)
    with rasterio.open(
        scene / "broken_S1Hand.tif",
        "w",
        driver="GTiff",
        dtype="float32",
        width=11,
        height=19,
        count=2,
        crs="EPSG:4326",
        transform=rasterio.Affine(0.01, 0, 70, 0, -0.01, 20),
    ) as dest:
        dest.write(np.full((2, 19, 11), -13, dtype=np.float32))

    def broken_export(*args):
        raise geoio.GeoError("simulated vector write failure")

    monkeypatch.setattr(geoio, "write_water_geojson", broken_export)
    with closing(db.connect(tmp_path / "registry.sqlite3")) as conn:
        db.init_db(conn)
        weights = tmp_path / "weights.pth"
        model = SmallUNet(channels=(2, 4, 8, 16))
        torch.save({"channels": [2, 4, 8, 16], "state_dict": model.state_dict()}, weights)
        with pytest.raises(geoio.GeoError, match="simulated vector write failure"):
            with db.begin_run(conn, "analyze", console=False) as run:
                inference.export_scene_flood(
                    conn,
                    run,
                    "broken",
                    weights,
                    tile_size=9,
                    make_label_comparison=False,
                )
        assert db.get_run(conn, run.run_id)["status"] == "failed"
    assert list(root.iterdir()) == []
