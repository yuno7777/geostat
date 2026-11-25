"""Opt-in smoke test with a pinned public Sen1Floods11 scene."""

from __future__ import annotations

import os
import shutil
import urllib.request
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.features import rasterize
from shapely.geometry import shape

from geostat import api, cli, db, exposure, geoio, inference, settings
from geostat.provenance import sha256_hex
from geostat.unet import SmallUNet

SCENE = "Spain_7387658"
RASTERS = {
    "S1": (
        1601573071280175,
        1509867,
        "2a69a686bdb5aa7ed0736d8a428a18fdf819f2b2862db75ff3e5f7b902b25207",
    ),
    "Label": (
        1601572623312222,
        19124,
        "d7f6c03457e5191733d46f9d52f8342af1d73cfbeafeeba100794443b01f79fd",
    ),
}
FIXTURE = Path(__file__).parent / "fixtures" / "spain_7387658_osm.json"


def stage_rasters() -> dict[str, Path]:
    cached = os.environ.get("GEOSTAT_SMOKE_DATA_DIR")
    base = settings.RAW_DIR / settings.DATASET_NAME / settings.DATASET_VERSION
    assets = {}
    files = []
    for layer, (generation, size, sha256) in RASTERS.items():
        folder = f"{layer}Hand"
        filename = f"{SCENE}_{folder}.tif"
        object_name = f"{settings.DATASET_PREFIX}/{folder}/{filename}"
        url = f"{settings.GCS_PUBLIC_URL}/{settings.DATASET_NAME}/{object_name}?generation={generation}"
        path = base / folder / filename
        path.parent.mkdir(parents=True)
        if cached:
            shutil.copyfile(Path(cached) / filename, path)
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read(size + 1))
        assert path.stat().st_size == size, f"unexpected size for {filename}"
        assert sha256_hex(path) == sha256, f"SHA-256 mismatch for {filename}"
        assets[layer] = path
        files.append({"name": url, "path": str(path), "size": size, "sha256": sha256})
    geoio.save_json(base / "MANIFEST.json", {"files": files})
    return assets


def register_smoke_checkpoint(path: Path) -> str:
    channels = [2, 4, 8, 16]
    model = SmallUNet(channels=channels)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    torch.nn.init.constant_(model.head.bias, 1.0)
    torch.save({"channels": channels, "state_dict": model.state_dict()}, path)
    with closing(db.connect()) as conn:
        with db.begin_run(conn, "train", config={"threshold": 0.5}) as run:
            db.upsert_model(
                conn,
                "smoke-constant",
                "unet",
                "{}",
                checkpoint_path=str(path),
                checkpoint_sha256=sha256_hex(path),
                created_by_run=run.run_id,
            )
        return run.run_id


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("GEOSTAT_REAL_DATA_SMOKE") != "1",
    reason="set GEOSTAT_REAL_DATA_SMOKE=1 to run the pinned public data smoke test",
)
def test_real_scene_cli_cog_geojson_and_api(api_client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", api.DB_PATH)
    monkeypatch.setattr(cli, "DB_PATH", api.DB_PATH)
    monkeypatch.setattr(inference, "RAW_DIR", settings.RAW_DIR)
    assets = stage_rasters()

    # A fresh fixture holds this test's empty SQLite registry and artifact roots.
    assert cli.main(["init-db"]) == 0
    assert cli.main(["ingest", "--chips", SCENE]) == 0
    assert cli.main(["validate", "--scene", SCENE]) == 0

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        train_id = register_smoke_checkpoint(tmp_path / "smoke.pth")
        assert cli.main(["analyze", "--scene", SCENE, "--run", train_id]) == 0
    finally:
        torch.set_num_threads(previous_threads)

    def unexpected_network(*args, **kwargs):
        pytest.fail("exposure smoke test attempted to refresh the pinned OSM fixture")

    payload = geoio.load_json(FIXTURE)
    cache = exposure.cache_dir(SCENE)
    geoio.save_json(cache / "response.json", payload)
    geoio.save_json(cache / "response_meta.json", payload["_geostat_meta"])
    monkeypatch.setattr(exposure.urllib.request, "urlopen", unexpected_network)
    assert cli.main(["exposure", "--scene", SCENE]) == 0

    def get_json(route):
        response = api_client.get(route)
        assert response.status_code == 200, (route, response.text)
        return response.json()

    scene = get_json(f"/api/scenes/{SCENE}")
    assert scene["s1_asset_id"]["sha256"] == RASTERS["S1"][2]
    assert scene["label_asset_id"]["sha256"] == RASTERS["Label"][2]
    assert scene["s1_asset_id"]["bbox_wgs84"] == payload["_geostat_meta"]["bbox_wgs84"]
    analysis = get_json(f"/api/analyses/{SCENE}")
    assert analysis["metrics"]["n_valid"] == 262057
    assert analysis["metrics"]["tp"] == 153006

    with (
        rasterio.open(assets["S1"]) as radar,
        rasterio.open(analysis["artifacts"]["prob_cog"]) as probability,
        rasterio.open(analysis["artifacts"]["mask_cog"]) as mask,
    ):
        for product, dtype in ((probability, "float32"), (mask, "uint8")):
            assert product.crs == radar.crs
            assert product.transform == radar.transform
            assert product.shape == radar.shape == (512, 512)
            assert product.dtypes == (dtype,)
            assert product.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        assert np.isnan(probability.nodata)
        assert mask.nodata == 255
        np.testing.assert_allclose(probability.read(1), 1 / (1 + np.exp(-1)), rtol=1e-6)
        transform = radar.transform
        expected = mask.read(1)
        assert np.all(expected == 1)

    reference = get_json(f"/api/layers/reference/{SCENE}")
    assert reference["type"] == "FeatureCollection"
    assert reference["features"]
    flood = get_json(f"/api/layers/flood/{SCENE}")
    assert flood == geoio.load_json(Path(analysis["artifacts"]["flood_geojson"]))
    geoms = [shape(feature["geometry"]) for feature in flood["features"]]
    assert geoms and all(geom.is_valid for geom in geoms)
    np.testing.assert_array_equal(
        rasterize(geoms, out_shape=expected.shape, transform=transform),
        expected,
    )

    summary = get_json(f"/api/exposure/{SCENE}")["summary"]
    assert summary == analysis["exposure"]
    assert summary["roads"]["segments_total"] == 2
    assert 0 < summary["roads"]["flooded_m"] < summary["roads"]["total_m"]
    assert summary["buildings"]["total"] == 1
    for kind, exposed_key in (("roads", "segments_flooded"), ("buildings", "flooded")):
        layer = get_json(f"/api/layers/{kind}/{SCENE}")
        flooded = sum(feature["properties"]["flooded"] for feature in layer["features"])
        assert summary[kind][exposed_key] == flooded > 0
    cog = Path(analysis["artifacts"]["prob_cog"])
    download = api_client.get(f"/files/{analysis['run_id']}/{cog.name}")
    assert download.status_code == 200
    assert download.content == cog.read_bytes()
