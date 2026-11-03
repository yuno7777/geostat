from __future__ import annotations

import pytest

from geostat import geoio
from geostat.tiling import (
    TilingConfig,
    TilingError,
    build_manifest,
    manifest_sha256,
    scene_tiles,
    split_disjointness_check,
    split_of,
)


def test_split_of():
    cfg = TilingConfig(name="test_cfg", val_events=["Ghana"], test_events=["Bolivia"])
    assert split_of("Bolivia", cfg) == "test"
    assert split_of("Ghana", cfg) == "val"
    assert split_of("Spain", cfg) == "train"


def test_scene_tiles_calculation():
    meta = geoio.RasterMeta(
        path="dummy.tif",
        crs="EPSG:4326",
        width=512,
        height=512,
        band_count=2,
        dtype="float32",
        nodata=float("nan"),
        has_nan_nodata=True,
        resolution_x=0.0001,
        resolution_y=0.0001,
        transform=(0.0001, 0, 0, 0, -0.0001, 0),
        bounds=(0, -0.0512, 0.0512, 0),
    )
    cfg = TilingConfig(name="test", tile_size=256, stride=256)
    tiles = scene_tiles(meta, cfg)
    assert len(tiles) == 4
    assert tiles[0]["row"] == 0 and tiles[0]["col"] == 0
    assert tiles[3]["row"] == 1 and tiles[3]["col"] == 1


def test_split_disjointness():
    cfg = TilingConfig(name="test", val_events=["Ghana"], test_events=["Bolivia"])
    meta = geoio.RasterMeta(
        path="dummy.tif", crs="EPSG:4326", width=256, height=256, band_count=2,
        dtype="float32", nodata=float("nan"), has_nan_nodata=True, resolution_x=1,
        resolution_y=1, transform=(1, 0, 0, 0, -1, 0), bounds=(0, -256, 256, 0)
    )
    scenes = [
        {"id": "Bolivia_101", "event": "Bolivia", "meta": meta},
        {"id": "Ghana_202", "event": "Ghana", "meta": meta},
        {"id": "Spain_303", "event": "Spain", "meta": meta},
    ]
    manifest = build_manifest(cfg, scenes, code_revision={"source_tree_sha256": "dummy"})
    # Should not raise
    split_disjointness_check(manifest)

    # Test that overlapping events raise error
    with pytest.raises(TilingError):
        bad_cfg = TilingConfig(name="bad", val_events=["Bolivia"], test_events=["Bolivia"])
        build_manifest(bad_cfg, scenes, code_revision={"source_tree_sha256": "dummy"})


def test_manifest_sha_is_stable():
    cfg = TilingConfig(name="test", val_events=["Ghana"], test_events=["Bolivia"])
    meta = geoio.RasterMeta(
        path="dummy.tif", crs="EPSG:4326", width=256, height=256, band_count=2,
        dtype="float32", nodata=float("nan"), has_nan_nodata=True, resolution_x=1,
        resolution_y=1, transform=(1, 0, 0, 0, -1, 0), bounds=(0, -256, 256, 0)
    )
    scenes = [{"id": "Bolivia_101", "event": "Bolivia", "meta": meta}]
    m1 = build_manifest(cfg, scenes, code_revision={"source_tree_sha256": "dummy"})
    m2 = build_manifest(cfg, scenes, code_revision={"source_tree_sha256": "dummy"})
    assert manifest_sha256(m1) == manifest_sha256(m2)
