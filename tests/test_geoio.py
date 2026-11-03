from __future__ import annotations

from pathlib import Path
import rasterio
from rasterio.transform import from_origin

from geostat import geoio


def test_read_meta(synthetic_raster_pair: tuple[Path, Path]):
    s1_path, label_path = synthetic_raster_pair
    s1_meta = geoio.read_meta(s1_path)
    label_meta = geoio.read_meta(label_path)

    assert s1_meta.width == 64
    assert s1_meta.height == 64
    assert s1_meta.band_count == 2
    assert s1_meta.has_nan_nodata

    assert label_meta.width == 64
    assert label_meta.height == 64
    assert label_meta.band_count == 1
    assert not label_meta.has_nan_nodata


def test_assert_alignment_success(synthetic_raster_pair: tuple[Path, Path]):
    s1_path, label_path = synthetic_raster_pair
    s1_meta = geoio.read_meta(s1_path)
    label_meta = geoio.read_meta(label_path)

    res = geoio.assert_alignment(s1_meta, label_meta)
    assert res["aligned"] is True
    assert len(res["problems"]) == 0


def test_assert_alignment_detects_mismatch(tmp_path: Path):
    s1_path = tmp_path / "s1.tif"
    label_path = tmp_path / "label.tif"

    with rasterio.open(
        s1_path, "w", driver="GTiff", height=64, width=64, count=2, dtype="float32",
        crs="EPSG:4326", transform=from_origin(0, 0, 1, 1)
    ) as dst:
        pass

    with rasterio.open(
        label_path, "w", driver="GTiff", height=32, width=32, count=1, dtype="int16",
        crs="EPSG:3857", transform=from_origin(10, 10, 2, 2)
    ) as dst:
        pass

    s1_meta = geoio.read_meta(s1_path)
    label_meta = geoio.read_meta(label_path)
    res = geoio.assert_alignment(s1_meta, label_meta)
    assert res["aligned"] is False
    assert len(res["problems"]) > 0


def test_polygonize_water(synthetic_raster_pair: tuple[Path, Path]):
    _, label_path = synthetic_raster_pair
    meta = geoio.read_meta(label_path)
    labels = geoio.labels(label_path)
    geoms = geoio.polygonize_water(labels, meta.transform)
    assert len(geoms) > 0
    for g in geoms:
        assert g.is_valid
        assert not g.is_empty
