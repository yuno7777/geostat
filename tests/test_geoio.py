from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import shape

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
        s1_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ):
        pass

    with rasterio.open(
        label_path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=from_origin(10, 10, 2, 2),
    ):
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


def test_projected_export_and_unsigned_nodata(tmp_path):
    path = tmp_path / "projected.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="uint8",
        nodata=255,
        crs="EPSG:3857",
        transform=from_origin(111319.490793, 111325.142866, 100, 100),
    ) as dest:
        dest.write(np.array([[[1, 255, 0], [1, 0, 0]]], dtype=np.uint8))
    label = geoio.read_window(path, 0, 0, 2, 3)[0]
    assert label.tolist() == [[1, -1, 0], [1, 0, 0]]
    assert geoio.label_value_counts(path) == {"-1": 1, "0": 3, "1": 2}
    with pytest.raises(geoio.GeoError, match="inside"):
        geoio.read_window(path, 0, 0, 2, 4)
    meta = geoio.read_meta(path)
    collection = geoio.water_feature_collection(label, meta, {"flood": 1})
    geom = shape(collection["features"][0]["geometry"])
    assert geom.bounds[0] == pytest.approx(1)
    assert geom.bounds[3] == pytest.approx(1)
    assert geoio.water_feature_collection(np.empty((0, 0)), meta, {})["features"] == []


def test_nodata_mask_is_exact_and_nonfinite_is_invalid(tmp_path):
    path = tmp_path / "sar.tif"
    data = np.array([[[-9999.0, -9999.01, np.inf]]], dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=3,
        count=1,
        dtype="float32",
        nodata=-9999,
        crs="EPSG:4326",
        transform=from_origin(10, 10, 0.01, 0.01),
    ) as dest:
        dest.write(data)
    values = geoio.read_window(path, 0, 0, 1, 3)
    assert np.isnan(values[0, 0, [0, 2]]).all()
    assert values[0, 0, 1] == data[0, 0, 1]
    assert geoio.missing_sar_fraction(path) == pytest.approx(2 / 3)


def test_alignment_requires_known_crs(synthetic_raster_pair):
    meta = replace(geoio.read_meta(synthetic_raster_pair[0]), crs="unknown")
    assert not geoio.assert_alignment(meta, meta)["aligned"]


def test_disk_polygonization_preserves_seams_holes_and_connectivity(tmp_path):
    from rasterio.features import rasterize
    from shapely.geometry import shape

    wet = np.zeros((40, 40), dtype=np.uint8)
    wet[10:35, 10:35] = 1
    wet[15:25, 15:25] = 0
    wet[9, 9] = 1  # Diagonal contact alone is separate under four-way connectivity.
    source = tmp_path / "wet mask.tif"
    output = tmp_path / "wet shapes.geojson"
    transform = rasterio.Affine(0.01, 0, 70, 0, -0.01, 20)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        dtype="uint8",
        width=40,
        height=40,
        count=1,
        crs="EPSG:4326",
        transform=transform,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        for _, window in dst.block_windows(1):
            dst.write(wet[window.toslices()], 1, window=window)
    assert geoio.write_water_geojson(source, output) == 2
    geoms = [shape(feature["geometry"]) for feature in geoio.load_json(output)["features"]]
    assert sorted(len(g.interiors) for g in geoms) == [0, 1]
    np.testing.assert_array_equal(rasterize(geoms, out_shape=wet.shape, transform=transform), wet)
