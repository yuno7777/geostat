from __future__ import annotations

import numpy as np
import pytest
from pyproj import Geod
from shapely import union_all
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from geostat.exposure import build_overpass_query, exposure_summary
from geostat.spatial import FloodIndex, bounds_wgs84, polygons_wgs84, wgs84_area_m2, wgs84_length_m


def test_spatial_index_matches_union_on_seeded_overlapping_polygons():
    rng = np.random.default_rng(42)
    polygons = [box(x, y, x + 0.01, y + 0.01) for x, y in rng.uniform(0, 0.1, (50, 2))]
    dissolved = union_all(polygons)
    index = FloodIndex(polygons)
    for start, stop in rng.uniform(-0.03, 0.13, (50, 2, 2)):
        line = LineString([start, stop])
        expected = line.intersection(dissolved)
        actual = index.clip(line)
        if actual is None:
            assert expected.is_empty
        else:
            # Dissolving clips may introduce collinear vertices, which can slightly
            # change geodesic path length for a line at nonzero latitude.
            assert actual.length == pytest.approx(expected.length, abs=1e-12)


def test_dateline_cut_and_overlay():
    raw = box(179.5, -0.25, 180.5, 0.25)
    split = list(polygons_wgs84(raw, "EPSG:4326"))
    assert len(split) == 2
    assert all(g.is_valid and -180 <= g.bounds[0] < g.bounds[2] <= 180 for g in split)
    out = exposure_summary(
        split,
        [
            {"geometry": LineString([(179, 0), (-179, 0)]), "highway": "primary"},
            {"geometry": LineString([(-1, 0), (1, 0)]), "highway": "secondary"},
        ],
        [{"id": "seam", "geometry": box(179.9, -0.1, 180.1, 0.1)}],
    )
    assert out["roads"]["segments_flooded"] == 1
    assert out["roads"]["flooded_m"] == pytest.approx(111319.5, abs=0.1)
    assert out["buildings"]["flooded"] == 1
    assert bounds_wgs84((179.5, -0.25, 180.5, 0.25), "EPSG:4326") == (179.5, -0.25, -179.5, 0.25)
    query = build_overpass_query((179.5, -0.25, -179.5, 0.25))
    assert query.count('way["highway"]') == 2
    assert query.count('way["building"]') == 2


def test_polar_and_mixed_geometry_measurements():
    geod = Geod(ellps="WGS84")
    path = LineString([(0, 89, 10), (180, 89, 20)])
    assert wgs84_length_m(GeometryCollection([path, Point(0, 90)])) == pytest.approx(
        geod.inv(0, 89, 180, 89)[2]
    )
    polar_ring = Polygon([(-135, 89), (-45, 89), (45, 89), (135, 89)])
    assert wgs84_area_m2(polar_ring) == pytest.approx(
        abs(geod.geometry_area_perimeter(polar_ring)[0])
    )
    a = box(10, 10, 11, 11)
    b = box(12, 10, 13, 11).reverse()
    assert wgs84_area_m2(MultiPolygon([a, b])) == pytest.approx(wgs84_area_m2(a) + wgs84_area_m2(b))
    shell = box(10, 10, 11, 11)
    hole = box(10.1, 10.1, 10.2, 10.2)
    assert wgs84_area_m2(Polygon(shell.exterior, [hole.exterior])) == pytest.approx(
        wgs84_area_m2(shell) - wgs84_area_m2(hole)
    )


def test_dry_scene_null_empty_and_touch_only_inputs():
    road = LineString([(-1, -1), (0, 0)])
    roads = [{"geometry": road}, {"geometry": None}, {"geometry": LineString()}]
    dry = exposure_summary([None, Polygon()], roads, [{"id": "missing", "geometry": None}])
    assert dry["roads"]["total_m"] > 0
    assert dry["roads"]["flooded_m"] == dry["buildings"]["flooded_area_m2"] == 0
    touched = exposure_summary(
        [box(0, 0, 1, 1)], roads, [{"id": "touch", "geometry": box(1, 0, 2, 1)}]
    )
    assert touched["roads"]["segments_flooded"] == touched["buildings"]["flooded"] == 0


def test_degenerate_flood_rejected():
    with pytest.raises(ValueError, match="invalid"):
        FloodIndex([Polygon([(0, 0), (0.5, 0), (1, 0)])])
    with pytest.raises(ValueError, match="latitude"):
        FloodIndex([box(10, 89, 11, 91)])
