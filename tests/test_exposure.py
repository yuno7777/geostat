from __future__ import annotations

from shapely.geometry import box, LineString, Polygon

from geostat.exposure import exposure_summary, parse_osm_elements


def test_parse_osm_elements():
    elements = [
        {
            "type": "way",
            "id": 101,
            "tags": {"highway": "residential", "name": "Main St"},
            "geometry": [{"lon": -65.1, "lat": -15.1}, {"lon": -65.1, "lat": -15.0}],
        },
        {
            "type": "way",
            "id": 102,
            "tags": {"building": "yes"},
            "geometry": [
                {"lon": -65.1, "lat": -15.1},
                {"lon": -65.1, "lat": -15.11},
                {"lon": -65.11, "lat": -15.11},
                {"lon": -65.11, "lat": -15.1},
                {"lon": -65.1, "lat": -15.1},
            ],
        },
    ]

    res = parse_osm_elements(elements)
    assert len(res["roads"]) == 1
    assert len(res["buildings"]) == 1
    assert res["roads"][0]["highway"] == "residential"
    assert res["buildings"][0]["building"] == "yes"


def test_exposure_summary():
    # Flood polygon covering -65.15 to -65.05 and -15.15 to -15.05
    flood_poly = box(-65.15, -15.15, -65.05, -15.05)
    road_line = LineString([(-65.1, -15.2), (-65.1, -15.0)])  # passes through flood poly
    building_poly = box(-65.12, -15.12, -65.08, -15.08)       # inside flood poly

    roads = [{"id": "osm:way/1", "geometry": road_line, "highway": "residential"}]
    buildings = [{"id": "osm:way/2", "geometry": building_poly, "building": "yes"}]

    summary = exposure_summary([flood_poly], roads, buildings)

    assert summary["roads"]["total_km"] > 0
    assert summary["roads"]["flooded_km"] > 0
    assert summary["roads"]["flooded_km"] < summary["roads"]["total_km"]
    assert summary["buildings"]["total"] == 1
    assert summary["buildings"]["flooded"] == 1
