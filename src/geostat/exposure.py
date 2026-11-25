from __future__ import annotations

import json
import math
from collections import defaultdict
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import LineString, Polygon, shape
from shapely.geometry.base import BaseGeometry

import geostat.settings as settings
from geostat import geoio
from geostat.provenance import utcnow
from geostat.spatial import BBox, FloodIndex, wgs84_area_m2, wgs84_length_m

OSM_UA = "GeoStat/0.1 (research; contact via GitHub repo)"
OSM_TIMEOUT_S = 180
OSM_RETRIES = 3
OSM_RETRY_SLEEP = 8.0
HIGHWAY_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "residential",
    "service", "unclassified", "track", "footway", "path", "cycleway", "pedestrian",
}


def cache_dir(scene: str) -> Path:
    scene = settings.path_component(scene, "scene")
    d = settings.CACHE_DIR / "osm" / f"scene={scene}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _query_bounds(bbox: BBox) -> tuple[BBox, ...]:
    west, south, east, north = bbox
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south < north <= 90):
        raise ValueError("OSM bounds must be finite WGS84 longitude/latitude with south < north")
    if west == east or (west == 180 and east == -180):
        raise ValueError("OSM bounds must have a positive longitude width")
    if east < west:
        return tuple(
            (w, south, e, north) for w, e in ((west, 180.0), (-180.0, east)) if w < e
        )
    return (bbox,)


def build_overpass_query(bbox: BBox) -> str:
    terms: list[str] = []
    for west, south, east, north in _query_bounds(bbox):
        coords = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"
        terms.extend(f'way["{tag}"]({coords});' for tag in ("highway", "building"))
    return "[out:json][timeout:120];(" + "".join(terms) + ");out body geom;"


def fetch_overpass(bbox: tuple[float, float, float, float], scene: str, *, refresh: bool = False) -> tuple[dict[str, Any], Path]:
    d = cache_dir(scene)
    resp_path = d / "response.json"
    meta_path = d / "response_meta.json"
    query = build_overpass_query(bbox)
    url = settings.OVERPASS_URL
    if resp_path.exists() and meta_path.exists() and not refresh:
        meta = geoio.load_json(meta_path)
        if meta.get("bbox_wgs84") == list(bbox):
            return geoio.load_json(resp_path), meta_path
    data = urllib.parse.urlencode({"data": query, "info": "true"}).encode()
    last: Optional[Exception] = None
    for attempt in range(1, OSM_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": OSM_UA}, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - URL from settings only
                payload = json.load(resp)
            meta = {
                "scene": scene,
                "bbox_wgs84": list(bbox),
                "query": query,
                "url": url,
                "retrieved_at": utcnow(),
                "attempt": attempt,
                "status": resp.status,
                "elements": len(payload.get("elements", [])),
            }
            geoio.save_json(resp_path, payload)
            geoio.save_json(meta_path, meta)
            return payload, meta_path
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < OSM_RETRIES:
                time.sleep(1.0)
    # Fallback to official OSM 0.6 map API
    if len(_query_bounds(bbox)) != 1:
        raise OSError(f"Overpass is required for an antimeridian-crossing scene {scene}: {last}") from last
    try:
        minx, miny, maxx, maxy = _query_bounds(bbox)[0]
        osm_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}"
        req = urllib.request.Request(osm_url, headers={"User-Agent": OSM_UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_data = resp.read()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_data)
        nodes = {n.attrib["id"]: {"lat": float(n.attrib["lat"]), "lon": float(n.attrib["lon"])} for n in root.findall("node")}
        elements = []
        for w in root.findall("way"):
            geom = [nodes[nd.attrib["ref"]] for nd in w.findall("nd") if nd.attrib["ref"] in nodes]
            tags = {t.attrib["k"]: t.attrib["v"] for t in w.findall("tag")}
            elements.append({"type": "way", "id": int(w.attrib["id"]), "tags": tags, "geometry": geom})
        payload = {"elements": elements}
        meta = {
            "scene": scene,
            "bbox_wgs84": list(bbox),
            "query": osm_url,
            "url": "https://api.openstreetmap.org/api/0.6/map",
            "retrieved_at": utcnow(),
            "attempt": 1,
            "status": 200,
            "elements": len(elements),
        }
        geoio.save_json(resp_path, payload)
        geoio.save_json(meta_path, meta)
        return payload, meta_path
    except Exception as exc:
        last = exc
    raise IOError(f"overpass fetch failed for {scene}: {last}")


def parse_osm_elements(elements: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    roads: list[dict[str, Any]] = []
    buildings: list[dict[str, Any]] = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        geometry = el.get("geometry") or []
        coords = [(g["lon"], g["lat"]) for g in geometry if "lon" in g and "lat" in g]
        if not coords or any(
            not isinstance(lon, (float, int)) or not isinstance(lat, (float, int))
            or not -180 <= lon <= 180 or not -90 <= lat <= 90 for lon, lat in coords
        ):
            continue
        el_type = el.get("type")
        if el_type == "relation":
            continue  # geometries of relations in `out body geom` are multipolygon segments; skipped in MVP
        highway = tags.get("highway")
        building = tags.get("building")
        if highway:
            if len(coords) < 2:
                continue
            try:
                geom: BaseGeometry = LineString(coords)
            except Exception:
                continue
            if not geom.is_valid:
                continue
            roads.append({
                "id": f"osm:{el.get('type')}/{el.get('id')}",
                "smth": None,
                "highway": highway,
                "name": tags.get("name"),
                "geometry": geom,
            })
        elif building:
            if len(coords) < 4:
                continue
            try:
                geom = Polygon(coords)
            except Exception:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty or not geom.is_valid:
                continue
            buildings.append({
                "id": f"osm:{el.get('type')}/{el.get('id')}",
                "building": building if isinstance(building, str) else "yes",
                "name": tags.get("name"),
                "geometry": geom,
            })
    return {"roads": roads, "buildings": buildings}


def exposure_summary(
    flood_polys: Sequence[BaseGeometry | None],
    roads: list[dict[str, Any]],
    buildings: list[dict[str, Any]],
) -> dict[str, Any]:
    index = FloodIndex(flood_polys)
    n_road_flooded = 0
    road_classes_flooded: defaultdict[str, float] = defaultdict(float)

    def road_lengths() -> Iterator[float]:
        nonlocal n_road_flooded
        for road in roads:
            geom = road.get("geometry")
            if not isinstance(geom, BaseGeometry) or geom.is_empty:
                continue
            clipped = index.clip(geom)
            if clipped is not None:
                flooded = wgs84_length_m(clipped)
                if flooded > 0:
                    road_classes_flooded[str(road.get("highway") or "other")] += flooded
                    n_road_flooded += 1
            yield wgs84_length_m(geom)

    road_total_m = math.fsum(road_lengths())
    road_flooded_m = math.fsum(road_classes_flooded.values())
    flooded_buildings: list[dict[str, Any]] = []
    n_buildings_flooded = 0

    def building_areas() -> Iterator[float]:
        nonlocal n_buildings_flooded
        for building in buildings:
            geom = building.get("geometry")
            if not isinstance(geom, BaseGeometry) or geom.is_empty:
                continue
            clipped = index.clip(geom)
            if clipped is None:
                continue
            area = wgs84_area_m2(clipped)
            if area <= 0:
                continue
            n_buildings_flooded += 1
            if len(flooded_buildings) < 50:
                flooded_buildings.append({
                    "id": building["id"],
                    "building": building.get("building"),
                    "name": building.get("name"),
                    "flooded_area_m2": round(area, 1),
                })
            yield area

    flooded_area_m2 = math.fsum(building_areas())
    return {
        "roads": {
            "total_m": round(road_total_m, 1),
            "total_km": round(road_total_m / 1000.0, 3),
            "flooded_m": round(road_flooded_m, 1),
            "flooded_km": round(road_flooded_m / 1000.0, 3),
            "segments_total": len(roads),
            "segments_flooded": n_road_flooded,
            "flooded_by_class": {key: round(value, 1) for key, value in road_classes_flooded.items()},
        },
        "buildings": {
            "total": len(buildings),
            "flooded": n_buildings_flooded,
            "flooded_area_m2": round(flooded_area_m2, 1),
            "flooded_sample": flooded_buildings,
        },
        "n_flood_polys": len(flood_polys),
    }


def run_exposure(
    run: Any,
    scene: str,
    bbox: tuple[float, float, float, float],
    flood_geojson_path: Path,
    *,
    refresh_osm: bool = False,
) -> dict[str, Any]:
    payload, meta_path = fetch_overpass(bbox, scene, refresh=refresh_osm)
    parsed = parse_osm_elements(payload.get("elements", []))
    fc = geoio.load_json(flood_geojson_path)
    flood_polys = [shape(f["geometry"]) for f in fc["features"] if f.get("geometry") is not None]
    summary = exposure_summary(flood_polys, parsed["roads"], parsed["buildings"])
    summary["scene"] = scene
    summary["osm"] = {
        "retrieved_at": json.loads(meta_path.read_text(encoding="utf-8"))["retrieved_at"],
        "url": settings.OVERPASS_URL,
        "license": "ODbL 1.0",
        "attribution": "© OpenStreetMap contributors",
        "unknown_relation_skipped": True,
    }
    return summary
