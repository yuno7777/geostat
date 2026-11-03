from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import LineString, Polygon, shape
from shapely.geometry.base import BaseGeometry

import geostat.settings as settings
from geostat import geoio
from geostat.provenance import utcnow

OSM_UA = "GeoStat/0.1 (research; contact via GitHub repo)"
OSM_TIMEOUT_S = 180
OSM_RETRIES = 3
OSM_RETRY_SLEEP = 8.0
HIGHWAY_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "residential",
    "service", "unclassified", "track", "footway", "path", "cycleway", "pedestrian",
}


def cache_dir(scene: str) -> Path:
    d = settings.CACHE_DIR / "osm" / f"scene={scene}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_overpass_query(bbox: tuple[float, float, float, float]) -> str:
    minx, miny, maxx, maxy = bbox
    b = f"{miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f}"
    return (
        "[out:json][timeout:120];"
        f"("
        f'way["highway"]({b});'
        f'relation["highway"]({b});'
        f'way["building"]({b});'
        f'relation["building"]({b});'
        ");"
        "out body geom;"
    )


def fetch_overpass(bbox: tuple[float, float, float, float], scene: str, *, refresh: bool = False) -> tuple[dict[str, Any], Path]:
    d = cache_dir(scene)
    resp_path = d / "response.json"
    meta_path = d / "response_meta.json"
    if resp_path.exists() and meta_path.exists() and not refresh:
        return json.loads(resp_path.read_text(encoding="utf-8")), meta_path
    query = build_overpass_query(bbox)
    url = settings.OVERPASS_URL
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
            resp_path.write_text(json.dumps(payload), encoding="utf-8")
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            return payload, meta_path
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < OSM_RETRIES:
                time.sleep(1.0)
    # Fallback to official OSM 0.6 map API
    try:
        minx, miny, maxx, maxy = bbox
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
        resp_path.write_text(json.dumps(payload), encoding="utf-8")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
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
        if not coords:
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
    flood_polys: list[BaseGeometry],
    roads: list[dict[str, Any]],
    buildings: list[dict[str, Any]],
) -> dict[str, Any]:
    flood_union = flood_polys[0]
    for g in flood_polys[1:]:
        flood_union = flood_union.union(g)
    road_total_m = 0.0
    road_flooded_m = 0.0
    n_road_segments = len(roads)
    n_road_flooded = 0
    road_classes_flooded: dict[str, float] = {}
    for r in roads:
        geom: BaseGeometry = r["geometry"]
        total = geoio.wgs84_length_m(geom)
        road_total_m += total
        inter = geom.intersection(flood_union)
        if inter.is_empty:
            continue
        flooded_len = geoio.wgs84_length_m(inter)
        if flooded_len <= 0:
            continue
        road_flooded_m += flooded_len
        n_road_flooded += 1
        cls = str(r.get("highway") or "other")
        road_classes_flooded[cls] = round(road_classes_flooded.get(cls, 0.0) + flooded_len, 1)
    buildings_total = len(buildings)
    flooded_buildings: list[dict[str, Any]] = []
    flooded_area_m2 = 0.0
    for b in buildings:
        geom: BaseGeometry = b["geometry"]
        inter = geom.intersection(flood_union)
        if inter.is_empty:
            continue
        a = geoio.wgs84_area_m2(inter)
        if a <= 0:
            continue
        flooded_area_m2 += a
        flooded_buildings.append({
            "id": b["id"],
            "building": b.get("building"),
            "name": b.get("name"),
            "flooded_area_m2": round(a, 1),
        })
    return {
        "roads": {
            "total_m": round(road_total_m, 1),
            "total_km": round(road_total_m / 1000.0, 3),
            "flooded_m": round(road_flooded_m, 1),
            "flooded_km": round(road_flooded_m / 1000.0, 3),
            "segments_total": n_road_segments,
            "segments_flooded": n_road_flooded,
            "flooded_by_class": road_classes_flooded,
        },
        "buildings": {
            "total": buildings_total,
            "flooded": len(flooded_buildings),
            "flooded_area_m2": round(flooded_area_m2, 1),
            "flooded_sample": flooded_buildings[:50],
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
    flood_polys = [shape(f["geometry"]) for f in fc["features"]]
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
