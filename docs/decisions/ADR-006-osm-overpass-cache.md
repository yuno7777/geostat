# ADR-006 — OSM via Overpass API with per-scene response caching; geodesic measurement

**Status:** Accepted (2026-11-03)

## Context
Exposure analysis needs local roads/buildings. Options considered: Geofabrik country
extracts (hundreds of MB per country, overkill per chip), OSMnx (extra conda/graph
stack, heavy for Windows), Overpass API (on-demand, small bbox, JSON with inline
geometry).

## Decision
Query Overpass per scene bbox: `(way/relation{bbox}[highway|building]; out body geom;)`.
Cache the raw JSON response per scene under `data/cache/osm/` together with retrieval
timestamp + query text. Parse into shapely geometries; intersect in EPSG:4326; lengths
and areas via `pyproj.Geod` (WGS-84 ellipsoid), not local UTM.

## Consequences
- (+) No daemon; small per-scene response (~10 KB–2 MB typical).
- (+) Response is delta-stable only at retrieval time; quarantine with timestamp is
  necessary and done.
- (−) Overpass is a shared public resource; be a good citizen: small queries, cached,
  no polling loops. Documented; used only when explicitly requested per scene.
- (−) OSM completeness varies (see DATA_CATALOG.md §2). Exposure is presented as
  ODM-obtained counts, never "every building".

## Reason for geodesics
Avoiding local UTM zone selection bugs and zone-edge distortions for a 5 km bbox; the
ellipsoid math is robust and simple.

## References
exposure.py; tests/test_exposure.py; DATA_CATALOG.md §2.
