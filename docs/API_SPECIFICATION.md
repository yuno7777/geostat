# GeoStat — API Specification

Base: `http://127.0.0.1:8787` (default). Format: JSON (all responses
`application/json`, UTF-8). No auth in MVP (single-analyst, localhost bind; see
SECURITY.md threat model). Read-only; no mutation endpoints in MVP; all writes go
through CLI commands that own DB transactions atomic per run.

## Endpoints

### `GET /api/health`
`200` `{ "ok": true, "version": "0.1.0", "python": "3.12.x", "db_path": "data/geostat.sqlite3", "data_dir_exists": true, "has_scenes": true|false }`
No side effects.

### `GET /api/study-areas`
`200` — list of `{ "id", "name", "slug", "description", "bbox": [minx,miny,maxx,maxy], "crs": "EPSG:4326", "n_scenes": int }`
Empty DB ⇒ `[]`.

### `GET /api/study-areas/{id}`
`200` — study area + scenes `{ "id", "name", "bbox", "crs", "scenes": [scene-lite] }`
`404` `{ "detail": "Study area not found" }`.

### `GET /api/scenes`
Query: `study_area`: optional id filter.
`200` — `{ "scenes": [ { "id": chip_id, "study_area_id", "event": country, "acquired_date", "s1_asset_id", "label_asset_id", "split_ref": "train|valid|test|none", "bbox" } ] }`

### `GET /api/scenes/{id}`
`200` — full scene incl. asset metadata (sha256, crs, bounds, nodata) for both S1 and label.
`404` if unknown id.

### `GET /api/runs`
Query: `kind=analysis|training|exposure|baseline|evaluate`, `scene_id`, `limit` (≤ 200).
`200` — newest-first list of run summaries `{ "id", "kind", "status", "started_at", "finished_at", "model_id", "metrics_keys": [...], "config_keys": [...] }`.

### `GET /api/runs/{id}`
`200` — full run (config, metrics, artifact paths, env, code_revision, log tail up to last 200 lines).
`404` if unknown id.

### `GET /api/analyses/{scene_id}`
`200` — the bundle the map client needs for a scene, referencing the latest successful
`analysis` run: `{ "scene": {...}, "run": {...}, "layers": { "flood": "/api/layers/flood/{scene}", "reference":"/api/layers/reference/{scene}", "roads":"/api/layers/roads/{scene}", "buildings":"/api/layers/buildings/{scene}" }, "exposure": {...} | null, "metrics": per-scene metrics | null, "artifacts": {"prob_cog": "/files/…", "mask_cog": "/files/…", "geojson": "/files/…" } }`
`404` if scene unknown; `200` with `"run": null` if not yet analyzed; `206` partial when some layers missing (state explicit in JSON, no silent empties).

### `GET /api/layers/flood/{scene}`
`200` — GeoJSON FeatureCollection of flood polygons (properties `{flood: 1}`) of latest analysis run, plus `run_id` in an `x-geostat-run` response header.
`.` (404 — no analysis yet, empty body `{ "type":"FeatureCollection", "features": [] }`)? **No**: returns `204` with empty body for "no data yet"; a scene never analyzed yields `404`; an analyzed scene with zero polygons yields `200` with a zero-feature collection (explicit).

### `GET /api/layers/reference/{scene}`
`200` — GeoJSON FeatureCollection of reference water polygons from the hand label, properties `{reference: 1, nodata_excluded: true}` with nodata polygons excluded. Also 404/204 rules above, plus specific: 404 if scene has no label asset registered.

### `GET /api/layers/roads/{scene}` / `GET /api/layers/buildings/{scene}`
`200` — GeoJSON FeatureCollections from the cached OSM response; each feature carries only `id`, `name` (if present), `highway`/`building` class, and `exposed: 0|1`.
- 404 if that scene has no OSM cache (never fetched);
- 204 if cache exists but zero matching elements (honest empty, not an error);
- properties include `osm_attribution` string.

### `GET /api/exposure/{scene}`
`200` — latest exposure summary JSON for that scene (same content as
`artifacts/runs/…/exposure.json`), with attribution + retrieval timestamp.
`404` if never computed.

### `GET /files/{run_id}/{filename}`
Returns a static artifact file from `artifacts/runs/{run_id}/` — **path traversal safe**
(`..` and nested separators rejected with `400`; file must already be registered in the
run row's `artifact_paths`, not just exist on disk).

### `GET /` and `/{static:path}`
Serves the MapLibre UI, then its static assets (js/css/vendor files).

## Error contract

All errors are FastAPI `HTTPException`:
- `400` with a message; `404` with `{detail}`; never 500 with a stack trace for user input
  errors. 500 responses do not include filesystem paths or secrets (asserted by test).
- Any 500 in log is prefixed `[api]` with root cause; API re-raises as HTTPException(500, "unknown error").

## Versioning

`/api/*` is v0 (unstable). Breaking change ⇒ new prefix. There is no `/v1` yet.
