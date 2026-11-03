# GeoStat — Architecture

**Status: Implemented (Phase 1–7).** Living doc; changes recorded in CHANGELOG.md.

## 1. High-level shape

GeoStat is a **CLI-driven pipeline + read-only API** for reproducible flood-scene
analysis. Modules are dependency-ordered; there is no web framework in the data layer
and no ORM. The product targets a single self-hosted analyst box (Windows/Linux), not a
distributed deployment.

```text
                 ┌──────────────────────────────────────────────┐
                 │  CLI commands (python -m geostat.cli …)      │
                 │  download → ingest → validate → split-manifest│
                 │  → train → evaluate → analyze → serve         │
                 └──────┬──────────────────────────────┬────────┘
                        │ writes                       │ reads
                        ▼                              ▼
        ┌────────────────────────────────┐   ┌──────────────────────────┐
        │ data/           artifacts/     │   │ SQLite  geostat.sqlite3  │
        │  raw/ (immutable, checksummed) │   │  assets, scenes, models  │
        │  cache/ (OSM + tiles index)    │   │  runs, exposures, audit  │
        │  reference/ (metadata/splits)  │   └──────────┬───────────────┘
        └────────────────────────────────┘              │
                                                        ▼
                                        ┌────────────────────────────┐
                                        │  FastAPI (read-only)       │
                                        │  + static frontend/        │
                                        │  MapLibre browse UI        │
                                        └────────────────────────────┘
```

## 2. Modules (src/geostat)

| Module | Owns |
|---|---|
| `settings.py` | Path roots, constants, environment config; never mutates state. |
| `provenance.py` | sha256/md5, code-revision (source-tree sha256 + optional git HEAD format), environment snapshot (OS/CPU/RAM/GPU/torch). |
| `db.py` | SQLite schema (`CREATE TABLE IF NOT EXISTS`), helper `record_run`, `log` and query helpers. |
| `geoio.py` | Open rasters, extract metadata, alignment/geometry validation, nan-safe reads with `-1` nodata encoding, matrix helpers. |
| `tiling.py` | Tile-grid math, manifest JSON schema + deterministic hashing, event-disjoint split assignment for tiles, alignment manifests. |
| `dataset.py` | torch `Dataset` over tile manifests; reads windows from GeoTIFFs; normalization; deterministic augmentation; seed helpers. |
| `metrics.py` | Binary confusion matrix + precision/recall/F1/IoU per event; nodata exclusion. |
| `baseline.py` | VH threshold baseline (`water if VH < t`); threshold search maximizing IoU on a given split. |
| `unet.py` | Small U-Net (channels 8/16/32/64) for S1 2-band input; ~600k params. |
| `train.py` | `train_model`: BCE+Dice, AdamW, cosine, seeded; early stop; checkpoint JSON sidecar. |
| `inference.py` | Infer over full scene, stitch tiles into prob mask; signal threshold; write shuffled COGs (prob float32 nodata nan, binary uint8 nodata 255) + GeoJSON polygons. |
| `exposure.py` | Overpass query + cache; OSM parse → shapely; flooded length/area via geodesic math; summary dict. |
| `ingest.py` | Reads `data/raw` manifests → DB rows for assets, scenes, study areas; validates alignment first. |
| `api.py` | FastAPI app (read endpoints + static files + layer GeoJSONs; no arbitrary uploads). |
| `cli.py` | Argparse wiring for all above; each command logs + records a `runs` row. |

## 3. Data lifecycle / jobs / run provenance

Every CLI command that mutates state creates a **run row** in SQLite with:

- `config_json` (exact inputs incl. seed, batch size, thresholds, splits, paths).
- `code_revision` (source tree sha256, or git sha + dirty state once commits exist).
- `env` (hardware + torch + python versions).
- artifact paths (manifest, model, cog, geojson, exposure report).
- status timing + per-line log tail in `run_events`.

Analyzed scenes reference the *specific* `run_{kind}` that produced their layers, so a
map never shows a layer without its generating config. `audit` records actor/action,
which today is always the CLI/script because there is no public write API (security).

## 4. Data contracts

### Raster
- Scene chips are EPSG:4326, 512×512, 10 m/px, float32 dB bands `[VV, VH]`, NaN nodata.
- Must share transform/bounds/resolution into their label; validated pre-ingest and
  again before any training/inference touching them.
- All derived rasters keep the source transform/CRS/shape; only `driver` may change
  (GeoTIFF→COG) and dtype/ndim at export when both consistent.

### Tile manifest (`artifacts/manifest_{slug}.json`)
```json
{
  "schema": "geostat.tile-manifest/1",
  "dataset": "sen1floods11.v1.1",
  "tile_size": 256, "stride": 256, "split_rule": "event-disjoint",
  "val_events": ["..."], "test_events": ["..."],
  "seed": 1337,
  "code_revision": {"source_tree_sha256": "..."},
  "tiles": [{"scene":"...","tile":"...","split":"train|val|test","row":0,"col":0,...}]
}
```
Determinism: iterating dictionaries sorted; JSON with `sort_keys=True`; same config +
same +code_revision ⇒ identical sha256.

### Training checkpoint sidecar (`model_{name}.json`)
Model architecture, input spec, loss, optimizer, seeds, environment, dataset manifest
hash, and split configuration. Saved next to `.pt` weights.

## 5. API surface (read-only for MVP)

Full specification: [API_SPECIFICATION.md](API_SPECIFICATION.md). Auth: none for local
single-analyst MVP (SECURITY.md threat model). CORS: same-origin only; localhost by default.

## 6. Non-goals re-confirmed

No message queue, no kubernetes, no celery, no electron desktop app, no arbitrary raster
upload (explicitly rejected in SECURITY.md because GDAL memory/file DoS vectors on
arbitrary input are out of MVP scope).
