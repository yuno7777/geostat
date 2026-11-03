# GeoStat — Database Schema (SQLite, MVP)

DDL lives in `src/geostat/db.py` (`init_db`), applied idempotently on every CLI
command start. No migration framework in MVP; schema changes are additive
(`ADD COLUMN`) and ADR-001 documents the Postgres/Postgres-PostGIS upgrade path.

```sql
CREATE TABLE IF NOT EXISTS study_areas (
  id            TEXT PRIMARY KEY,          -- slug
  name          TEXT NOT NULL,
  description   TEXT,
  bbox_minx REAL, bbox_miny REAL, bbox_maxx REAL, bbox_maxy REAL,
  crs           TEXT NOT NULL DEFAULT 'EPSG:4326',
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
  id            TEXT PRIMARY KEY,          -- {sha256:16}-{kind[:4]}
  kind          TEXT NOT NULL,             -- s1_raw|label_yaml|model_weights|cog_raster|geojson|osm_cache
  path          TEXT NOT NULL,
  sha256        TEXT NOT NULL,
  size_bytes    INTEGER NOT NULL,
  dataset       TEXT,
  dataset_version TEXT,
  source_url    TEXT,
  license       TEXT,
  attribution   TEXT,
  crs           TEXT,
  width INTEGER, height INTEGER, band_count INTEGER, dtype TEXT,
  nodata        TEXT,                      -- sqlite-safe string, e.g. 'nan' or '255'
  resolution_x REAL, resolution_y REAL,
  minx REAL, miny REAL, maxx REAL, maxy REAL,
  acquired_at   TEXT,
  created_at    TEXT NOT NULL,
  meta_json     TEXT                       -- reserved
);

CREATE TABLE IF NOT EXISTS scenes (
  id            TEXT PRIMARY KEY,          -- e.g. 'Bolivia_103757'
  study_area_id TEXT NOT NULL REFERENCES study_areas(id),
  event_name    TEXT,
  acquired_date TEXT,                      -- from Sen1Floods11_Metadata.geojson
  s1_asset_id   TEXT REFERENCES assets(id),
  label_asset_id TEXT REFERENCES assets(id),
  official_split TEXT,                     -- train|valid|test|none from Sen1Floods11 CSVs
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
  id            TEXT PRIMARY KEY,          -- slug/version
  kind          TEXT NOT NULL,             -- unet|vh_threshold
  config_json   TEXT NOT NULL,
  checkpoint_path TEXT, checkpoint_sha256 TEXT,
  summary_json  TEXT,                      -- final metric summary
  created_at    TEXT NOT NULL,
  created_by_run TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  id            TEXT PRIMARY KEY,          -- run-<slug>-<ts> (timestamp UTC 'YYYYmmddTHHMMSSZ')
  kind          TEXT NOT NULL,             -- download|ingest|validate|manifest|train|evaluate|analyze|exposure|api_smoke
  status        TEXT NOT NULL,             -- started|completed|failed
  scene_id      TEXT,
  model_id      TEXT,
  config_json   TEXT,                      -- exact inputs
  metrics_json  TEXT,                      -- exact computed metrics (or null)
  artifact_paths_json TEXT,
  env_json      TEXT,                      -- hardware + versions snapshot
  code_revision TEXT NOT NULL,             -- sha256 or git:<ref>
  seed          INTEGER,
  started_at    TEXT NOT NULL, finished_at TEXT,
  error         TEXT,
  notes_json    TEXT
);

CREATE TABLE IF NOT EXISTS run_events (
  run_id TEXT NOT NULL REFERENCES runs(id),
  ts     TEXT NOT NULL,
  level  TEXT NOT NULL,                    -- info|warn|error
  message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exposures (
  run_id      TEXT PRIMARY KEY REFERENCES runs(id),
  scene_id    TEXT NOT NULL,
  summary_json TEXT NOT NULL,             -- geodesic counts/lengths + attribution + retrieval ts
  osm_dataset_date TEXT
);

CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
  details_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_run_scene ON runs(scene_id, kind, started_at);
CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind);
```

## Relationship / purpose notes

- `assets` are content-addressed (id derived from sha256) so re-ingesting is idempotent
  and cheap.
- `scenes` associates `{S1 raster, label, study area, event; official split ref}`. The
  *analysis tile split* is a separate concern (manifest) — a scene could in principle
  be official-test but our-train; both are recorded, and any **experiment-only** fact
  lives on the **manifest**, never on `scenes`.
- `runs` is the reproducibility ledger: a run row alone is enough to re-execute (config
  + code_revision + env + seed).
- `run_events` is the raw log; `exposures` summarizes upper query results (kept separate
  for query ergonomics of the API).
- `audit` — now CLI-only actor ("cli"), later records any API mutation (none in MVP).

## What is deliberately **not** in the DB

- No raster blobs (per doc contract rule #5); DB stores metadata + paths only.
- No vectors of full OSM response (kept as files in cache; parsed features go straight
  into returned GeoJSON on demand, not into SQL).
