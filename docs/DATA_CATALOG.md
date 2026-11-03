# GeoStat — Data Catalog

Every external data asset used by GeoStat is listed here with source, license, terms,
and known limitations. Registration in-app also stores the same facts in SQLite
(`assets`, `scenes`) — this document is the human-readable contract.

---

## 1. Sen1Floods11 v1.1 (primary dataset)

| Field | Value |
|---|---|
| Provider | Cloud to Street (Public Benefit Corporation) |
| Canonical source | `gs://sen1floods11/` (GCS bucket, public read) — v1.1 restructured to be COG-compliant |
| HTTP access used | `https://storage.googleapis.com/sen1floods11/v1.1/…` (verified reachable 2026-11-03) |
| Product name / version | Sen1Floods11, v1.1 |
| Acquired dates | Event-specific; per-event fields (`s1_date`, `s2_date`) in `Sen1Floods11_Metadata.geojson` (12 features / 12 flood events) |
| Reference paper | Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. 2020. *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR-W 2020, pp. 210-211. |
| License | **No LICENSE file in the upstream repo** (checked 2026-11-03, root GitHub listing). Released openly for research use by Cloud to Street; **citation required**; questions to `support@cloudtostreet.info`. GeoStat treats this as *research-grade, attribution-required* data; it is not redistributed by GeoStat. |
| Files used | `v1.1/data/flood_events/HandLabeled/S1Hand/*.tif` (446 files, ~696 MB total), `v1.1/data/flood_events/HandLabeled/LabelHand/*.tif` (446 files, ~2.4 MB), `v1.1/splits/flood_handlabeled/*.csv` (train/valid/test seed columns), `v1.1/Sen1Floods11_Metadata.geojson` |
| Files intentionally **not** used | S2OutHand (Sentinel-2 L1C OA), S1OutHand (Sentinel-1 GRD), JRCWaterHand (permanent water) — out of MVP S1 path. |
| Schema (per chip) | 512×512, EPSG:4326, 10 m/px ground resolution (≈ 0.0000898315°); S1 = Float32 2-band `[VV, VH]` ×10 m dB; Label = Int16 1-band `{-1, 0, 1}` where -1 = no data/not valid, 0 = not water, 1 = water. |
| Units | S1 bands: dB (negatives typical, e.g. −48 to 0 dB). |
| CRS | EPSG:4326 (per README: all chips WGS 84). |
| Labels semantics | Hand-labeled water vs. not-water per pixel; nodata −1 must be excluded in loss/metrics (pinned in code, asserted in tests). |
| Checksums | sha256 for every file serialized into `data/raw/sen1floods11/v1.1/MANIFEST.json` and into the SQLite `assets` table; md5 (origin `md5Hash` field from GCS JSON API) cross-checked except where prohibited by tooling gaps. |
| Known quality issues | Hand labels not exhaustive (sparse open water; typos in per-pixel precision unknown) — treat labels as ground truth for the MVP; size of usable chips shrinks when −1 rows removed; event–country naming awkward for multi-country events like `Mekong`. |

### 1.1 Event metadata observed (exact values from `Sen1Floods11_Metadata.geojson`)

24 flood events? No — 12 features: `Bolivia, Colombia, Ghana, India, Cambodia, Nigeria,
Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA`. Note: chip names use `Mekong_`
(Cambodia flood). Official train/valid/test split CSVs cover only 10 events (Bolivia,
Colombia, Cambodia are excluded from `flood_handlabeled` splits — we use Bolivia as our
event-disjoint holdout test event, per ADR-004).

- Bolivia: s1_date 2018/02/15, orbit DESCENDING, rel_orbit_num 156, VH_thresh −20.44 dB.
- Sample counts (exact, from sample ingest): Bolivia_103757 → nodata 174067, notwater 52715, water 35362.

---

## 2. OpenStreetMap (infrastructure)

| Field | Value |
|---|---|
| Provider | OpenStreetMap contributors |
| Endpoint used | Overpass API (`https://overpass-api.de/api/interpreter`, configurable override) |
| Query | Per scene bbox: all `way`/`relation` with `highway=*` or `building=*`; mode `out body geom;` response includes geometry inline (no node resolution needed). |
| License | ODbL 1.0. Attribution string required on any public-facing map: `© OpenStreetMap contributors` (link `https://www.openstreetmap.org/copyright`). Persisted per run into `exposures.osm_attribution`. |
| Retrieval timestamp | Stored per response in `data/cache/osm/scene={chip_id}/response_meta.json`. |
| Known limitations | OSM completeness is geographic; urban centers rich, other places sparse. Building polygons may be missing for common informal settlements → exposure can undercount. OSM data is not a safety-rated source; treat "flooded buildings count" as **OSM-visible** buildings only. |
| Redundancy | none in MVP; query cached locally per scene; rate-limit respected by periodic queries and small bboxes. |

Basemap tiles: MapLibre raster style hits OSM.org tiles (or slate-gray blank style when
offline). Attribution rendered in UI footnote. Governed by OSM tile-usage policy
("heavy use discouraged in apps"); acceptable for a self-hosted analysis tool.

---

## 3. Data governance rules enforced in code

1. Every *raw* asset carries: `source_url`, `dataset`, `dataset_version`, `sha256`,
   `size_bytes`, `crs`, `band_count`, `dtype`, `nodata`, `acquired_at` (if known),
   `license` label, `attribution` text, `created_at`.
2. `data/` and `artifacts/` are gitignored; large raster files never enter a git repo.
3. Checksums recomputed on every read of a registered raw asset: mismatch ⇒ hard error
   (integrity, not just provenance bookkeeping).
4. `code_revision` recorded on every run — currently source-tree SHA (see ADR-003 §content);
   once the repo is committed to git this will become git SHA + dirty flag.
5. Derived artifacts (COG, GeoJSON, exposure JSON) record the SHA(s) of their parent raw
   asset(s) and their config in SQLite (many-to-many `run_events`-log is not the ledger;
   the ledger is `runs` + `run_artifacts` info through `artifact_paths` JSON).
