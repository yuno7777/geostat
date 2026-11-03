# GeoStat — Changelog

All notable changes, chronological. Newest entries at top. Doc-only updates get one line.

## 2026-11-03 (build session, day 1)

- **Phase 0 verification.** Environment recon captured in DOCUMENTING.md §2; sample chips
  (Bolivia_103757, Bolivia_129334) fetched from official GCS and verified end-to-end with
  rasterio (alignment, CRS, nodata, label distribution). Sample + reference files stored
  under `data/`. E-001/E-002 recorded in EXPERIMENTS_AND_BENCHMARKS.md.
- **Phase 1 scaffolding.** `pyproject.toml` (+ hatchling, src-layout, pytest config),
  `.gitignore`, README, and full documentation set:
  ARCHITECTURE, IMPLEMENTATION_PLAN, DATA_CATALOG, GEO_PIPELINE, MODEL_CARD,
  API_SPECIFICATION, DATABASE_SCHEMA, TESTING_STRATEGY, SECURITY, DEPLOYMENT,
  EXPERIMENTS_AND_BENCHMARKS, CHANGELOG + ADR-001..007.
- **Environment:** Python 3.12.14 via uv; torch 2.15.0+cpu (CUDA wheels abstracted by
  ADR-005 after 2.6 GB wheel fetch timed out); rasterio 1.4.4 / numpy 2.2.6 / pyproj 3.8.0
  / shapely 2.2.0; fastapi 0.162.0 / pydantic 2.14.2; pytest 8.4.2.
- **Phase 2–8 completion & verification (2026-11-03).**
  - Ingest & asset registry with SHA-256 and MD5 integrity verification (`download.py`, `ingest.py`, `db.py`).
  - Deterministic tile manifest pipeline with leakage-safe event-disjoint splits (`tiling.py`).
  - Baseline threshold search and evaluation on train/val splits (`baseline.py`).
  - Lightweight U-Net training and evaluation on held-out out-of-sample event (`unet.py`, `train.py`, `metrics.py`), achieving 0.6302 test IoU.
  - Scene-level inference and export generating probability COGs, binary mask COGs, and GeoJSON polygons (`inference.py`).
  - Infrastructure exposure intersecting flood masks with OpenStreetMap roads and buildings using geodesic pyproj measures (`exposure.py`).
  - FastAPI read-only service with full JSON schema contracts and static file serving (`api.py`).
  - Static MapLibre GL frontend client (`frontend/`) with layer toggles, metrics, exposure, and artifact downloads.
  - Comprehensive automated test suite (`tests/`) covering geo I/O, tiling, metrics, unet, inference, exposure, API contracts, and security path-traversal prevention (27/27 tests green).
- Known gaps: no git repo yet (commit pending explicit user approval), no GPU path in
  recorded results, no Docker.

<!-- subsequent entries appended per phase verification -->
