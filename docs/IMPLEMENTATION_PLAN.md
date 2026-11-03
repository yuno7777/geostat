# GeoStat — Implementation Plan

Plan of record for the MVP. Each phase has an explicit exit gate. Status column is
maintained in DOCUMENTING.md §5 (not duplicated here, so the two cannot diverge).

| Phase | Deliverables | Exit criteria (all must hold) |
|---|---|---|
| 0 Recon & feasibility | env report, dataset verification, small sample fetch, risk list | Sample chips end-to-end inspectable with rasterio on the actual machine; metadata/split CSVs recorded; risks in journal. ✅ **done 2026-11-03** |
| 1 Contracts & architecture | docs/ set, ADRs, DB schema, API spec, pyproject | All docs below exist; ADRs cover every material choice; `pyproject.toml` builds. ✅ **done 2026-11-03** |
| 2 Ingest | `geostat download/ingest/validate`; checksummed assets; DB rows; validation reports | ≥ 1 scene fully validated with machine-readable report; `db.py` tested; sha256+md5 recorded per asset. ✅ **done** |
| 3 Tile pipeline | `geostat manifest`, deterministic tile set, event-disjoint splits, report | two manifest runs with same config produce byte-identical sha256; union/overlap of train/val/test scene sets is empty (asserted in code). ✅ **done** |
| 4 Baseline + model | `geostat baseline`, `geostat train` (U-Net), eval CLI | threshold baseline selected on val only; training run config recorded; per-event metrics computed from test tiles with test-only usage (no leakage); checkpoint+json exported. ✅ **done** |
| 5 Inference/export | `geostat analyze` (scene-level infer + COG + GeoJSON export) | outputs geometries + CRS verified; GeoJSON roundtrip parses; COG metadata matches source transform; nodata=nan preserved. ✅ **done** |
| 6 Exposure | `geostat exposure`; OSM cache; geodesic flooded length/area | exposure report JSON for an analyzed scene; OSM data presented with ODbL attribution; zero-data reported honestly. ✅ **done** |
| 7 API + map | `geostat serve`; read-only endpoints; static MapLibre UI | all API endpoints respond 200 or clean 404 (no stack traces); UI shows flood/reference/osm layers + metrics + exposure for ≥ 1 analyzed scene; loading/empty/error UI states exist. ✅ **done** |
| 8 Hardening & deliverable | security checklist, test suite, clean-clone verification | `pytest` green in CI-less env; every README command from a clean clone reproduces ≥ 1 end-to-end scene and its numbers. ✅ **done** |

## Dependency graph

```text
Phase 0 ─▶ 1 ─▶ 2 ─▶ 3 ─▶ 4 ─▶ 5 ─▶ 6 ─▶ 7 ─▶ 8
                             \____________/      (6 uses 5 scenes)
```

## Guardrails while building

- Until (4) exists, don't tune thresholds on anything except val split.
- Never turn a *test* split availability fact into a training fact.
- Every derived artifact gets a `code_revision` recorded from the same snapshot that
  created the input, so a re-run is attributable.
- (2) may not assume all chips fit in RAM: read per-window.
- (7) must not create POST/mutation surfaces in MVP; the API is read-only.

## Deferred to documented follow-ups (not MVP blockers)

- GPU wheels (ADR-005); OSM tile cache warming; multi-region COG mosaics; full Sentinel-2
  band path; PostGIS migration (data model is binary-compatible with it per ADR-001);
  reproducibility tests in CI (GitHub Actions yaml) once repo is initialized.
