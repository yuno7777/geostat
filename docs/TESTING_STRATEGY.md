# GeoStat — Testing Strategy

Framework: **pytest**, small/unit tests utilising **synthetic raster fixtures** only —
no network or real dataset needed. Integration tests against the real dataset are
opt-in via `-m integration`.

## Where tests live

```text
tests/
  conftest.py          # tmp-path fixtures; deterministic seeds; synthetic raster builders
  test_geoio.py        # metadata extraction, alignment rules, nodata semantics
  test_tiling.py       # determinism, overlap, edges, split disjointness
  test_metrics.py      # known-answer metric cases
  test_baseline.py     # threshold selection, IoU math, boundary cases
  test_unet.py         # shapes, gradients flow, parameter count, mask-loss correctness
  test_inference.py    # prob-map stitching, exports roundtrip, GeoJSON validity
  test_exposure.py     # synthetic OSM JSON parse, geodesic lengths/areas sanity
  test_api.py          # FastAPI endpoints, empty/partial/full states, error contract
  test_security.py     # path traversal, oversized payload, safe-file-registry rules
```

## Command

```powershell
pytest                        # all unit tests
pytest -m integration         # needs data/ (downloads gitignored) — real dataset path
pytest -q                     # quiet
```

## Rules for test data

1. Synthetic rasters generated via `rasterio` in tmp dirs; no megabytes of fixture
   committed. A 512×512 float32 fixture is fine; larger arrays avoided.
2. Every geospatial test asserts **CRS/type/shape/transform** of outputs, not just
   data equality — geometry bugs (flip-y, skew) often break equality only subtly.
3. Ahead of any test that writes a database, call `init_db` from the same module so
   schema drift is caught immediately.
4. Any test that needs **determinism** is seeded explicitly (`random.seed`, `np.random.seed`,
   `torch.manual_seed`); nondeterminism is called out in the test comment.
5. No test mutates the repository cwd; use `tmp_path`.

## Coverage areas (mapping to the risk list)

| Risk class | Covered by |
|---|---|
| CRS/transform sneaky flip | `test_geoio.py::test_alignment_rejects_flip`, `test_inference exports roundtrip` |
| nodata-as-class | `test_metrics excludes nodata`, `test_unet mask-loss excludes nodata` |
| Tile seam artefacts | `test_inference stitching with stride == tile size yields no seams` |
| Leak across splits | `test_tiling.split_disjointness` |
| Determinism | `test_tiling.manifest_sha_bootstrap_is_stable` |
| Threshold tuning leakage | `test_baseline train_only_selection` (asserts tuning-data access guard) |
| GroupNorm / mask-loss correctness | `test_unet` synthetic forward/backward with nodata mask |
| Wrong geodesic length/area | `test_exposure geodesic sanity` |
| API contract stability | `test_api.py` golden shape assertions per endpoint, not exact text match |
| Path traversal on /files | `test_security.files_blocks_upward_traversal`, oversized names, unregistered-file names |

## Markers / hooks

- `@pytest.mark.integration` — asserts a network/download-dependent path; excluded by
  default so unit CI never touches the internet. (In `_pytest.ini`, not part of "unit".)
- No `pytest-asyncio`: FastAPI tested through `TestClient` sync wrapper (fastapi starlette).

## Exit criterion met by

- All tests green on a clean clone runtime.
- `pytest -m integration` covers ingest → manifest → baseline → train(use tiny model) →
  analyze → exposure; each step writes to tmp dir and DB fixture, so a repeatable
  end-to-end data-scene run is achievable in minutes.
