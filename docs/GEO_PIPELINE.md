# GeoStat — Geospatial Pipeline (CRS / resampling / tiling / export)

Scope of this doc: exact rules for geometry, CRS, resampling, nodata, tiling, and
exported artifacts. Everything here is enforced in code (`geoio.py`, `tiling.py`,
`inference.py`, `exposure.py`).

## 1. CRS and units

- All Sen1Floods11 raster chips are **EPSG:4326** at ~10 m/px (8.983152841195215e-05 deg).
- No re-projection is performed on the raster path for MVP. This is deliberate
  (ADR-003): the dataset ships a single CRS, so re-projection would only add error.
- Measurement (lengths/areas in the exposure stage) is **geodesic**, via
  `pyproj.Geod` on the WGS-84 ellipsoid. Report `m` and `m²`; no local UTM reprojection
  for MVP (avoid zone-selection bugs).

## 2. Nodata and label semantics (strict)

- S1 chips carry NaN nodata for pixels outside the valid footprint.
- Label chips: −1 = nodata/not valid, 0 = not-water, 1 = water.
- Loss masks nodata pixels exactly; metrics ignore nodata pixels; polygon extraction
  never emits nodata pixels.
- 0 is a real class ("not water"), never conflated with missingness.

## 3. Pixel alignment contract (always checked pre-use)

Two grids (image and label) are considered aligned iff:
1. same CRS, same width, same height, same band count expectation per kind;
2. `abs(a - b) < 1e-9` for every affine six-tuple (transform), resolution, bounds.
Violations are hard errors (no silent resample). Checked in `geoio.assert_alignment`.

## 4. Tiling (phase 3)

- Tile grid = {size, stride} defined per config (`tile_size=256, stride=256` default).
- Tile ids are stable, derived from (scene_id, row, col) — no dataset-global offsets so
  re-ordering is never relied on.
- Deterministic manifest generation:
  - dict iteration sorted, floats rounded to 6 decimals in JSON when compared,
  - `sort_keys=True`, `separators=(",",":")` so manifest is byte-stable,
  - manifest contains `code_revision` (source-tree sha at build time) so a later code
    change is visible in a diff.
- **Leakage-safe split.** Split unit = event (country prefix). All tiles of one scene
  belong to one split (their scene's event). Assertions in code guarantee absence of
  cross-split overlap of scenes — verified by unit test.
- Band computation: per tile `valid_frac` (fraction not-nodata pixels), `water_frac`,
  stored per tile for analyses (class imbalance reporting).

## 5. Normalization / preprocessing

Per band: `db_clipped = clip(x, -30.0, 5.0)`; then `norm = (db_clipped + 30.0) / 35.0`
→ range [0,1]. Both are recorded in run config; no other transformation (radiometric or
per-scene histogram scaling) is applied. Enhancements to another scheme require a new
ADR.

## 6. Model output → raster back-mapping (inference)

Tile inference runs on windows of `tile_size`; result stitching:
- prob map `float32 [0, 1]`, shape H×W from source raster;
- stride=tile_size ⇒ **no seams** (exactly non-overlapping); windows always fully inside
  the raster (`assert`), so ghost regions cannot appear;
- binary mask = (prob ≥ threshold), threshold from val-split tuning (else 0.5), recorded.

## 7. Exports

| Artifact | Format | CRS | Dtype | Nodata | Notes |
|---|---|---|---|---|---|
| `flood_prob_{scene}.tif` | COG (via rasterio COG driver) | source CRS | float32 | NaN | has `NEWBIT/skirt_tiles`? no; plain COG with integer scaling absent |
| `flood_mask_{scene}.tif` | COG | source CRS | uint8 | 255 | values 0,1; mask excludes NaN-source pixels |
| `flood_{scene}.geojson` | GeoJSON FeatureCollection (RFC 7946) | EPSG:4326 | polygon | — | polygons from binary mask via `rasterio.features.shapes`; validated by shapely; properties `{flood: 1}` only |
| `exposure_{scene}.json` | JSON summary | — | — | — | geodesic lengths/areas + counts + attribution + retrieval timestamps |

Roundtrip test in pytest: COG back-opened with rasterio must show identical transform,
CRS, bounds, shape; polygon count > 0 only when mask has positive pixels.

## 8. Vectorization

`rasterio.features.shapes(mask.astype('uint8'), mask == 1)`; output polygons are in
pixel space → immediately projected via the raster transform. Geometries validated
(“is_valid＝False ⇒ hard error”); holes/nested shapes are preserved naturally; a
tolerance-based simplify (1e-9 deg) is applied only for a stable output size; verified
that land indicator area after simplification remains within 0.1% of unsimplified
for the MVP test scenes (asserted for each on export in warn-only spirit).
