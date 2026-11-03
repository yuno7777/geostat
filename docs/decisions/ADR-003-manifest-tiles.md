# ADR-003 — Tiles are windows over source rasters, addressed by a manifest

**Status:** Accepted (2026-11-03)

## Context
Training on 512×512 chips directly would fix resolution/context; a tile/manifest
approach enables 256² tiles + augmentation. The naive alternative (materialize every
tile into a new raster on disk) doubles storage and risks georeferencing drift, plus
generates hundreds of files per scene.

## Decision
Tiles are **not** new rasters. A tile manifest stores `(scene, row, col, size, stride,
split, valid_frac, water_frac)` rows; the Dataset reads windows via
`rasterio.DatasetReader.read(band, window=Window(...))`. GeoTIFFs stay
immutable/content-addressed.

## Consequences
- (+) Zero duplicated storage; tiles always inherit exact source georeferencing.
- (+) Manifest is byte-stable and hashed; determinism testable (manifest sha256).
- (−) Random access per item has opening-time cost (rasterio open per read). Mitigated
  by keeping an open DatasetReader per scene in memory within a worker, so an epoch's
  reads are buffered by the OS rather than re-opened.
- (−) Occasional edge-less tiles at the chip bottom/right (512 not divisible by 256? It
  is: 512 = 2×256; but if a custom config sets size not dividing 512, edge handling:
  tiles that overflow are clipped and recorded with their actual height/width; inference
  stitching handles ragged tiles by writing at their true position).
- No silent re-projection — windows never resample (index-space). If a config ever asks
  for a different CRS, an ADR must first change this.

## References
GEO_PIPELINE.md §4; tests/test_tiling.py.
