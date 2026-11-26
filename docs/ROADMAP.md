# GeoStat Windows and Linux roadmap

**Goal.** Make GeoStat a reproducible flood mapping and infrastructure exposure
workstation on Windows and Linux. Analysts should be able to process large
scenes, inspect model and data quality, and share portable results.

CPU is the baseline on both platforms. Validate user-facing CLI and API
behavior on the Windows and Linux runner matrix. Measure performance on named
hardware and runner classes. Treat new scientific outputs as experimental until
a held-out evaluation verifies their assumptions and limitations.

## Milestone 1 · Reliable scene processing

- [x] Add a pinned real-data smoke test for the CLI, COGs, GeoJSON, and API.
- [x] Add Windows and Linux jobs for Python 3.12 and 3.13 and exercise temporary paths with spaces and Unicode.
- [x] Write inference COGs from tiled scratch rasters and accumulate reference metrics by window. Retain the in-memory Python API.
- [x] Stream GeoJSON features from the complete on-disk wet mask and check connectivity and holes across tile boundaries.
- [x] Add repeatable benchmarks for export time, process memory, and staging bytes by stage, with reports from Windows and Linux CI.
- [ ] Evaluate tiled polygon stitching on larger, highly fragmented scenes using the benchmark reports.
- [ ] Add resumable `analyze-batch` with bounded concurrency, progress, cancellation and input fingerprinting.
- [ ] Record artifact hashes and add `geostat doctor` for GDAL drivers, CPU/GPU and storage checks.

## Milestone 2 · Scientific validity and inputs

- [ ] Evaluate event-disjoint splits across regions, land cover and flood sizes. Publish precision, recall, IoU and uncertainty intervals.
- [ ] Benchmark overlap and blending against non-overlapping inference.
- [ ] Distinguish new flooding from permanent water using a reference image or water layer. Export confidence, unknown pixels and observation quality.
- [ ] Accept user VV/VH GeoTIFFs and STAC inputs with explicit grid, unit and resampling contracts.
- [ ] Report inundated area, flooded road length within the area of interest and affected building fractions. Support administrative grouping, OSM multipolygons and user-provided infrastructure layers.

## Milestone 3 · Analyst workflow

- [ ] Add draw/upload for areas of interest, pre/post map comparison, pixel confidence and exposure filters.
- [ ] Serve COG previews and viewport vector tiles. Keep portable GeoJSON for small exports.
- [ ] Export GeoPackage, GeoParquet, CSV and STAC metadata.
- [ ] Add offline project folders and cached data, models and infrastructure. Provide equivalent PowerShell and Bash workflows.
- [ ] Expose API job progress, cancellation and pagination. Evaluate shared storage and multi-user operation after the workstation workflow is stable.

## Milestone 4 · Releases and operation

- [ ] Build and test installed wheels on both platforms and publish tagged versions with checksums. Verify GPU installation separately.
- [ ] Record per-stage timing and memory, tile throughput, cache hits and diagnostic events keyed by scene and run.
- [ ] Add performance regression gates tied to runner class and output equivalence.
- [ ] Document native workstation installs, headless Linux scheduled execution, and an optional standardized container.
