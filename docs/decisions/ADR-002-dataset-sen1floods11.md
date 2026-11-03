# ADR-002 — Dataset = Sen1Floods11 v1.1, Sentinel-1 hand-labeled path only

**Status:** Accepted (2026-11-03)

## Context
The product contract needs real, georeferenced flood imagery with reference labels,
small per-file granularity, and an open/public origin. Candidates: Sen1Floods11,
Sentinel-1/2 L2A GRD + own labels, Copernicus EMS mapping, own scrape.

## Decision
Adopt **Sen1Floods11 v1.1 hand-labeled Sentinel-1 chips** as the sole source for the
MVP vertical slice; ignore the S2 path and JRC permanent-water layers for now. Fetch
from the **official GCS bucket** (`gs://sen1floods11`) rather than mirrors when
possible.

## Reasons
- Hand labels directly tied to each chip; no label-invention risk.
- Consistent CRS (EPSG:4326), 10 m/px, 512×512 chips; S1 only — small (696 MB total).
- Academic reference version (Bonafilia et al., CVPR-W 2020) with established
  train/valid/test split CSVs for comparability.
- Per-event metadata (dates, orbit, VH threshold) available.

## Consequences
- (+) Strong provenance; checksummed; no scraping.
- (−) Geographic domain fixed to 12 events in the dataset; generalization outside is
  unverified (documented in MODEL_CARD).
- (−) License is "open for research, cite the paper" with **no explicit LICENSE file**
  discovered — treat as research-use, no redistribution. Recorded in DATA_CATALOG.md.

## References
DATA_CATALOG.md; EXPERIMENTS_AND_BENCHMARKS.md E-001/E-002.
