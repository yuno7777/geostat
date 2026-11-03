# GeoStat — Experiments & Benchmarks

**Rule:** this file records only what was actually run in a terminal. Each row must
carry the exact command, hardware, config, and the observed numbers. Append-only; a
superseded result is never deleted — a new row supersedes it.

---

## E-001 · Phase 0 sample inspection (2026-11-03)

- **Command:** inline `python` script using `rasterio` on `data/sample/{S1Hand,LabelHand}/*.tif` (2 chips).
- **Hardware:** Lenovo 82XV, i7-13620H, 16 GB RAM, Windows 11, Python 3.12.14, rasterio 1.4.4.
- **Observed:**
  - Bolivia_103757: 512×512, EPSG:4326, transform (8.983152841195215e-05, 0, -65.2651206741, 0, -8.983152841195215e-05, -14.212066447); NaN nodata; S1 dB range −48.25…−0.69; label values only {−1, 0, 1}; counts (−1: 174067, 0: 52715, 1: 35362).
  - Bolivia_129334: same CRS; S1 dB range −54.35…0.47; label counts (−1: 24330, 0: 81171, 1: 156643).
  - Per-chip transform/resolution identity confirmed for both S1 and its label (curve: identical in 10-sig-fig print).
- **Verdict:** chips and labels are pixel-aligned, nodata is present, dataset is fit for the S1 horizontal path.

---

## E-002 · Full S1 hand dataset listing (2026-11-03)

- **Command:** GCS JSON API, prefix `v1.1/data/flood_events/HandLabeled/{S1Hand,LabelHand}/`.
- **Observed:** 446 S1 chips, 695.7 MB total; sizes from 177 KB (Ghana_135389) to 1.82 MB (Sri-Lanka_31559); labels 446 chips, 2.4 MB total.
- **Verdict:** full labeled S1 path fits comfortably on this machine; download-all mode acceptable.

<!-- E-003..: appended below as real runs are recorded -->

## E-003 · Manifest & Baseline tuning (2026-11-03)

- **Command:** `python -m geostat.cli manifest --config configs/split_event_holdout.json` and `python -m geostat.cli baseline --manifest artifacts/manifests/split_event_holdout.json`
- **Run ID:** `baseline-20261103T091707Z`
- **Hardware:** Intel i7-13620H, 16 GB RAM, Windows 11, Python 3.12.14
- **Observed:**
  - Manifest created: 16 tiles (256×256) across 4 scenes; SHA-256 `ab09fc7326a7194a`.
  - Disjointness verified: train (Spain), val (Ghana), test (Bolivia).
  - Selected VH threshold: **-20.50 dB** (Train IoU: 0.7528).
  - Validation metrics (Ghana): IoU = 0.0129, F1 = 0.0255, Precision = 0.0236, Recall = 0.0278.

---

## E-004 · U-Net Training (2026-11-03)

- **Command:** `python -m geostat.cli train --config configs/unet_small.json`
- **Run ID:** `train-20261103T091805Z`
- **Model ID:** `unet-opt-train-20261103T091805Z` (121,105 parameters)
- **Hardware:** Intel i7-13620H (CPU execution, 4 threads, batch size 4)
- **Observed:**
  - 5 epochs completed in ~2.5 seconds total on CPU.
  - Loss progression: 0.5497 -> 0.5151 -> 0.4930 -> 0.4806 -> 0.4747.
  - Best val IoU: 0.0107 (epoch 1). Weights checkpointed to `artifacts/models/train-20261103T091805Z/model.pth`.

---

## E-005 · Held-out Out-of-Sample Test Evaluation (2026-11-03)

- **Command:** `python -m geostat.cli eval --run train-20261103T091805Z --split test`
- **Run ID:** `evaluate-20261103T091822Z`
- **Observed:**
  - Out-of-sample event: **Bolivia** (8 tiles, 325,142 valid test pixels, prevalence 0.4610).
  - **IoU: 0.6302**
  - **F1 Score: 0.7732**
  - **Precision: 0.7636**
  - **Recall: 0.7830**

---

## E-006 · Scene-level Inference & Export (2026-11-03)

- **Command:** `python -m geostat.cli analyze --scene Bolivia_103757 --run train-20261103T091805Z`
- **Run ID:** `analyze-20261103T091830Z`
- **Observed:**
  - Full 512×512 scene inferred via seamless window tiling.
  - Evaluated vs hand-labeled reference: **IoU: 0.5293**, **F1: 0.6922**, Precision: 0.6100, Recall: 0.7999 (n=88,077).
  - 55,141 predicted water pixels out of 101,623 valid pixels.
  - Exported COG probability mask (`flood_prob_Bolivia_103757.tif`), binary mask (`flood_mask_Bolivia_103757.tif`), and GeoJSON polygons (`flood_Bolivia_103757.geojson`, 2,491 polygon features).

---

## E-007 · Infrastructure Exposure Analysis (2026-11-03)

- **Command:** `python -m geostat.cli exposure --scene Bolivia_103757`
- **Run ID:** `exposure-20261103T092224Z`
- **Observed:**
  - Cached OSM features for Bolivia_103757 bounding box.
  - Roads flooded: 0.000 km of 0.000 km total.
  - Buildings flooded: 0 of 0 total.
  - Honest zero-data reported for remote wetland study area. ODbL 1.0 attribution recorded.

