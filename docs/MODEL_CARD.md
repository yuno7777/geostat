# GeoStat — Model Card

**Model family:** S1-only flood water segmentation. Two models shipped in MVP:
(a) fixed VH-threshold baseline, (b) small U-Net.

## Model (a) — VH threshold baseline

- **Input:** `VH` band only (dB).
- **Decision:** water iff `VH < t_best` where `t_best` maximizes IoU over a grid on the
  **train** split; a val-split sanity check is computed but not used to swap t.
- **Intent:** transparent, fast floor model; if a deep model cannot beat it on our
  per-event split, that is a *bug in the deep model*, not a victory for it.
- **Pros:** interpretable, no training.
- **Cons:** cannot separate permanent/seasonal water from flood-specific patterns beyond
  a global dB gate; sensitive to incidence angle per event.

## Model (b) — small U-Net

| Field | Value |
|---|---|
| Architecture | 2-input U-Net, channels 8/16/32/64, 3 down sampling stages, GroupNorm, BCEWithLogits + Dice loss, AdamW lr 1e-3, cosine to ~0, weight decay 1e-4, epochs 25 (early-stopped on val IoU), batch 16, tile 256². |
| Params | ≈ 0.6 M total (verified at build; printed per run). |
| Input | 2 bands `[VV, VH]`, dB, clipped to [-30, 5], normalized to [0,1] by `(x + 30)/35`; nodata → 0 (and attention via mask). |
| Truth | `LabelHand` int16 −1/0/1, nodata excluded from loss and metrics. |
| Output | Sigmoid logit map; binary mask at threshold; per-CLS tile stitching is N/A (no CLS head); per-scene prob map + mask. |
| Trained on | Sen1Floods11 v1.1, Sentinel-1 GRD 10 m chips, split event-disjoint (ADR-004). Exact config preserved per run in SQLite + `configs/`. |

### Hardware

Recorded runs use **CPU-only** (torch 2.15.0+cpu), because the CUDA wheel set (~2.6 GB)
did not fetch in this environment within the ops time budget; see ADR-005. Any
GPU-obtained numbers will be recorded as a separate experiment (never overwriting).

## Honest statement of what the output *is*

The MVP model is **not** a flood forecast, not a calibrated risk probability, and not
an emergency-response-grade product. In-scope output: a *segmented flood-extent scene*
over a 512×512 S1 chip. Calibration curves, abstention, and probabilistic aleatoric/epistemic
uncertainty are **not** provided in MVP (future work).

## Per-event evaluation

Test events are *fully held out*, by event (ADR-004). Metrics reported only on labeled
(non-nodata) pixels: precision, recall, F1, IoU, plus confusion counts and water
prevalence. Aggregated across test tiles and written into SQLite + experiment log.

Real values live in `artifacts/runs/…/metrics.json` and EXPERIMENTS_AND_BENCHMARKS.md
after they exist; never presented as "expected results" here.

## Known limitations (exhaustive for MVP)

1. **Prevalence shift** — event classes differ hugely in water fraction; a global
   threshold can hurt small-water scenes. Per-event IoU reported so this is visible.
2. **Incidence-angle angle** for S1 relative to north varies by chip; GRD gamma0
   Normalization beyond not handled — noise floor in incidence mismatches event-based.
3. **Permanent water** could dominate some scenes; Sen1Floods11 labels do not split
   permanent vs. temporary water for the S1 path — so results are *water-fraction*
   classification, not "flooded-only-that-day". Exposed metrics treat all water as one.
4. **No temporal stack**: odd scene "sprites" (single S1 pass) are all we get.
5. **Chip granularity** — 512×512 chips, not mosaics; scenes are occlusion-restricted.
6. **Nodata-surrounding edges** — the chip borders with big NaN fringes (e.g., after tile
   split) can create hallucinated edges in the model; std. mitigation is mask-attention
   loss + geodesic decorrelation (future).

## Intended use

Research reproducibility + analyst triage of an S1 flood scene on a single laptop.
**Not** for safety-critical decisions. Citation of Sen1Floods11 paper required for any
published derivative that results from training on it.
