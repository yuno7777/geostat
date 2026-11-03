# ADR-005 — Recorded results run on CPU-only torch; GPU path documented, not used

**Status:** Accepted (2026-11-03); revisit when network/GPU wheels are viable.

## Context
The build machine has an RTX 4060 (8 GB) with driver 617.42, but torch CUDA wheels for
Windows are a single ~2.6 GB package bundling CUDA libs. Fetching it inside this session
exceeded the practical tool timeouts (opencode shell cap) twice; a half-installed CUDA
wheel is worse than a clean CPU wheel.

## Decision
All recorded experiments (E-001 onward) run on **torch CPU** (torch 2.15.0+cpu). The
GPU path is not silently claimed to work; `torch.cuda.is_available() == False` is
printed into every run env snapshot. If GPU wheels arrive (network change or lighter
cu126 wheel), they get their own experiment rows in EXPERIMENTS_AND_BENCHMARKS.md —
never replacing CPU numbers.

## Consequences
- (+) Honest and reproducible today; no claims hidden behind untested parallelism.
- (+) Training config sized for CPU: tile 256², batch 8—not 32, epochs 25 max, small
  U-Net (~0.6 M params). Record printed wall times.
- (−) Slower runs; input-layer performance results (e.g., tiles/s) are CPU facts.
- (+) Upgrade path is one command and is in DEPLOYMENT.md.

## References
EXPERIMENTS_AND_BENCHMARKS.md; DEPLOYMENT.md §1; MODEL_CARD.md.
