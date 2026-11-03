# ADR-004 — leakage-safe evaluation uses an **event-disjoint holdout**

**Status:** Accepted (2026-11-03)

## Context
The official Sen1Floods11 `flood_train_data.csv` / `flood_valid_data.csv` /
`flood_test_data.csv` split removes Bolivia/Colombia/Cambodia — those events never
appear in the official splits, so a "use the official test split" workflow silently
*ignores whole geographies*. Also, scene-level splits (the official ones) allow events
from the same flood hotspot to show up in train *and* test, which can leak.

## Decision
Primary evaluation uses **event-disjoint holdout**: specific events (country prefixes)
are fully excluded from training and used for validation/test. Chosen holdout for MVP:
`test_events = [Bolivia, Spain]`, `val_events = [India, Ghana]`, all others train.
Official splits are still ingested and shown (`scenes.official_split`) for
comparability but are not the primary metric source.

## Consequences
- (+) Test events are out-of-sample geographies, matching honest map-use reality.
- (+) Per-event metric reporting is meaningful (two distinct test events).
- (−) Less training data than possible.
- (−) Threshold baseline tuning must occur on train only — val is test-adjacent and the
  tuning guard is unit-tested to enforce "no test pixels".

## References
DOCUMENTING.md §10; tests/test_tiling.py::test_split_disjointness; MODEL_CARD.md.
