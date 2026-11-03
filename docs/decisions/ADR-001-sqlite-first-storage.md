# ADR-001 — SQLite as the MVP storage engine, PostGIS-safe schema

**Status:** Accepted (2026-11-03)

## Context
MVP needs: metadata for ~450 raster chips, run traces, exposure summaries, and a
FastAPI query surface. PostgreSQL+PostGIS is the classic choice, but it adds a daemon,
container/DDL lifecycle, and install friction on a single-analyst laptop. No MVP query
actually needs spatial indexing: spatial work happens in shapely/pyproj on in-memory
feature collections, and the rasters live on the filesystem (never in DB rows).

## Decision
Use a single SQLite file (`data/geostat.sqlite3`) with parameterized SQL and an
idempotent `init_db`. Schema is designed for later migration: no SQLite-specific
column types, no triggers, no R*Tree usage.

## Consequences
- (+) One command install; single file backup/restore; zero admin.
- (+) DB stores only metadata; big assets remain files, checksummed.
- (−) No concurrent writers beyond one process — fine for CLI-only writes (ADR-007).
- (+) Migration path to Postgres/PostGIS later is a straightforward DDL port when a
  measured need (multi-user, big vector volumes) shows up. Adding it earlier would be
  the classic YAGNI trap for a single analyst.

## References
DATABASE_SCHEMA.md; ARCHITECTURE.md §1.
