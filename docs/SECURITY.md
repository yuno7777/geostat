# GeoStat — Security Notes

Scope: single-analyst, self-hosted MVP. The API ships **without authentication by
design**; binding to `127.0.0.1` by default. Any network-facing deployment must be put
behind a reverse proxy with its own authn/authz — documented here so the boundary is
explicit.

## Threat model (explicit, MVP scope)

| Asset | Threat | Mitigation in effect |
|---|---|---|
| Filesystem | Path traversal on `/files/{run_id}/{filename}` | Reject `..` anywhere, absolute paths, separators, and any filename not registered in the run's `artifact_paths` JSON (not merely existing on disk). Unit-tested (`tests/test_security.py`). |
| Filesystem | Arbitrary file upload → GDAL-driver DoS / decompression bomb | **No upload endpoint exists in MVP.** Ingestion paths are built by the CLI with allow-listed sources (official GCS bucket). |
| In-process | Huge rasters / OOM | Per-scene (512×512×2 float32) + per-tile reads bounded by tile manifest; no unbounded `read()`; max concurrency 1 in training by design. |
| Secrets | Leakage via logs / API errors | API statements `HTTPException` only; log paths redact home-abs paths (`settings.py` swallows `C:\Users\...` in public error strings). No secrets are used anywhere in MVP (no external credentials — GCS public read, OSM public API), so the secret surface is empty by construction. |
| Data governance | Attribution + license drift | Every external layer ships `license` + `attribution` in DB rows + UI footer; deletion path not implemented for raw data in MVP (documented). |
| Process | SQL injection | Parameterized SQL everywhere (`?`). No string-built SQL. Unit-tested lightly (test_api). |
| Process | Shell injection | No `subprocess` call from the app shell path. CLI commands do not interpolate user text into shell. |
| Network | SSRF via API | No user-supplied URL inputs accepted by API; OSM URL is a settings constant. |
| Frontend | XSS in map popups | Figures are numbers chosen from enumerations; UI avoids `innerHTML` with arbitrary strings. |

## Retention & deletion

`data/raw/` is treated as **content-addressed, append-only**; re-running the same ingest
is idempotent (checksum re-verified). No automated removal path exists for MVP; the
operator may delete `data/` and `artifacts/` folders manually, then re-run. Documented
in DEPLOYMENT.md.

## Auth / multi-user

Deferred explicitly. If multi-user is ever adopted: (1) never expose public write API
without OIDC; (2) `assets`/`runs` rows associate `study_area_id` and apply per-area
ACL; (3) log lines gain a `actor` field (already structured for it).

## Dependencies and supply chain

- PyPI wheels only from standard indexes; hashes default (verified by uv resolver).
- `torch` from PyPI CPU wheels (CPU-only decision per ADR-005); when switching to CUDA
  index, re-verify checksums. No binary layer built in-repo.
- No runtime SRIA; `uv` — the single tool creating environment.

## Known gaps (honest)

- No TLS anywhere (localhost only).
- No rate limiting on API (not needed for localhost MVP).
- `docs/SECURITY.md` not yet integrated with a real code scanner; a low-effort
  `bandit`/`safety` pass is a follow-up, not a blocker for MVP (rejected now because a
  new dependency supply chain would itself need review).
