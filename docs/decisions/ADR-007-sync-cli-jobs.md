# ADR-007 — Synchronous CLI jobs recorded in a `runs` ledger; no worker framework

**Status:** Accepted (2026-11-03)

## Context
The build prompt requires processing jobs + job logs. Options: Celery/RQ/Redis stack,
or FastAPI BackgroundTasks plus DB, or synchronous CLI-driven jobs each recorded as a
`runs` row with timing/config/env/artifacts/log-tail.

## Decision
Every mutating CLI command wraps its work as a run row: `runs` + `run_events` capture
status, exact config, timestamps, env snapshot, artifacts, and error. There is no job
queue, no daemon, no long-running worker in MVP. The API is strictly read-only
(ADR-001) and reflects whatever the CLI wrote.

## Consequences
- (+) Simple; each run is fully reproducible from its row (config_json + code_revision).
- (+) Works on a laptop; no daemon for an analyst.
- (−) The API cannot *start* jobs; a user cannot sit and watch progress from the browser.
  Accepted for MVP; a follow-up could add a tiny `POST /api/jobs/shell` behind auth,
  but only with a real security review first.

## References
ARCHITECTURE.md §3; DATABASE_SCHEMA.md; API_SPECIFICATION.md (read-only note).
