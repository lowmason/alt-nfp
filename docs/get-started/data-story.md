# The data story

Understanding where the data lives — and doesn't live — is essential before working with
alt-nfp. This page explains the three-part contract that governs data access throughout the
codebase.

## The vintage store lives in MinIO/S3, not on disk

alt-nfp is designed to run in a **container environment with a minimal disk footprint**. Every
persistent artifact lives in object storage selected by environment variables:

- `NFP_STORE_URI` — the Hive-partitioned CES/QCEW vintage store
- `NFP_SNAPSHOTS_URI` — hash-pinned ModelData `.npz` snapshots
- `NFP_DATA_URI` — indicators, consensus forecasts, and release-date schedules
- `NFP_PROVIDERS_URI` — proprietary provider data (a separate store, not seeded by this repo)

When any of these variables is **unset**, the code falls back to a local `data/` subdirectory.
That fallback is intentional for development and CI — but on a production deployment (Bloomberg)
all four must point to S3.

All path resolution goes through `nfp_lookups.paths`. No other package constructs data paths
directly — this is a hard rule enforced at code review.

## `data/` is proprietary and gitignored

The `data/` directory contains real-time BLS data vintages captured over time. This is
**proprietary operational data** — it is gitignored and never committed to the repository.

This repo is **public**. The code, specs, and documentation are open; the data are not.

Consequences for contributors:

- You cannot reproduce a full production run without store credentials.
- The test suite is written to **self-skip** any test that requires the vintage store when
  the store is unavailable. Missing store = skipped tests, not failures.
- Network tests (live BLS/FRED fetches) are decorated with `@pytest.mark.network` and are
  excluded from CI:

    ```bash
    uv run pytest -m "not network" --no-cov       # local suite — no live fetches
    uv run pytest -m "network" --no-cov           # run network tests explicitly
    ```

## Examples in these docs use illustrative data

The vintage store encodes real non-farm payroll revisions captured from BLS in real time. That
data is not in this repo.

All examples and code snippets in this documentation use **synthetic or illustrative values**
— never real vintage data. When a snippet shows an employment level, a revision number, or a
vintage date, treat it as a stand-in chosen for clarity, not a production observation.

## Rebuild vs. capture

The store is **reconstructable from public BLS sources** for its CES/QCEW backbone (2017+).
The one-time bootstrap script (`scripts/bootstrap_store.py`) performs this reconstruction.
After bootstrap, the everyday CLI (`alt-nfp update`) appends new monthly captures
incrementally.

The split matters:

- **Bootstrap** — bulk triangular reconstruction, run once; targets a scratch prefix
  (`s3://…/store-rebuild`) and is promoted to canonical via a deliberate copy-then-delete
  cutover. Never writes directly to the canonical store.
- **Capture** — real-time BLS API current-print for month T, appended to the canonical store
  by `alt-nfp update`. Idempotent; safe to run more than once for the same month.

Proprietary provider data (`NFP_PROVIDERS_URI`) is **not** reconstructable from public sources
and is not seeded by this repo.

## Summary

| What | Where | Public? |
|---|---|---|
| Code, specs, docs | This repo | Yes |
| CES/QCEW vintage store | `NFP_STORE_URI` (S3) | No (proprietary operational data) |
| ModelData snapshots | `NFP_SNAPSHOTS_URI` (S3) | No |
| Provider data | `NFP_PROVIDERS_URI` (S3) | No |
| `data/` on disk | Local fallback (dev/CI only) | Gitignored |
