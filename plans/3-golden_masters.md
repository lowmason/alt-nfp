# Implementation Plan: A1 — Golden-Master Censoring Fixtures

> **Status: ✅ COMPLETE (2026-06-12).** All steps done:
>
> - 10 fixtures generated read-only from the old repo
>   (`scripts/generate_golden_masters.py` run with the old venv's
>   interpreter): 9 censored panels (11,020–17,996 rows) + `provider_G`
>   (83 rows), uploaded to `s3://alt-nfp/golden/a1/`; manifest committed at
>   `packages/nfp-ingest/tests/golden/a1_manifest.json`.
> - `packages/nfp-ingest/tests/test_golden_masters.py`: **11/11 pass** —
>   9 value-identical panel comparisons, 1 negative master (2026-02-12
>   raises on the shutdown ref-gap), 1 provider comparison. Verified to
>   skip cleanly without `.env` (CI mode).
> - Old repo's `data/providers/` rsynced to local `data/providers/`
>   (gitignored) for the provider comparison.
> - Full suite: 372 passed / 1 intentional skip; ruff clean.
> - Frontier finding documented below (2026-02-12 → negative master;
>   positive frontier moved to 2026-01-12, which also exercises the
>   shutdown revision-fallback path).

Phase A1 of `plans/0-port_and_staged_plan.md`: generate censored panels in
the **old** repo (`~/Projects/alt_nfp`, frozen reference) for as-of dates
that exercise the known censoring edge cases; the new repo must reproduce
them value-identical. These masters are the safety net for the A2 seam
refactors and the regression gate for everything that touches the data layer.

## Scope decision: what a master is (and isn't)

- A master is the output of the **data layer**: `build_panel(start_year=2012,
  end_year=2026, as_of_ref=D, providers=[])` — the rank-based censored
  CES+QCEW panel (PANEL_SCHEMA). Both repos share this API (ported by copy).
- Provider rows are pinned by **one** uncensored fixture per provider
  (`ingest_provider(cfg)` output — payroll ingestion + QCEW-weighted
  compositing). Provider *knowability* (publication lags, staleness masking)
  is layer-2 logic living in the old repo's `nfp_models.panel_adapter`; it
  has no data-layer home until A2's `model_data(as_of=D)`. The
  stale-provider as-of date is included in the date set now so the panel
  fixture exists, but the staleness *behavior* gets golden-mastered in A2.
- Cyclical indicators are out of scope (FRED inputs, layer-2 masking — A2).

## Deviation from the plan of record

The port plan says "commit them as fixtures". This repo is **public** and
provider-derived fixtures are proprietary, so: fixtures live in
**`s3://alt-nfp/golden/a1/`** (same MinIO as the canonical store), and only
`packages/nfp-ingest/tests/golden/a1_manifest.json` is committed — dates,
row counts, schema, sha256 of each fixture file, provider config fields, and
generator provenance (old-repo commit, polars version). Tests compare
**values** (sorted-frame equality), not file bytes, so the manifest hashes
are transport-integrity only — value comparison is robust to parquet
encoding drift between polars 1.38 (old venv) and 1.41 (new).

## Fixture dates (day-12 BLS convention, store frontier 2026-02-16)

| as_of_ref | Edge case |
|---|---|
| 2020-05-12 | COVID era break: Apr-2020 collapse at rev-0 |
| 2023-07-12 | Mid-sample control month |
| 2024-09-12 | QCEW Q1 max-revision rule (Q1-2024 pub 2024-08-21) |
| 2024-12-12 | QCEW Q2 rule (Q2-2024 pub 2024-11-20) |
| 2025-02-12 | January benchmark print (pub 2025-02-07) |
| 2025-03-12 | QCEW Q3 rule (Q3-2024 pub 2025-02-19) |
| 2025-07-12 | QCEW Q4 rule (Q4-2024 pub 2025-06-04) |
| 2025-11-12 | Stale-provider month (behavior gated in A2) |
| 2026-01-12 | Frontier + shutdown rev-fallback (doubled rev-1 → fallback path) |
| 2026-02-12 | **Negative master**: must raise (see below) |

QCEW dates straddle one full publication cycle so each quarter's
max-revision rule {Q1:4, Q2:3, Q3:2, Q4:1} is hit with a fresh release.

**Frontier finding (discovered during generation):** the originally chosen
frontier 2026-02-12 is *correctly unbuildable* — the 2025 government
shutdown left Oct/Nov-2025 CES supersector detail unpublished until the
2026-02-16 make-up print, so the censored selection has a ref-month gap and
`_validate_censored_selection` fail-fasts (in both repos, by design). It is
pinned as a **negative master**: the test asserts `build_panel(as_of_ref=
2026-02-12)` raises `ValueError` mentioning "ref_date gap". The positive
frontier master moved to 2026-01-12, which additionally exercises the
shutdown-induced revision-fallback path (national series with doubled
rev-1).

## Steps

### 3.1 Generation (old repo, read-only)

`scripts/generate_golden_masters.py` (lives in this repo) run with the old
repo's existing venv interpreter directly
(`~/Projects/alt_nfp/.venv/bin/python`) — **not** `uv run`, which would sync
and mutate the frozen repo. The script:

1. Builds the 9 censored panels via the old repo's `build_panel`
   (old local store, `providers=[]`, explicit `start_year=2012,
   end_year=2026`).
2. Builds one provider fixture per old-repo `PROVIDERS` entry via
   `ingest_provider` (imports `nfp_models` config from the old venv).
3. Writes everything + `a1_manifest.json` to a staging dir in **this** repo
   (`data/golden_staging/`, gitignored). Nothing is written to the old repo.

### 3.2 Publication

Upload staging → `s3://alt-nfp/golden/a1/` (s3fs, same env contract as the
store); commit the manifest to `packages/nfp-ingest/tests/golden/`.

### 3.3 Provider inputs for the new repo

`rsync` the old repo's `data/providers/` → this repo's `data/providers/`
(gitignored), so the new repo can run `ingest_provider` for the comparison.

### 3.4 The golden-master test (new repo)

`packages/nfp-ingest/tests/test_golden_masters.py`:

- Skips when the store or the golden prefix is unreachable (same probe
  pattern as `test_store_coverage`); CI therefore skips.
- Per date: `build_panel(start_year=2012, end_year=2026, as_of_ref=D)`
  against the canonical S3 store, compare to the fixture
  (`pl.read_parquet(s3://…, storage_options=…)`): identical schema, row
  count vs manifest, and sorted-frame value equality.
- Provider: reconstruct `ProviderConfig` from manifest fields, run
  `ingest_provider` against local `data/providers/`, compare the same way;
  self-skips if provider files absent.

### 3.5 Gate

All golden-master tests pass against the canonical store; suite + ruff
green. Annotate A1 in `plans/0-port_and_staged_plan.md`.

## Risks / notes

- Old (1.38) vs new (1.41) polars could in principle differ in float
  arithmetic; value comparison will surface it immediately and any diff
  becomes an A1 finding, not a silent drift.
- The old repo's panel pulls from its local store, the new repo from
  `s3://alt-nfp/store` — verified value-identical (plans/2), so a master
  mismatch means *code*, not data.
- `end_year` is pinned to 2026 in both generator and test because
  `build_panel` defaults it to `date.today().year`.
