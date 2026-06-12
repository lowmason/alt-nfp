# nfp-ingest

Data ingestion, vintage store management, panel construction, and compositing.

## Overview

Transforms raw downloaded data into analysis-ready panels. Provides:
- **Vintage store** (`vintage_store.py`): Hive-partitioned parquet read/write with rank-based horizon censoring. `store_path` accepts a local `Path` or an `s3://` `UPath` (default `VINTAGE_STORE_PATH` from `nfp_lookups.paths`, which is MinIO/S3 when `NFP_STORE_URI` is set); Polars I/O passes `storage_options_for(store_path)` and `mkdir` is guarded by `is_remote()`
- **Panel construction** (`panel.py`): `build_panel()` assembles CES + QCEW + provider data into a unified panel
- **CES/QCEW ingestion** (`ces_national.py`, `ces_state.py`, `qcew.py`): source-specific transformers
- **Provider ingestion** (`payroll.py`): auto-detects cell-level vs national providers
- **Compositing** (`compositing.py`): QCEW-weighted national compositing for cell-level providers
- **Indicator store** (`indicators.py`): download + read cyclical indicator parquets
- **Model data** (`model_data.py`): `build_model_data(as_of=D)` — the single entry point answering "what was knowable on D" (A2). Layer-1 `build_panel(as_of_ref=D)` + layer-2 extraction (best-available CES selection with vintage remap, QCEW noise multipliers incl. post-COVID boundary inflation, provider pub-lag censoring, cyclical pub-lag masking). Knobs in `ModelDataConfig` (defaults frozen from the reference settings). No plotting concerns, no acquisition imports.
- **Snapshots** (`snapshots.py`): hash-pinned ModelData artifacts (`.npz` arrays + embedded JSON meta) under `NFP_SNAPSHOTS_URI` (S3) or `data/snapshots/`. `content_hash` is over array bytes + canonical meta — never npz file bytes (zip timestamps). `alt-nfp snapshot --as-of D [--grid-end E]` writes them. Schema v2 (A3): provider meta carries `error_model` so `nfp_model.data.from_snapshot` can rebuild likelihood structure; v1 snapshots are read with an `"iid"` fallback.
- **Release dates** (`release_dates/`): config and vintage date builder

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.10)
- **Dependencies**: numpy, polars
- **Build**: hatchling
- **Internal deps**: `nfp-lookups` (schemas, industry, geography, revision schedules, provider config), `nfp-download` (BLS/FRED clients)

## Key Commands

```bash
# Run ingest tests
pytest tests/

# Lint
ruff check src/nfp_ingest/
```

## Package Structure

```
src/nfp_ingest/
├── __init__.py
├── base.py                 # validate_panel(), empty_panel() — uses schemas from nfp_lookups
├── vintage_store.py        # read/write vintage store, transform_to_panel(), rank-based censoring
├── ces_national.py         # CES national-level ingestion
├── ces_state.py            # CES state-level ingestion
├── qcew.py                 # QCEW ingestion (4 input streams, industry hierarchy)
├── payroll.py              # Provider index ingestion (auto-detects cell-level → compositing)
├── compositing.py          # QCEW-weighted national compositing for cell-level providers
├── indicators.py           # download_indicators(), read_indicator() — FRED cyclical indicators
├── panel.py                # build_panel(), save_panel(), load_panel()
├── aggregate.py            # Geographic aggregation (FIPS → division → region)
├── tagger.py               # Tag estimates with source/vintage metadata
├── releases.py             # Release management, combine_estimates()
└── release_dates/
    ├── __init__.py
    ├── config.py            # Release date path config (VINTAGE_DATES_PATH, etc.)
    └── vintage_dates.py     # build_vintage_dates() from release_dates.parquet
```

## Code Style

- **Formatter**: black (line length 100)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- Line length limit: 100 characters

## Key Patterns

- **Rank-based horizon censoring** (`vintage_store.py`): `transform_to_panel(lf, as_of_ref=D)` applies two-layer censoring: (1) `vintage_date <= D` + `ref_date < D` filtering prevents lookahead; (2) rank-based selection picks the correct revision per recency rank. CES uses `_select_ces_at_horizon` (rank 1→rev-0, rank 2→rev-1, rank 3+→rev-2 with `benchmark_revision=0`). QCEW uses `_select_qcew_at_horizon` with quarter-dependent revision rules. `_validate_censored_selection` runs fail-fast checks before data reaches the sampler. Growth is computed *before* rank selection to preserve per-vintage measurement error semantics.
- **QCEW-weighted compositing** (`compositing.py`): cell-level provider parquets (4 Census regions x 11 supersectors = 44 cells) are composited into a national growth series. `load_qcew_weights()` computes shares from vintage store data. `redistribute_weights()` reallocates from uncovered cells. `compute_provider_composite()` returns synthetic national employment (base=100).
- **Provider auto-detection** (`payroll.py`): `_is_cell_level()` checks `geographic_type='region'`; cell-level data routes through compositing, national data enters directly.
- **Panel schema validation**: `validate_panel()` enforces the schema from `nfp_lookups.schemas`. All panel DataFrames must pass validation before use.
- **MIN_PSEUDO_ESTABS_PER_CELL**: filtering threshold for cell-level compositing, defined in `nfp_lookups`.

## Test Mapping

Tests live in `tests/` within this package:
- `test_ingest.py` — panel validation & schema tests
- `test_new_ingest.py` — new ingest module tests
- `test_release_dates.py` — release date parsing/scraping tests
- `test_vintage_store.py` — vintage store + rank-based censoring + validation guards
- `test_compositing.py` — QCEW-weighted compositing tests
- `test_store_coverage.py` — store data-integrity + CES censored diagonal invariant
- `test_golden_masters.py` — A1 golden masters: censored panels vs frozen-reference fixtures in `s3://…/golden/a1/` (manifest in `tests/golden/`); self-skips without store env
- `test_model_data_golden.py` — A2 golden masters: `build_model_data` arrays/frames vs frozen-reference `panel_to_model_data` outputs in `s3://…/golden/a2/`; self-skips without store env + local providers/indicators
- `test_snapshots.py` — snapshot content hash (deterministic, order-insensitive, corruption-detecting), round-trip, and build-twice hash stability
- `test_cyclical_indicators.py` — NOT yet ported: depends on `nfp_models.panel_adapter`; comes over when knowability logic moves into the data layer (Phase A2)
- `test_fred.py` — lives in `packages/nfp-download/tests/` (imports only `nfp_download.fred`)
