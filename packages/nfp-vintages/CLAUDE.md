# nfp-vintages

Vintage data pipeline for CES, QCEW, and SAE employment data.

## Overview

End-to-end pipeline: download → process → build for managing real-time data vintages. Provides:
- **Download** (`download/`): fetch CES triangular revision CSVs and QCEW bulk files from BLS
- **Processing** (`processing/`): transform raw downloads into revision-tagged parquet
- **Store builder** (`build_store.py`): merge revisions + current estimates into `data/store/`
- **Views** (`views.py`): `real_time_view()`, `final_view()`, `specific_vintage_view()`
- **Evaluation** (`evaluation.py`): `vintage_diff()`, noise multiplier construction
- **CLI**: `alt-nfp` (or `python -m nfp_vintages`)

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.10)
- **Dependencies**: httpx, polars, typer (CLI)
- **Build**: hatchling
- **Internal deps**: `nfp-lookups` (industry, geography, revision schedules), `nfp-download` (HTTP client), `nfp-ingest` (vintage store read/write)

## Key Commands

```bash
# Run all vintage pipeline steps
uv run alt-nfp

# Individual steps
uv run alt-nfp download            # Download CES triangular + QCEW bulk
uv run alt-nfp download-indicators # Download FRED cyclical indicators
uv run alt-nfp process             # Scrape BLS calendar + process revisions
uv run alt-nfp current             # Fetch current BLS estimates (benchmark-revised)
uv run alt-nfp build               # Merge revisions + current → data/store/
uv run alt-nfp build --releases PATH

# Run vintage tests
pytest tests/

# Lint
ruff check src/nfp_vintages/
```

## Package Structure

```
src/nfp_vintages/
├── __init__.py             # Exports: real_time_view, final_view, vintage_diff
├── __main__.py             # CLI entry point (typer app)
├── views.py                # real_time_view(), final_view(), specific_vintage_view()
├── evaluation.py           # vintage_diff(), build_noise_multiplier_vector()
├── build_store.py          # Merge revisions + current → Hive-partitioned store
├── download/
│   ├── __init__.py
│   ├── ces.py              # Download cesvinall.zip (CES triangular revision CSVs)
│   └── qcew.py             # Download QCEW yearly singlefile ZIPs from BLS
└── processing/
    ├── __init__.py
    ├── ces_national.py     # CES national vintage processing
    ├── qcew.py             # QCEW vintage processing (4 input streams)
    ├── sae_states.py       # State and Area Employment processing
    └── combine.py          # Combine vintage files
```

## Code Style

- **Formatter**: black (line length 100)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- Line length limit: 100 characters

## Key Patterns

- **Vintage store format**: Hive-partitioned parquet at `data/store/`, partitioned by `(source, seasonally_adjusted)`. Sources: `ces`, `qcew`, `sae`.
- **QCEW bulk download** (`download/qcew.py`): downloads yearly singlefile ZIPs, filters to `own_code in {0,1,2,3,5}` and `agglvl_code in {10,11,14,15,50,51,54,55}`. Saves as `qcew_bulk.parquet`.
- **QCEW processing** (`processing/qcew.py`): four input streams: (1) total all-ownership, (2) private 2-digit NAICS, (3) government by ownership (→ sectors 91/92/93), (4) manufacturing 3-digit NAICS (→ durable 31 / nondurable 32). Employment units converted from persons to thousands.
- **CES processing** (`processing/ces_national.py`): parses triangular revision CSV structure from `cesvinall/`, assigns vintage dates from release schedule.
- **SAE processing** (`processing/sae_states.py`): State and Area Employment, fetched via httpx.
- **Views** (`views.py`): pure Polars operations on vintage DataFrames. `real_time_view()` returns what was known at a given date. `final_view()` returns latest available revision.
- **Evaluation** (`evaluation.py`): `vintage_diff()` computes revision magnitudes. `build_noise_multiplier_vector()` constructs empirical noise multipliers by source and revision. Uses `nfp_lookups.revision_schedules` for CES/QCEW revision specs.
- **CLI** (`__main__.py`): typer app with subcommands: `download`, `download-indicators`, `process`, `current`, `build`. Each step is idempotent.

## Data Layout

```
data/
├── store/                  # Output: Hive-partitioned vintage store
│   ├── source=ces/
│   ├── source=qcew/
│   └── source=sae/
├── downloads/              # Input: raw fetched files
│   ├── ces/cesvinall/      # CES triangular revision CSVs
│   ├── qcew/               # QCEW bulk + revisions
│   └── releases/           # Scraped BLS schedule HTML
└── intermediate/           # Pipeline byproducts
    ├── ces_revisions.parquet
    ├── qcew_revisions.parquet
    ├── revisions.parquet
    ├── release_dates.parquet
    └── vintage_dates.parquet
```

## Test Mapping

Tests from the monorepo `tests/` that belong here:
- `test_vintages.py` — vintage view & evaluation tests
