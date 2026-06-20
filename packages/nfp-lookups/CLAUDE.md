# nfp-lookups

Static reference data, schemas, and path configuration for the NFP monorepo.

## Overview

This is the foundation package with no internal dependencies. It provides:
- **Schemas**: `PANEL_SCHEMA`, `VINTAGE_STORE_SCHEMA`, `CES_VINTAGE_SCHEMA`, `QCEW_VINTAGE_SCHEMA`
- **Industry hierarchy**: NAICS → sector → supersector → domain mappings, CES series ID construction
- **Geography hierarchy**: FIPS → state → division → region mappings
- **Revision schedules**: CES and QCEW vintage timing, noise multipliers by revision number
- **Benchmark revisions**: Historical actual BLS benchmark revision amounts
- **Path config**: Canonical data directory layout (`BASE_DIR`, `DATA_DIR`, `STORE_DIR`, etc.)
- **Provider config**: `ProviderConfig` dataclass, `CYCLICAL_INDICATORS` definitions
- **Government interventions** (`government.py`, Track B): `GovIntervention` dataclass +
  the **announcement-dated** `KNOWN_INTERVENTIONS` table, `get_known_interventions_as_of(as_of)`
  (the lookahead-safe censor — a DATE comparison), `intervention_column()` (change-space
  `pulse`/`box`/`tc` shape encoders), and `GOVERNMENT_INDICATORS` (FRED `90–93` SA entries,
  diagnostics-only). Reference data for the government-wedge model. Spec: `specs/completed/government_wedge.md`.

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.12)
- **Dependencies**: numpy, polars, universal-pathlib + s3fs (S3-backed store paths)
- **Build**: hatchling

## Key Commands

```bash
# Run lookups tests
pytest src/nfp_lookups/tests/

# Lint
ruff check src/nfp_lookups/
```

## Package Structure

```
src/nfp_lookups/
├── __init__.py
├── paths.py                # BASE_DIR, DATA_DIR, STORE_DIR, DOWNLOADS_DIR, INTERMEDIATE_DIR, etc.
├── schemas.py              # PANEL_SCHEMA, VINTAGE_STORE_SCHEMA, CES/QCEW_VINTAGE_SCHEMA
├── industry.py             # NAICS → supersector → domain hierarchy + CES series ID map
│                           #   NAICS3_TO_MFG_SECTOR, SINGLE_SECTOR_SUPERSECTORS,
│                           #   GOVT_OWNERSHIP_TO_SECTOR, CES_SECTOR_TO_NAICS
├── series_ids.py           # BLS LABSTAT series-ID grammar (CE/SM/EN): build_series_id, parse_series_id
├── geography.py            # FIPS_TO_DIVISION, FIPS_TO_REGION, STATES, REGION_NAMES, etc.
├── revision_schedules.py   # CES_REVISIONS, QCEW_REVISIONS, get_noise_multiplier, vintage date helpers
├── benchmark_revisions.py  # BENCHMARK_REVISIONS dict (historical actuals)
├── provider_config.py      # ProviderConfig dataclass, CYCLICAL_INDICATORS
└── government.py           # GovIntervention, KNOWN_INTERVENTIONS, intervention encoders (Track B)
```

## Code Style

- **Formatter**: black (line length 100)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- Line length limit: 100 characters

## Key Patterns

- **Industry hierarchy** (`industry.py`): Three-level mapping: NAICS sectors → BLS supersectors (10, 20, ..., 90) → domains (05–08). Special cases: manufacturing splits into durable (31) / nondurable (32) via `NAICS3_TO_MFG_SECTOR`; government maps ownership codes to sectors 91/92/93 via `GOVT_OWNERSHIP_TO_SECTOR`. `SINGLE_SECTOR_SUPERSECTORS` (20/50/80) produce sector rows (23/51/81) directly from NAICS. QCEW national has 38 industry combos; CES has 35.
- **Revision schedules** (`revision_schedules.py`): `get_noise_multiplier(source, rev)` returns the empirical noise multiplier for a given source and revision number. CES has revisions 0–2; QCEW has quarter-dependent revision counts (`Q1: 4, Q2: 3, Q3: 2, Q4: 1`). `revision_schedules.py` must NOT import from other packages — vintage dates path should be passed as a parameter.
- **ProviderConfig**: dataclass defining provider name, file paths, error structure (`iid`/`ar1`), optional `birth_file`. Used by ingest and models packages.
- **CYCLICAL_INDICATORS**: dict mapping indicator names to FRED series IDs and metadata (frequency, publication lag). Used by both ingest (download) and models (censoring).
- **Schemas are Polars-native**: defined as `dict[str, pl.DataType]` for use with `pl.DataFrame.cast()` and validation.
- **Paths** (`paths.py`): every path constant derives from `BASE_DIR`. Discovery precedence: `NFP_BASE_DIR` env var (set before first import) → walk up to the first dir containing `packages/` + `pyproject.toml` → fixed-depth fallback for editable installs. Includes pipeline artifact paths (`RELEASE_DATES_PATH`, `VINTAGE_DATES_PATH`, `RELEASES_DIR`, `VINTAGE_STORE_PATH`) — other packages must import paths from here, never define their own.
- **Store location** (`paths.py`): `VINTAGE_STORE_PATH` is the switch point — a `upath.UPath` (S3 via s3fs) when `NFP_STORE_URI` is set (with `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_ENDPOINT_URL`, region default `us-east-1`), else the local `STORE_DIR`. `storage_options_for(path)` builds the Polars/object_store options for a remote path (`aws_allow_http` derived from an `http://` endpoint); `is_remote(path)` guards directory-only operations like `mkdir`. Env is read at import time.
- **Series-ID grammar** (`series_ids.py`): `build_series_id()` / `parse_series_id()` for CE/SM/EN. Pure reference data; `nfp_download.bls` re-exports it. This package imports nothing from other `nfp_*` packages — keep it that way.

## Test Mapping

Tests live in `src/nfp_lookups/tests/` within this package:
- `test_lookups.py` — industry hierarchy & revision schedule tests
- `test_series_ids.py` — BLS series-ID grammar (registry, build/parse) tests
- `test_paths.py` — base-dir discovery (env override, marker walk, fallback), store location (`NFP_STORE_URI` → UPath, `storage_options_for`), derived layout
- `test_revision_schedules.py` — noise multiplier & vintage timing tests
- `test_provider_config.py` — `ProviderConfig` dataclass tests
- `test_benchmark_revisions.py` — historical benchmark revision lookup tests
