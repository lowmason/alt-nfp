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

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.10)
- **Dependencies**: numpy, polars (minimal footprint)
- **Build**: hatchling

## Key Commands

```bash
# Run lookups tests
pytest tests/

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
│                           #   GOVT_OWNERSHIP_TO_SECTOR, _CES_SECTOR_TO_NAICS
├── geography.py            # FIPS_TO_DIVISION, FIPS_TO_REGION, STATES, REGION_NAMES, etc.
├── revision_schedules.py   # CES_REVISIONS, QCEW_REVISIONS, get_noise_multiplier, vintage date helpers
├── benchmark_revisions.py  # BENCHMARK_REVISIONS dict (historical actuals)
└── provider_config.py      # ProviderConfig dataclass, CYCLICAL_INDICATORS
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

## Test Mapping

Tests from the monorepo `tests/` that belong here:
- `test_lookups.py` — industry hierarchy & revision schedule tests
