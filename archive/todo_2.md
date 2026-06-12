# TODO: Data Infrastructure for Hierarchical Vintage-Tracked Observation Panel

## Context

The current `src/alt_nfp/data.py` loads CES, QCEW, and payroll provider CSVs as flat national-level index series, joins them on `ref_date`, and computes growth rates into a dict of numpy arrays. This works for the current national-level model but does not support:

- **Industry hierarchy** (domain → supersector → sector) needed for cell-level estimation
- **Vintage tracking** for QCEW (asymmetric revision schedule) and CES (3 monthly revisions + annual benchmark)
- **Real-time information set reconstruction** for proper pseudo-out-of-sample evaluation
- **Vintage-dependent measurement error** in the model (earlier vintages are noisier)

This TODO adds three new subpackages (`lookups/`, `ingest/`, `vintages/`) alongside the existing code. The current `data.py` and `model.py` continue to work unchanged; the new infrastructure will be integrated into the model builder in a follow-up task.

### Data Source Architecture

**Current (real-time) data** is pulled from the BLS API via the `eco-stats` library (https://github.com/lowmason/eco-stats). The `BLSClient` wraps the BLS Public Data API v2. CES data uses series IDs with the format `CE{S|U}{supersector:2}{industry:6}{datatype:2}` (e.g., `CES0000000001` = total nonfarm all employees SA). QCEW data is accessed through BLS's separate QCEW open data access endpoint (not the timeseries API). Add `eco-stats` as a dependency.

**Historical revision data** will be provided as parquet files conforming to schemas defined in this TODO (Task 2.7). The parquet schemas are designed first; actual data files will be constructed to match.

**Publication date schedules** for CES and QCEW will be derived from BLS web pages provided separately. The revision schedules module (Task 1.3) encodes the structural lag patterns, while actual publication dates can be loaded from a supplementary calendar file.

## References

- `src/alt_nfp/config.py` — existing `ProviderConfig`, paths, model constants
- `src/alt_nfp/data.py` — existing data loading (will eventually be replaced by `ingest/`)
- `src/alt_nfp/model.py` — existing PyMC model (consumes dict from `data.py`)
- `docs/` — methodology docs for measurement equation details
- `eco-stats` repo — BLS API client: https://github.com/lowmason/eco-stats

## Code Style

- Python 3.12, line length 100, formatter: black, linter: ruff
- Use **Polars** (not pandas) for all DataFrames
- Prefer **single quotes** over double quotes for strings
- **Two blank lines** after class and function definitions (PEP 8)
- Type hints on all public functions
- Docstrings on all public functions and classes (numpy-style)

---

## Task 1: `src/alt_nfp/lookups/` — Static Reference Tables

These are methodology decisions encoded as code. They change when we change our classification scheme, not when data updates arrive.

### 1.1 Create `src/alt_nfp/lookups/__init__.py`

Re-export the key objects:
- `INDUSTRY_HIERARCHY` (LazyFrame)
- `QCEW_REVISIONS`, `CES_REVISIONS`
- Index-builder functions

### 1.2 Create `src/alt_nfp/lookups/industry.py`

Contains the canonical BLS NAICS → supersector → domain mapping as a Polars DataFrame.

**`INDUSTRY_HIERARCHY`**: A `pl.LazyFrame` with columns:
- `sector_code` (str) — 2-digit NAICS sector code, using simplified codes: **'31'** (not '31-33'), **'44'** (not '44-45'), **'48'** (not '48-49'). Full list: '21', '23', '31', '42', '44', '48', '22', '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81'
- `sector_title` (str) — e.g., 'Mining', 'Construction', 'Manufacturing', etc.
- `supersector_code` (str) — BLS CES supersector code (2-digit, matches the first two digits of CES industry codes): '10', '20', '30', '40', '50', '55', '60', '65', '70', '80'
- `supersector_title` (str) — BLS supersector name
- `domain_code` (str) — 'G' (goods-producing) or 'S' (service-providing)
- `domain_title` (str)

The nesting (supersector_code → supersector_title → constituent sector_codes):
- **Goods-producing (G):**
  - 10 → Mining and Logging → sectors: '21' (Mining; logging is part of NAICS 11 but assigned here per BLS convention)
  - 20 → Construction → sectors: '23'
  - 30 → Manufacturing → sectors: '31' (covers NAICS 31-33)
- **Service-providing (S):**
  - 40 → Trade, Transportation, and Utilities → sectors: '42' (Wholesale), '44' (Retail, covers NAICS 44-45), '48' (Transportation & Warehousing, covers NAICS 48-49), '22' (Utilities)
  - 50 → Information → sectors: '51'
  - 55 → Financial Activities → sectors: '52' (Finance & Insurance), '53' (Real Estate)
  - 60 → Professional and Business Services → sectors: '54' (Professional & Technical), '55' (Management of Companies), '56' (Administrative & Waste)
  - 65 → Private Education and Health Services → sectors: '61' (Educational Services), '62' (Health Care & Social Assistance)
  - 70 → Leisure and Hospitality → sectors: '71' (Arts & Entertainment), '72' (Accommodation & Food)
  - 80 → Other Services → sectors: '81'

Note: BLS CES supersector codes are the first 2 digits of the 8-digit CES industry code. These map to CES series IDs: e.g., supersector '30' → series `CES3000000001` (Manufacturing, all employees, SA). The sector codes above use simplified single-prefix forms rather than NAICS range notation.

**`CES_SERIES_MAP`**: A dict or DataFrame mapping (supersector_code, seasonal_adjustment, data_type) to BLS series IDs. This enables the eco-stats `BLSClient` to pull the right series. For national employment (data_type='01'):
```python
# Example: supersector '30' (Manufacturing)
# SA:  CES3000000001
# NSA: CEU3000000001
```

Generate series IDs programmatically: `f'CE{"S" if sa else "U"}{supersector_code}00000001'`

**Index-builder functions** that return numpy integer arrays for PyMC hierarchical indexing:
- `supersector_to_domain_idx() -> np.ndarray` — length n_supersectors, maps each supersector to its parent domain index
- `sector_to_supersector_idx() -> np.ndarray` — length n_sectors, maps each sector to its parent supersector index
- `get_domain_codes() -> list[str]` — sorted unique domain codes
- `get_supersector_codes() -> list[str]` — sorted unique supersector codes
- `get_sector_codes() -> list[str]` — sorted unique sector codes

### 1.3 Create `src/alt_nfp/lookups/revision_schedules.py`

**`RevisionSpec`** frozen dataclass:
- `revision_number: int` — 0 = initial publication, 1+ = revisions, -1 = benchmark
- `lag_months: int` — months from end of reference period to publication
- `noise_multiplier: float` — relative to final vintage (final = 1.0)

**`QCEW_REVISIONS: dict[str, list[RevisionSpec]]`** keyed by reference quarter ('Q1' through 'Q4'):

```
# QCEW revision schedule by reference quarter.
# Q1 is revised 4 times (5 total vintages: initial + 4 revisions)
# Q2 is revised 3 times (4 total vintages)
# Q3 is revised 2 times (3 total vintages)
# Q4 is revised 1 time  (2 total vintages)
#
# This asymmetry arises from the annual file structure: each annual
# QCEW release revises all quarters of its reference year, so earlier
# quarters accumulate more revision opportunities.
```

- Q1: 5 vintages (v0 through v4), lags ≈ 5, 8, 14, 20, 26 months
- Q2: 4 vintages (v0 through v3), lags ≈ 5, 11, 17, 23 months
- Q3: 3 vintages (v0 through v2), lags ≈ 5, 14, 20 months
- Q4: 2 vintages (v0 through v1), lags ≈ 8, 17 months

Noise multipliers (priors to be calibrated, start with reasonable defaults):
- v0 is noisiest (multiplier ~1.5–2.0), final vintage = 1.0, intermediate vintages interpolated

**`CES_REVISIONS: list[RevisionSpec]`**:
- v0 (advance/first-print): lag≈1 month, noise_multiplier≈3.0
- v1 (second estimate): lag≈2 months, noise_multiplier≈2.0
- v2 (third estimate): lag≈3 months, noise_multiplier≈1.5
- v-1 (benchmark): lag≈13 months, noise_multiplier≈1.0

**Helper functions:**
- `get_qcew_vintage_date(ref_quarter: str, ref_year: int, revision: int) -> date` — compute the approximate publication date for a specific QCEW vintage
- `get_ces_vintage_date(ref_month: date, revision: int) -> date` — compute the approximate publication date for a specific CES vintage
- `get_noise_multiplier(source: str, revision_number: int) -> float` — lookup noise multiplier for a given source and revision

**`PublicationCalendar`** (optional, for exact dates):
```python
@dataclass
class PublicationCalendar:
    """Actual BLS publication dates, loaded from a supplementary file.

    The structural lag_months in RevisionSpec gives approximate dates.
    This class provides exact dates when available, parsed from BLS
    schedule pages provided separately.
    """
    ces_release_dates: pl.DataFrame  # columns: ref_month, revision_number, pub_date
    qcew_release_dates: pl.DataFrame  # columns: ref_quarter, ref_year, revision_number, pub_date

    @classmethod
    def from_parquet(cls, path: Path) -> 'PublicationCalendar':
        ...
```

If no calendar file exists, vintage_date computation falls back to the structural lag approximations.

### 1.4 Create `src/alt_nfp/lookups/geography.py`

Placeholder for future geographic hierarchy (region → division → state). For now, just create the file with a module docstring and a `# TODO: implement when cell-level estimation is added` comment. No implementation needed yet.

---

## Task 2: `src/alt_nfp/ingest/` — Raw Data to Vintage-Tagged Observations

Each source has idiosyncratic formats but produces the same output schema: a long-format Polars DataFrame where each row is one observation of employment growth for one industry unit in one period from one source at one vintage.

### 2.1 Create `src/alt_nfp/ingest/__init__.py`

Re-export: `build_panel`, `validate_panel`, `PANEL_SCHEMA`

### 2.2 Create `src/alt_nfp/ingest/base.py`

**`PANEL_SCHEMA`** — dict mapping column names to Polars dtypes. The unified observation panel has these columns:

| Column | Type | Description |
|--------|------|-------------|
| `period` | `pl.Date` | First of reference month |
| `industry_code` | `pl.Utf8` | Supersector code (for CES/payroll) or sector code (for QCEW) |
| `industry_level` | `pl.Utf8` | `'supersector'` or `'sector'` — which hierarchy level |
| `source` | `pl.Utf8` | `'ces_sa'`, `'ces_nsa'`, `'qcew'`, `'pp1'`, `'pp2'`, ... |
| `source_type` | `pl.Utf8` | `'official_sa'`, `'official_nsa'`, `'census'`, `'payroll'` |
| `growth` | `pl.Float64` | Month-over-month log employment growth rate |
| `employment_level` | `pl.Float64` | Index level (for weighting/diagnostics), nullable |
| `is_seasonally_adjusted` | `pl.Boolean` | |
| `vintage_date` | `pl.Date` | Date this value became known/available |
| `revision_number` | `pl.Int32` | 0=initial, 1+=revisions, -1=benchmark |
| `is_final` | `pl.Boolean` | True if this is the latest available revision |
| `publication_lag_months` | `pl.Int32` | Months from reference period to vintage_date |
| `coverage_ratio` | `pl.Float64` | Payroll coverage vs QCEW (null for official sources) |

**`validate_panel(df: pl.DataFrame) -> pl.DataFrame`** — asserts:
- All required columns present with correct dtypes
- No duplicate (period, source, industry_code, revision_number) combinations
- `growth` values are finite where not null
- `revision_number` is consistent with source type
- Returns the input DataFrame if valid, raises ValueError otherwise

### 2.3 Create `src/alt_nfp/ingest/qcew.py`

Two ingestion paths:

**A) Current data via BLS API: `fetch_qcew_current(start_year: int, end_year: int) -> pl.DataFrame`**

Uses the eco-stats `BLSClient` or direct QCEW open data access to pull the latest available QCEW data. QCEW uses a separate API from the BLS timeseries API — data is accessed via CSV slices at URLs like:
```
https://data.bls.gov/cew/data/api/{YEAR}/{QTR}/industry/{INDUSTRY}.csv
```

Pull national-level (`area_fips = 'US000'`), private ownership (`own_code = 5`), by 2-digit NAICS industry. Extract `month1_emplvl`, `month2_emplvl`, `month3_emplvl` for monthly employment levels.

- QCEW is quarterly but provides monthly employment. Compute month-over-month log growth rates from the three monthly employment levels within each quarter, plus the cross-quarter seam (month3 of previous quarter → month1 of current quarter).
- Set `industry_level = 'sector'`, `industry_code` = sector code from the hierarchy lookup (e.g., '31', '44', '48', not NAICS range notation)
- For current API pulls, set `revision_number` based on how many revisions have likely occurred given the current date and the QCEW revision schedule. Or default to `revision_number = 0` and `is_final = False` since this is the latest available.
- Set `source = 'qcew'`, `source_type = 'census'`, `is_seasonally_adjusted = False`

**B) Historical revisions from parquet: `load_qcew_vintages(path: Path) -> pl.DataFrame`**

Reads parquet files conforming to the schema defined in Task 2.7. Transforms into PANEL_SCHEMA format.

**`ingest_qcew(api_client=None, vintage_dir: Path | None = None) -> pl.DataFrame`**

Orchestrator: loads historical vintages from parquet if available, fetches current data from API, combines, deduplicates (preferring the explicitly versioned vintage data over API-pulled data for overlapping periods), validates.

### 2.4 Create `src/alt_nfp/ingest/ces.py`

Two ingestion paths:

**A) Current data via BLS API: `fetch_ces_current(start_year: int, end_year: int) -> pl.DataFrame`**

Uses eco-stats `BLSClient.get_series()` to pull CES data for all supersectors. Build series IDs programmatically from `CES_SERIES_MAP`:
```python
# For each supersector, pull SA and NSA employment:
# SA:  CES{supersector_code}00000001  (e.g., CES3000000001)
# NSA: CEU{supersector_code}00000001  (e.g., CEU3000000001)
# Also pull total private: CES0500000001 / CEU0500000001
```

The BLS API v2 accepts up to 50 series per request and returns JSON with year, period, value. Parse into employment levels, compute log growth rates.

- Set `industry_level = 'supersector'`, `industry_code` = supersector code
- Current API data represents the latest available revision. Set `revision_number` based on how recently the reference month was: if the reference month is within the last 1 month, it's v0; 2 months ago → v1; 3+ months ago → v2; if a benchmark has been applied → v-1. Alternatively, always set current pulls to `is_final = True` since we can't reconstruct earlier vintages from the API.
- Generate two rows per observation: `source = 'ces_sa'` and `source = 'ces_nsa'`

**B) Historical revisions from parquet: `load_ces_vintages(path: Path) -> pl.DataFrame`**

Reads parquet files conforming to the schema defined in Task 2.7.

**`ingest_ces(api_client=None, vintage_dir: Path | None = None) -> pl.DataFrame`**

Orchestrator: combines historical vintages + current API data, deduplicates, validates.

### 2.5 Create `src/alt_nfp/ingest/payroll.py`

**`ingest_provider(raw_dir: Path | None, config: ProviderConfig) -> pl.DataFrame`**

Reads payroll provider index files. Payroll provider data are real-time and not revised, so:
- `revision_number = 0`, `is_final = True`
- `vintage_date` = the date the data became available (approximate as ref_date + 3 weeks if not explicitly tracked)
- `source = config.name.lower()` (e.g., 'pp1')
- `source_type = 'payroll'`
- `is_seasonally_adjusted = False`
- `industry_level = 'supersector'` (or as available)

Fall back to loading from `DATA_DIR / config.file` (the existing flat CSVs) if `raw_dir` doesn't exist. This maintains backward compatibility.

### 2.6 Create `src/alt_nfp/ingest/panel.py`

**`build_panel(raw_dir: Path | None = None, use_legacy: bool = False, use_api: bool = True) -> pl.DataFrame`**

Three modes:
1. **Legacy mode** (`use_legacy=True`): loads from existing `data/*.csv` files using simplified ingestion (single vintage per source, no revision tracking). Backward-compatible path that reproduces current behavior.
2. **API mode** (`use_api=True`, default): fetches current data from BLS API via eco-stats, loads historical vintage parquet files from `raw_dir` if available, combines.
3. **Offline mode** (`use_api=False`): only reads from local parquet vintage files and existing CSVs. For reproducible builds without network access.

```python
def build_panel(
    raw_dir: Path | None = None,
    use_legacy: bool = False,
    use_api: bool = True,
) -> pl.DataFrame:
    ...
```

Also include:

**`save_panel(panel: pl.DataFrame, output_dir: Path) -> None`** — writes `observation_panel.parquet` and `panel_manifest.json` (containing build timestamp, row count, source counts, date range, git hash if available).

**`load_panel(panel_dir: Path) -> pl.DataFrame`** — reads the parquet file back.

### 2.7 Historical Revision Parquet Schemas

Define the expected schemas for historical revision data. These parquet files will be constructed separately to conform to these schemas, then placed in `data/raw/vintages/`.

**`QCEW_VINTAGE_SCHEMA`** — for `data/raw/vintages/qcew_vintages.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| `ref_year` | `pl.Int32` | Reference year |
| `ref_quarter` | `pl.Int32` | Reference quarter (1-4) |
| `ref_month` | `pl.Int32` | Month within quarter (1, 2, or 3) |
| `area_fips` | `pl.Utf8` | FIPS code ('US000' for national, state codes for state-level) |
| `industry_code` | `pl.Utf8` | NAICS sector code using simplified form: '31' not '31-33', '44' not '44-45', '48' not '48-49' |
| `own_code` | `pl.Int32` | Ownership code (5 = private) |
| `employment` | `pl.Int64` | Monthly employment level |
| `revision_number` | `pl.Int32` | 0 = initial publication, 1+ = subsequent revisions |
| `vintage_date` | `pl.Date` | Actual publication/release date of this vintage |

Notes:
- Each unique (ref_year, ref_quarter, ref_month, area_fips, industry_code, own_code, revision_number) identifies one observation.
- Q1 data will have up to 5 rows per cell (v0–v4), Q2 up to 4, Q3 up to 3, Q4 up to 2.
- Employment is the raw level (not growth rate); growth rates are computed during ingestion.
- `industry_code` values should match the `sector_code` values in `INDUSTRY_HIERARCHY` (Task 1.2).

**`CES_VINTAGE_SCHEMA`** — for `data/raw/vintages/ces_vintages.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| `ref_date` | `pl.Date` | First of reference month |
| `supersector_code` | `pl.Utf8` | BLS CES supersector code ('10', '20', '30', ..., '80') |
| `seasonal_adjustment` | `pl.Utf8` | 'S' (seasonally adjusted) or 'U' (not) |
| `employment` | `pl.Float64` | Employment level (thousands) |
| `revision_number` | `pl.Int32` | 0 = first print, 1 = second, 2 = third, -1 = benchmark |
| `vintage_date` | `pl.Date` | Actual release date of this vintage |
| `bls_series_id` | `pl.Utf8` | Original BLS series ID (e.g., 'CES3000000001') |

Notes:
- Each unique (ref_date, supersector_code, seasonal_adjustment, revision_number) identifies one observation.
- Most reference months will have up to 4 rows (v0, v1, v2, v-1). Not all revisions may be available for all historical months.
- Employment is in thousands (matching BLS publication units); growth rates computed during ingestion.
- `supersector_code` values should match `INDUSTRY_HIERARCHY` (Task 1.2).
- Include total private ('05') in addition to individual supersectors.

**`PUBLICATION_CALENDAR_SCHEMA`** — for `data/raw/vintages/publication_calendar.parquet` (optional):

| Column | Type | Description |
|--------|------|-------------|
| `source` | `pl.Utf8` | 'ces' or 'qcew' |
| `ref_period` | `pl.Utf8` | Reference period identifier: 'YYYY-MM' for CES, 'YYYYqQ' for QCEW |
| `revision_number` | `pl.Int32` | Which vintage |
| `publication_date` | `pl.Date` | Actual publication date from BLS schedule |

Notes:
- Derived from BLS release schedule web pages (to be provided separately).
- Used by `PublicationCalendar` (Task 1.3) to override structural lag approximations with exact dates.

Document these schemas in a `data/raw/vintages/README.md` file that also includes example rows and instructions for constructing the parquet files.

---

## Task 3: `src/alt_nfp/vintages/` — Vintage Views and Real-Time Reconstruction

Views are functions over the observation panel, not separate materialized tables.

### 3.1 Create `src/alt_nfp/vintages/__init__.py`

Re-export: `real_time_view`, `final_view`, `vintage_diff`

### 3.2 Create `src/alt_nfp/vintages/views.py`

**`real_time_view(panel: pl.LazyFrame, as_of: date) -> pl.LazyFrame`**

Returns the latest vintage available for each (period, source, industry_code) as of `as_of`. Logic:
- Filter to `vintage_date <= as_of`
- Sort by `revision_number` descending within each group
- Keep first (highest revision_number) per (period, source, industry_code)

**`final_view(panel: pl.LazyFrame) -> pl.LazyFrame`**

Returns only rows where `is_final == True`. This is the best-available data for full-sample estimation.

**`specific_vintage_view(panel: pl.LazyFrame, source: str, revision_number: int) -> pl.LazyFrame`**

Filter to a specific source and revision stage. Useful for studying revision properties.

### 3.3 Create `src/alt_nfp/vintages/evaluation.py`

**`vintage_diff(panel: pl.LazyFrame, source: str, rev_a: int, rev_b: int) -> pl.LazyFrame`**

Compute the revision between two vintages of the same source: `growth(rev_b) - growth(rev_a)`. Returns a frame with columns (period, industry_code, revision_from, revision_to, growth_diff). Useful for analyzing revision properties and calibrating noise multipliers.

**`build_noise_multiplier_vector(panel_view: pl.DataFrame) -> np.ndarray`**

For a given panel view (output of `real_time_view` or `final_view`), produce a per-observation noise multiplier array by looking up each row's (source, revision_number) in the revision schedule. Returns a numpy array aligned to the panel rows, suitable for passing to the PyMC model as observation-level noise scaling.

---

## Task 4: Tests

### 4.1 Create `tests/test_lookups.py`

- `test_industry_hierarchy_completeness` — all 10 BLS supersectors present, all sectors map to a valid supersector, all supersectors map to a valid domain
- `test_industry_hierarchy_no_orphans` — no sector codes without a supersector parent
- `test_sector_codes_simplified` — verify sector codes use simplified forms: '31' not '31-33', '44' not '44-45', '48' not '48-49'
- `test_ces_series_id_generation` — verify that generated CES series IDs have correct format (e.g., 'CES3000000001' for supersector '30' SA)
- `test_index_builders_shapes` — `supersector_to_domain_idx()` has length n_supersectors, values in range [0, n_domains); similarly for sector_to_supersector_idx
- `test_revision_schedule_counts` — QCEW: Q1 has 5 vintages, Q2 has 4, Q3 has 3, Q4 has 2; CES has 4 entries
- `test_revision_schedule_monotonic_noise` — noise multipliers decrease with revision number (later revisions are less noisy)
- `test_revision_schedule_monotonic_lag` — lag_months increases with revision number

### 4.2 Create `tests/test_ingest.py`

- `test_validate_panel_good` — a well-formed mini DataFrame passes validation
- `test_validate_panel_duplicate` — duplicate (period, source, industry_code, revision_number) raises ValueError
- `test_validate_panel_missing_column` — missing required column raises ValueError
- `test_qcew_vintage_parquet_schema` — verify a sample parquet with correct schema loads and transforms correctly
- `test_ces_vintage_parquet_schema` — verify a sample parquet with correct schema loads and transforms correctly
- `test_legacy_panel_builds` — `build_panel(use_legacy=True)` produces a valid panel from existing `data/*.csv` files (skip if data files not present)

### 4.3 Create `tests/test_vintages.py`

- `test_real_time_view_filters_correctly` — create a panel with v0 and v1 for the same observation; `real_time_view` with `as_of` between v0 and v1 vintage_dates returns only v0; with `as_of` after v1 returns v1
- `test_final_view` — only rows with `is_final=True` are returned
- `test_vintage_diff` — known revision between v0 and v1 is computed correctly
- `test_noise_multiplier_vector_shape` — output array length matches panel row count

---

## Task 5: Wire Up and Document

### 5.1 Update `src/alt_nfp/__init__.py`

Add imports for the new subpackages so they're accessible:
```python
from .lookups import INDUSTRY_HIERARCHY, QCEW_REVISIONS, CES_REVISIONS
from .ingest import build_panel, validate_panel, PANEL_SCHEMA
from .vintages import real_time_view, final_view, vintage_diff
```

### 5.2 Update `pyproject.toml`

Add `eco-stats` as a dependency. Since it's installed from source (git), add:
```toml
dependencies = [
    # ... existing deps ...
    "eco-stats @ git+https://github.com/lowmason/eco-stats.git",
]
```

### 5.3 Update CLAUDE.md

Add the new subpackages to the Project Structure section:
```
src/alt_nfp/
├── __init__.py
├── config.py
├── data.py              # existing — legacy data loading
├── model.py             # existing — PyMC model
├── lookups/             # NEW — static reference tables
│   ├── __init__.py
│   ├── industry.py      # NAICS → supersector → domain hierarchy + CES series ID map
│   ├── revision_schedules.py  # QCEW & CES vintage schedules + publication calendar
│   └── geography.py     # placeholder for future geo hierarchy
├── ingest/              # NEW — raw data → observation panel
│   ├── __init__.py
│   ├── base.py          # PANEL_SCHEMA, validate_panel
│   ├── qcew.py          # QCEW ingestion (BLS API + historical parquet)
│   ├── ces.py           # CES ingestion (BLS API + historical parquet)
│   ├── payroll.py       # provider index ingestion
│   └── panel.py         # build_panel, save/load
├── vintages/            # NEW — vintage views & evaluation
│   ├── __init__.py
│   ├── views.py         # real_time_view, final_view
│   └── evaluation.py    # vintage_diff, noise multiplier builder
├── backtest.py          # existing
├── sampling.py          # existing
└── sensitivity.py       # existing
```

### 5.4 Add directory structure and schema documentation

Create the directory skeleton (with `.gitkeep` files):
```
data/raw/
├── vintages/
│   ├── .gitkeep
│   └── README.md        # Documents parquet schemas from Task 2.7
└── providers/
    ├── pp1/.gitkeep
    └── pp2/.gitkeep
```

Also create `data/panels/.gitkeep` and add `data/panels/*.parquet` to `.gitignore`.

The `data/raw/vintages/README.md` should contain:
- The three parquet schemas (QCEW, CES, publication calendar) with column descriptions
- Example rows for each schema
- Notes on the QCEW asymmetric revision structure
- Notes on CES series ID format
- Instructions: "Place parquet files here conforming to these schemas. The ingestion pipeline will read them automatically."

---

## Ordering and Dependencies

Work through the tasks in order: Task 1 → Task 2 → Task 3 → Task 4 → Task 5.

Within each task, subtasks can be done in listed order. Task 2 depends on Task 1 (ingestion uses revision schedules and industry hierarchy). Task 3 depends on Task 2 (vintage views operate on the panel schema). Task 4 tests all prior work. Task 5 is integration.

## What This Does NOT Change

- `src/alt_nfp/data.py` — untouched, continues to work for existing national-level model
- `src/alt_nfp/model.py` — untouched, still consumes dict from `data.py`
- `pp_estimation_v2.py` / `alt_nfp_estimation_v3.py` — untouched
- Existing `data/*.csv` files — untouched, used by legacy mode

The new infrastructure runs in parallel. A follow-up TODO will wire the observation panel into a new hierarchical model builder that replaces the current `data.py → model.py` pipeline.