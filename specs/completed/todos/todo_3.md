# TODO: Port BLS Download Infrastructure into `alt-nfp/src/alt_nfp/ingest/bls/`

## Context and Rationale

The `eco-stats` library (https://github.com/lowmason/eco-stats) provides a general-purpose Python client for four federal statistical APIs (BEA, BLS, Census, FRED). For the alt-nfp project, we only need BLS download functionality for three specific programs: QCEW, CES National, and CES State. Rather than taking `eco-stats` as a full dependency—which pulls in BEA, Census, and FRED clients we'll never use—we should port the relevant BLS infrastructure directly into `alt-nfp` as a self-contained `ingest/bls/` subpackage.

This subpackage handles **downloading raw data from BLS**. It is distinct from the higher-level `ingest/ces.py` and `ingest/qcew.py` modules (defined in `TODO_data_infrastructure.md`) that **transform** downloaded data into the vintage-tracked `PANEL_SCHEMA`. The separation keeps download logic (HTTP, caching, parsing) cleanly decoupled from domain logic (vintage tagging, growth rate computation, schema conformance).

### What the existing `ingest/` TODO already defines

`TODO_data_infrastructure.md` specifies `ingest/ces.py` and `ingest/qcew.py` as transformation modules. The more recent hierarchical-industry conversation expanded these with dual ingestion paths: a "fetch current from API" path and a "load historical vintages from parquet" path. The fetch functions within those modules assumed `eco-stats` as a dependency. This TODO replaces that assumption by giving alt-nfp its own BLS download layer.

### Why CES National and CES State must be separate

The existing `TODO_data_infrastructure.md` defines a single `ingest/ces.py`. This is insufficient because CES National (BLS program prefix `CE`) and CES State & Area (prefix `SM`) are fundamentally different statistical products:

CES National estimates are produced from roughly 121,000 sampled businesses across 500+ detailed NAICS estimation cells, with independent birth/death modeling and direct sample-based estimation for every published series. CES State estimates use the same underlying sample but employ a top-down estimation structure at the Estimation Super Sector level, with over 55% of series relying on small area models rather than direct sampling. State estimates undergo full QCEW replacement during benchmarking (replacing the entire level), while national uses March-only wedge-back. State birth/death factors are raked down from national ARIMA forecasts rather than independently modeled. State data releases two weeks after national. These differences create materially different measurement characteristics—revision magnitudes, model-dependence, seasonal factor instability—that require distinct parameterization in the Bayesian model.

At the download layer, they are also different: CES National uses the `CE` flat file prefix with series IDs like `CES0000000001`, while CES State uses the `SM` prefix with series IDs like `SMU36000000000000001` that encode state, area, and supersector/industry codes.

### Why QCEW needs its own download path (not just EN flat files)

QCEW data is available through two mechanisms. First, the EN flat files at `download.bls.gov/pub/time.series/en/` provide LABSTAT-format time series. Second, the QCEW data API at `data.bls.gov/cew/data/api/{YEAR}/{QTR}/industry/{INDUSTRY}.csv` provides quarterly CSV files with full detail including monthly employment levels, wage data, and establishment counts by area, ownership, and industry.

The CSV API is the correct download path for alt-nfp because it provides monthly employment within quarters (`month1_emplvl`, `month2_emplvl`, `month3_emplvl`), supports filtering by area FIPS (`US000` for national), ownership code (`5` for private), and industry at any NAICS level. The EN flat files are useful as a secondary access path for mapping tables and series metadata.

## References

These are the source files from eco-stats to draw from, plus the existing alt-nfp context:

**eco-stats source files (in project knowledge):**
- `src/eco_stats/api/bls/client.py` — `BLSClient` with JSON API + flat file access + convenience methods
- `src/eco_stats/api/bls/flat_files.py` — `BLSFlatFileClient` with caching, download, TSV parsing
- `src/eco_stats/api/bls/programs.py` — `BLSProgram`, `SeriesField`, full program registry (keep only CE, SM, EN)
- `src/eco_stats/api/bls/series_id.py` — `parse_series_id()`, `build_series_id()`
- `src/eco_stats/api/bls/__init__.py` — package re-exports
- `pyproject.toml` — dependencies (only `requests`, `polars` needed)

**alt-nfp existing context:**
- `TODO_data_infrastructure.md` — defines `ingest/base.py`, `ingest/ces.py`, `ingest/qcew.py`
- `src/alt_nfp/config.py` — existing `ProviderConfig`, paths
- `src/alt_nfp/data.py` — current data loading (will be replaced)

**Project methodology docs (in project knowledge):**
- `CES_Methodology__A_Complete_Technical_Reference_for_NFP_Nowcasting.md`
- `CES_State_Methodology__A_Complete_Reference_for_Bayesian_Nowcasting_of_State-Level_Employment_Data.md`
- `The_Quarterly_Census_of_Employment_and_Wages__A_Comprehensive_Technical_Reference.md`

## Code Style

- Python 3.12, line length 100, formatter: black, linter: ruff
- Use **Polars** (not pandas) for all DataFrames
- Prefer **single quotes** over double quotes for strings
- **Two blank lines** after class and function definitions (PEP 8)
- Type hints on all public functions
- Docstrings on all public functions and classes (numpy-style)

---

## Proposed Module Structure

```
src/alt_nfp/ingest/
├── __init__.py              # existing (from TODO_data_infrastructure.md)
├── base.py                  # existing (PANEL_SCHEMA, validate_panel)
├── bls/                     # NEW — BLS download infrastructure
│   ├── __init__.py          # re-exports: fetch_qcew, fetch_ces_national, fetch_ces_state
│   ├── _http.py             # shared HTTP session, user-agent, caching
│   ├── _programs.py         # CE, SM, EN program definitions + series ID utils
│   ├── qcew.py              # QCEW CSV API download
│   ├── ces_national.py      # CES National (CE) download via flat files + JSON API
│   └── ces_state.py         # CES State (SM) download via flat files + JSON API
├── ces_national.py          # transform CES National downloads → PANEL_SCHEMA
├── ces_state.py             # transform CES State downloads → PANEL_SCHEMA (NEW)
├── qcew.py                  # transform QCEW downloads → PANEL_SCHEMA
├── payroll.py               # existing (from TODO_data_infrastructure.md)
└── panel.py                 # existing (build_panel orchestrator)
```

The `bls/` subpackage is the download layer. The top-level `ces_national.py`, `ces_state.py`, and `qcew.py` are the transform layer. Each transform module calls into `bls/` for its "fetch current from API" path and reads local files for its "load historical vintages" path.

---

## Task 1: Create `src/alt_nfp/ingest/bls/_http.py` — Shared HTTP Infrastructure

Port from `eco-stats/src/eco_stats/api/bls/flat_files.py`. This is the foundation that all three download modules share.

### 1.1 `_USER_AGENT` constant

BLS blocks default Python user agents. Define a descriptive UA string:
```python
_USER_AGENT = (
    'alt-nfp/0.1.0 '
    '(Python; +https://github.com/lowmason/alt-nfp) '
    'requests/{req_version}'
).format(req_version=requests.__version__)
```

### 1.2 `BLSHttpClient` class

Adapt `BLSFlatFileClient` from eco-stats but generalize it to handle both flat-file (LABSTAT) downloads and QCEW CSV API requests. The class should provide:

**Constructor**: `__init__(self, api_key: str | None = None, cache_dir: str = '.cache/bls', cache_ttl: int = 86_400)`
- Creates a `requests.Session()` with the custom user agent
- Stores `api_key` for JSON API v2 access (optional; v1 works without key but has lower rate limits)
- Sets up local cache directory and TTL

**Flat file methods** (ported from eco-stats `BLSFlatFileClient`):
- `get_flat_file(self, prefix: str, filename: str) -> list[dict[str, str]]` — download and parse a tab-delimited flat file from `download.bls.gov/pub/time.series/{prefix}/{filename}`. Uses local file cache with TTL.
- `get_mapping(self, prefix: str, mapping_name: str) -> list[dict[str, str]]` — convenience wrapper: `get_flat_file(prefix, f'{prefix.lower()}.{mapping_name}')`
- `get_data(self, prefix: str, file_suffix: str = '0.Current') -> list[dict[str, str]]` — convenience wrapper for data files: `get_flat_file(prefix, f'{prefix.lower()}.data.{file_suffix}')`
- `get_series_list(self, prefix: str, **filters: str) -> list[dict[str, str]]` — downloads the series file and optionally filters rows

**JSON API method** (ported from eco-stats `BLSClient.get_series()`):
- `get_series(self, series_ids: list[str], start_year: str | None = None, end_year: str | None = None) -> pl.DataFrame` — POST to `https://api.bls.gov/publicAPI/v2/timeseries/data/` (v2 if api_key provided, v1 otherwise). Returns Polars DataFrame with columns: `series_id`, `date`, `year`, `period`, `period_name`, `value`. The BLS API accepts up to 50 series per request and supports 10-year windows (v1) or 20-year windows (v2).

**QCEW CSV API method** (new, not in eco-stats):
- `get_qcew_csv(self, year: int, quarter: int, industry: str) -> pl.DataFrame` — GET from `https://data.bls.gov/cew/data/api/{year}/{quarter}/industry/{industry}.csv`. Returns raw Polars DataFrame parsed from the CSV response. Uses cache with same TTL logic.

**Internal methods** (ported from eco-stats):
- `_cache_path(self, filename: str) -> str`
- `_is_cache_valid(self, path: str) -> bool`
- `_download_and_parse_tsv(self, url: str, filename: str) -> list[dict[str, str]]`
- `@staticmethod _parse_tsv(text: str) -> list[dict[str, str]]` — tab-delimited parser that strips whitespace from headers and values

**Context manager support**: `__enter__` / `__exit__` to close the session.

### Implementation notes

- Port the TSV parsing logic verbatim from eco-stats `_parse_tsv()` — it handles BLS's quirky whitespace-padded fields
- The QCEW CSV API returns standard CSV (not TSV), so use `pl.read_csv()` or `io.StringIO` + Polars for that path
- Cache QCEW CSV responses using a key like `qcew_{year}_{quarter}_{industry}.csv`
- The JSON API response parsing should follow eco-stats `_parse_api_response()` — extract from `response['Results']['series']`, build rows with `series_id`, `year`, `period`, `period_name`, `value`

---

## Task 2: Create `src/alt_nfp/ingest/bls/_programs.py` — Program Definitions

Port from `eco-stats/src/eco_stats/api/bls/programs.py` and `series_id.py`, keeping **only CE, SM, EN**.

### 2.1 `SeriesField` class

Port verbatim from eco-stats. Lightweight class with `name`, `start`, `end`, `description`, plus `length` property and `extract(series_id)` method.

### 2.2 `BLSProgram` class

Port verbatim from eco-stats. Contains `prefix`, `name`, `description`, `fields`, `mapping_files`.

### 2.3 Program registry: `PROGRAMS` dict

Register exactly three programs:

**CE — Current Employment Statistics (National)**:
```python
# prefix='CE', 13-character series IDs
# Fields: prefix(1-2), seasonal(3), supersector(4-5), industry(6-11), data_type(12-13)
# mapping_files: ['datatype', 'industry', 'seasonal', 'series', 'supersector']
```

**SM — State and Area Employment, Hours, and Earnings**:
```python
# prefix='SM', 20-character series IDs
# Fields: prefix(1-2), seasonal(3), state(4-5), area(6-10),
#         supersector_industry(11-18), data_type(19-20)
# mapping_files: ['area', 'datatype', 'industry', 'seasonal',
#                 'series', 'state', 'supersector']
```

**EN — Quarterly Census of Employment and Wages**:
```python
# prefix='EN', 17-character series IDs
# Fields: prefix(1-2), seasonal(3), area(4-8), data_type(9),
#         size(10), ownership(11), industry(12-17)
# mapping_files: ['area', 'datatype', 'industry', 'ownership',
#                 'seasonal', 'series', 'size']
```

Copy field definitions exactly from eco-stats `programs.py`.

### 2.4 Helper functions

- `get_program(prefix: str) -> BLSProgram` — lookup by prefix, raise `KeyError` if not found
- `list_programs() -> dict[str, str]` — returns `{prefix: name}` for the three registered programs
- `parse_series_id(series_id: str) -> dict[str, str]` — port from eco-stats `series_id.py`
- `build_series_id(program: str, **components: str) -> str` — port from eco-stats `series_id.py`

---

## Task 3: Create `src/alt_nfp/ingest/bls/qcew.py` — QCEW Download

This module provides a clean interface for downloading QCEW data from the BLS QCEW CSV API. It does NOT transform data into `PANEL_SCHEMA`—that is the job of `ingest/qcew.py`.

### 3.1 `fetch_qcew(years: list[int], quarters: list[int] | None = None, industries: list[str] | None = None, area_fips: str = 'US000', ownership_code: str = '5', client: BLSHttpClient | None = None) -> pl.DataFrame`

Downloads QCEW data for the specified year/quarter/industry combinations.

**Parameters:**
- `years` — list of reference years (e.g., `[2020, 2021, 2022, 2023, 2024]`)
- `quarters` — list of quarters (1-4), defaults to `[1, 2, 3, 4]`
- `industries` — list of NAICS industry codes to request (e.g., `['10', '1011', '1012', '1013', '1021', '1022', '1023', '1024', '1025', '1026', '1029']`). The QCEW CSV API uses its own industry code scheme. If `None`, default to the 2-digit private-sector NAICS codes needed for the sector-level model.
- `area_fips` — area FIPS code, defaults to `'US000'` for national
- `ownership_code` — ownership filter, defaults to `'5'` (private)
- `client` — optional `BLSHttpClient` instance (creates a default if `None`)

**Returns:** Polars DataFrame with raw QCEW columns including `area_fips`, `own_code`, `industry_code`, `year`, `qtr`, `month1_emplvl`, `month2_emplvl`, `month3_emplvl`, `total_qtrly_wages`, `avg_wkly_wage`, `qtrly_estabs`, plus other raw fields.

**Implementation:**
- Iterate over year × quarter combinations
- For each combination, call `client.get_qcew_csv(year, quarter, industry)` for each industry
- Filter response rows by `area_fips` and `own_code`
- Concatenate all results into a single DataFrame
- Handle missing data gracefully (some year/quarter combos may not be published yet)
- Log which year/quarter/industry combos were fetched vs. missing

### 3.2 `QCEW_INDUSTRY_CODES` constant

A dict mapping 2-digit NAICS sector codes to the QCEW industry code format. The QCEW CSV API uses codes like `10` (Total, all industries), `1011` (NAICS 11), `1012` (NAICS 21), etc. Document the mapping so downstream code can translate between NAICS codes, QCEW API codes, and the project's sector codes from `lookups/industry.py`.

### 3.3 `fetch_qcew_annual_files(year: int, client: BLSHttpClient | None = None) -> pl.DataFrame`

Optional: download full annual QCEW data files from `bls.gov/cew/downloadable-data-files.htm`. These are larger but provide complete coverage. This is a convenience method for bulk historical downloads that may be added later (mark as `# TODO` placeholder for now).

---

## Task 4: Create `src/alt_nfp/ingest/bls/ces_national.py` — CES National Download

### 4.1 `CES_SERIES_MAP` constant

A dict mapping supersector codes to their CES series ID roots. For each supersector, we need SA and NSA employment (data type `01`):

```python
# SA:  CES{ss_code}00000001  (seasonal='S')
# NSA: CEU{ss_code}00000001  (seasonal='U')
#
# Supersector codes: '05' (total private), '10' (mining/logging),
# '20' (construction), '30' (manufacturing),
# '40' (trade/transp/util), '50' (information),
# '55' (financial), '60' (prof/business),
# '65' (education/health), '70' (leisure/hospitality),
# '80' (other services), '00' (total nonfarm)
```

Build series IDs programmatically using `build_series_id('CE', seasonal=..., supersector=..., industry='000000', data_type='01')`.

### 4.2 `fetch_ces_national(start_year: int | None = None, end_year: int | None = None, supersectors: list[str] | None = None, include_nsa: bool = True, client: BLSHttpClient | None = None) -> pl.DataFrame`

Downloads CES national employment data for all (or specified) supersectors.

**Strategy — use flat files as primary, JSON API as fallback for recent months:**

The flat file path (`ce.data.0.AllCESSeries` or individual data files) provides full history without rate limits. The JSON API provides the most recent data if the flat file cache is stale. Prefer flat files because they give complete history in a single download.

**Implementation:**
- Build the list of target series IDs from `CES_SERIES_MAP` (both SA and NSA if `include_nsa=True`)
- Download `ce.data.0.AllCESSeries` via `client.get_data('CE', '0.AllCESSeries')`
- Parse into Polars DataFrame with columns: `series_id`, `year`, `period`, `value`, `date`
- Filter to target series IDs and date range
- Add derived columns: `supersector_code` (extracted from series_id), `seasonal_code` (S or U), `data_type` (extracted)
- Cast `value` to Float64, `year` to Int64
- Derive `date` column from year + period using the pay-period-including-the-12th convention (day=12 for CE)

**Returns:** Polars DataFrame with columns: `series_id`, `date`, `year`, `period`, `value`, `supersector_code`, `is_seasonally_adjusted`.

### 4.3 `fetch_ces_national_via_api(series_ids: list[str], start_year: int, end_year: int, client: BLSHttpClient | None = None) -> pl.DataFrame`

Fallback method using the JSON API for cases where flat files are insufficient (e.g., you need data from the current month that hasn't been added to flat files yet). Handles the 50-series-per-request and 10/20-year-window constraints by batching requests.

---

## Task 5: Create `src/alt_nfp/ingest/bls/ces_state.py` — CES State Download

### 5.1 `SM_SERIES_MAP` constant or builder function

CES State series IDs are more complex: `SM{seasonal}{state}{area}{supersector_industry}{data_type}`. For state-level total employment by supersector:

```python
# SA:  SMS{state}00000{ss_code}00000001  (seasonal='S', area='00000'=statewide)
# NSA: SMU{state}00000{ss_code}00000001  (seasonal='U')
#
# state = 2-digit FIPS code ('01'=AL, '02'=AK, ..., '56'=WY)
# ss_code = 8-digit supersector/industry code (e.g., '00000000' total nonfarm)
```

Provide a builder function: `build_state_series_ids(states: list[str], supersectors: list[str], seasonal: str = 'S', data_type: str = '01') -> list[str]`

### 5.2 `fetch_ces_state(states: list[str] | None = None, start_year: int | None = None, end_year: int | None = None, supersectors: list[str] | None = None, include_nsa: bool = True, client: BLSHttpClient | None = None) -> pl.DataFrame`

Downloads CES State employment data.

**Strategy — flat files as primary:**

Download `sm.data.0.Current` via `client.get_data('SM', '0.Current')`. This file is larger than the CE equivalent but still manageable. Filter to target state/supersector/data_type combinations.

**Parameters:**
- `states` — list of 2-digit state FIPS codes, defaults to all 50 states + DC
- `supersectors` — list of supersector codes, defaults to all 10 + total nonfarm + total private
- Other params as in CES National

**Returns:** Polars DataFrame with columns: `series_id`, `date`, `year`, `period`, `value`, `state_fips`, `area_code`, `supersector_code`, `is_seasonally_adjusted`.

### 5.3 State and area mapping helpers

- `fetch_state_mapping(client: BLSHttpClient | None = None) -> pl.DataFrame` — downloads `sm.state` mapping file, returns DataFrame of state FIPS → state name
- `fetch_area_mapping(client: BLSHttpClient | None = None) -> pl.DataFrame` — downloads `sm.area` mapping file for MSA codes

---

## Task 6: Create `src/alt_nfp/ingest/bls/__init__.py` — Package Exports

```python
'''
BLS data download infrastructure for alt-nfp.

Provides download functions for three BLS programs:
- QCEW (Quarterly Census of Employment and Wages)
- CES National (Current Employment Statistics, national)
- CES State (Current Employment Statistics, state and area)

Ported from eco-stats (https://github.com/lowmason/eco-stats),
retaining only the BLS functionality needed for NFP nowcasting.
'''

from alt_nfp.ingest.bls._http import BLSHttpClient
from alt_nfp.ingest.bls._programs import (
    BLSProgram,
    SeriesField,
    build_series_id,
    get_program,
    list_programs,
    parse_series_id,
)
from alt_nfp.ingest.bls.ces_national import fetch_ces_national
from alt_nfp.ingest.bls.ces_state import fetch_ces_state
from alt_nfp.ingest.bls.qcew import fetch_qcew

__all__ = [
    'BLSHttpClient',
    'BLSProgram',
    'SeriesField',
    'build_series_id',
    'fetch_ces_national',
    'fetch_ces_state',
    'fetch_qcew',
    'get_program',
    'list_programs',
    'parse_series_id',
]
```

---

## Task 7: Update Transform Modules to Use `bls/`

### 7.1 Split `ingest/ces.py` → `ingest/ces_national.py` + `ingest/ces_state.py`

The existing `TODO_data_infrastructure.md` defines a single `ingest/ces.py`. Split this into two files.

**`ingest/ces_national.py`**: Contains `ingest_ces_national()` orchestrator with two paths:
- **API path**: calls `bls.fetch_ces_national()` to get current data, then transforms to `PANEL_SCHEMA`
- **Vintage path**: reads archived release files from `raw_dir / 'ces_national'` (as currently specified in the TODO)
- Combines, deduplicates, validates

**`ingest/ces_state.py`** (NEW): Contains `ingest_ces_state()` orchestrator with:
- **API path**: calls `bls.fetch_ces_state()` to get current data, transforms to `PANEL_SCHEMA`
- **Vintage path**: reads archived state release files (format TBD — state releases follow a different schedule than national, published on the 5th Friday after the 12th of the reference month)
- Must handle the fact that state CES estimates are not constrained to sum to national
- Set `source = 'ces_state_sa'` / `'ces_state_nsa'` to distinguish from national
- Set `industry_level = 'supersector'` (state CES publishes at supersector level)
- Include `state_fips` as an additional column (extend `PANEL_SCHEMA` or add as metadata)

### 7.2 Update `ingest/qcew.py` to use `bls.fetch_qcew()`

Replace the planned `fetch_qcew_current()` function with a call to `bls.fetch_qcew()`. The transform logic (filtering by area/ownership, extracting monthly employment, computing growth rates, assigning vintage metadata) stays in `ingest/qcew.py`.

### 7.3 Update `ingest/__init__.py`

Add re-exports for `ingest_ces_national`, `ingest_ces_state`, and the `bls` subpackage.

### 7.4 Update `ingest/panel.py`

The `build_panel()` function should call `ingest_ces_national()`, `ingest_ces_state()` (when state estimation is enabled), and `ingest_qcew()` separately.

---

## Task 8: Tests

### 8.1 `tests/ingest/bls/test_programs.py`

Port from eco-stats `tests/test_bls.py`, keeping only CE/SM/EN tests:
- `test_parse_series_id_ce()` — roundtrip `'CES0000000001'`
- `test_parse_series_id_sm()` — roundtrip a SM series ID
- `test_parse_series_id_en()` — roundtrip an EN series ID
- `test_build_series_id_defaults()` — zero-fills missing components
- `test_unknown_prefix_raises()` — KeyError for non-CE/SM/EN prefixes

### 8.2 `tests/ingest/bls/test_http.py`

- `test_parse_tsv_basic()` — port from eco-stats
- `test_parse_tsv_strips_whitespace()` — port from eco-stats
- `test_parse_tsv_empty()` — port from eco-stats
- `test_cache_path_sanitization()` — filenames with slashes get sanitized
- No network tests (mock HTTP responses or test only parsing logic)

### 8.3 `tests/ingest/bls/test_downloads.py`

Integration tests (mark with `@pytest.mark.network` so they can be skipped in CI):
- `test_fetch_qcew_single_quarter()` — fetch one quarter, verify expected columns
- `test_fetch_ces_national_total_nonfarm()` — fetch total nonfarm, verify data shape
- `test_fetch_ces_state_single_state()` — fetch one state, verify series ID parsing

---

## Implementation Order

1. **Task 1** (`_http.py`) — foundation, no dependencies within alt-nfp
2. **Task 2** (`_programs.py`) — depends on nothing, pure data definitions
3. **Task 6** (`__init__.py`) — wire up the package
4. **Task 3** (`qcew.py`) — depends on Tasks 1-2
5. **Task 4** (`ces_national.py`) — depends on Tasks 1-2
6. **Task 5** (`ces_state.py`) — depends on Tasks 1-2
7. **Task 8** (tests) — after each module is implemented
8. **Task 7** (update transforms) — after download layer is complete and tested

Tasks 3, 4, 5 are independent of each other and can be implemented in parallel.

---

## What NOT to Port from eco-stats

- BEA client (`bea_client.py`) — not needed
- Census client (`census_client.py`) — not needed
- FRED client (`fred_client.py`) — not needed
- The `EcoStats` unified wrapper (`__main__.py`) — not needed
- The `utils/` helpers (`validate_date`, `calculate_percent_change`, etc.) — Polars handles these natively
- BLS programs other than CE, SM, EN — not needed (AP, BD, CI, CU, CW, IP, JT, LA, LN, PC, WP)
- The `examples/` directory — not applicable
- Legacy pandas support — Polars only
