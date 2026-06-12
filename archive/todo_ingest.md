# TODO: Port Missing Functionality from `bls-estimates` into `alt_nfp/ingest`

## Context

The [`bls-estimates`](https://github.com/lowmason/bls-estimates) repo is a
standalone pipeline that downloads current BLS estimates (QCEW, CES National,
SAE), tags them with vintage/revision metadata, aggregates state-level data to
Census Regions and Divisions, and combines everything into a unified
`releases.parquet`.

The [`alt_nfp/src/alt_nfp/ingest`](https://github.com/lowmason/alt_nfp) module
handles a similar job but is model-focused: it downloads BLS data, transforms it
to `PANEL_SCHEMA` (growth rates, revision metadata), and feeds the Bayesian
state space model.  Several capabilities in `bls-estimates` are absent or
incomplete in `alt_nfp/ingest` and would strengthen the data pipeline — especially
for Releases 2 (industry) and 3 (geography).

**Source of truth for each capability:**

| Capability | `bls-estimates` location | `alt_nfp` location |
|---|---|---|
| BLS JSON API client | `bls/client.py` | `ingest/bls/_http.py` (equivalent) |
| QCEW CSV client | `bls/qcew.py` | `ingest/bls/_http.py` (partial) |
| BLS program registry | `bls/programs.py` | `ingest/bls/_programs.py` (equivalent) |
| Series ID builder/parser | `bls/series_id.py` | `ingest/bls/_programs.py` (equivalent) |
| CES National download | `download/ces.py` | `ingest/bls/ces_national.py` (equivalent) |
| CES State download | `download/sae.py` | `ingest/bls/ces_state.py` (partial) |
| QCEW download | `download/qcew.py` | `ingest/bls/qcew.py` (partial) |
| Industry mapping (CES ↔ QCEW) | `series/industries.py` | `lookups/industry.py` (partial) |
| EN series ID helpers | `series/builder.py` | **missing** |
| Geography lookups | `processing/geography.py` | `lookups/geography.py` (placeholder) |
| Geographic aggregation | `processing/aggregate.py` | **missing** |
| Release date scraping | `release_dates/` | **missing** |
| Vintage date construction | `release_dates/vintage_dates.py` | **missing** |
| Vintage tagging | `vintage/tagger.py` | `ingest/vintage_store.py` (different approach) |
| Combined output | `processing/combine.py` | **missing** |

**Priority order:** Tasks 1–3 are foundational and should be done first. Tasks
4–6 depend on Tasks 1–3 and unlock Release 2/3 capabilities. Tasks 7–8 are
integration/output improvements.

## Code Style

- Python 3.12, line length 100, formatter: black, linter: ruff
- Use **Polars** (not pandas) for all DataFrames
- Prefer **single quotes** over double quotes for strings
- **Two blank lines** after class and function definitions (PEP 8)
- Type hints on all public functions
- Docstrings on all public functions and classes (numpy-style)

---

## Task 1: Release Date Scraping and Vintage Date Construction

### What exists

`alt_nfp` has no equivalent of `bls-estimates/release_dates/`. The vintage
store (`ingest/vintage_store.py`) reads pre-built Hive-partitioned parquet but
has no mechanism to generate vintage_dates from BLS news release pages. Without
this, the pipeline depends on externally produced vintage metadata.

### What to port

The full `bls-estimates/release_dates/` module:

- **`config.py`**: Publication definitions (CES → `empsit`, SAE → `laus`,
  QCEW → `cewqtr`) with index URLs and frequency metadata.
- **`scraper.py`**: Async index page fetcher (`fetch_index`), archive link
  parser (`parse_index_page`), and concurrent HTML downloader
  (`download_all`). Uses `httpx` + `lxml`/`BeautifulSoup`.
- **`parser.py`**: Extracts vintage dates from downloaded HTML embargo lines
  using regex. Parses ref_date from filenames
  (`{publication}_{yyyy}_{mm}.htm`).
- **`vintage_dates.py`**: The core logic that builds a complete vintage_dates
  DataFrame from release_dates with publication-specific revision semantics:
  - CES: revisions 0, 1, 2 + benchmark (Jan Y+1 triggers
    benchmark_revision=1 for all year Y months).
  - SAE: revisions 0, 1 + two benchmark generations (March Y+1 and Y+2).
  - QCEW: revisions 0–4 with asymmetric schedule (Q1 gets 4 revisions, Q4
    gets 1).
  - Government shutdown special cases (CES/SAE Oct 2025 released with Nov
    2025; SAE Sep 2013 released with Oct 2013).
  - Supplemental release dates for early-2010 gaps.

### Where to put it

`src/alt_nfp/ingest/release_dates/` — mirror the `bls-estimates` structure but
adapt imports to use `alt_nfp.ingest.bls` infrastructure.

### Deliverables

- `ingest/release_dates/__init__.py`
- `ingest/release_dates/config.py`
- `ingest/release_dates/scraper.py`
- `ingest/release_dates/parser.py`
- `ingest/release_dates/vintage_dates.py`
- CLI integration or a `build_vintage_dates()` function callable from the panel
  builder.

### Dependencies

- `httpx`, `beautifulsoup4`, `lxml` (add to `pyproject.toml` if not already
  present).

---

## Task 2: Geography Lookups — Census Regions and Divisions

### What exists

`alt_nfp/lookups/geography.py` is a placeholder:

```python
# TODO: implement when cell-level estimation is added
```

### What to port

`bls-estimates/processing/geography.py` contains:

- `STATES`: all 50 states + DC + Puerto Rico (52 two-digit FIPS codes).
- `FIPS_TO_REGION`: state FIPS → Census Region (1–4).
- `FIPS_TO_DIVISION`: state FIPS → Census Division (01–09).
- Puerto Rico assignment: Region 3 (South), Division 05 (South Atlantic).

### Where to put it

`src/alt_nfp/lookups/geography.py` — replace the placeholder.

### Deliverables

- `STATES` list.
- `FIPS_TO_REGION` and `FIPS_TO_DIVISION` dicts.
- Optionally expose as a Polars LazyFrame (`GEOGRAPHY_HIERARCHY`) for
  consistency with `INDUSTRY_HIERARCHY` in `lookups/industry.py`.

### Tests

- Verify all 52 FIPS codes map to valid region and division codes.
- Verify Puerto Rico assignment.

---

## Task 3: CES ↔ QCEW Industry Cross-Mapping and EN Series IDs

### What exists

`alt_nfp/lookups/industry.py` has a clean NAICS → supersector → domain
hierarchy and CES series ID construction, but is missing:

- QCEW NAICS slice codes (the `'10'`, `'1011'`, `'1012'`, ... codes used by
  the QCEW CSV API).
- EN (QCEW) series ID construction for cross-referencing.
- A unified `IndustryEntry` that ties CES codes to QCEW codes.

`alt_nfp/ingest/bls/qcew.py` has its own `_QCEW_TO_SECTOR` mapping, but
it's local and doesn't connect back to the lookup tables.

### What to port

From `bls-estimates/series/industries.py`:

- The `IndustryEntry` dataclass with fields: `industry_code`,
  `industry_type` (domain/supersector/sector), `industry_name`, `ces_code`,
  `qcew_naics`, `en_industry`.
- `INDUSTRY_MAP`: the complete list of entries spanning domain, supersector,
  and sector levels.
- The QCEW NAICS lookup tables: `_QCEW_NAICS_BY_SUPERSECTOR`,
  `_QCEW_NAICS_BY_SECTOR`.

From `bls-estimates/series/builder.py`:

- `en_series_id(industry_entry, area, ownership)` → EN series ID string.
- `en_series_id_for_state(industry_entry, state_fips, ownership)` → state EN
  series ID.

### Where to put it

Extend `src/alt_nfp/lookups/industry.py`:

- Add `qcew_naics` and `en_industry` fields to the existing hierarchy rows
  (or create a parallel `IndustryEntry` dataclass).
- Add `INDUSTRY_MAP` as a list of `IndustryEntry` objects.
- Add `en_series_id()` and `en_series_id_for_state()` as module-level
  functions (these use `build_series_id` from `ingest/bls/_programs.py`).

### Cleanup

- Remove the local `_QCEW_TO_SECTOR` and `_NAICS_TO_SECTOR` dicts from
  `ingest/bls/qcew.py` and `ingest/qcew.py` once the canonical mapping lives
  in `lookups/industry.py`.

---

## Task 4: State-Level QCEW Download

### What exists

`alt_nfp/ingest/bls/qcew.py`'s `fetch_qcew()` filters to a single
`area_fips` (default `'US000'` for national). It cannot download state-level
QCEW data.

### What to port

`bls-estimates/download/qcew.py` downloads QCEW industry slices and retains
**both** national and state-level rows:

- National: `area_fips='US000'`, `agglvl_code` in `('10', '11')`.
- State: `area_fips` matching `XX000` pattern (first 2 chars in `STATES`),
  `agglvl_code` in `('50', '51')`.
- Each row gets a proper geographic_type (`'national'` or `'state'`) and
  geographic_code.
- EN series IDs are assigned via the industry mapping (Task 3).

### Where to put it

Extend `src/alt_nfp/ingest/bls/qcew.py`:

- Add a `fetch_qcew_by_geography()` or modify `fetch_qcew()` to accept an
  `area_fips` parameter that can be `'all'` (national + all states).
- Add state-row filtering logic using the `STATES` list from
  `lookups/geography.py` (Task 2).

Also extend `src/alt_nfp/ingest/qcew.py` to handle state-level rows in the
`PANEL_SCHEMA` transformation.

### Dependencies

- Task 2 (geography lookups for `STATES`).
- Task 3 (industry mapping for EN series IDs).

---

## Task 5: QCEW Area and Size Slicing

### What exists

`alt_nfp/ingest/bls/_http.py`'s `get_qcew_csv()` only supports industry
slices (`/year/quarter/industry/{code}.csv`).

### What to port

`bls-estimates/bls/qcew.py`'s `QCEWClient` supports three slice types:

- `get_industry(industry_code, ...)` — slice by industry (already
  supported).
- `get_area(area_code, ...)` — slice by area (e.g., `'US000'`,
  `'01000'`).
- `get_size(size_code, ...)` — slice by size class (Q1 only).

The area and size slice types use different URL patterns:
`/{year}/{quarter}/area/{code}.csv` and `/{year}/{quarter}/size/{code}.csv`.

### Where to put it

Extend `BLSHttpClient.get_qcew_csv()` to accept a `slice_type` parameter
(default `'industry'`):

```python
def get_qcew_csv(
    self,
    year: int,
    quarter: int,
    slice_code: str,
    slice_type: str = 'industry',
) -> pl.DataFrame:
```

Or add separate `get_qcew_area()` and `get_qcew_size()` methods.

### Why it matters

- Area slices provide a more efficient way to get all industries for a
  specific geography (useful for Release 3).
- Size slices are needed for the large-firm coverage analysis and sample
  composition diagnostics.

---

## Task 6: Geographic Aggregation — State → Region/Division

### What exists

`alt_nfp` has no aggregation of state-level data to Census Regions or
Divisions.

### What to port

`bls-estimates/processing/aggregate.py`:

- `_aggregate_geo(state_df, region_dict, division_dict)`: Takes a DataFrame
  with national + state rows, adds region and division rows by mapping
  state FIPS via `FIPS_TO_REGION` and `FIPS_TO_DIVISION`, and summing
  employment within group columns.
- `aggregate_qcew_sae(qcew_path, sae_path)`: Reads tagged parquet files,
  applies `_aggregate_geo`, overwrites files.

### Where to put it

`src/alt_nfp/ingest/aggregate.py` — a new module that:

- Takes a DataFrame with state-level rows.
- Produces region and division rows using `lookups/geography.py` mappings.
- Can operate on both vintage store data and current-vintage panels.

### Dependencies

- Task 2 (geography lookups).
- Task 4 (state-level QCEW data to aggregate).

### Note

This is primarily needed for Release 3 (geographic decomposition) but the
infrastructure should be built early.

---

## Task 7: Vintage Tagging Pipeline

### What exists

`alt_nfp/ingest/vintage_store.py` reads and transforms a Hive-partitioned
vintage store, but the store must be populated externally. There is no
automated pipeline to:

1. Download BLS estimates (current).
2. Tag them with vintage_date, revision, benchmark_revision.
3. Append the tagged rows to the vintage store.

### What to port

`bls-estimates/vintage/tagger.py`:

- `_latest_vintage_lookup(vintage_df, publication)`: Per ref_date, takes
  max revision, max benchmark_revision, max vintage_date.
- `tag_estimates(vintage_dates_path)`: Joins vintage metadata onto estimate
  DataFrames.

### Where to put it

`src/alt_nfp/ingest/tagger.py` — adapts the bls-estimates tagging logic to
work with `alt_nfp`'s schemas:

- Reads `vintage_dates.parquet` (from Task 1).
- Tags current-vintage download DataFrames with revision metadata.
- Feeds tagged rows into `append_to_vintage_store()`.

### Deliverables

- `ingest/tagger.py` with `tag_and_append()` function.
- Integration into the panel builder pipeline.

### Dependencies

- Task 1 (vintage_dates construction).

---

## Task 8: Combined Releases Output

### What exists

`alt_nfp/ingest/panel.py` builds a `PANEL_SCHEMA` DataFrame (growth-rate
oriented, model-ready). There is no equivalent of bls-estimates'
`releases.parquet` — a levels-based, source-unified file suitable for
revision analysis and diagnostics.

### What to port

`bls-estimates/processing/combine.py`:

- `COMBINED_COLUMNS` and `COMBINED_SCHEMA`: unified column set (source,
  seasonally_adjusted, geographic_type, geographic_code, industry_type,
  industry_code, ref_date, vintage_date, revision, benchmark_revision,
  employment).
- `combine_estimates(qcew_path, ces_path, sae_path)`: Reads all tagged
  estimate parquets, selects unified columns, casts to schema, and writes
  `releases.parquet`.

### Where to put it

`src/alt_nfp/ingest/releases.py` — produces a `releases.parquet` alongside
the existing `observation_panel.parquet`. This file:

- Stores employment **levels** (not growth rates).
- Retains full vintage/revision metadata.
- Is suitable for revision analysis, diagnostic plots, and benchmark
  prediction work.
- Can be derived from the vintage store via a simple projection.

### Why it matters

The benchmark prediction work and revision diagnostics need levels-based
data with full vintage tracking. Having a standardized releases file makes
this much easier.

---

## Summary of Dependencies

```
Task 1 (release dates) ──────┐
                              ├─→ Task 7 (vintage tagging)
Task 2 (geography) ──────────┤
                              ├─→ Task 4 (state QCEW) ──→ Task 6 (aggregation)
Task 3 (industry mapping) ───┤
                              └─→ Task 5 (QCEW area/size slicing)

Task 7 (vintage tagging) ────→ Task 8 (combined releases)
```

Tasks 1, 2, and 3 are independent of each other and can be done in parallel.