# TODO: Port `bls-revisions` Vintage-Building Pipeline into `alt_nfp`

## Context

The [`bls-revisions`](https://github.com/lowmason/bls-revisions) repo provides a
complete three-stage pipeline for building historical revision data:

1. **Release date scraping** — download BLS news-release archive pages and extract
   the embargo (vintage) dates that tell us *when* each data point was first
   published.
2. **Data downloading** — fetch the CES triangular-revision spreadsheets and the
   QCEW revisions CSV from bls.gov.
3. **Processing** — read the raw files, attach vintage dates, and produce tidy
   revision parquets for CES national, SAE state-level (via ALFRED), and QCEW.

A separate repo ([`bls-estimates`](https://github.com/lowmason/bls-estimates))
provides the *current* CES/SAE/QCEW estimates as `releases.parquet`. The
`scratch.ipynb` notebook in `bls-revisions` combines `revisions.parquet` (historical)
with `releases.parquet` (current) into the Hive-partitioned
`./data/raw/vintages/vintage_store` that `alt_nfp` consumes.

**This pipeline has already been run** — the `vintage_store` exists and is in use.
The goal of this TODO is to bring the logic into `alt_nfp` so it can be re-run
in-repo if the vintage store ever needs to be rebuilt or extended, without
depending on external repos.

### What `alt_nfp` already has

- `ingest/bls/` — downloads *current* CES/SAE/QCEW data from BLS APIs (ported
  from `eco-stats`)
- `ingest/vintage_store.py` — reads, writes, appends, and compacts the
  Hive-partitioned vintage store
- `vintages/` — view functions and evaluation over the observation panel

### What's missing

The entire pipeline to *build* the vintage store from scratch: release-date
scraping, revision data downloading, revision processing, combining sources,
and merging with current estimates.

## References

- `bls-revisions` repo — https://github.com/lowmason/bls-revisions
  - `src/bls_revisions/release_dates/` — scraping + vintage date assignment
  - `src/bls_revisions/download/` — CES vintage files + QCEW revisions CSV
  - `src/bls_revisions/processing/` — CES national, SAE states, QCEW, combiner
  - `src/bls_revisions/_client.py` — HTTP client with retry
  - `scratch.ipynb` — final combine step (revisions + releases → vintage_store)
- `bls-estimates` repo — https://github.com/lowmason/bls-estimates
  - Provides `releases.parquet` (current estimates with `current=1` flag)
- `alt_nfp` existing code:
  - `src/alt_nfp/ingest/bls/_http.py` — existing BLS HTTP client (potential merge target)
  - `src/alt_nfp/ingest/vintage_store.py` — existing store reader/writer
  - `src/alt_nfp/config.py` — `DATA_DIR` and project paths

## Code Style

- Python 3.12, line length 100, formatter: black, linter: ruff
- Use **Polars** (not pandas) for all DataFrames
- Prefer **single quotes** over double quotes for strings
- **Two blank lines** after class and function definitions (PEP 8)
- Type hints on all public functions
- Docstrings on all public functions and classes (numpy-style)

---

## Task 1: Port Release Date Scraping into `vintages/release_dates/`

### What this does

The release-dates subpackage scrapes BLS news-release archive index pages for
three publications (CES/empsit, SAE/laus, QCEW/cewqtr), downloads individual
release HTML files, and parses the embargo line to extract the date each
reference period was first published. This produces `release_dates.parquet`
(one row per publication × ref_date → vintage_date).

### Source files to port

| `bls-revisions` source | Target in `alt_nfp` |
|---|---|
| `release_dates/config.py` | `vintages/release_dates/config.py` |
| `release_dates/parser.py` | `vintages/release_dates/parser.py` |
| `release_dates/scraper.py` | `vintages/release_dates/scraper.py` |
| `release_dates/read.py` | `vintages/release_dates/read.py` |
| `release_dates/vintage_dates.py` | `vintages/release_dates/vintage_dates.py` |
| `release_dates/__main__.py` | `vintages/release_dates/__main__.py` |

### Adaptation notes

- Update all path constants to use `alt_nfp.config.DATA_DIR` instead of
  hard-coded `Path('data/...')` relative to CWD.
- Store scraped HTML files under `DATA_DIR / 'raw' / 'releases' / {publication}/`.
- Store output parquets under `DATA_DIR / 'raw' / 'vintages/'`.
- The async scraper uses `httpx.AsyncClient` — keep as-is, it handles BLS
  rate limits well.
- The `SUPPLEMENTAL_RELEASE_DATES` list (CES/SAE Jan–Mar 2016) should be
  preserved for gap-filling.
- Add `httpx`, `beautifulsoup4`, and `lxml` to project dependencies if not
  already present.

### Acceptance criteria

- `python -m alt_nfp.vintages.release_dates` runs the full scrape → parse →
  build pipeline and produces `release_dates.parquet` and `vintage_dates.parquet`.
- Round-trip test: parse a sample release HTML and verify extracted vintage date.
- `read_release_dates()` and `read_vintage_dates()` return correctly typed
  DataFrames.

---

## Task 2: Port CES Triangular Revision Downloader

### What this does

Scrapes the BLS CES vintage-data page (`bls.gov/web/empsit/cesvindata.htm`),
discovers links to `cesvinall.zip` and `cesvin*.xlsx` files, downloads them,
and extracts the zip into `data/raw/ces/cesvinall/`.

### Source files to port

| `bls-revisions` source | Target in `alt_nfp` |
|---|---|
| `download/ces.py` | `vintages/download/ces.py` |

### Adaptation notes

- Reuse `ingest/bls/_http.py`'s retry logic or port `_client.py`'s
  `get_with_retry()` as a shared utility in `vintages/_client.py`.
- Output directory: `DATA_DIR / 'raw' / 'ces' / 'cesvinall'`.
- The CES vintage page layout is stable but could change; the link discovery
  uses BeautifulSoup and is robust to minor HTML changes.

### Acceptance criteria

- `download_ces()` fetches and extracts all triangular CSV files.
- Idempotent: skips files that already exist locally.

---

## Task 3: Port QCEW Revisions CSV Downloader

### What this does

Downloads the single QCEW revisions CSV from
`bls.gov/cew/revisions/qcew-revisions.csv` (2017–present) containing initial
and revised employment counts by state and quarter.

### Source files to port

| `bls-revisions` source | Target in `alt_nfp` |
|---|---|
| `download/qcew.py` | `vintages/download/qcew.py` |

### Adaptation notes

- Output: `DATA_DIR / 'raw' / 'qcew' / 'qcew-revisions.csv'`.
- This is a single HTTP GET with retry — straightforward port.

### Acceptance criteria

- `download_qcew()` saves the CSV to the expected path.

---

## Task 4: Port CES National Triangular Revision Processing

### What this does

Reads every `tri_{code}_{SA|NSA}.csv` file from `cesvinall/`, extracts the
k=0, 1, 2 revision diagonals from the triangular matrix, joins vintage dates
from `vintage_dates.parquet`, and writes `ces_revisions.parquet`.

The triangular matrix has rows indexed by (year, month) = reference period and
columns `Jan_10`, `Feb_10`, ... = vintage (publication month). The main diagonal
is the initial estimate; the first sub-diagonal is the first revision, etc.

### Source files to port

| `bls-revisions` source | Target in `alt_nfp` |
|---|---|
| `processing/ces_national.py` | `vintages/processing/ces_national.py` |

### Key data structures

- `CES_DOMAIN`, `CES_SUPERSECTOR`, `CES_SECTOR` — industry code tuples already
  partially overlap with `alt_nfp/lookups/industry.py`. Consolidate or
  cross-reference rather than duplicating.
- `_build_schema()` inspects sample CSV to discover available `Mon_YY` columns.
- `read_triangular_ces()` computes diagonal offsets and extracts revision values.

### Adaptation notes

- The diagonal extraction logic is the most complex piece — port carefully with
  unit tests on a small synthetic triangular matrix.
- Join vintage dates from the parquet produced by Task 1.
- Output schema matches `VINTAGE_STORE_SCHEMA` in
  `ingest/vintage_store.py`: `source`, `seasonally_adjusted`, `geographic_type`,
  `geographic_code`, `industry_type`, `industry_code`, `ref_date`, `vintage_date`,
  `revision`, `benchmark_revision`, `employment`.

### Acceptance criteria

- Processes all available industry × adjustment files.
- Output passes uniqueness assertion on the full key.
- Diagonal extraction produces correct values on a 5×5 synthetic test matrix.

---

## Task 5: Port SAE State-Level Revision Fetching from ALFRED

### What this does

For each state × industry combination, queries the FRED/ALFRED API for available
vintage dates, downloads wide-format observations (one column per vintage), and
extracts the initial (first non-null) and latest (last non-null) employment
level for every reference date. Produces `sae_revisions.parquet`.

### Source files to port

| `bls-revisions` source | Target in `alt_nfp` |
|---|---|
| `processing/ces_states.py` | `vintages/processing/sae_states.py` |

### Key components

- `FIPS_TO_ABBREV` dict — consider moving to `lookups/geography.py` (which
  currently exists but is minimal).
- `INDUSTRIES` list — 34 industry × adjustment combos per state, with two
  FRED series naming conventions (short-form `{ABBREV}{SUFFIX}` vs long-form
  `SMU{FIPS}00000{CES_CODE}01`).
- `build_series_df()` — generates the full cross product of states × industries
  × adjustments.
- `fetch_batch_sae_revisions()` — batch download with checkpointing, retry,
  and exponential backoff. Handles 429s and FRED 400/404 gracefully.
- `compute_initial_and_latest_levels()` — extracts initial and latest from
  ALFRED wide-format.

### Adaptation notes

- Requires `FRED_API_KEY` environment variable.
- This is the slowest pipeline step (~3,700 series × ~1s each). The
  checkpointing logic is essential for resumability.
- The `_split_revisions()` helper splits into revision-0 (initial) and
  revision-1 (latest) DataFrames — straightforward.
- Consider whether the existing `ingest/bls/ces_state.py` (which fetches
  *current* state CES data from BLS APIs) can share the FIPS/industry lookup
  tables.

### Acceptance criteria

- `build_series_df()` produces ~3,700 rows (54 geos × 34 industries × 2 adj).
- Checkpoint file is written every 100 series and cleaned up on completion.
- Skipped series (404s) are logged but do not halt the pipeline.

---

## Task 6: Port QCEW Revision Processing

### What this does

Reads the QCEW revisions CSV, filters to employment rows, maps area names to
FIPS codes, unpivots revision columns (Initial, First Revised, …, Fourth Revised)
into long format, joins vintage dates, and writes `qcew_revisions.parquet`.

### Source files to port

| `bls-revisions` source | Target in `alt_nfp` |
|---|---|
| `processing/qcew.py` | `vintages/processing/qcew.py` |

### Adaptation notes

- Needs `geographic_codes.csv` reference file — already present in bls-revisions
  as `data/reference/geographic_codes.csv`. Port this file into
  `DATA_DIR / 'raw' / 'reference'` or `lookups/`.
- The `STATES` list from `processing/__init__.py` overlaps with existing
  `lookups/geography.py` — consolidate.
- QCEW employment values in the CSV are in units (not thousands); the processing
  divides by 1000 to match CES conventions.

### Acceptance criteria

- Output contains revision 0–4 rows for Q1, 0–3 for Q2, 0–2 for Q3, 0–1 for Q4.
- Geographic codes correctly mapped from area names including US national (00)
  and Puerto Rico (72).

---

## Task 7: Port Vintage Series Combiner and Geographic Aggregation

### What this does

Reads `ces_revisions.parquet`, `sae_revisions.parquet`, and
`qcew_revisions.parquet`, vertically concatenates them, then creates region-
and division-level aggregates by summing state employment within each
geographic grouping. Writes `revisions.parquet`.

### Source files to port

| `bls-revisions` source | Target in `alt_nfp` |
|---|---|
| `processing/vintage_series.py` | `vintages/processing/combine.py` |

### Adaptation notes

- `_load_geo_lookups()` reads `geographic_codes.csv` for FIPS → region/division
  mapping. Share with Task 6.
- The `GROUP_COLS` list defines the uniqueness key — should match
  `VINTAGE_STORE_SCHEMA` keys in `ingest/vintage_store.py`.
- Output is `revisions.parquet` (historical revisions only, without current
  estimates).

### Acceptance criteria

- Output includes geographic_type in {national, state, region, division}.
- No duplicate rows on the full key.

---

## Task 8: Port the Vintage Store Builder (scratch.ipynb Logic)

### What this does

The final step that merges historical revisions with current estimates into the
Hive-partitioned `vintage_store`. From `scratch.ipynb`:

1. Read `revisions.parquet` (from Task 7) with `current=0`.
2. Read `releases.parquet` (from `bls-estimates`) with `current=1`.
3. Concatenate, normalize `industry_type` (set to `'national'` where
   `industry_code == '00'`).
4. Group-by deduplicate: for rows sharing the same full key, keep the *last*
   value (current estimates take precedence over revisions).
5. Write to `vintage_store` as Hive-partitioned parquet with partition columns
   `['source', 'seasonally_adjusted']`.

### Target file

`vintages/build_store.py`

### Adaptation notes

- The `releases.parquet` from `bls-estimates` contains the same schema as
  `revisions.parquet` but with only the most recent vintage per series. The
  `current` flag distinguishes origin.
- The deduplication logic (sort by `current` ascending → group_by → take last)
  ensures that where a revision and a current release overlap, the current
  release wins.
- This should be callable both as a module (`python -m alt_nfp.vintages.build_store`)
  and programmatically.
- Consider adding a parameter for the `releases.parquet` path since
  `bls-estimates` is a separate repo.

### Acceptance criteria

- Produces a valid Hive-partitioned parquet at `DATA_DIR / 'raw' / 'vintages' / 'vintage_store'`.
- Partition directories are `source={ces,sae,qcew}/seasonally_adjusted={true,false}`.
- `ingest.vintage_store.read_vintage_store()` successfully reads the output.

---

## Task 9: Shared HTTP Client and Dependency Cleanup

### What this does

The `bls-revisions` repo has `_client.py` with `create_client()` and
`get_with_retry()` using `httpx` with HTTP/2, browser-like headers, BLS API
key injection, and exponential backoff on 429/5xx. The existing
`ingest/bls/_http.py` in `alt_nfp` already has its own `BLSHttpClient`. These
should be reconciled.

### Options

**Option A:** Extend `ingest/bls/_http.py` to expose a standalone
`get_with_retry()` function that the vintage-building code can import.

**Option B:** Create `vintages/_client.py` as a thin wrapper, keeping the
vintage pipeline self-contained.

### Adaptation notes

- The key difference: `bls-revisions` uses a raw `httpx.Client` with a
  standalone retry function; `alt_nfp` uses a class-based `BLSHttpClient`.
  Either approach works.
- The async scraper in Task 1 uses `httpx.AsyncClient` — this is separate from
  the sync retry client and should stay async.
- Dependencies to ensure: `httpx[http2]`, `beautifulsoup4`, `lxml`.

### Acceptance criteria

- No duplicated retry logic between `ingest/bls/` and `vintages/`.
- `BLS_API_KEY` environment variable works for both download paths.

---

## Task 10: CLI Entry Point

### What this does

Port the unified CLI from `bls-revisions/__main__.py` into alt_nfp so the full
pipeline can be run with a single command.

### Target

`vintages/__main__.py` (invocable as `python -m alt_nfp.vintages`)

### Subcommands

```
python -m alt_nfp.vintages              # Run all steps
python -m alt_nfp.vintages release      # Scrape release dates
python -m alt_nfp.vintages download     # Download CES + QCEW revision files
python -m alt_nfp.vintages process      # Process CES, SAE, QCEW revisions
python -m alt_nfp.vintages build        # Combine + build vintage_store
```

### Acceptance criteria

- Running with no arguments executes the full pipeline end-to-end.
- Each subcommand can be run independently (assuming prerequisites exist).
- Requires `FRED_API_KEY` only for the `process` step (SAE fetching).

---

## Dependency Summary

Tasks are mostly independent but have a natural execution order:

```
Task 1 (release dates)  ──┐
Task 2 (CES download)   ──┤
Task 3 (QCEW download)  ──┼── Task 4 (CES processing)  ──┐
                           │   Task 5 (SAE processing)  ──┤
                           │   Task 6 (QCEW processing) ──┼── Task 7 (combine) ── Task 8 (build store)
                           │                               │
Task 9 (HTTP client)   ────┘                               │
Task 10 (CLI)          ────────────────────────────────────┘
```

Tasks 4, 5, and 6 depend on Task 1 (vintage dates) for the join step. Tasks 4
and 6 depend on Tasks 2 and 3 respectively for raw data. Task 7 depends on
Tasks 4–6 for the source parquets. Task 8 depends on Task 7 and also on
`bls-estimates` for current estimates. Task 9 should be done early since
Tasks 1–3 need HTTP infrastructure.

### New files created

```
src/alt_nfp/vintages/
    __init__.py                     (update)
    __main__.py                     (Task 10)
    _client.py                      (Task 9)
    build_store.py                  (Task 8)
    release_dates/
        __init__.py
        config.py                   (Task 1)
        parser.py                   (Task 1)
        scraper.py                  (Task 1)
        read.py                     (Task 1)
        vintage_dates.py            (Task 1)
        __main__.py                 (Task 1)
    download/
        __init__.py
        ces.py                      (Task 2)
        qcew.py                     (Task 3)
    processing/
        __init__.py
        ces_national.py             (Task 4)
        sae_states.py               (Task 5)
        qcew.py                     (Task 6)
        combine.py                  (Task 7)
```

### New dependencies

- `httpx[http2]` (may already be present)
- `beautifulsoup4`
- `lxml`
- `FRED_API_KEY` env var (for Task 5 only)