# nfp-download

HTTP clients and scrapers for downloading raw BLS and FRED data.

## Overview

Generic download layer — no data transformation, just fetching. Provides:
- **BLS API client** (`bls/`): structured HTTP layer for CES and QCEW series
- **FRED client** (`fred.py`): single-series JSON API downloader with retry
- **HTTP retry client** (`client.py`): shared httpx client with exponential backoff
- **Release date scraper** (`release_dates/`): BLS publication schedule HTML scraping + parsing

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.10)
- **HTTP**: httpx (async, HTTP/2), requests (BLS legacy)
- **Parsing**: BeautifulSoup4 + lxml
- **Build**: hatchling
- **Internal deps**: `nfp-lookups` (geography for QCEW state filtering)

## Key Commands

```bash
# Run download tests
pytest tests/

# Lint
ruff check src/nfp_download/
```

## Package Structure

```
src/nfp_download/
├── __init__.py
├── client.py               # create_client(), get_with_retry() — shared httpx utilities
├── fred.py                 # fetch_fred_series(series_id) — FRED JSON API, requires FRED_API_KEY
├── bls/
│   ├── __init__.py
│   ├── _http.py            # BLSHttpClient — CSV download transport for BLS API
│   ├── _programs.py        # BLS program definitions (CES, QCEW series ID structure)
│   ├── ces_national.py     # fetch_ces_national() — CES national series
│   ├── ces_state.py        # fetch_ces_state() — CES state series
│   └── qcew.py             # fetch_qcew(), fetch_qcew_with_geography()
└── release_dates/
    ├── __init__.py
    ├── scraper.py           # Async scraper for BLS publication schedule HTML pages
    └── parser.py            # Parse scraped HTML into release date records
```

## Code Style

- **Formatter**: black (line length 100)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- Line length limit: 100 characters

## Key Patterns

- **FRED client** (`fred.py`): `fetch_fred_series(series_id)` returns a Polars DataFrame with `(ref_date, value)`. Uses httpx with exponential-backoff retry. Requires `FRED_API_KEY` env var.
- **BLS HTTP client** (`bls/_http.py`): `BLSHttpClient` handles CSV downloads from BLS. No internal dependencies beyond `_programs.py` for series ID construction.
- **BLS programs** (`bls/_programs.py`): `build_series_id()` and `parse_series_id()` encode/decode BLS series ID conventions. Pure functions, no I/O.
- **Release date scraper** (`release_dates/scraper.py`): async httpx scraper that fetches BLS schedule HTML. Config values (URLs, start year, output dirs) should be passed as parameters, not imported from other packages.
- **All download functions should accept output paths as parameters** rather than importing path constants, to maintain package independence.
- **`@pytest.mark.network`**: tests requiring network access are marked; deselect with `-m "not network"`.

## Test Mapping

Tests from the monorepo `tests/` that belong here:
- `tests/ingest/bls/test_downloads.py` — BLS download integration tests
- `tests/ingest/bls/test_http.py` — BLS HTTP client tests
- `tests/ingest/bls/test_programs.py` — Series ID construction/parsing tests
- `test_fred.py` — FRED client tests
