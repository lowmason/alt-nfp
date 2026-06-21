# nfp-download

HTTP clients and scrapers for downloading raw BLS and FRED data.

## Overview

Generic download layer — no data transformation, just fetching. Provides:
- **BLS API client** (`bls/`): structured HTTP layer for CES and QCEW series
- **FRED client** (`fred.py`): single-series JSON API downloader with retry
- **HTTP retry client** (`client.py`): shared httpx client with exponential backoff, plus Chrome-impersonating curl_cffi session for www.bls.gov
- **Release date scraper** (`release_dates/`): BLS publication schedule HTML scraping + parsing

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.12)
- **HTTP**: httpx (async, HTTP/2); curl_cffi (Chrome impersonation) for www.bls.gov pages only
- **Parsing**: BeautifulSoup4 + lxml
- **Build**: hatchling
- **Internal deps**: `nfp-lookups` (geography for QCEW state filtering)

## Key Commands

```bash
# Run download tests
pytest src/nfp_download/tests/

# Lint
ruff check src/nfp_download/
```

## Package Structure

```
src/nfp_download/
├── __init__.py
├── client.py               # create_client(), create_impersonating_session(), get_with_retry()
├── fred.py                 # fetch_fred_series(series_id) — FRED JSON API, requires FRED_API_KEY
├── bls/
│   ├── __init__.py
│   ├── _http.py            # BLSHttpClient — CSV download transport for BLS API
│   ├── _programs.py        # Back-compat shim → nfp_lookups.series_ids (series-ID grammar)
│   ├── ces_national.py     # fetch_ces_national() — CES national series
│   ├── ces_state.py        # fetch_ces_state() — CES state series
│   ├── qcew.py             # fetch_qcew(), fetch_qcew_with_geography()
│   └── bulk.py             # download_ces (cesvinall.zip), download_qcew (revisions CSV), download_qcew_bulk (yearly singlefiles) — moved from nfp-vintages in A2
└── release_dates/
    ├── __init__.py
    ├── scraper.py           # Async scraper for BLS publication schedule HTML pages
    ├── feed.py              # parse_feed()/fetch_feed() — BLS empsit/cewqtr RSS (curl_cffi impersonation, for watch)
    └── parser.py            # Parse scraped HTML into release date records
```

## Code Style

- **Formatter**: black (line length 100)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- Line length limit: 100 characters

## Key Patterns

- **FRED client** (`fred.py`): `fetch_fred_series(series_id)` returns a Polars DataFrame with `(ref_date, value)`. Uses httpx with exponential-backoff retry. Requires `FRED_API_KEY` env var.
- **BLS HTTP client** (`bls/_http.py`): `BLSHttpClient` handles CSV downloads from BLS. No internal dependencies beyond `_programs.py` for series ID construction. `cache_dir` defaults to `None` ⇒ a per-process `tempfile.mkdtemp` (atexit-cleaned), so it never writes under the CWD/`./data` on Bloomberg's container (plans/15 Task 8); pass an explicit `cache_dir=` to persist across runs.
- **Series-ID grammar**: `build_series_id()` / `parse_series_id()` live in `nfp_lookups.series_ids` (pure reference data); `bls/_programs.py` re-exports them for back-compat.
- **Release date scraper** (`release_dates/scraper.py`): async scraper for BLS schedule HTML. Transport is curl_cffi `AsyncSession(impersonate='chrome')` via `create_session()` — www.bls.gov (Akamai) fingerprints TLS, so httpx/plain curl get 403 regardless of headers. Callers catch the re-exported `FetchError`. Config values (URLs, start year, output dirs) should be passed as parameters, not imported from other packages.
- **Sync www.bls.gov transport** (`client.py`): `create_impersonating_session()` is the sync counterpart of the scraper's `create_session()`, used by nfp-vintages for the CES vintage zip and QCEW revisions CSV. `get_with_retry()` accepts either an `httpx.Client` or this session (same `get`/`raise_for_status` surface). Non-www.bls.gov hosts (api.bls.gov, data.bls.gov, FRED) stay on httpx.
- **`download.bls.gov` flat-file UA** (`bls/_http.py`): `BLSHttpClient` fetches LABSTAT flat files over plain httpx (no impersonation needed), but Akamai there enforces two User-Agent rules verified live: (1) it 403s any UA containing the `github.com` token — URL *or* email domain (`mac.com`/`example.com` pass); (2) the heavy bulk files (`ce.data.0.AllCESSeries`, ~47 MB) additionally require a contact email — a bare product token 403s on them though it passes on small overview files. `_build_user_agent()` emits `alt-nfp/0.1.0 (<email>)`, defaulting the email to a non-PII placeholder and overridable via `BLS_CONTACT_EMAIL` (set in `.env` for real acquisition). This is a *distinct* mechanism from www.bls.gov's TLS fingerprinting above — don't conflate them.
- **All download functions should accept output paths as parameters** rather than importing path constants, to maintain package independence.
- **`@pytest.mark.network`**: tests requiring network access are marked; deselect with `-m "not network"`.

## Test Mapping

Tests live in `src/nfp_download/tests/` within this package:
- `src/nfp_download/tests/bls/test_downloads.py` — BLS download integration tests (network-marked)
- `src/nfp_download/tests/bls/test_http.py` — BLS HTTP client tests
- `src/nfp_download/tests/bls/test_programs.py` — re-export smoke test (grammar tests live in nfp-lookups)
- `src/nfp_download/tests/bls/test_bulk_network.py` — live www.bls.gov bulk-download transport tests (network-marked)
- `src/nfp_download/tests/release_dates/test_scraper_network.py` — live BLS scraper transport tests (network-marked)
- `src/nfp_download/tests/test_fred.py` — FRED client tests
- `src/nfp_download/tests/test_client.py` — HTTP client retry logic tests
