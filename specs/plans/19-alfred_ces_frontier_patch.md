# ALFRED CES Frontier Patch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the CES vintage-store gap (`vintage_date > 2026-02-11`, which the BLS `cesvinall` CSVs lack) by fetching the `(0,0)/(1,0)/(2,0)` prints from the FRED ALFRED API and appending them to `source='ces'` via the store's existing append path.

**Architecture:** A new ALFRED fetch layer (`nfp_download/alfred.py`) supplies real-time vintage matrices; a new builder (`nfp_ingest/ces_alfred.py`) extracts the three monthly prints and shapes `VINTAGE_STORE_SCHEMA` rows, joining schedule `vintage_date`s from the existing calendar; a capture wrapper reuses `append_to_vintage_store` (idempotent anti-join) + `_detect_corrected_levels`; a script drives dry-run → apply. Nothing in the `cesvinall`/`ces_builder` path or downstream changes.

**Tech Stack:** Python 3.12, httpx, polars, FRED/ALFRED JSON API (`FRED_API_KEY`), pytest, uv workspace.

## Global Constraints

- **Package boundaries**: `nfp-download` imports no other `nfp_*` package; `nfp-ingest` may import `nfp-download` + `nfp-lookups`. No upward imports. (CLAUDE.md "Boundaries".)
- **Paths**: never construct store paths; use `nfp_lookups.paths` (`VINTAGE_STORE_PATH`, `VINTAGE_DATES_PATH`, `storage_options_for`, `is_remote`).
- **Store-write safety (memory `store-write-test-safety`)**: tests MUST pass an explicit local `store_path=tmp_path`. NEVER let a test write to the real store — `conftest.py` auto-loads live prod creds. A store-writing fn against real MinIO once wiped the canonical store.
- **No `./data` writes (plans/15)**: any scratch goes to `tempfile`; the script takes `--store` (a URI or path) — never hardcode `data/`.
- **Docstrings**: 100% public-symbol docstring coverage (`interrogate` docs gate). Every new public function/class needs a numpydoc-style docstring. Run `uv run --group docs interrogate -c pyproject.toml packages` before finishing.
- **Lint**: `ruff check .` clean (line length 100; E,W,F,I,B,C4,UP).
- **Network tests**: mark live-API tests `@pytest.mark.network`; the default suite runs `-m "not network"`.
- **ALFRED facts (verified, do not re-derive)**: deep SA aggregates live under aliases (PAYEMS/USPRIV/MANEMP/…), NOT systematic `CES…01`; `output_type=2` returns wide columns named `{SERIES_ID}_{YYYYMMDD}`; `output_type=4` needs explicit `realtime_start`/`realtime_end`. The full SA + NSA resolution tables are in Task 1.

---

## File Structure

- `packages/nfp-download/src/nfp_download/alfred.py` — NEW. Resolution tables + ALFRED fetch primitives + title-verify. No `nfp_*` imports.
- `packages/nfp-download/src/nfp_download/tests/test_alfred.py` — NEW. Unit (parsing/resolve) + network-marked live tests.
- `packages/nfp-ingest/src/nfp_ingest/ces_alfred.py` — NEW. `extract_prints` (§5 rule) + `build_ces_alfred_window` (store-schema rows).
- `packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py` — NEW. Synthetic-matrix extraction + builder tests (no network).
- `packages/nfp-ingest/src/nfp_ingest/capture.py` — MODIFY. Add `capture_ces_alfred_window()` reusing the existing append/compact/corrected-level path.
- `packages/nfp-ingest/src/nfp_ingest/tests/test_capture_alfred.py` — NEW. tmp_path store: append, idempotence, dry-run, corrected-level.
- `scripts/patch_ces_alfred.py` — NEW. CLI driver: dry-run report → apply.

---

## Task 1: ALFRED resolution tables + `resolve_series_id`

**Files:**
- Create: `packages/nfp-download/src/nfp_download/alfred.py`
- Test: `packages/nfp-download/src/nfp_download/tests/test_alfred.py`

**Interfaces:**
- Produces: `CES_SERIES_SA: dict[tuple[str, str], str]`, `CES_SERIES_NSA: dict[tuple[str, str], str]` (keyed `(industry_type, industry_code)` in **store NAICS codes**), and `resolve_series_id(industry_type: str, industry_code: str, *, sa: bool) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# packages/nfp-download/src/nfp_download/tests/test_alfred.py
import pytest
from nfp_download.alfred import CES_SERIES_SA, CES_SERIES_NSA, resolve_series_id


def test_resolution_tables_cover_all_30_keys():
    expected = {
        ("total", "00"), ("total", "05"), ("domain", "06"), ("domain", "08"),
        *[("supersector", c) for c in
          ("10", "20", "30", "40", "50", "55", "60", "65", "70", "80")],
        *[("sector", c) for c in
          ("21", "22", "31", "32", "42", "44", "48", "52", "53", "54",
           "55", "56", "61", "62", "71", "72")],
    }
    assert set(CES_SERIES_SA) == expected
    assert set(CES_SERIES_NSA) == expected


def test_resolve_sa_aggregates_use_aliases():
    assert resolve_series_id("total", "00", sa=True) == "PAYEMS"
    assert resolve_series_id("supersector", "30", sa=True) == "MANEMP"
    assert resolve_series_id("sector", "42", sa=True) == "USWTRADE"


def test_resolve_nsa_systematic_and_paynsa():
    assert resolve_series_id("total", "00", sa=False) == "PAYNSA"
    assert resolve_series_id("supersector", "30", sa=False) == "CEU3000000001"
    assert resolve_series_id("sector", "42", sa=False) == "CEU4142000001"


def test_resolve_unknown_key_raises():
    with pytest.raises(KeyError):
        resolve_series_id("sector", "99", sa=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-download/src/nfp_download/tests/test_alfred.py -v`
Expected: FAIL with `ModuleNotFoundError: nfp_download.alfred`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-download/src/nfp_download/alfred.py
"""FRED/ALFRED real-time vintage access for national CES series.

Resolution tables map each vintage-store ``(industry_type, industry_code)``
key (NAICS-coded sectors) to its ALFRED series id. SA aggregates have NO
systematic ``CES…01`` archive, so they resolve to FRED friendly aliases
(PAYEMS, USPRIV, MANEMP, …); NSA resolves to systematic ``CEU…01`` except
``00`` (``PAYNSA``). All ids verified live 2026-06-27/28.

This module imports no ``nfp_*`` package (download-layer boundary).
"""
from __future__ import annotations

# (industry_type, industry_code NAICS) -> ALFRED SA series id.
CES_SERIES_SA: dict[tuple[str, str], str] = {
    ("total", "00"): "PAYEMS",
    ("total", "05"): "USPRIV",
    ("domain", "06"): "USGOOD",
    ("domain", "08"): "CES0800000001",
    ("supersector", "10"): "USMINE",
    ("supersector", "20"): "USCONS",
    ("supersector", "30"): "MANEMP",
    ("supersector", "40"): "USTPU",
    ("supersector", "50"): "USINFO",
    ("supersector", "55"): "USFIRE",
    ("supersector", "60"): "USPBS",
    ("supersector", "65"): "USEHS",
    ("supersector", "70"): "USLAH",
    ("supersector", "80"): "USSERV",
    ("sector", "21"): "CES1021000001",
    ("sector", "22"): "CES4422000001",
    ("sector", "31"): "DMANEMP",
    ("sector", "32"): "NDMANEMP",
    ("sector", "42"): "USWTRADE",
    ("sector", "44"): "USTRADE",
    ("sector", "48"): "CES4300000001",
    ("sector", "52"): "CES5552000001",
    ("sector", "53"): "CES5553000001",
    ("sector", "54"): "CES6054000001",
    ("sector", "55"): "CES6055000001",
    ("sector", "56"): "CES6056000001",
    ("sector", "61"): "CES6561000001",
    ("sector", "62"): "CES6562000001",
    ("sector", "71"): "CES7071000001",
    ("sector", "72"): "CES7072000001",
}

# NSA: 29/30 systematic CEU{8digit}01; only 00 -> PAYNSA.
CES_SERIES_NSA: dict[tuple[str, str], str] = {
    ("total", "00"): "PAYNSA",
    ("total", "05"): "CEU0500000001",
    ("domain", "06"): "CEU0600000001",
    ("domain", "08"): "CEU0800000001",
    ("supersector", "10"): "CEU1000000001",
    ("supersector", "20"): "CEU2000000001",
    ("supersector", "30"): "CEU3000000001",
    ("supersector", "40"): "CEU4000000001",
    ("supersector", "50"): "CEU5000000001",
    ("supersector", "55"): "CEU5500000001",
    ("supersector", "60"): "CEU6000000001",
    ("supersector", "65"): "CEU6500000001",
    ("supersector", "70"): "CEU7000000001",
    ("supersector", "80"): "CEU8000000001",
    ("sector", "21"): "CEU1021000001",
    ("sector", "22"): "CEU4422000001",
    ("sector", "31"): "CEU3100000001",
    ("sector", "32"): "CEU3200000001",
    ("sector", "42"): "CEU4142000001",
    ("sector", "44"): "CEU4200000001",
    ("sector", "48"): "CEU4300000001",
    ("sector", "52"): "CEU5552000001",
    ("sector", "53"): "CEU5553000001",
    ("sector", "54"): "CEU6054000001",
    ("sector", "55"): "CEU6055000001",
    ("sector", "56"): "CEU6056000001",
    ("sector", "61"): "CEU6561000001",
    ("sector", "62"): "CEU6562000001",
    ("sector", "71"): "CEU7071000001",
    ("sector", "72"): "CEU7072000001",
}


def resolve_series_id(industry_type: str, industry_code: str, *, sa: bool) -> str:
    """Return the ALFRED series id for a vintage-store industry key.

    Parameters
    ----------
    industry_type : str
        One of ``'total'``, ``'domain'``, ``'supersector'``, ``'sector'``.
    industry_code : str
        Store NAICS-coded industry code (e.g. ``'42'`` for wholesale).
    sa : bool
        ``True`` for seasonally adjusted (CES/alias), ``False`` for NSA (CEU/PAYNSA).

    Returns
    -------
    str
        The resolved ALFRED series id.

    Raises
    ------
    KeyError
        If ``(industry_type, industry_code)`` is not a stored CES key.
    """
    table = CES_SERIES_SA if sa else CES_SERIES_NSA
    return table[(industry_type, industry_code)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/nfp-download/src/nfp_download/tests/test_alfred.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-download/src/nfp_download/alfred.py packages/nfp-download/src/nfp_download/tests/test_alfred.py
git commit -m "feat(download): ALFRED CES series resolution tables (SA+NSA)"
```

---

## Task 2: ALFRED fetch primitives + title-verify

**Files:**
- Modify: `packages/nfp-download/src/nfp_download/alfred.py`
- Test: `packages/nfp-download/src/nfp_download/tests/test_alfred.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `get_vintage_dates(client: httpx.Client, series_id: str, *, api_key: str, start: str | None = None) -> list[str]`
  - `fetch_vintage_matrix(client, series_id, *, api_key, vintage_dates: list[str], observation_start: str, chunk_size: int = 100) -> pl.DataFrame` — long frame `(ref_date: Date, vintage_date: Date, value: Float64)`.
  - `verify_series_concept(client, series_id, *, api_key, sa: bool) -> tuple[str, bool]` — `(title, seasonal_ok)`.
  - `verify_ces_series(client, industry_type: str, industry_code: str, *, sa: bool, api_key: str) -> tuple[str, bool]` — **public** title-verify for a store key (resolves the id + checks SA flag + concept substring); keeps the title map private to this package so `nfp_ingest` imports no download-private name (CLAUDE.md boundary rule).
  - `_title_matches(industry_type, industry_code, title) -> bool` — pure substring check (unit-testable without network).

- [ ] **Step 1: Write the failing test** (parsing is unit-testable without network)

```python
# append to test_alfred.py
import polars as pl
from nfp_download.alfred import _matrix_from_observations, _title_matches


def test_title_matches_accepts_right_concept_rejects_swap():
    # The USSERV->08 mis-map that title-verify must catch: USSERV is "Other Services".
    assert _title_matches("supersector", "80", "All Employees, Other Services")
    assert not _title_matches("domain", "08", "All Employees, Other Services")
    assert _title_matches("domain", "08", "All Employees, Private Service-Providing")


def test_matrix_parses_sid_yyyymmdd_columns():
    # output_type=2 wide shape: date + {SID}_{YYYYMMDD} columns, "." = missing.
    obs = [
        {"date": "2026-02-01", "PAYEMS_20260306": "159000", "PAYEMS_20260403": "159050"},
        {"date": "2026-03-01", "PAYEMS_20260306": ".", "PAYEMS_20260403": "159200"},
    ]
    long = _matrix_from_observations(obs)
    assert long.columns == ["ref_date", "vintage_date", "value"]
    # 3 non-null cells (the "." dropped)
    assert long.height == 3
    feb = long.filter(pl.col("ref_date") == pl.date(2026, 2, 1)).sort("vintage_date")
    assert feb["vintage_date"].to_list() == [pl.date(2026, 3, 6), pl.date(2026, 4, 3)]
    assert feb["value"].to_list() == [159000.0, 159050.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-download/src/nfp_download/tests/test_alfred.py::test_matrix_parses_sid_yyyymmdd_columns -v`
Expected: FAIL with `ImportError: cannot import name '_matrix_from_observations'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to alfred.py
import logging
import time

import httpx
import polars as pl

logger = logging.getLogger(__name__)

ALFRED_BASE = "https://api.stlouisfed.org/fred"

# Expected /series title substring per key — the title-verify guard that caught
# the USSERV(=Other Services)->08 mis-map during design.
_EXPECTED_TITLE_SUBSTR: dict[tuple[str, str], str] = {
    ("total", "00"): "Total Nonfarm",
    ("total", "05"): "Total Private",
    ("domain", "06"): "Goods-Producing",
    ("domain", "08"): "Private Service-Providing",
    ("supersector", "10"): "Mining and Logging",
    ("supersector", "20"): "Construction",
    ("supersector", "30"): "Manufacturing",
    ("supersector", "40"): "Trade, Transportation",
    ("supersector", "50"): "Information",
    ("supersector", "55"): "Financial Activities",
    ("supersector", "60"): "Professional and Business",
    ("supersector", "65"): "Education and Health",
    ("supersector", "70"): "Leisure and Hospitality",
    ("supersector", "80"): "Other Services",
    ("sector", "21"): "Mining",
    ("sector", "22"): "Utilities",
    ("sector", "31"): "Durable Goods",
    ("sector", "32"): "Nondurable Goods",
    ("sector", "42"): "Wholesale Trade",
    ("sector", "44"): "Retail Trade",
    ("sector", "48"): "Transportation and Warehousing",
    ("sector", "52"): "Finance and Insurance",
    ("sector", "53"): "Real Estate",
    ("sector", "54"): "Professional, Scientific",
    ("sector", "55"): "Management of Companies",
    ("sector", "56"): "Administrative",
    ("sector", "61"): "Educational Services",
    ("sector", "62"): "Health Care",
    ("sector", "71"): "Arts, Entertainment",
    ("sector", "72"): "Accommodation and Food",
}


def _request_json(client: httpx.Client, path: str, params: dict, max_retries: int = 6) -> dict:
    """GET a FRED JSON endpoint with exponential backoff on 429/5xx/timeouts."""
    params = {**params, "file_type": "json"}
    last: httpx.Response | None = None
    for attempt in range(max_retries):
        try:
            last = client.get(f"{ALFRED_BASE}/{path}", params=params, timeout=60.0)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            time.sleep(min(2**attempt, 60))
            logger.warning("[%s] retrying ...", type(exc).__name__)
            continue
        if last.status_code == 429 or last.status_code >= 500:
            time.sleep(min(2**attempt, 60))
            continue
        last.raise_for_status()
        return last.json()
    assert last is not None
    last.raise_for_status()
    return last.json()


def get_vintage_dates(
    client: httpx.Client, series_id: str, *, api_key: str, start: str | None = None
) -> list[str]:
    """Return ALFRED vintage date strings for *series_id* (optionally from *start*)."""
    payload = _request_json(
        client, "series/vintagedates", {"series_id": series_id, "api_key": api_key}
    )
    vds = payload.get("vintage_dates", [])
    return [v for v in vds if start is None or v >= start]


def _matrix_from_observations(observations: list[dict]) -> pl.DataFrame:
    """Reshape ``output_type=2`` wide observations to long ``(ref_date, vintage_date, value)``.

    Wide columns are named ``{SERIES_ID}_{YYYYMMDD}``; the ``"."`` sentinel and
    empty strings are dropped (a ref month is absent from vintages before its
    first release).
    """
    if not observations:
        return pl.DataFrame(
            schema={"ref_date": pl.Date, "vintage_date": pl.Date, "value": pl.Float64}
        )
    wide = pl.from_dicts(observations)
    vcols = [c for c in wide.columns if c != "date"]
    return (
        wide.unpivot(index="date", on=vcols, variable_name="vcol", value_name="value")
        .with_columns(pl.col("value").cast(pl.Utf8).str.strip_chars())
        .filter((pl.col("value") != ".") & (pl.col("value") != "")
                & pl.col("value").is_not_null())
        .with_columns(
            pl.col("date").str.to_date().alias("ref_date"),
            pl.col("vcol").str.extract(r"(\d{8})$").str.to_date("%Y%m%d").alias("vintage_date"),
            pl.col("value").cast(pl.Float64),
        )
        .select("ref_date", "vintage_date", "value")
    )


def fetch_vintage_matrix(
    client: httpx.Client,
    series_id: str,
    *,
    api_key: str,
    vintage_dates: list[str],
    observation_start: str,
    chunk_size: int = 100,
) -> pl.DataFrame:
    """Fetch ``output_type=2`` observations and return a long vintage matrix.

    Chunks ``vintage_dates`` (the API caps the request size), concatenates the
    per-chunk long frames.
    """
    frames: list[pl.DataFrame] = []
    for i in range(0, len(vintage_dates), chunk_size):
        chunk = vintage_dates[i : i + chunk_size]
        payload = _request_json(
            client,
            "series/observations",
            {
                "series_id": series_id,
                "api_key": api_key,
                "output_type": 2,
                "vintage_dates": ",".join(chunk),
                "observation_start": observation_start,
            },
        )
        obs = payload.get("observations", [])
        if obs:
            frames.append(_matrix_from_observations(obs))
    if not frames:
        return pl.DataFrame(
            schema={"ref_date": pl.Date, "vintage_date": pl.Date, "value": pl.Float64}
        )
    return pl.concat(frames, how="vertical")


def verify_series_concept(
    client: httpx.Client, series_id: str, *, api_key: str, sa: bool
) -> tuple[str, bool]:
    """Return ``(title, seasonal_ok)`` from the FRED ``/series`` metadata.

    ``seasonal_ok`` is ``True`` iff the series' ``seasonal_adjustment_short``
    matches the requested *sa* flag. The caller cross-checks *title* against the
    expected concept (the title-verify gate).
    """
    payload = _request_json(client, "series", {"series_id": series_id, "api_key": api_key})
    meta = payload["seriess"][0]
    seasonal_ok = meta["seasonal_adjustment_short"] == ("SA" if sa else "NSA")
    return meta["title"], seasonal_ok


def _title_matches(industry_type: str, industry_code: str, title: str) -> bool:
    """True iff *title* contains the expected concept substring for the key."""
    return _EXPECTED_TITLE_SUBSTR[(industry_type, industry_code)] in title


def verify_ces_series(
    client: httpx.Client,
    industry_type: str,
    industry_code: str,
    *,
    sa: bool,
    api_key: str,
) -> tuple[str, bool]:
    """Title-verify the resolved CES series for a store key (public boundary fn).

    Resolves ``(industry_type, industry_code, sa)`` to its ALFRED id, fetches the
    ``/series`` metadata, and returns ``(title, ok)`` where *ok* is ``True`` iff
    BOTH the seasonal-adjustment flag matches and the title contains the expected
    concept substring. Keeps all title-verify logic in the download layer so
    ``nfp_ingest`` never imports a download-private name.

    Parameters
    ----------
    client : httpx.Client
        Shared HTTP client.
    industry_type, industry_code : str
        Vintage-store key (NAICS-coded sectors).
    sa : bool
        Seasonal-adjustment flag.
    api_key : str
        FRED API key.

    Returns
    -------
    tuple[str, bool]
        ``(title, ok)``.
    """
    series_id = resolve_series_id(industry_type, industry_code, sa=sa)
    title, seasonal_ok = verify_series_concept(client, series_id, api_key=api_key, sa=sa)
    return title, seasonal_ok and _title_matches(industry_type, industry_code, title)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/nfp-download/src/nfp_download/tests/test_alfred.py -v`
Expected: PASS (all unit tests).

- [ ] **Step 5: Add a network-marked live smoke test**

```python
# append to test_alfred.py
import os

@pytest.mark.network
def test_live_payems_resolves_and_fetches():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        pytest.skip("FRED_API_KEY not set")
    import httpx
    from nfp_download.alfred import get_vintage_dates, verify_series_concept
    with httpx.Client() as client:
        title, ok = verify_series_concept(client, "PAYEMS", api_key=key, sa=True)
        assert ok and "Total Nonfarm" in title
        vds = get_vintage_dates(client, "PAYEMS", api_key=key, start="2026-01-01")
        assert vds and all(v >= "2026-01-01" for v in vds)
```

- [ ] **Step 6: Run the default (non-network) suite to confirm the marked test is skipped**

Run: `uv run pytest packages/nfp-download/src/nfp_download/tests/test_alfred.py -m "not network" -v`
Expected: PASS; the live test deselected.

- [ ] **Step 7: Commit**

```bash
git add packages/nfp-download/src/nfp_download/alfred.py packages/nfp-download/src/nfp_download/tests/test_alfred.py
git commit -m "feat(download): ALFRED fetch primitives + title-verify"
```

---

## Task 3: `extract_prints` — the §5 rule with real-time guard

**Files:**
- Create: `packages/nfp-ingest/src/nfp_ingest/ces_alfred.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py`

**Interfaces:**
- Produces: `extract_prints(matrix: pl.DataFrame, *, max_gap_days: int = 70) -> pl.DataFrame` — input long `(ref_date, vintage_date, value)`, output `(ref_date, revision: UInt8, vintage_date, value)` for `revision ∈ {0,1,2}`.

- [ ] **Step 1: Write the failing test**

```python
# packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py
import datetime as dt
import polars as pl
from nfp_ingest.ces_alfred import extract_prints


def _row(ref, vint, val):
    return {"ref_date": dt.date(*ref), "vintage_date": dt.date(*vint), "value": float(val)}


def test_first_three_appearances_are_rev_0_1_2():
    m = pl.DataFrame([
        _row((2026, 2, 1), (2026, 3, 6), 159000),   # rev0
        _row((2026, 2, 1), (2026, 4, 3), 159050),   # rev1
        _row((2026, 2, 1), (2026, 5, 8), 159040),   # rev2
        _row((2026, 2, 1), (2026, 6, 5), 159040),   # 4th appearance -> dropped
    ])
    out = extract_prints(m).sort("revision")
    assert out["revision"].to_list() == [0, 1, 2]
    assert out["value"].to_list() == [159000.0, 159050.0, 159040.0]


def test_no_value_dedup_keeps_unchanged_revision():
    # rev1 == rev0 value (rounding); positional rule must still emit it as rev1.
    m = pl.DataFrame([
        _row((2026, 2, 1), (2026, 3, 6), 159000),
        _row((2026, 2, 1), (2026, 4, 3), 159000),  # same value, real revision
        _row((2026, 2, 1), (2026, 5, 8), 159010),
    ])
    out = extract_prints(m).sort("revision")
    assert out["revision"].to_list() == [0, 1, 2]
    assert out.filter(pl.col("revision") == 1)["vintage_date"].item() == dt.date(2026, 4, 3)


def test_real_time_guard_drops_back_history_artifact():
    # ref 2003-01 first appears only in a 2011 vintage (gap ~8y) -> artifact, dropped.
    artifact = pl.DataFrame([
        _row((2003, 1, 1), (2011, 3, 4), 130000),
        _row((2003, 1, 1), (2011, 4, 1), 130100),
        _row((2003, 1, 1), (2011, 5, 6), 130200),
    ])
    assert extract_prints(artifact).height == 0


def test_frontier_month_with_only_rev0_kept():
    m = pl.DataFrame([_row((2026, 5, 1), (2026, 6, 5), 160000)])
    out = extract_prints(m)
    assert out["revision"].to_list() == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py -v`
Expected: FAIL with `ModuleNotFoundError: nfp_ingest.ces_alfred`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/nfp-ingest/src/nfp_ingest/ces_alfred.py
"""Build CES vintage-store rows from ALFRED for the frontier-patch window.

Implements the spec §5 extraction (1st/2nd/3rd appearance as-published, no
value-dedup, real-time guard) and shapes ``VINTAGE_STORE_SCHEMA`` rows whose
``vintage_date`` comes from the existing release calendar (values from ALFRED,
dates from the schedule). Spec: ``specs/alfred_ces_vintages.md``.
"""
from __future__ import annotations

import polars as pl


def extract_prints(matrix: pl.DataFrame, *, max_gap_days: int = 70) -> pl.DataFrame:
    """Extract the three monthly prints ``(0,0)/(1,0)/(2,0)`` from a vintage matrix.

    The 1st/2nd/3rd appearance of each ref month (in ``vintage_date`` order, **no
    value-dedup**) is revision 0/1/2. The **real-time guard** keeps a ref month
    only when its first appearance lands within *max_gap_days* of ``ref_date`` —
    dropping back-history artifacts (a shallow series' first archived vintage
    carries years-old history).

    Parameters
    ----------
    matrix : pl.DataFrame
        Long frame ``(ref_date: Date, vintage_date: Date, value: Float64)``.
    max_gap_days : int
        Maximum ``vintage_date - ref_date`` (days) for a genuine first print.

    Returns
    -------
    pl.DataFrame
        ``(ref_date, revision: UInt8, vintage_date, value)`` for ``revision ∈ {0,1,2}``.
    """
    ranked = matrix.sort("ref_date", "vintage_date").with_columns(
        pl.col("vintage_date").rank("ordinal").over("ref_date").alias("_rk")
    )
    prints = ranked.filter(pl.col("_rk") <= 3).with_columns(
        (pl.col("_rk") - 1).cast(pl.UInt8).alias("revision")
    )
    genuine = (
        prints.filter(pl.col("revision") == 0)
        .filter(
            (pl.col("vintage_date") - pl.col("ref_date")).dt.total_days() <= max_gap_days
        )
        .select("ref_date")
    )
    return (
        prints.join(genuine, on="ref_date", how="inner")
        .select("ref_date", "revision", "vintage_date", "value")
        .sort("ref_date", "revision")
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-ingest/src/nfp_ingest/ces_alfred.py packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py
git commit -m "feat(ingest): ALFRED print extraction with real-time guard"
```

---

## Task 4: `build_ces_alfred_window` — store-schema rows for the gap

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/ces_alfred.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py`

**Interfaces:**
- Consumes: `extract_prints`; `nfp_download.alfred` (`CES_SERIES_SA`/`CES_SERIES_NSA`, `get_vintage_dates`, `fetch_vintage_matrix`, `verify_ces_series` — the **public** title-verify); `nfp_lookups.industry.ownership_for`; `nfp_lookups.schemas.VINTAGE_STORE_SCHEMA`.
- Produces: `build_ces_alfred_window(*, store_frontier: date, through: date, calendar: pl.DataFrame, api_key: str, adjustments: tuple[bool, ...] = (True, False), keys: list[tuple[str, str]] | None = None, fetch: Callable | None = None) -> pl.DataFrame` — `VINTAGE_STORE_SCHEMA` rows. The `fetch` seam (default real ALFRED) lets tests inject a stub.

- [ ] **Step 1: Write the failing test** (inject a stub `fetch`, no network)

```python
# append to test_ces_alfred.py
import datetime as dt
import polars as pl
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_ingest.ces_alfred import build_ces_alfred_window


def _calendar():
    # ces, bmr=0; ref_date day-12; only Feb-2026 rev0/1/2 in-window.
    return pl.DataFrame({
        "publication": ["ces"] * 3,
        "ref_date": [dt.date(2026, 2, 12)] * 3,
        "revision": [0, 1, 2],
        "benchmark_revision": [0, 0, 0],
        "vintage_date": [dt.date(2026, 3, 6), dt.date(2026, 4, 3), dt.date(2026, 5, 8)],
    })


def _stub_fetch(series_id, *, sa, key):
    # Returns (title_ok, matrix) for one series; Feb-2026 three genuine prints.
    matrix = pl.DataFrame({
        "ref_date": [dt.date(2026, 2, 1)] * 3,
        "vintage_date": [dt.date(2026, 3, 6), dt.date(2026, 4, 3), dt.date(2026, 5, 8)],
        "value": [159000.0, 159050.0, 159040.0],
    })
    return matrix


def test_build_window_shapes_store_rows_for_one_key():
    rows = build_ces_alfred_window(
        store_frontier=dt.date(2026, 2, 11),
        through=dt.date(2026, 6, 28),
        calendar=_calendar(),
        api_key="x",
        adjustments=(True,),
        keys=[("total", "00")],
        fetch=_stub_fetch,
    )
    assert list(rows.columns) == list(VINTAGE_STORE_SCHEMA)
    assert rows.height == 3  # rev 0/1/2 for Feb-2026
    r = rows.sort("revision")
    assert r["revision"].to_list() == [0, 1, 2]
    assert r["ref_date"].unique().to_list() == [dt.date(2026, 2, 12)]   # calendar day-12
    assert r["vintage_date"].to_list() == [dt.date(2026, 3, 6), dt.date(2026, 4, 3), dt.date(2026, 5, 8)]
    assert r["employment"].to_list() == [159000.0, 159050.0, 159040.0]
    assert r["ownership"].unique().to_list() == ["total"]
    assert r["source"].unique().to_list() == ["ces"]
    assert r["seasonally_adjusted"].unique().to_list() == [True]


def test_build_window_excludes_cohorts_at_or_before_frontier():
    # frontier AFTER all window vintages -> nothing to add.
    rows = build_ces_alfred_window(
        store_frontier=dt.date(2026, 5, 8),
        through=dt.date(2026, 6, 28),
        calendar=_calendar(),
        api_key="x",
        adjustments=(True,),
        keys=[("total", "00")],
        fetch=_stub_fetch,
    )
    assert rows.height == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py -k build_window -v`
Expected: FAIL with `ImportError: cannot import name 'build_ces_alfred_window'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to ces_alfred.py (top-of-file imports)
from __future__ import annotations

from collections.abc import Callable
from datetime import date

import httpx
import polars as pl
from nfp_download.alfred import (
    CES_SERIES_NSA,
    CES_SERIES_SA,
    fetch_vintage_matrix,
    get_vintage_dates,
    verify_ces_series,
)
from nfp_lookups.industry import ownership_for
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


def _default_fetch(api_key: str) -> Callable[..., pl.DataFrame]:
    """Return a real-ALFRED fetch closure ``(series_id, *, sa, key) -> matrix``.

    Title-verifies through the public ``verify_ces_series`` (raising on a SA-flag
    or concept-substring mismatch — no download-private import), pulls vintage
    dates, and returns the long vintage matrix. Shares one client.
    """
    client = httpx.Client(http2=True)

    def fetch(series_id: str, *, sa: bool, key: tuple[str, str]) -> pl.DataFrame:
        itype, code = key
        title, ok = verify_ces_series(client, itype, code, sa=sa, api_key=api_key)
        if not ok:
            raise ValueError(
                f"title-verify failed for {series_id} ({key}): title={title!r}"
            )
        obs_start = "2024-01-01"
        vds = get_vintage_dates(client, series_id, api_key=api_key, start=obs_start)
        return fetch_vintage_matrix(
            client, series_id, api_key=api_key, vintage_dates=vds, observation_start=obs_start
        )

    return fetch


def build_ces_alfred_window(
    *,
    store_frontier: date,
    through: date,
    calendar: pl.DataFrame,
    api_key: str,
    adjustments: tuple[bool, ...] = (True, False),
    keys: list[tuple[str, str]] | None = None,
    fetch: Callable[..., pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Build ``VINTAGE_STORE_SCHEMA`` rows for the cohorts ALFRED must patch.

    The window is the calendar's CES ``benchmark_revision=0`` cohorts with
    ``store_frontier < vintage_date <= through``. For each resolved series the
    §5 prints are extracted and joined to the window on ``(ref-month, revision)``
    — values from ALFRED, ``vintage_date``/``ref_date`` from the calendar.

    Parameters
    ----------
    store_frontier : datetime.date
        The store's current max CES ``vintage_date``; cohorts ``<=`` it are skipped.
    through : datetime.date
        Upper bound on the window's ``vintage_date`` (typically today).
    calendar : pl.DataFrame
        Release calendar with ``publication, ref_date, revision, benchmark_revision,
        vintage_date`` (e.g. ``vintage_dates.parquet``).
    api_key : str
        FRED API key (used only by the default fetch).
    adjustments : tuple[bool, ...]
        Which seasonal adjustments to build (``True`` SA, ``False`` NSA).
    keys : list[tuple[str, str]] or None
        Restrict to these ``(industry_type, industry_code)`` keys (default: all 30).
    fetch : Callable or None
        ``(series_id, *, sa, key) -> long matrix``; defaults to real ALFRED.

    Returns
    -------
    pl.DataFrame
        Rows conforming to ``VINTAGE_STORE_SCHEMA`` (may be empty).
    """
    fetch = fetch or _default_fetch(api_key)

    window = (
        calendar.filter(
            (pl.col("publication") == "ces")
            & (pl.col("benchmark_revision") == 0)
            & pl.col("revision").is_in([0, 1, 2])
            & (pl.col("vintage_date") > store_frontier)
            & (pl.col("vintage_date") <= through)
        )
        .select(
            "ref_date",
            "revision",
            "vintage_date",
            pl.col("ref_date").dt.truncate("1mo").alias("_m"),
        )
    )
    if window.is_empty():
        return pl.DataFrame(schema=VINTAGE_STORE_SCHEMA)

    out: list[pl.DataFrame] = []
    for sa in adjustments:
        table = CES_SERIES_SA if sa else CES_SERIES_NSA
        for key, series_id in table.items():
            if keys is not None and key not in keys:
                continue
            itype, code = key
            matrix = fetch(series_id, sa=sa, key=key)
            prints = extract_prints(matrix).with_columns(
                pl.col("ref_date").dt.truncate("1mo").alias("_m")
            )
            joined = window.join(
                prints.select("_m", "revision", "value"), on=["_m", "revision"], how="inner"
            )
            if joined.is_empty():
                continue
            out.append(
                joined.with_columns(
                    pl.lit("national").alias("geographic_type"),
                    pl.lit("00").alias("geographic_code"),
                    pl.lit(ownership_for(itype, code)).alias("ownership"),
                    pl.lit(itype).alias("industry_type"),
                    pl.lit(code).alias("industry_code"),
                    pl.col("revision").cast(pl.UInt8),
                    pl.lit(0, dtype=pl.UInt8).alias("benchmark_revision"),
                    pl.col("value").alias("employment"),
                    pl.lit(None, dtype=pl.Utf8).alias("size_class_type"),
                    pl.lit(None, dtype=pl.Utf8).alias("size_class_code"),
                    pl.lit("ces").alias("source"),
                    pl.lit(sa).alias("seasonally_adjusted"),
                )
            )

    if not out:
        return pl.DataFrame(schema=VINTAGE_STORE_SCHEMA)
    return (
        pl.concat(out, how="vertical")
        .select(list(VINTAGE_STORE_SCHEMA))
        .cast(VINTAGE_STORE_SCHEMA)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py -v`
Expected: PASS (all `ces_alfred` tests).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-ingest/src/nfp_ingest/ces_alfred.py packages/nfp-ingest/src/nfp_ingest/tests/test_ces_alfred.py
git commit -m "feat(ingest): build ALFRED CES window rows joined to the calendar"
```

---

## Task 5: `capture_ces_alfred_window` — frontier + dry-run + append

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/capture.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_capture_alfred.py`

**Interfaces:**
- Consumes: `build_ces_alfred_window`; `read_vintage_store`, `append_to_vintage_store`, `compact_partition`; `_detect_corrected_levels`, `CaptureResult`; `VINTAGE_DATES_PATH`, `storage_options_for`.
- Produces: `capture_ces_alfred_window(*, through: date, store_path: Path = VINTAGE_STORE_PATH, api_key: str | None = None, dry_run: bool = False, calendar: pl.DataFrame | None = None, builder: Callable | None = None) -> CaptureResult`. The `calendar`/`builder` seams keep it hermetically testable.

- [ ] **Step 1: Write the failing test** (tmp_path store — NEVER the real store)

```python
# packages/nfp-ingest/src/nfp_ingest/tests/test_capture_alfred.py
import datetime as dt
import polars as pl
import pytest
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_ingest.capture import capture_ces_alfred_window
from nfp_ingest.vintage_store import append_to_vintage_store, read_vintage_store


def _store_row(ref, vint, rev, emp, sa=True):
    return {
        "geographic_type": "national", "geographic_code": "00", "ownership": "total",
        "industry_type": "total", "industry_code": "00", "ref_date": ref,
        "vintage_date": vint, "revision": rev, "benchmark_revision": 0,
        "employment": float(emp), "size_class_type": None, "size_class_code": None,
        "source": "ces", "seasonally_adjusted": sa,
    }


def _seed_store(tmp_path):
    # Existing frontier: Jan-2026 rev0 only (vintage 2026-02-11).
    df = pl.DataFrame(
        [_store_row(dt.date(2026, 1, 12), dt.date(2026, 2, 11), 0, 158000)]
    ).cast(VINTAGE_STORE_SCHEMA)
    append_to_vintage_store(df, tmp_path)
    return tmp_path


def _builder_two_new_rows(**_kw):
    # Jan-2026 rev1/rev2 — the missing cohorts.
    return pl.DataFrame([
        _store_row(dt.date(2026, 1, 12), dt.date(2026, 3, 6), 1, 157800),
        _store_row(dt.date(2026, 1, 12), dt.date(2026, 4, 3), 2, 157820),
    ]).cast(VINTAGE_STORE_SCHEMA)


def test_apply_appends_missing_rows(tmp_path):
    _seed_store(tmp_path)
    res = capture_ces_alfred_window(
        through=dt.date(2026, 6, 28), store_path=tmp_path, api_key="x",
        builder=_builder_two_new_rows, calendar=pl.DataFrame(),
    )
    assert res.appended == 2
    stored = read_vintage_store(tmp_path, source="ces", seasonally_adjusted=True).collect()
    assert sorted(stored["revision"].to_list()) == [0, 1, 2]


def test_idempotent_rerun_appends_zero(tmp_path):
    _seed_store(tmp_path)
    capture_ces_alfred_window(through=dt.date(2026, 6, 28), store_path=tmp_path,
                              api_key="x", builder=_builder_two_new_rows, calendar=pl.DataFrame())
    res2 = capture_ces_alfred_window(through=dt.date(2026, 6, 28), store_path=tmp_path,
                                     api_key="x", builder=_builder_two_new_rows, calendar=pl.DataFrame())
    assert res2.appended == 0


def test_dry_run_writes_nothing(tmp_path):
    _seed_store(tmp_path)
    res = capture_ces_alfred_window(through=dt.date(2026, 6, 28), store_path=tmp_path,
                                    api_key="x", dry_run=True,
                                    builder=_builder_two_new_rows, calendar=pl.DataFrame())
    assert res.appended == 0
    stored = read_vintage_store(tmp_path, source="ces", seasonally_adjusted=True).collect()
    assert stored.height == 1  # only the seed row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture_alfred.py -v`
Expected: FAIL with `ImportError: cannot import name 'capture_ces_alfred_window'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to capture.py imports
from collections.abc import Callable

import polars as pl
from nfp_lookups.paths import VINTAGE_DATES_PATH, storage_options_for

from nfp_ingest.ces_alfred import build_ces_alfred_window


def _ces_store_frontier(store_path: Path) -> date:
    """Max CES ``vintage_date`` currently in the store (or a low sentinel if empty)."""
    part = store_path / "source=ces"
    if not part.exists():
        return date(1900, 1, 1)
    fr = (
        read_vintage_store(store_path, source="ces")
        .select(pl.col("vintage_date").max())
        .collect()
    )
    val = fr.item() if fr.height else None
    return val or date(1900, 1, 1)


def capture_ces_alfred_window(
    *,
    through: date,
    store_path: Path = VINTAGE_STORE_PATH,
    api_key: str | None = None,
    dry_run: bool = False,
    calendar: pl.DataFrame | None = None,
    builder: Callable[..., pl.DataFrame] | None = None,
) -> CaptureResult:
    """Patch the CES store frontier from ALFRED through *through* and append it.

    Computes the store's CES ``vintage_date`` frontier, builds the missing
    ``(0,0)/(1,0)/(2,0)`` cohorts from ALFRED (spec §5/§6), flags corrected
    levels, then (unless *dry_run*) appends + compacts each touched
    ``(ces, sa)`` partition. Idempotent: a re-run appends 0 via the anti-join.

    Parameters
    ----------
    through : datetime.date
        Upper bound on the window's ``vintage_date`` (typically today).
    store_path : Path
        Hive-partitioned store root. Tests MUST pass a local ``tmp_path``.
    api_key : str or None
        FRED API key; falls back to ``FRED_API_KEY``.
    dry_run : bool
        If ``True``, compute + flag corrections but write nothing (``appended=0``).
    calendar : pl.DataFrame or None
        Release calendar; defaults to reading ``VINTAGE_DATES_PATH``.
    builder : Callable or None
        Injection seam for ``build_ces_alfred_window`` (tests pass a stub).

    Returns
    -------
    CaptureResult
        Rows appended, corrected-level records, and rows skipped.
    """
    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key and builder is None:
        raise RuntimeError("FRED_API_KEY is required for ALFRED capture.")

    if calendar is None:
        calendar = pl.read_parquet(
            str(VINTAGE_DATES_PATH), storage_options=storage_options_for(VINTAGE_DATES_PATH)
        )

    frontier = _ces_store_frontier(store_path)
    build = builder or build_ces_alfred_window
    rows = build(store_frontier=frontier, through=through, calendar=calendar, api_key=key)
    if rows.is_empty():
        return CaptureResult(appended=0, corrected=[], skipped=0)

    corrected: list[CorrectedLevel] = []
    sas = rows["seasonally_adjusted"].unique().to_list()
    for sa in sas:
        part = rows.filter(pl.col("seasonally_adjusted") == sa)
        cl = _detect_corrected_levels(part, store_path, source="ces", seasonally_adjusted=sa)
        for c in cl:
            logger.warning(
                "CORRECTED-LEVEL ces sa=%s ref=%s code=%s rev=%s stored=%s incoming=%s",
                sa, c.ref_date, c.industry_code, c.revision,
                c.stored_employment, c.incoming_employment,
            )
        corrected.extend(cl)

    if dry_run:
        return CaptureResult(appended=0, corrected=corrected, skipped=rows.height)

    appended = append_to_vintage_store(rows, store_path)
    for sa in sas:
        compact_partition(store_path, source="ces", seasonally_adjusted=sa)
    return CaptureResult(appended=appended, corrected=corrected, skipped=rows.height - appended)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture_alfred.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the ingest suite to confirm no regression**

Run: `uv run pytest packages/nfp-ingest -m "not network and not slow" --no-cov -q`
Expected: PASS (existing capture/store tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add packages/nfp-ingest/src/nfp_ingest/capture.py packages/nfp-ingest/src/nfp_ingest/tests/test_capture_alfred.py
git commit -m "feat(ingest): capture_ces_alfred_window — frontier patch via existing append path"
```

---

## Task 6: `patch_ces_alfred.py` driver + live dry-run verification

**Files:**
- Create: `scripts/patch_ces_alfred.py`

**Interfaces:**
- Consumes: `capture_ces_alfred_window`; resolves `--store` to a `Path`/`UPath` like `bootstrap_store.py`.

- [ ] **Step 1: Write the script**

```python
# scripts/patch_ces_alfred.py
"""One-shot ALFRED CES frontier patch.

Dry-run (default) reports what would be appended; --apply writes. Point --store
at a SCRATCH prefix first (validate), then canonical. Never writes ./data.

    uv run python scripts/patch_ces_alfred.py --through 2026-06-12            # dry-run, canonical (read-only)
    uv run python scripts/patch_ces_alfred.py --apply --store s3://alt-nfp/store-rebuild
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")  # noqa: E402

from nfp_ingest.capture import capture_ces_alfred_window  # noqa: E402
from nfp_lookups.paths import VINTAGE_STORE_PATH  # noqa: E402


def _store(uri: str | None):
    """Resolve a --store argument to a Path/UPath (default VINTAGE_STORE_PATH)."""
    if uri is None:
        return VINTAGE_STORE_PATH
    if uri.startswith(("s3://", "s3a://")):
        from upath import UPath

        return UPath(uri)
    return Path(uri)


def main() -> None:
    """Parse args and run the ALFRED CES frontier patch (dry-run unless --apply)."""
    p = argparse.ArgumentParser(description="ALFRED CES frontier patch")
    p.add_argument("--through", type=date.fromisoformat, default=date.today(),
                   help="Upper bound on vintage_date (YYYY-MM-DD); default today.")
    p.add_argument("--store", default=None, help="Store URI/path (default VINTAGE_STORE_PATH).")
    p.add_argument("--apply", action="store_true", help="Write (default: dry-run).")
    args = p.parse_args()

    res = capture_ces_alfred_window(
        through=args.through, store_path=_store(args.store), dry_run=not args.apply
    )
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] appended={res.appended} skipped={res.skipped} "
          f"corrected={len(res.corrected)}")
    for c in res.corrected:
        print(f"  CORRECTED ref={c.ref_date} code={c.industry_code} rev={c.revision} "
              f"stored={c.stored_employment} incoming={c.incoming_employment}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint the script**

Run: `uv run ruff check scripts/patch_ces_alfred.py`
Expected: clean.

- [ ] **Step 3: Live dry-run against the canonical store (READ-ONLY — no --apply)**

Run: `uv run python scripts/patch_ces_alfred.py --through 2026-06-12`
Expected: prints `[DRY-RUN] appended=0 skipped=N corrected=M` with `N > 0` (the Feb–May 2026 cohorts across the 30 keys × SA/NSA). Review any `CORRECTED` lines. **Do not pass `--apply` to canonical** — validate on a scratch store first per spec §7.5.

- [ ] **Step 4: Commit**

```bash
git add scripts/patch_ces_alfred.py
git commit -m "feat(scripts): patch_ces_alfred driver (dry-run/apply)"
```

---

## Task 7: Docstring gate + lint + full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Interrogate docstring coverage (100% — docs-deploy gate)**

Run: `uv run --group docs interrogate -c pyproject.toml packages`
Expected: 100%. If a new public symbol lacks a docstring, add a numpydoc-style one and re-run.

- [ ] **Step 2: Lint the whole tree**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 3: Run the fast suite**

Run: `uv run pytest -m "not network and not slow" --no-cov -q`
Expected: PASS.

- [ ] **Step 4: Commit any doc/lint fixes**

```bash
git add -A
git commit -m "chore: docstring + lint pass for ALFRED CES patch"
```

---

## Self-Review (completed during authoring)

- **Spec coverage**: §3 resolution → Task 1; §4 primitives → Task 2; §5 extraction+guard → Task 3; §5/§6 window build + calendar dates + NAICS keying → Task 4; §7 frontier/dry-run/append/corrected-level → Task 5; §7 driver + live dry-run → Task 6; §8 docstring/lint → Task 7. NSA (§3) covered by the `CES_SERIES_NSA` table in Task 1.
- **Type consistency**: `extract_prints` returns `(ref_date, revision: UInt8, vintage_date, value)`, consumed by `build_ces_alfred_window` (joins on `_m`, `revision`); `build_ces_alfred_window` returns `VINTAGE_STORE_SCHEMA`, consumed by `capture_ces_alfred_window` → `append_to_vintage_store` (validates the schema). `CaptureResult`/`CorrectedLevel` reused unchanged from `capture.py`.
- **Placeholder scan**: none — every step carries runnable code/commands.
