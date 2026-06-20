# CLI Production Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape the `alt-nfp` CLI into a production month-T workflow — capture each month's BLS current print via the API and append it to the vintage store, with feed-driven cron automation — and move the one-time historical rebuild to a script.

**Architecture:** Four CLI commands (`update` / `status` / `snapshot` / `watch`) over the existing `append_to_vintage_store` / `compact_partition` store primitives; the bulk bootstrap moves to `scripts/bootstrap_store.py`. New capture adapter in `nfp-ingest`, status reader + CLI in `nfp-vintages`, RSS feed reader in `nfp-download`. **Firewall:** no changes to `nfp-model`, `transform_to_panel`, `build_model_data`, or the A1/A2/A3 golden paths.

**Tech Stack:** Python 3.12, Typer, Polars 1.41.2, httpx / curl_cffi (BLS/FRED), uv workspace. Tests: pytest (TDD, `-m "not network"`); ruff/black (line length 100).

**Spec:** [specs/cli_production_workflow.md](../cli_production_workflow.md)

**Execution order:** Phases run **1 → 9 in order** (dependency-ordered). Phase 1 relocates the QCEW acquire helpers so Phase 4's capture can import them legally; Phases 2 (ukey) and 3 (calendar) are the correctness/dependency floor; Phases 4–6 build the capture adapters + `update`; Phases 7–8 add `status` / `watch`; Phase 9 retires the legacy lineage and adds the bootstrap script.

## Cross-phase adjustments (apply during execution)

A consistency review across phase boundaries surfaced three reconciliations the executor must honor:

1. **polars deprecation:** use `nulls_equal=True` (not the deprecated `join_nulls=True`) in every null-aware join — the pinned polars 1.41.2 warns on the old spelling. (Already applied throughout this plan.)
2. **`process` retirement:** the Phase 3 calendar test must call `advance_release_calendar()` **directly** — do *not* couple it to the `process` command, which Phase 9 deletes. If any `process`-coupled test is written, Phase 9 Task 9.2 must delete it so the green-suite gate holds.
3. **Import placement (ruff E402):** in the Phase 6/7/8 tasks that *extend* an existing test file, add any new imports at the **top** of that file, not after class definitions.
4. **QCEW capture stub:** Phase 4 creates `nfp_ingest/capture.py` with `capture_ces_print` **and a `capture_qcew_quarter` stub** (`def capture_qcew_quarter(as_of, *, store_path=VINTAGE_STORE_PATH): raise NotImplementedError`) so Phase 5's `update` body and its tests can reference/monkeypatch the symbol before Phase 6 replaces the stub with the real implementation. (Build order is 5 → 6, but `update` orchestrates both sources.)

---


## Phase 1 — Relocate QCEW acquire helpers to `nfp_ingest/qcew_acquire.py`

**Why this phase first (spec §14 step 1):** `capture.py` (Phase 4, in `nfp-ingest`) must
call the QCEW acquire helpers, but they currently live in `nfp-vintages/rebuild_store.py`,
which sits **above** `nfp-ingest` in the dependency chain. An upward import of private names
is illegal. This phase moves the acquire layer down into a new **public** `nfp_ingest/qcew_acquire.py`
(legal: it imports only `nfp_download.client` + `nfp_lookups` + `nfp_ingest.qcew_crosswalk`),
renaming the two entry points public, and rewires the two existing callers
(`rebuild_store.py`, `__main__.py:build-rebuild`) to consume them from there — keeping the
whole suite green. This is a **refactor-move**: TDD = write a test asserting the new public
symbols exist + behave, run-fail (ImportError), move code, run-pass.

**Move boundary.** The two public entry points depend on a private cluster that moves *with*
them: `_fetch_qcew_csv`, `_prep_area_raw`, `_size_raw_to_native`, and the constants
`_REBUILD_START_YEAR`, `_QCEW_LEVELS_REQUIRED`, `_SIZE_AGGLVL_KEEP`, `_SIZE_AGGLVL_OFFSET`.
`_SERIES_IDENTITY_KEY` stays in `rebuild_store.py` (it is a `compose_rebuild_panel` helper at
`rebuild_store.py:510,534,540`, **not** an acquire helper). The existing tests
`test_rebuild_acquire.py` (imports `_prep_area_raw`/`_size_raw_to_native`/`_fetch_qcew_csv`/
`_QCEW_LEVELS_REQUIRED`) and `test_rebuild_gates.py:1921` (imports `_acquire_qcew_levels`)
keep importing from `nfp_vintages.rebuild_store`, so `rebuild_store.py` **re-exports** the moved
private names + a private alias for each renamed public fn (Task 1.2) — no edits to those test
files.

---

### Task 1.1: Create the new public module `nfp_ingest/qcew_acquire.py`

**Files:**
- Create: `packages/nfp-ingest/src/nfp_ingest/qcew_acquire.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_qcew_acquire.py`

The new module relocates the acquire layer **verbatim** (bodies copied exactly from
`rebuild_store.py:92-433`), with the two entry points renamed public:
`_acquire_qcew_levels` → `acquire_qcew_levels`, `_acquire_qcew_size_native` →
`acquire_qcew_size_native`. The only body edits are the renamed `RuntimeError` strings (they
self-reference the old names). All imports are legal for `nfp-ingest`: `httpx`/`polars`
(direct), `nfp_lookups.industry.QCEW_AREA_NATIONAL`, deferred `nfp_download.client`
(create_client/get_with_retry), deferred `nfp_ingest.qcew_crosswalk.build_qcew_panel`.

- [ ] **Step 1: Write the failing test** — assert the new public symbols exist, the
  import-legality holds (the module imports no `nfp_vintages`), and the pure transforms behave
  on a tiny synthetic CSV-shaped frame. Network tests are marked `@pytest.mark.network`.

```python
"""Tests for the relocated QCEW acquire layer (nfp_ingest.qcew_acquire).

Phase 1 of the CLI production workflow (specs/cli_production_workflow.md §5.2, §14
step 1) relocates the two QCEW acquire entry points from nfp_vintages/rebuild_store.py
into this PUBLIC nfp_ingest module so capture.py (also in nfp-ingest) can call them
without an illegal upward import of private names.

Unit tests (no network) cover the pure-transform helpers and import-legality.
A @pytest.mark.network test fetches one real BLS area slice.

CRITICAL SAFETY: no test here writes to any store; no test calls the full network
fetch loops acquire_qcew_levels()/acquire_qcew_size_native() in CI.
"""

from __future__ import annotations

import polars as pl
import pytest


def _area_raw(rows: list[dict]) -> pl.DataFrame:
    """Build an all-string raw area frame (as _fetch_qcew_csv returns)."""
    cols = list(rows[0].keys())
    schema = dict.fromkeys(cols, pl.Utf8)
    return pl.DataFrame({c: [str(r[c]) for r in rows] for c in cols}, schema=schema)


def _area_row(
    *,
    own_code: str = "5",
    industry_code: str = "1013",
    agglvl_code: str = "13",
    month1: int = 10_000,
    month2: int = 10_100,
    month3: int = 10_200,
) -> dict:
    return {
        "area_fips": "US000",
        "own_code": own_code,
        "industry_code": industry_code,
        "agglvl_code": agglvl_code,
        "year": "2024",
        "qtr": "1",
        "month1_emplvl": str(month1),
        "month2_emplvl": str(month2),
        "month3_emplvl": str(month3),
        "disclosure_code": "",
        "total_qtrly_wages": "9999999",
    }


class TestPublicSymbolsExist:
    """The two acquire entry points must exist PUBLIC on nfp_ingest.qcew_acquire."""

    def test_acquire_qcew_levels_is_public_callable(self):
        from nfp_ingest.qcew_acquire import acquire_qcew_levels

        assert callable(acquire_qcew_levels)

    def test_acquire_qcew_size_native_is_public_callable(self):
        from nfp_ingest.qcew_acquire import acquire_qcew_size_native

        assert callable(acquire_qcew_size_native)


class TestImportLegality:
    """The module must NOT import nfp_vintages (it sits above nfp-ingest)."""

    def test_module_has_no_nfp_vintages_import(self):
        import inspect

        import nfp_ingest.qcew_acquire as mod

        src = inspect.getsource(mod)
        assert "nfp_vintages" not in src, (
            "qcew_acquire.py must not import nfp_vintages (illegal upward import)"
        )


class TestPrepAreaRaw:
    """_prep_area_raw moved verbatim — pure transform, no network."""

    def _prep(self, rows: list[dict]) -> pl.DataFrame:
        from nfp_ingest.qcew_acquire import _prep_area_raw

        return _prep_area_raw(_area_raw(rows))

    def test_filter_keeps_private_and_total(self):
        rows = [
            _area_row(own_code="5"),
            _area_row(own_code="0"),
            _area_row(own_code="1"),
            _area_row(own_code="2"),
            _area_row(own_code="3"),
        ]
        result = self._prep(rows)
        assert result.height == 2
        assert set(result["own_code"].to_list()) == {"5", "0"}

    def test_required_columns_exact(self):
        from nfp_ingest.qcew_acquire import _QCEW_LEVELS_REQUIRED

        result = self._prep([_area_row()])
        assert set(result.columns) == set(_QCEW_LEVELS_REQUIRED)

    def test_emplvl_cast_to_int64_and_revision_zero(self):
        result = self._prep([_area_row(month1=12345, month2=23456, month3=34567)])
        assert result["month1_emplvl"].dtype == pl.Int64
        assert result["revision"][0] == 0

    def test_hyphenated_industry_code_preserved(self):
        result = self._prep([_area_row(industry_code="44-45", agglvl_code="14")])
        assert result["industry_code"][0] == "44-45"


@pytest.mark.network
class TestAcquireLevelsNetwork:
    """Fetch ONE real QCEW area slice through the relocated helper (maintainer-run)."""

    def test_single_slice_non_empty(self):
        from nfp_download.client import create_client

        from nfp_ingest.qcew_acquire import _fetch_qcew_csv

        url = "https://data.bls.gov/cew/data/api/2024/1/area/US000.csv"
        with create_client() as session:
            raw = _fetch_qcew_csv(session, url)
        assert raw is not None
        assert raw.height > 0
```

- [ ] **Step 2: Run the test, verify it fails** — `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_qcew_acquire.py -q --no-cov -m "not network"`.
  Expected: **FAIL** — `ModuleNotFoundError: No module named 'nfp_ingest.qcew_acquire'`
  (the module does not exist yet, so every test errors at import).

- [ ] **Step 3: Implement** — create `qcew_acquire.py` with the relocated bodies (copied
  exactly from `rebuild_store.py:92-433`; entry points renamed public; error strings updated).

```python
"""QCEW acquire layer — area + size BLS API slice fetchers (relocated from nfp-vintages).

Relocated PUBLIC from ``nfp_vintages/rebuild_store.py`` (CLI production workflow
spec §5.2, §14 step 1) so ``nfp_ingest.capture`` can call the QCEW acquire helpers
without an illegal upward import of private names: ``nfp-vintages`` sits above
``nfp-ingest`` in the dependency chain, but these helpers import only
httpx/polars/``nfp_lookups``/``nfp_download.client``/``nfp_ingest.qcew_crosswalk`` —
all legal for ``nfp-ingest``.

The two entry points fetch public BLS API slices over plain httpx (``data.bls.gov``
needs no Akamai impersonation; only www.bls.gov is fingerprinted) and transform them
into frames ready for :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel`
(levels) and :func:`~nfp_ingest.size_class.build_size_class_panel` (size).
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any

import httpx
import polars as pl
from nfp_lookups.industry import QCEW_AREA_NATIONAL

logger = logging.getLogger(__name__)

# First year of the rebuild scope.  The rebuild covers 2017-present so that
# QCEW revisions pre-2017 (not needed by the model) aren't fetched.
_REBUILD_START_YEAR: int = 2017

# Columns required by build_qcew_panel (nfp_ingest.qcew_crosswalk._REQUIRED_COLUMNS).
# We select these from each raw area-slice CSV before concat.
_QCEW_LEVELS_REQUIRED = (
    "area_fips",
    "own_code",
    "industry_code",
    "agglvl_code",
    "year",
    "qtr",
    "month1_emplvl",
    "month2_emplvl",
    "month3_emplvl",
    "revision",
)

# agglvl codes kept from the size-endpoint files. The duplicate family (61–64)
# carries the same industry_codes as 21–24 and would double-count, so keep only
# the 21–28 by-industry-detail tree. We keep the *full* 21–28 (not just the
# 23–26 that build_qcew_panel pulls): after the −10 remap, 21/22/27/28 → 11/12/
# 17/18, which build_qcew_panel simply ignores — harmless, and robust if BLS
# ever shifts which detail level a CES pull reads.
_SIZE_AGGLVL_KEEP: frozenset[str] = frozenset(str(a) for a in range(21, 29))

# The +10 shift that maps size agglvl to the equivalent area agglvl understood
# by build_qcew_panel (23→13 supersectors, 24→14 sectors, 25→15 3-digit,
# 26→16 4-digit). Applied as a vectorised subtraction inside _size_raw_to_native.
_SIZE_AGGLVL_OFFSET: int = 10


# ---------------------------------------------------------------------------
# Thin network helpers
# ---------------------------------------------------------------------------


def _fetch_qcew_csv(session: Any, url: str) -> pl.DataFrame | None:
    """GET *url* with retry; return a raw all-string DataFrame or ``None`` on 404.

    Uses :func:`~nfp_download.client.get_with_retry` for exponential back-off
    and rate-limit handling.  Returns ``None`` when the slice doesn't exist yet
    (404) so callers can skip it cleanly.  Re-raises all other HTTP errors.

    Parameters
    ----------
    session :
        An open :class:`httpx.Client` from
        :func:`~nfp_download.client.create_client`. ``data.bls.gov`` is a plain
        host (no Akamai TLS fingerprinting — that's www.bls.gov only), so it
        stays on httpx per the nfp-download transport convention.
    url : str
        Absolute BLS API URL.

    Returns
    -------
    pl.DataFrame or None
        All columns as ``Utf8`` (``infer_schema_length=0``); caller is
        responsible for casting numeric fields.  ``None`` on HTTP 404.
    """
    from nfp_download.client import get_with_retry

    try:
        r = get_with_retry(session, url)
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.debug("404 — skipping %s", url)
            return None
        raise

    return pl.read_csv(io.BytesIO(r.content), infer_schema_length=0)


# ---------------------------------------------------------------------------
# Levels acquire — area endpoint /api/{y}/{q}/area/US000.csv
# ---------------------------------------------------------------------------


def _prep_area_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare a raw area-endpoint CSV slice for :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel`.

    Filters to private establishments (``own_code == '5'``) and the total-covered
    row (``own_code == '0'``); drops government rows (``own_code`` in ``{'1','2','3'}``).
    Selects the columns required by ``build_qcew_panel``, casts the three
    ``month{1,2,3}_emplvl`` columns to ``Int64`` (the CSV is read as all-string
    to preserve codes like ``'44-45'`` and ``'US000'``), and attaches
    ``revision = 0`` (QCEW area-endpoint rows are revision-0 only for the
    rebuild scope).

    Parameters
    ----------
    df : pl.DataFrame
        Raw all-string frame from :func:`_fetch_qcew_csv`.

    Returns
    -------
    pl.DataFrame
        Frame with exactly :data:`_QCEW_LEVELS_REQUIRED` columns, private rows
        and the total-covered (``own_code='0'``) row only, ``month*_emplvl``
        as ``Int64``, ``revision`` as ``Int64``.
    """
    # No disclosure filter here (unlike the size path): the area endpoint's
    # all-sizes national aggregates at agglvl 10/13–16 are large cells BLS does
    # not suppress. Suppression only bites the finer size×industry cells (see
    # _size_raw_to_native step 2).
    return (
        df.filter(pl.col("own_code").is_in(["5", "0"]))
        .select(
            [c for c in _QCEW_LEVELS_REQUIRED if c != "revision"]
        )
        .with_columns(
            pl.col("month1_emplvl").cast(pl.Int64, strict=False),
            pl.col("month2_emplvl").cast(pl.Int64, strict=False),
            pl.col("month3_emplvl").cast(pl.Int64, strict=False),
            revision=pl.lit(0, pl.Int64),
        )
        .select(list(_QCEW_LEVELS_REQUIRED))
    )


def acquire_qcew_levels(
    start_year: int = _REBUILD_START_YEAR, end_year: int | None = None
) -> pl.DataFrame:
    """Acquire raw QCEW area-endpoint rows for crosswalk → :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel`.

    Fetches ``/api/{year}/{qtr}/area/US000.csv`` for every ``(year, quarter)``
    in ``[start_year, end_year]`` (all 4 quarters) over plain httpx
    (``data.bls.gov`` is not Akamai-fingerprinted, unlike www.bls.gov).
    Per-slice 404s are skipped (the current year may lack later quarters).

    All rows are tagged ``revision=0`` (decision A: per-industry QCEW data on
    the area endpoint is revision-0 for the rebuild scope).

    Parameters
    ----------
    start_year : int
        First reference year to fetch. Defaults to :data:`_REBUILD_START_YEAR`
        (2017). Narrow this for a small-window smoke build.
    end_year : int or None
        Last reference year (inclusive). Defaults to the current calendar year.

    Returns
    -------
    pl.DataFrame
        Concatenated raw rows ready for :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel`.
        Carries exactly :data:`_QCEW_LEVELS_REQUIRED` columns:
        ``area_fips``, ``own_code``, ``industry_code``, ``agglvl_code``,
        ``year``, ``qtr``, ``month{1,2,3}_emplvl`` (``Int64``), ``revision=0``.
    """
    from nfp_download.client import create_client

    last_year = end_year if end_year is not None else date.today().year
    slices: list[pl.DataFrame] = []

    with create_client() as session:
        for year in range(start_year, last_year + 1):
            for qtr in range(1, 5):
                url = f"https://data.bls.gov/cew/data/api/{year}/{qtr}/area/US000.csv"
                logger.info("Fetching QCEW levels %d Q%d ...", year, qtr)
                raw = _fetch_qcew_csv(session, url)
                if raw is None:
                    logger.info("  skipped (404)")
                    continue
                prepped = _prep_area_raw(raw)
                logger.info("  %d rows after private+total filter", prepped.height)
                slices.append(prepped)

    if not slices:
        raise RuntimeError(
            "acquire_qcew_levels: no QCEW area slices fetched — "
            "check network access and BLS API availability"
        )

    return pl.concat(slices)


# ---------------------------------------------------------------------------
# Size acquire — size endpoint /api/{y}/1/size/{size_code}.csv
# ---------------------------------------------------------------------------


def _size_raw_to_native(raw_size: pl.DataFrame) -> pl.DataFrame:
    """Transform a concatenated raw size-endpoint frame into the ``native`` format.

    Takes the combined frame for all size codes fetched across all years and
    applies the full transform pipeline:

    1. Filter ``own_code == '5'`` (private) and ``agglvl_code ∈ {'21'..'28'}``
       (drops the duplicate 61–64 family that would double-count).
    2. Drop rows where ``disclosure_code == 'N'`` (withheld cells; the
       suppressed employment value is zero, not a real zero — never sum it).
       Log the disclosure distribution.
    3. Normalize ``area_fips`` to :data:`~nfp_lookups.industry.QCEW_AREA_NATIONAL`
       (``'US000'``).  Size files are national-only but may carry a different
       area code; :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel` filters
       on ``area_fips == 'US000'``, so this normalisation is unconditional.
    4. Remap ``agglvl_code`` by subtracting :data:`_SIZE_AGGLVL_OFFSET` (so
       23→13, 24→14, 25→15, 26→16), making the size tree compatible with
       ``build_qcew_panel``'s pull tables.
    5. Cast ``month{1,2,3}_emplvl`` to ``Int64``; add ``revision = 0``.
    6. Run :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel` **once per
       ``size_code``** (avoids collapsing the size breakdown, since
       ``build_qcew_panel``'s internal grouping has no size axis).
    7. Attach ``size_code`` as a string column to each per-size output; concat.
    8. Assert all ``ref_date``s are in Q1 (months 1–3); fail loud on any
       ``size_code`` that produces zero rows.

    Parameters
    ----------
    raw_size : pl.DataFrame
        Concatenated raw size-endpoint CSVs (all-string schema) carrying at
        minimum: ``area_fips``, ``own_code``, ``industry_code``,
        ``agglvl_code``, ``disclosure_code``, ``year``, ``qtr``,
        ``month{1,2,3}_emplvl``, ``size_code``.

    Returns
    -------
    pl.DataFrame
        ``native`` frame conformant to
        :data:`~nfp_ingest.size_class._REQUIRED_COLUMNS`:
        ``(geographic_type, geographic_code, ownership, industry_type,
        industry_code, ref_date, vintage_date, revision, size_code,
        employment)`` — ready for
        :func:`~nfp_ingest.size_class.build_size_class_panel`.
    """
    from nfp_ingest.qcew_crosswalk import build_qcew_panel

    # --- Step 1: filter ownership + agglvl ---
    df = raw_size.filter(
        (pl.col("own_code") == "5") & pl.col("agglvl_code").is_in(list(_SIZE_AGGLVL_KEEP))
    )

    # --- Step 2: disclosure logging + drop suppressed rows ---
    total_rows = df.height
    disc_col = "disclosure_code" if "disclosure_code" in df.columns else None
    if disc_col is not None:
        disc_counts = (
            df.group_by(disc_col)
            .agg(pl.len().alias("n"))
            .sort(disc_col)
        )
        logger.info(
            "Disclosure distribution (private, agglvl 21-28): %s",
            {r[disc_col]: r["n"] for r in disc_counts.to_dicts()},
        )
        suppressed = df.filter(pl.col(disc_col) == "N").height
        if suppressed:
            logger.info(
                "Dropping %d suppressed (disclosure_code='N') rows out of %d",
                suppressed,
                total_rows,
            )
        df = df.filter(pl.col(disc_col).is_null() | (pl.col(disc_col) != "N"))
    else:
        logger.warning("disclosure_code column not found in size frame — skipping suppression filter")

    logger.info(
        "Size frame after ownership/agglvl/disclosure filter: %d rows", df.height
    )

    # --- Step 3: normalize area_fips to QCEW_AREA_NATIONAL ---
    df = df.with_columns(area_fips=pl.lit(QCEW_AREA_NATIONAL, pl.Utf8))

    # --- Steps 4 & 5: remap agglvl (-10), cast emplvl, add revision ---
    df = df.with_columns(
        agglvl_code=(pl.col("agglvl_code").cast(pl.Int64) - _SIZE_AGGLVL_OFFSET).cast(pl.Utf8),
        month1_emplvl=pl.col("month1_emplvl").cast(pl.Int64, strict=False),
        month2_emplvl=pl.col("month2_emplvl").cast(pl.Int64, strict=False),
        month3_emplvl=pl.col("month3_emplvl").cast(pl.Int64, strict=False),
        revision=pl.lit(0, pl.Int64),
    )

    # --- Step 6 & 7: per-size_code build_qcew_panel + re-tag size_code ---
    size_codes = sorted(df["size_code"].unique().to_list())
    native_parts: list[pl.DataFrame] = []

    for sc in size_codes:
        subset = df.filter(pl.col("size_code") == sc)
        panel = build_qcew_panel(subset)
        if panel.height == 0:
            raise RuntimeError(
                f"_size_raw_to_native: build_qcew_panel returned 0 rows for "
                f"size_code={sc!r} — the agglvl remap or industry filter is broken"
            )
        # _SERIES_KEYS from size_class.py + employment, plus size_code
        native_parts.append(
            panel.select(
                "geographic_type",
                "geographic_code",
                "ownership",
                "industry_type",
                "industry_code",
                "ref_date",
                "vintage_date",
                "revision",
                "employment",
            ).with_columns(size_code=pl.lit(sc, pl.Utf8))
        )

    if not native_parts:
        raise RuntimeError(
            "_size_raw_to_native: no size_codes found in frame after filtering"
        )

    native = pl.concat(native_parts)

    # --- Step 8: assert all ref_dates are Q1 ---
    non_q1 = native.filter(~pl.col("ref_date").dt.month().is_in([1, 2, 3]))
    if non_q1.height:
        sample = non_q1.head(3)["ref_date"].to_list()
        raise RuntimeError(
            f"_size_raw_to_native: {non_q1.height} non-Q1 ref_dates in native output "
            f"(e.g. {sample}); size-class data must be Q1-only"
        )

    return native


def acquire_qcew_size_native(
    start_year: int = _REBUILD_START_YEAR, end_year: int | None = None
) -> pl.DataFrame:
    """Acquire raw QCEW Q1 size-endpoint rows for :func:`~nfp_ingest.size_class.build_size_class_panel`.

    Fetches ``/api/{year}/1/size/{size_code}.csv`` for every ``(year, size_code)``
    in ``[start_year, end_year]`` (size codes 1–9) over plain httpx
    (``data.bls.gov`` needs no impersonation).  The Q1-only endpoint is implicit in the
    URL path (``/1/`` is quarter 1).  Per-slice 404s are skipped.

    The raw frames are concatenated and passed to :func:`_size_raw_to_native`,
    which applies the agglvl −10 remap, disclosure filtering, area_fips
    normalisation, and the per-size_code :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel`
    crosswalk.

    Parameters
    ----------
    start_year : int
        First reference year to fetch. Defaults to :data:`_REBUILD_START_YEAR`
        (2017). Narrow this for a small-window smoke build.
    end_year : int or None
        Last reference year (inclusive). Defaults to the current calendar year.

    Returns
    -------
    pl.DataFrame
        ``native`` frame conformant to
        :data:`~nfp_ingest.size_class._REQUIRED_COLUMNS` — ready for
        :func:`~nfp_ingest.size_class.build_size_class_panel`.
    """
    from nfp_download.client import create_client

    last_year = end_year if end_year is not None else date.today().year
    slices: list[pl.DataFrame] = []

    with create_client() as session:
        for year in range(start_year, last_year + 1):
            for size_code in range(1, 10):
                url = (
                    f"https://data.bls.gov/cew/data/api/{year}/1/size/{size_code}.csv"
                )
                logger.info("Fetching QCEW size %d size_code=%d ...", year, size_code)
                raw = _fetch_qcew_csv(session, url)
                if raw is None:
                    logger.info("  skipped (404)")
                    continue
                # Tag the size_code so _size_raw_to_native can partition by it.
                # The CSV itself carries a size_code column; we overwrite it
                # with the string form of the URL parameter so it's consistent.
                raw = raw.with_columns(size_code=pl.lit(str(size_code), pl.Utf8))
                logger.info("  %d rows", raw.height)
                slices.append(raw)

    if not slices:
        raise RuntimeError(
            "acquire_qcew_size_native: no QCEW size slices fetched — "
            "check network access and BLS API availability"
        )

    raw_size = pl.concat(slices, how="diagonal_relaxed")
    return _size_raw_to_native(raw_size)
```

- [ ] **Step 4: Run, verify pass** — `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_qcew_acquire.py -q --no-cov -m "not network"`.
  Expected: **PASS** (all non-network tests green). Also run the nfp-ingest suite and lint:
  `uv run pytest packages/nfp-ingest -q --no-cov -m "not network and not slow"` and
  `uv run ruff check packages/nfp-ingest/src/nfp_ingest/qcew_acquire.py packages/nfp-ingest/src/nfp_ingest/tests/test_qcew_acquire.py`.

- [ ] **Step 5: Commit** — stage only the two new files.

```bash
git add packages/nfp-ingest/src/nfp_ingest/qcew_acquire.py \
        packages/nfp-ingest/src/nfp_ingest/tests/test_qcew_acquire.py
git commit -m "feat(ingest): relocate QCEW acquire helpers to public nfp_ingest.qcew_acquire

Move acquire_qcew_levels/acquire_qcew_size_native (+ _fetch_qcew_csv,
_prep_area_raw, _size_raw_to_native, constants) down from
nfp-vintages/rebuild_store.py into a new PUBLIC nfp-ingest module so
capture.py can call them without an illegal upward import (spec §5.2/§14).
Callers rewired in the next commit.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.2: Rewire callers — `rebuild_store.py` (import + re-export) and `__main__.py:build-rebuild`

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/rebuild_store.py`
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_qcew_acquire.py` *(new — back-compat re-export guard; lives in nfp-vintages because it asserts the nfp-vintages re-export surface)*

`rebuild_store.py` deletes the seven moved definitions and **imports them back** from
`nfp_ingest.qcew_acquire`, aliasing the renamed entry points to their old private names so the
two existing test files keep importing from `nfp_vintages.rebuild_store` unchanged:
- `test_rebuild_acquire.py` imports `_prep_area_raw`, `_size_raw_to_native`, `_fetch_qcew_csv`,
  `_QCEW_LEVELS_REQUIRED` (all `@pytest.mark.network`-free unit tests + network probes).
- `test_rebuild_gates.py:1921` imports `_acquire_qcew_levels` (a `@pytest.mark.network` test).

`_SERIES_IDENTITY_KEY` and `compose_rebuild_panel`/`write_rebuild_store` are untouched.
`__main__.py:build-rebuild` (`__main__.py:256-261`) switches its import source from
`nfp_vintages.rebuild_store` (private names) to `nfp_ingest.qcew_acquire` (public names) and
updates the two call sites at `__main__.py:271,278`.

- [ ] **Step 1: Write the failing test** — assert the re-export surface and that
  `rebuild_store` no longer *defines* the acquire functions itself (they resolve to the
  `nfp_ingest.qcew_acquire` module), so the move is real, not a copy. Also assert the public
  aliasing is correct.

```python
"""Back-compat guard for the QCEW acquire relocation (spec §5.2, §14 step 1).

After Phase 1, the acquire helpers LIVE in nfp_ingest.qcew_acquire. rebuild_store.py
must re-export them (under their original private names) so the existing
test_rebuild_acquire.py / test_rebuild_gates.py imports keep resolving, AND the
re-exported objects must be the SAME objects as the nfp_ingest definitions
(proving a real move, not a stale copy).
"""

from __future__ import annotations


def test_private_aliases_resolve_to_ingest_definitions():
    from nfp_ingest.qcew_acquire import (
        acquire_qcew_levels,
        acquire_qcew_size_native,
    )

    from nfp_vintages.rebuild_store import (
        _acquire_qcew_levels,
        _acquire_qcew_size_native,
    )

    # Same object — rebuild_store re-exports, does not redefine.
    assert _acquire_qcew_levels is acquire_qcew_levels
    assert _acquire_qcew_size_native is acquire_qcew_size_native


def test_moved_private_helpers_reexported():
    from nfp_ingest import qcew_acquire

    from nfp_vintages.rebuild_store import (
        _fetch_qcew_csv,
        _prep_area_raw,
        _QCEW_LEVELS_REQUIRED,
        _size_raw_to_native,
    )

    assert _fetch_qcew_csv is qcew_acquire._fetch_qcew_csv
    assert _prep_area_raw is qcew_acquire._prep_area_raw
    assert _size_raw_to_native is qcew_acquire._size_raw_to_native
    assert _QCEW_LEVELS_REQUIRED == qcew_acquire._QCEW_LEVELS_REQUIRED


def test_acquire_defined_in_ingest_module_not_rebuild_store():
    from nfp_vintages.rebuild_store import _acquire_qcew_levels

    # The function's home module is nfp_ingest.qcew_acquire (moved, not copied).
    assert _acquire_qcew_levels.__module__ == "nfp_ingest.qcew_acquire"


def test_series_identity_key_stays_in_rebuild_store():
    # _SERIES_IDENTITY_KEY is a compose helper, NOT part of the acquire move.
    from nfp_vintages.rebuild_store import _SERIES_IDENTITY_KEY

    assert _SERIES_IDENTITY_KEY[0] == "geographic_type"
```

- [ ] **Step 2: Run the test, verify it fails** — `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_qcew_acquire.py -q --no-cov`.
  Expected: **FAIL** — `test_acquire_defined_in_ingest_module_not_rebuild_store` fails with
  `AssertionError` (`__module__` is still `'nfp_vintages.rebuild_store'`, since the functions
  are still defined there and not yet moved/aliased), and the `is`-identity tests fail because
  `rebuild_store`'s definitions are distinct objects from `nfp_ingest.qcew_acquire`'s.

- [ ] **Step 3: Implement** — edit `rebuild_store.py`: delete the seven moved defs/constants,
  add the import + private aliases; then edit `__main__.py:build-rebuild` to import the public
  names from `nfp_ingest.qcew_acquire`.

  **3a.** In `rebuild_store.py`, replace the now-unused `import io`, `from datetime import date`,
  `import httpx`, and `from nfp_lookups.industry import QCEW_AREA_NATIONAL` lines (they were
  only used by the moved code) — but **keep** them only if still referenced. After the move,
  `compose_rebuild_panel`/`write_rebuild_store` do **not** use `io`/`httpx`/`date`/
  `QCEW_AREA_NATIONAL`, so drop those four imports. Replace the module-level acquire block
  (`rebuild_store.py:38-433`, i.e. `_REBUILD_START_YEAR` through the end of
  `_acquire_qcew_size_native`) with a re-export block, leaving `_SERIES_IDENTITY_KEY` in place.

  Concretely, the top of `rebuild_store.py` becomes:

```python
from __future__ import annotations

import logging
from typing import Any

import polars as pl
from nfp_lookups.paths import (
    VINTAGE_STORE_PATH,
    is_canonical_store,
    is_remote,
    storage_options_for,
)
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

# Acquire layer relocated to nfp_ingest.qcew_acquire (CLI production workflow spec
# §5.2/§14): nfp-ingest sits below nfp-vintages, so capture.py can now import these
# without an illegal upward import. Re-exported here (private aliases) so the existing
# test_rebuild_acquire.py / test_rebuild_gates.py imports keep resolving.
from nfp_ingest.qcew_acquire import (  # noqa: F401  (re-export for back-compat)
    _fetch_qcew_csv,
    _prep_area_raw,
    _QCEW_LEVELS_REQUIRED,
    _REBUILD_START_YEAR,
    _size_raw_to_native,
)
from nfp_ingest.qcew_acquire import (
    acquire_qcew_levels as _acquire_qcew_levels,  # noqa: F401  (back-compat alias)
)
from nfp_ingest.qcew_acquire import (
    acquire_qcew_size_native as _acquire_qcew_size_native,  # noqa: F401
)

logger = logging.getLogger(__name__)

# The 6-column series identity that uniquely identifies one industry-month
# *independent* of the size axis, vintage, or revision.  Used as the anti-join
# key to detect which qcew_levels rows have size coverage.
#
# NOTE: do NOT include ``vintage_date`` or ``revision`` — a qcew_levels row at
# revision=1 and a size row at revision=0 cover the same series-month and the
# level row must still be dropped.
_SERIES_IDENTITY_KEY = [
    "geographic_type",
    "geographic_code",
    "ownership",
    "industry_type",
    "industry_code",
    "ref_date",
]
```

  Everything from the old `# --- Thin network helpers ---` banner through the end of
  `_acquire_qcew_size_native` (`rebuild_store.py:86-433`) is **deleted**; the file then
  continues with the unchanged `compose_rebuild_panel` (its `# --- Core compose function ---`
  banner) and `write_rebuild_store`. The module docstring at the top of the file is kept as-is.

  **3b.** In `__main__.py`, change the `build-rebuild` import block (`__main__.py:256-261`)
  from re-importing the private acquire names off `rebuild_store` to importing the **public**
  names off `nfp_ingest.qcew_acquire`, and update the two call sites:

```python
    from nfp_ingest.ces_builder import build_ces_panel
    from nfp_ingest.qcew_acquire import (
        acquire_qcew_levels,
        acquire_qcew_size_native,
    )
    from nfp_ingest.qcew_crosswalk import build_qcew_panel
    from nfp_ingest.size_class import build_size_class_panel

    from nfp_vintages.rebuild_store import (
        compose_rebuild_panel,
        write_rebuild_store,
    )
```

  and the two call sites at `__main__.py:271` and `__main__.py:278` become:

```python
    raw_qcew = acquire_qcew_levels(start_year=start_year, end_year=end_year)
```

```python
    size_native = acquire_qcew_size_native(start_year=start_year, end_year=end_year)
```

- [ ] **Step 4: Run, verify pass** — first the new back-compat test, then the two existing
  test files that import the re-exported names, then lint:
  - `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_qcew_acquire.py -q --no-cov` → **PASS**.
  - `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_rebuild_acquire.py -q --no-cov -m "not network"` → **PASS** (the moved `_prep_area_raw`/`_size_raw_to_native`/`_fetch_qcew_csv`/`_QCEW_LEVELS_REQUIRED` still import from `rebuild_store`).
  - `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_rebuild_gates.py -q --no-cov -m "not network"` → **PASS at collection** (the `_acquire_qcew_levels` import in the network test resolves via the alias; the network test itself is deselected).
  - Full non-network sweep of both touched packages:
    `uv run pytest packages/nfp-ingest packages/nfp-vintages -q --no-cov -m "not network and not slow"` → **PASS**.
  - Lint: `uv run ruff check packages/nfp-vintages/src/nfp_vintages/rebuild_store.py packages/nfp-vintages/src/nfp_vintages/__main__.py packages/nfp-vintages/src/nfp_vintages/tests/test_qcew_acquire.py` → clean (the `noqa: F401` re-export markers suppress the unused-import warnings; confirm no other E/W/F/I/B/UP findings).

- [ ] **Step 5: Commit** — stage only the rewired files + the new back-compat test.

```bash
git add packages/nfp-vintages/src/nfp_vintages/rebuild_store.py \
        packages/nfp-vintages/src/nfp_vintages/__main__.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_qcew_acquire.py
git commit -m "refactor(vintages): consume relocated QCEW acquire from nfp_ingest

rebuild_store.py drops the moved acquire defs and re-exports them (private
aliases) for back-compat with test_rebuild_acquire.py / test_rebuild_gates.py;
__main__.py:build-rebuild imports the public acquire_qcew_levels/
acquire_qcew_size_native from nfp_ingest.qcew_acquire. _SERIES_IDENTITY_KEY
(a compose helper) stays. No behaviour change.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

**Phase 1 done-when:**
- `nfp_ingest.qcew_acquire` exists with public `acquire_qcew_levels`/`acquire_qcew_size_native`
  (+ the moved private helpers/constants) and imports no `nfp_vintages`.
- `rebuild_store.py` re-exports the old private names (so the two existing test files are
  untouched and green) and keeps `_SERIES_IDENTITY_KEY` + the compose/write functions.
- `__main__.py:build-rebuild` calls the public helpers from `nfp_ingest.qcew_acquire`.
- Both package suites (`-m "not network and not slow"`) and ruff are green.
- **Firewall respected:** no edits to `nfp-model/*`, `transform_to_panel`, `model_data.py`,
  `first_print.py`, `wedge_data.py`, `a5.py`, or the A1/A2/A3 golden paths. No store writes in
  any new test (the only network paths are `@pytest.mark.network` and deselected in CI).

---


## Phase 2 — ukey under-keying fix (Decision A) in `vintage_store.py`

**Spec:** `specs/cli_production_workflow.md` §6.1 (Decision A — ukey under-keying fix) + §14 build-order step 2. Also touches §6.2 (append→compact policy) and §7 property 1 (idempotence regression).

**What this phase does.** The two incremental writers `append_to_vintage_store` (`packages/nfp-ingest/src/nfp_ingest/vintage_store.py:678`) and `compact_partition` (`:758`) dedup on a **7-column ukey** that excludes the three rebuilt-schema axes `ownership`, `size_class_type`, `size_class_code` (`VINTAGE_STORE_SCHEMA`, `nfp_lookups/schemas.py:125-144`). For QCEW Q1 size-class rows (same industry / quarter / revision, differing only by `size_class_code`) and for the `ownership` total-vs-private split, the current key **collapses distinct buckets into one** — silent data loss on append/compact. This phase extends **both** ukey lists to the full 10 columns.

**One subtlety that exceeds "add 3 lines" (called out per §6.1).** The new columns are **null** for every CES row and every non-Q1 QCEW row (`size_class_*`). Polars' anti-join in `append_to_vintage_store` (`vintage_store.py:732-736`) treats `null ≠ null` by default, so once nullable columns enter the append ukey the existing-row anti-join stops matching a re-appended identical row → dedup silently breaks and the §7 idempotence property fails. The fix therefore adds the 3 columns **plus** `nulls_equal=True` to the append anti-join (this is the current, non-deprecated keyword on the pinned polars 1.41.2; `join_nulls` is deprecated as of 1.24). `compact_partition` deduplicates with `DataFrame.unique(subset=ukey)`, which **already** treats nulls as equal, so compact needs the 3 columns only — no flag. This is a necessary part of the ukey fix, not separate scope.

**Firewall.** Edits are confined to the two writers in `vintage_store.py`. `transform_to_panel` / `read_vintage_store` and their own `_CES_SERIES_KEY` / `_QCEW_SERIES_KEY` / `dedup_key` are **not** touched; the A1/A2/A3 goldens only *read* the existing store and never call append/compact, so they are unaffected (§6.1).

**Test structure (per §14 step 2, TDD).** Two tasks, one per writer, so each edit is driven by its own valid red phase:
- **Task 2.1** edits `append_to_vintage_store` only. Its tests must trigger the anti-join (which only runs against an **existing** partition), so they use a **two-step append** — append bucket 1, then append the colliding buckets — because a single append of distinct buckets keeps them all (the anti-join has nothing to match yet).
- **Task 2.2** edits `compact_partition` only. Its test writes **two fragment files** then compacts (compact no-ops at ≤1 file).

All test rows are built **inline** (the existing `_make_vintage_df` helper cannot set ownership/size and is shaped for multi-month series, not same-ref-date buckets).

---

### Task 2.1: Extend the `append_to_vintage_store` ukey to 10 columns (+ `nulls_equal=True`)

**Files:**
- **Test:** `packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py` (EXTEND — add one test class)
- **Modify:** `packages/nfp-ingest/src/nfp_ingest/vintage_store.py` (`append_to_vintage_store` only)

- [ ] **Step 1: Write the failing tests** — append the following class to the END of `packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py`. It has three tests: (a) size-class collapse, (b) ownership collapse, (c) a null-row idempotence regression that pins the `nulls_equal=True` requirement. All build rows inline and use the two-step-append pattern.

```python
# ---------------------------------------------------------------------------
# append ukey under-keying fix (Decision A — spec §6.1)
# ---------------------------------------------------------------------------


def _store_row(
    *,
    source: str,
    sa: bool,
    ownership: str,
    industry_code: str,
    size_class_type=None,
    size_class_code=None,
    employment: float,
    industry_type: str = "supersector",
    geographic_type: str = "national",
    geographic_code: str = "00",
    ref_date: date = date(2023, 1, 1),
    vintage_date: date = date(2023, 7, 1),
    revision: int = 0,
    benchmark_revision: int = 0,
) -> dict:
    """A single VINTAGE_STORE_SCHEMA row built inline (ownership/size settable)."""
    return {
        "geographic_type": geographic_type,
        "geographic_code": geographic_code,
        "ownership": ownership,
        "industry_type": industry_type,
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date,
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "size_class_type": size_class_type,
        "size_class_code": size_class_code,
        "source": source,
        "seasonally_adjusted": sa,
    }


class TestAppendUkeySizeClassAndOwnership:
    """Spec §6.1: append must key on ownership + size_class_{type,code}."""

    def test_append_keeps_distinct_size_class_codes(self, tmp_path):
        """Rows differing ONLY by size_class_code must NOT collapse on append.

        Two-step append so the anti-join (which only runs vs an EXISTING
        partition) is exercised: bucket '1' is stored first, then buckets
        '2'/'3' (same 7-col legacy key, new size_class_code) are appended.
        Under the legacy 7-col ukey the second append returns 0 and the
        store keeps only size code '1'.
        """
        common = dict(
            source="qcew",
            sa=False,
            ownership="private",
            industry_code="10",
            size_class_type="size",
        )
        first = pl.DataFrame(
            [_store_row(**common, size_class_code="1", employment=100.0)],
            schema=VINTAGE_STORE_SCHEMA,
        )
        rest = pl.DataFrame(
            [
                _store_row(**common, size_class_code="2", employment=200.0),
                _store_row(**common, size_class_code="3", employment=300.0),
            ],
            schema=VINTAGE_STORE_SCHEMA,
        )

        assert append_to_vintage_store(first, tmp_path) == 1
        # New size buckets are genuinely new rows → both must append.
        assert append_to_vintage_store(rest, tmp_path) == 2

        pdir = tmp_path / "source=qcew" / "seasonally_adjusted=false"
        stored = pl.read_parquet(str(pdir / "*.parquet"))
        assert sorted(stored["size_class_code"].to_list()) == ["1", "2", "3"]

    def test_append_keeps_distinct_ownership(self, tmp_path):
        """Rows differing ONLY by ownership must NOT collapse on append.

        QCEW carries total (ownership='total') and private (ownership='private')
        for the SAME industry_code '00' — same 7-col legacy key, distinct
        ownership. Under the legacy ukey the second append returns 0.
        """
        common = dict(source="qcew", sa=False, industry_code="00", industry_type="total")
        total = pl.DataFrame(
            [_store_row(**common, ownership="total", employment=150_000.0)],
            schema=VINTAGE_STORE_SCHEMA,
        )
        private = pl.DataFrame(
            [_store_row(**common, ownership="private", employment=128_000.0)],
            schema=VINTAGE_STORE_SCHEMA,
        )

        assert append_to_vintage_store(total, tmp_path) == 1
        assert append_to_vintage_store(private, tmp_path) == 1

        pdir = tmp_path / "source=qcew" / "seasonally_adjusted=false"
        stored = pl.read_parquet(str(pdir / "*.parquet"))
        assert sorted(stored["ownership"].to_list()) == ["private", "total"]

    def test_append_idempotent_on_null_size_columns(self, tmp_path):
        """Regression: a CES row (ownership set, size_class_*=null) re-appended
        must still dedup to 0. Adding nullable columns to the ukey breaks the
        anti-join unless nulls_equal=True (null != null by default in polars).
        """
        row = pl.DataFrame(
            [
                _store_row(
                    source="ces",
                    sa=True,
                    ownership="private",
                    industry_code="05",
                    employment=150_000.0,
                )
            ],
            schema=VINTAGE_STORE_SCHEMA,
        )

        assert append_to_vintage_store(row, tmp_path) == 1
        # Identical re-append must skip (null size_class_* compared as equal).
        assert append_to_vintage_store(row, tmp_path) == 0
```

- [ ] **Step 2: Run the tests, verify they fail** — run:
  ```bash
  uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py::TestAppendUkeySizeClassAndOwnership -q --no-cov
  ```
  Expected: **FAIL**. `test_append_keeps_distinct_size_class_codes` fails with the second append returning `0` (legacy 7-col ukey anti-joins the new size buckets out) instead of `2`; `test_append_keeps_distinct_ownership` fails identically (second append returns `0`). `test_append_idempotent_on_null_size_columns` **passes** under the current 7-col ukey (it's the guard that will catch the `nulls_equal` regression in Step 4 if the flag is omitted).

- [ ] **Step 3: Implement** — in `packages/nfp-ingest/src/nfp_ingest/vintage_store.py`, extend the `append_to_vintage_store` ukey (the list at `:709-717`) with the three rebuilt-schema axes, and add `nulls_equal=True` to the anti-join (`:732-736`) so null size columns compare equal.

  Edit 1 — the ukey list (append the 3 lines before the closing `]`):
  ```python
    ukey = [
        "ref_date",
        "industry_type",
        "industry_code",
        "geographic_type",
        "geographic_code",
        "revision",
        "benchmark_revision",
        "ownership",
        "size_class_type",
        "size_class_code",
    ]
  ```

  Edit 2 — the anti-join (add `nulls_equal=True`; the new ukey columns are null
  for CES / non-Q1 QCEW rows and null != null by default would break dedup):
  ```python
            partition_df = partition_df.join(
                existing.select(ukey).unique(),
                on=ukey,
                how="anti",
                nulls_equal=True,
            )
  ```

- [ ] **Step 4: Run, verify pass** — run:
  ```bash
  uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py::TestAppendUkeySizeClassAndOwnership -q --no-cov
  ```
  Expected: **PASS** (all three). Then run the package suite (the pre-existing `TestAppendToVintageStore::test_append_dedup_skips_existing` exercises null size columns and would have regressed without `nulls_equal=True`) and the linter:
  ```bash
  uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py -q --no-cov
  uv run ruff check packages/nfp-ingest/src/nfp_ingest/
  ```
  Expected: **PASS** + clean lint (lines ≤ 100).

- [ ] **Step 5: Commit** — stage the two specific files only (never `git add -A`):
  ```bash
  git add packages/nfp-ingest/src/nfp_ingest/vintage_store.py \
          packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py
  git commit -m "fix(vintage-store): key append on ownership + size_class (Decision A)

Extend append_to_vintage_store ukey to 10 columns so QCEW Q1 size-class
buckets and the ownership total/private split no longer collapse on append.
Add nulls_equal=True to the anti-join so null size columns (CES / non-Q1
QCEW) still dedup. Spec cli_production_workflow.md §6.1."
  ```

---

### Task 2.2: Extend the `compact_partition` ukey to 10 columns

**Files:**
- **Test:** `packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py` (EXTEND — add one test class)
- **Modify:** `packages/nfp-ingest/src/nfp_ingest/vintage_store.py` (`compact_partition` only)

- [ ] **Step 1: Write the failing test** — append the following class to the END of `packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py` (it reuses the `_store_row` helper added in Task 2.1). It writes **two fragment files** of distinct size buckets, then compacts; under the legacy 7-col ukey compact collapses them to one.

```python
# ---------------------------------------------------------------------------
# compact ukey under-keying fix (Decision A — spec §6.1)
# ---------------------------------------------------------------------------


class TestCompactUkeySizeClass:
    """Spec §6.1: compact must key on ownership + size_class_{type,code}."""

    def test_compact_keeps_distinct_size_class_codes(self, tmp_path):
        """Distinct QCEW Q1 size buckets across two fragments must survive compaction.

        compact_partition no-ops at <=1 file, so write two fragment parquets
        (bucket '1' in one, buckets '2'/'3' in the other). Under the legacy
        7-col ukey unique(subset=ukey) collapses all three to a single row.
        """
        common = dict(
            source="qcew",
            sa=False,
            ownership="private",
            industry_code="10",
            size_class_type="size",
        )
        pdir = tmp_path / "source=qcew" / "seasonally_adjusted=false"
        pdir.mkdir(parents=True)

        frag_a = pl.DataFrame(
            [_store_row(**common, size_class_code="1", employment=100.0)],
            schema=VINTAGE_STORE_SCHEMA,
        )
        frag_b = pl.DataFrame(
            [
                _store_row(**common, size_class_code="2", employment=200.0),
                _store_row(**common, size_class_code="3", employment=300.0),
            ],
            schema=VINTAGE_STORE_SCHEMA,
        )
        frag_a.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "a.parquet")
        frag_b.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "b.parquet")

        compact_partition(tmp_path, "qcew", False)

        files = list(pdir.glob("*.parquet"))
        assert len(files) == 1
        assert files[0].name == "compacted.parquet"
        compacted = pl.read_parquet(files[0])
        assert sorted(compacted["size_class_code"].to_list()) == ["1", "2", "3"]
```

- [ ] **Step 2: Run the test, verify it fails** — run:
  ```bash
  uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py::TestCompactUkeySizeClass::test_compact_keeps_distinct_size_class_codes -q --no-cov
  ```
  Expected: **FAIL** — under the legacy 7-col ukey `unique(subset=ukey)` collapses the three size buckets, so `compacted["size_class_code"]` is `["1"]` (one row), not `["1", "2", "3"]`.

- [ ] **Step 3: Implement** — in `packages/nfp-ingest/src/nfp_ingest/vintage_store.py`, extend the `compact_partition` ukey (the list at `:797-805`) with the same three axes. No flag is needed here: `DataFrame.unique(subset=ukey)` (`:813`) already treats nulls as equal, so CES / non-Q1 QCEW rows still dedup correctly.
  ```python
    ukey = [
        "ref_date",
        "industry_type",
        "industry_code",
        "geographic_type",
        "geographic_code",
        "revision",
        "benchmark_revision",
        "ownership",
        "size_class_type",
        "size_class_code",
    ]
  ```

- [ ] **Step 4: Run, verify pass** — run:
  ```bash
  uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py::TestCompactUkeySizeClass -q --no-cov
  ```
  Expected: **PASS**. Then run the full vintage-store suite (the pre-existing `TestCompactPartition` tests use null size columns and must still pass — `unique` handles nulls) and the linter:
  ```bash
  uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py -q --no-cov
  uv run ruff check packages/nfp-ingest/src/nfp_ingest/
  ```
  Expected: **PASS** + clean lint.

- [ ] **Step 5: Commit** — stage the two specific files only (never `git add -A`):
  ```bash
  git add packages/nfp-ingest/src/nfp_ingest/vintage_store.py \
          packages/nfp-ingest/src/nfp_ingest/tests/test_vintage_store.py
  git commit -m "fix(vintage-store): key compact on ownership + size_class (Decision A)

Extend compact_partition ukey to 10 columns so QCEW Q1 size-class buckets
no longer collapse during compaction. unique(subset=ukey) already treats
nulls as equal, so CES / non-Q1 QCEW rows still dedup. Completes the §6.1
ukey fix across both incremental writers."
  ```


---


## Phase 3 — Calendar-advance callable `nfp_vintages/calendar.py`

Lifts the release-calendar scrape (`_build_release_calendar`, `__main__.py:61-168`) into a
public callable `advance_release_calendar()` in a new module
`nfp_vintages/calendar.py`, preserving the graceful-403 fallback verbatim (spec §5.0). The
legacy `process` command is **temporarily** rewired to call the new function so the suite
stays green until §9 deletes `process` (spec §10, §14 step 3). This is the §5.0 production
dependency: `update` (Phase 5) advances `vintage_dates.parquet` to `T` before every capture,
and an un-advanced calendar makes the CES tag join return null → a silent empty capture.

**Firewall:** does not touch `transform_to_panel`, `build_model_data`, `model_data.py`,
`first_print.py`, `wedge_data.py`, `a5.py`, or any A1/A2/A3 golden path. The lift is a pure
code move; `build_vintage_dates` (`nfp_ingest`) is consumed unchanged.

---

### Task 3.1: Lift `_build_release_calendar` into public `advance_release_calendar()`

**Files:**
- Create `packages/nfp-vintages/src/nfp_vintages/calendar.py`
- Test `packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py`

The lift is verbatim except the rename `_build_release_calendar` → `advance_release_calendar`
(now public) and a new module docstring. The graceful-403/parse fallback
(`__main__.py:97-127`) is preserved exactly. The function writes both `RELEASE_DATES_PATH`
and `VINTAGE_DATES_PATH` (`__main__.py:160-168`).

**Test seam (load-bearing).** `build_vintage_dates` (`vintage_dates.py:329`) binds
`RELEASE_DATES_PATH` in **its own module namespace** at import
(`vintage_dates.py:20`) and falls back to it (`path = release_dates_path or
RELEASE_DATES_PATH`, `vintage_dates.py:348`). `advance_release_calendar` imports the path
constants **in-function**, so those resolve at call time and are monkeypatchable via
`nfp_lookups.paths.*` — but `build_vintage_dates`'s binding is **not** reached by patching
`nfp_lookups.paths`. The test must therefore patch BOTH bindings:
`nfp_lookups.paths.RELEASE_DATES_PATH` (the writer) **and**
`nfp_ingest.release_dates.vintage_dates.RELEASE_DATES_PATH` (the reader). It also mocks the
async network functions (`fetch_index` → raise `FetchError`, exercising the graceful-403
fallback; `create_session` → trivial async CM) so no real BLS hit occurs. The leanest robust
assertion: even with no cached release dirs, `build_vintage_dates`'s
`SUPPLEMENTAL_RELEASE_DATES` + pre-scrape generators (`vintage_dates.py:352-369`) yield a
non-empty frame, so `VINTAGE_DATES_PATH` is written with `height > 0`.

- [ ] **Step 1: Write the failing test** — create
  `packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py`:

```python
"""Tests for advance_release_calendar — the §5.0 calendar-advance callable.

The unit test stubs the BLS network (fetch_index raises FetchError, exercising
the graceful-403 fallback) so no real bls.gov hit occurs, redirects every path
constant to tmp_path, and asserts VINTAGE_DATES_PATH is written non-empty. The
live path is marked @pytest.mark.network.
"""

from __future__ import annotations

import polars as pl
import pytest
from nfp_download.release_dates.scraper import FetchError


def _patch_paths(monkeypatch, tmp_path):
    """Redirect both the writer's and reader's path bindings to tmp_path."""
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    releases_dir = tmp_path / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    release_dates_path = intermediate / "release_dates.parquet"
    vintage_dates_path = intermediate / "vintage_dates.parquet"

    # advance_release_calendar imports these in-function → resolved at call time.
    monkeypatch.setattr("nfp_lookups.paths.RELEASES_DIR", releases_dir)
    monkeypatch.setattr("nfp_lookups.paths.RELEASE_DATES_PATH", release_dates_path)
    monkeypatch.setattr("nfp_lookups.paths.VINTAGE_DATES_PATH", vintage_dates_path)
    # build_vintage_dates binds RELEASE_DATES_PATH in ITS module at import — patch too.
    monkeypatch.setattr(
        "nfp_ingest.release_dates.vintage_dates.RELEASE_DATES_PATH",
        release_dates_path,
    )
    return release_dates_path, vintage_dates_path


def test_advance_release_calendar_writes_vintage_dates(monkeypatch, tmp_path):
    """With the scrape stubbed to a graceful-403 fallback, the calendar advance
    still builds and writes vintage_dates.parquet from supplemental/pre-scrape rows."""
    release_dates_path, vintage_dates_path = _patch_paths(monkeypatch, tmp_path)

    class _StubSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    def _stub_create_session(*args, **kwargs):
        return _StubSession()

    async def _stub_fetch_index(session, url):
        # Simulate BLS 403 — drives the cached-pages-only fallback (§5.0).
        raise FetchError("stubbed 403")

    monkeypatch.setattr(
        "nfp_download.release_dates.scraper.create_session", _stub_create_session
    )
    monkeypatch.setattr(
        "nfp_download.release_dates.scraper.fetch_index", _stub_fetch_index
    )

    from nfp_vintages.calendar import advance_release_calendar

    advance_release_calendar()

    assert release_dates_path.exists()
    assert vintage_dates_path.exists()
    vdf = pl.read_parquet(vintage_dates_path)
    assert vdf.height > 0
    assert set(vdf.columns) >= {
        "publication",
        "ref_date",
        "vintage_date",
        "revision",
        "benchmark_revision",
    }


@pytest.mark.network
def test_advance_release_calendar_live(monkeypatch, tmp_path):
    """Live BLS scrape path — redirected to tmp so it never clobbers prod."""
    _, vintage_dates_path = _patch_paths(monkeypatch, tmp_path)

    from nfp_vintages.calendar import advance_release_calendar

    advance_release_calendar()

    assert vintage_dates_path.exists()
    assert pl.read_parquet(vintage_dates_path).height > 0
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py::test_advance_release_calendar_writes_vintage_dates -q --no-cov`
  Expected: FAIL with `ModuleNotFoundError: No module named 'nfp_vintages.calendar'` (the
  module does not exist yet).

- [ ] **Step 3: Implement** — create
  `packages/nfp-vintages/src/nfp_vintages/calendar.py` with the verbatim lift (rename to the
  public name, new docstring; fallback logic identical to `__main__.py:90-127`):

```python
"""Release-calendar advance — scrape BLS schedule, build the vintage calendar.

Lifted from ``nfp_vintages.__main__._build_release_calendar`` (spec §5.0). The
public ``advance_release_calendar`` is the §5.0 production dependency: ``update``
advances ``vintage_dates.parquet`` to the as-of cutoff before every capture, and
the bootstrap script reuses it. The scrape degrades gracefully on a BLS 403 by
falling back to cached release pages already on disk.
"""

from __future__ import annotations


def advance_release_calendar() -> None:
    """Scrape the BLS publication schedule and build release/vintage parquets.

    Produces ``release_dates.parquet`` and ``vintage_dates.parquet`` in the
    intermediate directory. On a BLS 403/parse drift the per-publication scrape
    is skipped and the calendar is built from cached release pages on disk; only
    newly-published pages are missed.
    """
    import asyncio

    import polars as pl
    from nfp_download.release_dates.config import PUBLICATIONS
    from nfp_download.release_dates.parser import collect_release_dates
    from nfp_download.release_dates.scraper import (
        FetchError,
        ParseError,
        create_session,
        download_all,
        fetch_index,
        parse_index_page,
    )
    from nfp_ingest.release_dates.vintage_dates import (
        SUPPLEMENTAL_RELEASE_DATES,
        build_vintage_dates,
    )
    from nfp_lookups.paths import (
        RELEASE_DATES_PATH,
        RELEASES_DIR,
        VINTAGE_DATES_PATH,
    )

    async def _download_all_publications() -> None:
        async with create_session() as session:
            for pub in PUBLICATIONS:
                print(f'Fetching index for {pub.name}...')
                try:
                    html = await fetch_index(session, pub.index_url)
                except FetchError as e:
                    # Safety net: if BLS's bot detection changes again, the
                    # calendar can still be built from release pages already
                    # on disk; only newly published pages are missed.
                    print(
                        f'  WARNING: index fetch failed for {pub.name} ({e}); '
                        f'using cached release pages only'
                    )
                    continue
                try:
                    entries = parse_index_page(
                        html, pub.name, pub.series, pub.frequency,
                    )
                except ParseError as e:
                    # Page structure may have drifted; fall back to cached
                    # release pages already on disk so the rest of the calendar
                    # build can proceed. Only newly-published pages are missed.
                    print(
                        f'  WARNING: index parse failed for {pub.name} ({e}); '
                        f'using cached release pages only'
                    )
                    continue
                print(f'  Found {len(entries)} releases for {pub.name}')
                try:
                    paths = await download_all(entries, pub.name)
                except FetchError as e:
                    print(
                        f'  WARNING: release download failed for {pub.name} '
                        f'({e}); using cached release pages only'
                    )
                    continue
                print(f'  Downloaded {len(paths)} new files for {pub.name}')

    asyncio.run(_download_all_publications())

    print('Building release_dates...')
    rows = []
    for pub in PUBLICATIONS:
        pub_dir = RELEASES_DIR / pub.name
        if not pub_dir.exists():
            continue
        for row in collect_release_dates(pub.name, pub_dir):
            rows.append(row)

    df = pl.DataFrame(
        rows,
        schema={'publication': pl.Utf8, 'ref_date': pl.Date, 'vintage_date': pl.Date},
        orient='row',
    )
    supplemental = pl.DataFrame(
        [
            {'publication': p, 'ref_date': ref, 'vintage_date': vint}
            for p, ref, vint in SUPPLEMENTAL_RELEASE_DATES
        ],
        schema={'publication': pl.Utf8, 'ref_date': pl.Date, 'vintage_date': pl.Date},
    )
    existing_keys = df.select('publication', 'ref_date').unique()
    supplemental = supplemental.join(
        existing_keys, on=['publication', 'ref_date'], how='anti',
    )
    if supplemental.height > 0:
        df = pl.concat([df, supplemental])
    df = df.sort('publication', 'ref_date')

    RELEASE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RELEASE_DATES_PATH)
    print(f'Wrote {RELEASE_DATES_PATH} ({len(df)} rows)')

    print('Building vintage_dates...')
    vdf = build_vintage_dates()
    VINTAGE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    vdf.write_parquet(VINTAGE_DATES_PATH)
    print(f'Wrote {VINTAGE_DATES_PATH} ({len(vdf)} rows)')
```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py::test_advance_release_calendar_writes_vintage_dates -q --no-cov`
  Expected: PASS (the stubbed scrape falls back, supplemental + pre-scrape rows produce a
  non-empty `vintage_dates.parquet`). The `@pytest.mark.network` live test self-excludes under
  the `not network` marker. Then run the package suite + lint:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py -m "not network" -q --no-cov`
  and `uv run ruff check packages/nfp-vintages/src/nfp_vintages/calendar.py packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py`.
  Expected: all PASS, ruff clean.

- [ ] **Step 5: Commit** — stage only the new module and its test:
  `git add packages/nfp-vintages/src/nfp_vintages/calendar.py packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py`
  then commit:
  `git commit -m "feat(vintages): lift _build_release_calendar into public advance_release_calendar (§5.0)"`

---

### Task 3.2: Rewire `process` to call `advance_release_calendar`; remove the lifted helper

**Files:**
- Modify `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test `packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py` (EXTEND)

Complete the lift: delete the in-`__main__` `_build_release_calendar` def
(`__main__.py:61-168`) and rewire `process()` (`__main__.py:171-186`) to import and call
`advance_release_calendar` from the new module. This keeps `process` green until §9 (Phase 9)
deletes it. The test asserts `process` invokes `advance_release_calendar` (monkeypatching the
calendar advance + the three processing `main`s so no scrape or processing actually runs).

- [ ] **Step 1: Write the failing test** — append to
  `packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py`:

```python
def test_process_command_calls_advance_release_calendar(monkeypatch):
    """The legacy `process` command must delegate the calendar build to the
    lifted advance_release_calendar (kept green until §9 deletes `process`)."""
    import nfp_vintages.__main__ as cli

    calls: list[str] = []

    def _spy_advance() -> None:
        calls.append("advance")

    # process imports advance_release_calendar from nfp_vintages.calendar in-body.
    monkeypatch.setattr("nfp_vintages.calendar.advance_release_calendar", _spy_advance)
    # Stub the three processing mains so nothing heavy runs.
    monkeypatch.setattr(
        "nfp_vintages.processing.ces_triangular.main", lambda: None
    )
    monkeypatch.setattr("nfp_vintages.processing.qcew_bulk.main", lambda: None)
    monkeypatch.setattr("nfp_vintages.processing.combine.main", lambda: None)

    cli.process()

    assert calls == ["advance"]
    # The lifted helper must no longer live in __main__.
    assert not hasattr(cli, "_build_release_calendar")
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py::test_process_command_calls_advance_release_calendar -q --no-cov`
  Expected: FAIL — `process` still calls the in-`__main__` `_build_release_calendar` (so the
  monkeypatched `nfp_vintages.calendar.advance_release_calendar` is never invoked → `calls`
  is empty), and `_build_release_calendar` is still an attribute of the module
  (`assert not hasattr(...)` fails).

- [ ] **Step 3: Implement** — in `packages/nfp-vintages/src/nfp_vintages/__main__.py`,
  delete the entire `_build_release_calendar` def (`__main__.py:61-168`) and replace the
  `process` body's calendar step. The edited `process` becomes:

```python
@app.command()
def process() -> None:
    """Scrape BLS release calendar, then process CES/QCEW revisions."""
    from nfp_vintages.calendar import advance_release_calendar
    from nfp_vintages.processing.ces_triangular import main as ces_triangular_main
    from nfp_vintages.processing.combine import main as combine_main
    from nfp_vintages.processing.qcew_bulk import main as qcew_main

    print('=== Building BLS release calendar ===')
    advance_release_calendar()

    print('\n=== Processing CES national revisions ===')
    ces_triangular_main()
    print('\n=== Processing QCEW revisions ===')
    qcew_main()
    print('\n=== Combining revisions ===')
    combine_main()
```

  Note: the deleted `_build_release_calendar` block is `__main__.py:61-168` (the whole
  function between `@app.command("download-indicators")`'s `download_indicators` and the
  `@app.command()` `process` decorator). Removing it leaves `process` as the first definition
  after `download_indicators`. No other lines in `__main__.py` change in this task.

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py::test_process_command_calls_advance_release_calendar -q --no-cov`
  Expected: PASS (`process` now delegates to `advance_release_calendar`; the helper is gone
  from `__main__`). Then the full file + the package CLI smoke + lint:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py packages/nfp-vintages/src/nfp_vintages/tests/test_cli_snapshot.py -m "not network" -q --no-cov`
  and `uv run ruff check packages/nfp-vintages/src/nfp_vintages/__main__.py`.
  Expected: all PASS, ruff clean (confirms the `_build_release_calendar` deletion left no
  unused imports — its imports were function-local, so none leak to module scope).

- [ ] **Step 5: Commit** — stage only the CLI module and the extended test:
  `git add packages/nfp-vintages/src/nfp_vintages/__main__.py packages/nfp-vintages/src/nfp_vintages/tests/test_calendar.py`
  then commit:
  `git commit -m "refactor(vintages): rewire process to call advance_release_calendar; drop in-CLI helper (§10)"`


---


## Phase 4 — CES capture adapter (`nfp_ingest/capture.py`)

Implements spec §5.1 (the CES month-T real-time capture). This phase builds the
net-new **capture-to-store adapter** that bridges the legacy `COMBINED_SCHEMA`
emitted by `_fetch_ces_releases` (`releases.py:101`, 11 cols, no
`ownership`/`size_class_*`) to the rebuilt `VINTAGE_STORE_SCHEMA`
(`schemas.py:125-144`) that `append_to_vintage_store` hard-rejects anything missing
(`vintage_store.py:700-702`).

Scope of THIS phase (per the build order, §14 step 4): the **CES side only** —
`_remap_ces_to_store_schema`, `_detect_corrected_levels`, and `capture_ces_print`,
plus the `CaptureResult`/`CorrectedLevel` dataclasses. `capture_qcew_quarter` and
the `qcew_acquire.py` relocation are Phase 6 / a prior phase; this phase imports
neither. The `update` CLI wiring is Phase 5.

The IND-IMD-1 rev1/bmr0 drop (§5.1.2, `releases.py:159-185`) is already carried by
`_fetch_ces_releases`, which this adapter reuses for fetch+tag; the adapter does
not re-implement it. The capture writes the **same supersector set** `_fetch_ces_releases`
emits (NSA + SA, codes `00`/`05`/`10`/`20`/`30`/`40`/`50`/`55`/`60`/`65`/`70`/`80`),
matching `build_ces_panel`'s coverage (§5.1 final paragraph).

---

### Task 4.1: `_remap_ces_to_store_schema` — COMBINED_SCHEMA → VINTAGE_STORE_SCHEMA

**Files:**
- Create: `packages/nfp-ingest/src/nfp_ingest/capture.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py`

The remap derives `ownership` via `nfp_lookups.industry.ownership_for` keyed on the
**rebuilt** `(industry_type, industry_code)` pair (§5.1.3): `'00'`→(`total`,`total`),
`'05'`→(`total`,`private`), every other supersector code →(`supersector`,`private`).
It sets `size_class_type`/`size_class_code` to null (CES has no size dimension,
`schemas.py:136-139`) and re-stamps the legacy `industry_type` (`'national'`/`'domain'`/
`'supersector'`) with the rebuilt value. **Do not** reuse `releases.py:193-199`'s
legacy `'national'/'domain'` mapping (§5.1.3 explicit "Do not").

- [ ] **Step 1: Write the failing test** — synthetic `COMBINED_SCHEMA` frame, assert
  the remapped frame has exactly `VINTAGE_STORE_SCHEMA` columns/dtypes and the right
  `(industry_type, ownership)` per code.

```python
"""Tests for nfp_ingest.capture — CES month-T capture adapter (spec §5.1)."""

from datetime import date

import polars as pl
import pytest

from nfp_ingest import capture as _cap
from nfp_ingest.capture import (
    CaptureResult,
    CorrectedLevel,
    _detect_corrected_levels,
    _remap_ces_to_store_schema,
    capture_ces_print,
)
from nfp_ingest.releases import COMBINED_SCHEMA
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


def _combined_row(
    *,
    industry_type: str,
    industry_code: str,
    sa: bool = True,
    ref_date: date = date(2026, 1, 1),
    vintage_date: date = date(2026, 2, 6),
    revision: int = 0,
    benchmark_revision: int = 0,
    employment: float = 1000.0,
) -> dict:
    """One COMBINED_SCHEMA row (legacy 11-col CES release shape)."""
    return {
        "source": "ces",
        "seasonally_adjusted": sa,
        "geographic_type": "national",
        "geographic_code": "00",
        "industry_type": industry_type,
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date,
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
    }


def _combined_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=COMBINED_SCHEMA)


def test_remap_produces_vintage_store_schema():
    df = _combined_frame([
        _combined_row(industry_type="national", industry_code="00"),
        _combined_row(industry_type="domain", industry_code="05"),
        _combined_row(industry_type="supersector", industry_code="60"),
    ])

    out = _remap_ces_to_store_schema(df)

    assert out.columns == list(VINTAGE_STORE_SCHEMA)
    assert dict(zip(out.columns, out.dtypes)) == VINTAGE_STORE_SCHEMA


def test_remap_assigns_rebuilt_taxonomy_per_code():
    df = _combined_frame([
        _combined_row(industry_type="national", industry_code="00"),
        _combined_row(industry_type="domain", industry_code="05"),
        _combined_row(industry_type="supersector", industry_code="70"),
    ])

    out = _remap_ces_to_store_schema(df).sort("industry_code")
    got = {
        r["industry_code"]: (r["industry_type"], r["ownership"])
        for r in out.iter_rows(named=True)
    }

    assert got["00"] == ("total", "total")
    assert got["05"] == ("total", "private")
    assert got["70"] == ("supersector", "private")


def test_remap_nulls_size_class_columns():
    df = _combined_frame([_combined_row(industry_type="supersector", industry_code="40")])

    out = _remap_ces_to_store_schema(df)

    assert out["size_class_type"].null_count() == out.height
    assert out["size_class_code"].null_count() == out.height
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::test_remap_produces_vintage_store_schema -q --no-cov`
  Expected: FAIL with `ModuleNotFoundError: No module named 'nfp_ingest.capture'`
  (the file does not exist yet).

- [ ] **Step 3: Implement** — create `capture.py` with the module header, the two
  dataclasses (needed by later tasks, defined now so the module imports cleanly), and
  `_remap_ces_to_store_schema`. The rebuilt `industry_type` is `total` for `00`/`05`
  and `supersector` otherwise; `ownership` comes from `ownership_for` keyed on that
  rebuilt pair.

```python
"""CES (and QCEW) month-T capture-to-store adapter (spec §5.1).

Bridges the legacy ``COMBINED_SCHEMA`` CES release frame emitted by
:func:`nfp_ingest.releases._fetch_ces_releases` to the rebuilt
``VINTAGE_STORE_SCHEMA`` and appends it incrementally to the vintage store.
Production captures the current print BLS publishes for a month ``T`` and
appends it; the triangular bulk extract is never re-run here (that is the
bootstrap path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from nfp_lookups.industry import ownership_for
from nfp_lookups.paths import VINTAGE_STORE_PATH
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class CorrectedLevel:
    """A capture row whose ukey already exists in the store with a different level.

    Surfaced by :func:`_detect_corrected_levels` (spec §5.1.4 / §6.3): the store
    ukey excludes both ``vintage_date`` and ``employment``, so a re-stamped
    same-revision level would be silently dropped by the append anti-join. This
    record is the runtime detection signal — no auto-replacement is performed.
    """

    ref_date: date
    industry_code: str
    revision: int
    benchmark_revision: int
    stored_employment: float
    incoming_employment: float


@dataclass
class CaptureResult:
    """Outcome of a single ``capture_*`` call.

    Attributes
    ----------
    appended : int
        Rows actually written to the store (post anti-join).
    corrected : list[CorrectedLevel]
        Existing-ukey rows whose incoming level differs from the stored level.
    skipped : int
        Rows present in the capture but already in the store (anti-joined out).
    """

    appended: int
    corrected: list[CorrectedLevel]
    skipped: int


def _remap_ces_to_store_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Remap a ``COMBINED_SCHEMA`` CES frame to ``VINTAGE_STORE_SCHEMA``.

    Derives the rebuilt ``(industry_type, ownership)`` axes (spec §5.1.3):
    ``'00'``→(``total``, ``total``), ``'05'``→(``total``, ``private``), every
    other supersector code →(``supersector``, ``private``). ``size_class_*`` are
    null (CES has no size dimension). ``ownership`` is resolved through
    :func:`nfp_lookups.industry.ownership_for` on the rebuilt pair — never the
    legacy ``'national'/'domain'`` mapping in ``releases.py``.

    Parameters
    ----------
    df : pl.DataFrame
        A frame in ``nfp_ingest.releases.COMBINED_SCHEMA`` (CES release shape).

    Returns
    -------
    pl.DataFrame
        A frame in ``VINTAGE_STORE_SCHEMA`` column order and dtypes.
    """
    rebuilt_type = (
        pl.when(pl.col("industry_code").is_in(["00", "05"]))
        .then(pl.lit("total"))
        .otherwise(pl.lit("supersector"))
    )
    df = df.with_columns(rebuilt_type.alias("industry_type"))

    # ownership_for is keyed on the rebuilt (industry_type, industry_code) pair.
    pairs = (
        df.select("industry_type", "industry_code")
        .unique()
        .to_dicts()
    )
    own_map = {
        (p["industry_type"], p["industry_code"]): ownership_for(
            p["industry_type"], p["industry_code"]
        )
        for p in pairs
    }
    ownership = pl.struct("industry_type", "industry_code").map_elements(
        lambda s: own_map[(s["industry_type"], s["industry_code"])],
        return_dtype=pl.Utf8,
    )

    return (
        df.with_columns(
            ownership.alias("ownership"),
            pl.lit(None, dtype=pl.Utf8).alias("size_class_type"),
            pl.lit(None, dtype=pl.Utf8).alias("size_class_code"),
        )
        .select(list(VINTAGE_STORE_SCHEMA))
        .cast(VINTAGE_STORE_SCHEMA)
    )
```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py -q --no-cov`
  Expected: PASS (the 3 remap tests; the corrected-level/capture tests come in 4.2/4.3).
  Then the package suite + lint:
  `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py -q --no-cov`
  and `uv run ruff check packages/nfp-ingest/src/nfp_ingest/`.

- [ ] **Step 5: Commit** —
  `git add packages/nfp-ingest/src/nfp_ingest/capture.py packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py`
  then `git commit -m "feat(ingest): CES capture remap + result dataclasses (spec §5.1.3)"`

---

### Task 4.2: `_detect_corrected_levels` — incoming-vs-stored level comparison

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/capture.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py`

Per §5.1.4/§6.3: the append ukey excludes `vintage_date` **and** `employment`, so a
corrected same-revision level on an existing ukey is silently dropped by the anti-join.
`_detect_corrected_levels` compares each incoming row's `employment` against the stored
row for the same **extended** ukey (the 7-col ukey of `append_to_vintage_store:709-717`
plus the rebuilt axes `ownership`, `size_class_type`, `size_class_code` added in §6.1)
**before** the anti-join, returning a `CorrectedLevel` per divergence. It reads via
`read_vintage_store` (`vintage_store.py:336`), partition-pruned to `(source, sa)`, and
returns `[]` when the partition is absent.

- [ ] **Step 1: Write the failing test** — seed a tmp_path store with one CES row, then
  pass an incoming frame with the **same ukey but a different `employment`** and assert
  one `CorrectedLevel`; pass a matching-level frame and assert none; pass an empty store
  and assert none.

```python
def _store_row(
    *,
    industry_code: str = "05",
    industry_type: str = "total",
    ownership: str = "private",
    ref_date: date = date(2026, 1, 1),
    vintage_date: date = date(2026, 2, 6),
    revision: int = 0,
    benchmark_revision: int = 0,
    employment: float = 1000.0,
    sa: bool = True,
) -> dict:
    """One VINTAGE_STORE_SCHEMA row (CES headline)."""
    return {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": ownership,
        "industry_type": industry_type,
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date,
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "size_class_type": None,
        "size_class_code": None,
        "source": "ces",
        "seasonally_adjusted": sa,
    }


def _seed_store(store_path, rows: list[dict]) -> None:
    """Write VINTAGE_STORE_SCHEMA rows as a Hive-partitioned store under store_path."""
    df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
    for (source, sa), part in df.group_by(["source", "seasonally_adjusted"]):
        sa_str = str(sa).lower()
        pdir = store_path / f"source={source}" / f"seasonally_adjusted={sa_str}"
        pdir.mkdir(parents=True, exist_ok=True)
        part.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "data.parquet")


def test_detect_corrected_flags_changed_level(tmp_path):
    _seed_store(tmp_path, [_store_row(employment=1000.0)])
    incoming = pl.DataFrame(
        [_store_row(employment=1234.0)], schema=VINTAGE_STORE_SCHEMA
    )

    corrected = _detect_corrected_levels(
        incoming, tmp_path, source="ces", seasonally_adjusted=True
    )

    assert len(corrected) == 1
    cl = corrected[0]
    assert cl.ref_date == date(2026, 1, 1)
    assert cl.industry_code == "05"
    assert cl.stored_employment == 1000.0
    assert cl.incoming_employment == 1234.0


def test_detect_corrected_ignores_matching_level(tmp_path):
    _seed_store(tmp_path, [_store_row(employment=1000.0)])
    incoming = pl.DataFrame(
        [_store_row(employment=1000.0)], schema=VINTAGE_STORE_SCHEMA
    )

    corrected = _detect_corrected_levels(
        incoming, tmp_path, source="ces", seasonally_adjusted=True
    )

    assert corrected == []


def test_detect_corrected_empty_store_returns_empty(tmp_path):
    incoming = pl.DataFrame(
        [_store_row(employment=1000.0)], schema=VINTAGE_STORE_SCHEMA
    )

    corrected = _detect_corrected_levels(
        incoming, tmp_path, source="ces", seasonally_adjusted=True
    )

    assert corrected == []
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::test_detect_corrected_flags_changed_level -q --no-cov`
  Expected: FAIL with `ImportError: cannot import name '_detect_corrected_levels'`
  (helper not yet defined) — note the test file imports it at module top, so the whole
  file errors on collection until Step 3.

- [ ] **Step 3: Implement** — add `_detect_corrected_levels` to `capture.py`. The
  extended ukey matches the §6.1 writer key (7 base cols + `ownership`,
  `size_class_type`, `size_class_code`); the store partition is read with
  `read_vintage_store` pruned to `(source, sa)`. A null-safe join on the ukey surfaces
  divergent `employment`.

```python
from nfp_ingest.vintage_store import read_vintage_store

# Extended store ukey: the 7-col append/compact key (vintage_store.py:709-717)
# plus the rebuilt axes added in spec §6.1. Excludes vintage_date + employment.
_CES_CORRECTED_UKEY: list[str] = [
    "ref_date",
    "industry_type",
    "industry_code",
    "geographic_type",
    "geographic_code",
    "revision",
    "benchmark_revision",
    "ownership",
    "size_class_type",
    "size_class_code",
]


def _detect_corrected_levels(
    new_rows: pl.DataFrame,
    store_path: Path,
    source: str,
    seasonally_adjusted: bool,
) -> list[CorrectedLevel]:
    """Flag incoming rows whose ukey exists in the store with a *different* level.

    Compares each incoming ``employment`` against the stored value for the same
    extended ukey (spec §5.1.4 / §6.3), *before* the append anti-join would drop
    it. Returns one :class:`CorrectedLevel` per divergence; an absent partition
    yields ``[]``.

    Parameters
    ----------
    new_rows : pl.DataFrame
        Capture rows in ``VINTAGE_STORE_SCHEMA`` for one ``(source, sa)``.
    store_path : Path
        Root of the Hive-partitioned vintage store.
    source : str
        Source partition key (``'ces'``, ``'qcew'``).
    seasonally_adjusted : bool
        Seasonal-adjustment partition key.

    Returns
    -------
    list[CorrectedLevel]
        One record per same-ukey/different-level row, sorted by ref_date.
    """
    partition_dir = (
        store_path
        / f"source={source}"
        / f"seasonally_adjusted={str(seasonally_adjusted).lower()}"
    )
    if not partition_dir.exists():
        return []

    stored = (
        read_vintage_store(
            store_path, source=source, seasonally_adjusted=seasonally_adjusted
        )
        .select([*_CES_CORRECTED_UKEY, "employment"])
        .rename({"employment": "stored_employment"})
        .collect()
    )
    if stored.is_empty():
        return []

    joined = new_rows.select([*_CES_CORRECTED_UKEY, "employment"]).join(
        stored, on=_CES_CORRECTED_UKEY, how="inner", nulls_equal=True
    )
    diverged = joined.filter(
        pl.col("employment") != pl.col("stored_employment")
    ).sort("ref_date")

    return [
        CorrectedLevel(
            ref_date=r["ref_date"],
            industry_code=r["industry_code"],
            revision=r["revision"],
            benchmark_revision=r["benchmark_revision"],
            stored_employment=r["stored_employment"],
            incoming_employment=r["employment"],
        )
        for r in diverged.iter_rows(named=True)
    ]
```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py -q --no-cov`
  Expected: PASS (remap + 3 corrected-level tests). Then
  `uv run ruff check packages/nfp-ingest/src/nfp_ingest/`.

- [ ] **Step 5: Commit** —
  `git add packages/nfp-ingest/src/nfp_ingest/capture.py packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py`
  then `git commit -m "feat(ingest): corrected-level detection for CES capture (spec §5.1.4/§6.3)"`

---

### Task 4.3: `capture_ces_print` — fetch → tag → remap → censor → detect → append → compact

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/capture.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py`

`capture_ces_print(as_of, *, store_path)` (§5.1):
1. **Hard prereq** — raise `RuntimeError` if `BLS_API_KEY` is unset (§5.1.1, §13: a
   missing key silently returns an empty frame from `_fetch_ces_releases:127-129`, so a
   soft fallback converts a missing secret into silent data loss).
2. **Fetch + tag + IND-IMD-1 drop** — call `_fetch_ces_releases()`
   (`releases.py:101`), which already fetches via the JSON API, tags
   `vintage_date`/`revision`/`benchmark_revision` from the advanced calendar, and
   carries the IND-IMD-1 `(rev1,bmr0)` drop (`releases.py:159-185`). The deferred
   import keeps `load_dotenv` (run by the §5.0/Phase-5 CLI callback) effective before
   the BLS clients bind.
3. **Remap** to `VINTAGE_STORE_SCHEMA` via `_remap_ces_to_store_schema`.
4. **Censor** `vintage_date <= as_of` (knowability cutoff).
5. **Detect** corrected levels per `(source=ces, sa)` partition (§5.1.4), warn-logging
   each.
6. **Append** via `append_to_vintage_store` then **compact** each touched
   `(ces, sa)` partition (§6.2 — `update` always compacts after append).
7. Return `CaptureResult(appended, corrected, skipped)` where
   `skipped = censored_rows - appended`.

Unit tests **monkeypatch `_fetch_ces_releases`** (no network, no key) and use a
tmp_path store. The live-fetch path is a separate `@pytest.mark.network` test.

- [ ] **Step 1: Write the failing test** — three unit tests + one network test.

```python
def test_capture_ces_raises_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("BLS_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="BLS_API_KEY"):
        capture_ces_print(date(2026, 2, 6), store_path=tmp_path)


def test_capture_ces_appends_and_censors(tmp_path, monkeypatch):
    monkeypatch.setenv("BLS_API_KEY", "dummy-key")

    # Two CES prints: one knowable as of 2026-02-06, one stamped later (must be
    # censored out by vintage_date <= as_of).
    fetched = _combined_frame([
        _combined_row(
            industry_type="domain", industry_code="05",
            ref_date=date(2026, 1, 1), vintage_date=date(2026, 2, 6),
            employment=131_000.0,
        ),
        _combined_row(
            industry_type="domain", industry_code="05",
            ref_date=date(2026, 2, 1), vintage_date=date(2026, 3, 6),
            employment=131_200.0,
        ),
    ])
    monkeypatch.setattr(_cap, "_fetch_ces_releases", lambda: fetched)

    result = capture_ces_print(date(2026, 2, 6), store_path=tmp_path)

    assert isinstance(result, CaptureResult)
    assert result.appended == 1
    assert result.corrected == []

    stored = pl.read_parquet(
        tmp_path / "source=ces" / "seasonally_adjusted=true" / "*.parquet"
    )
    assert stored.height == 1
    assert stored["ref_date"].to_list() == [date(2026, 1, 1)]
    assert stored["ownership"].to_list() == ["private"]


def test_capture_ces_idempotent_second_run_appends_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("BLS_API_KEY", "dummy-key")
    fetched = _combined_frame([
        _combined_row(
            industry_type="domain", industry_code="05",
            ref_date=date(2026, 1, 1), vintage_date=date(2026, 2, 6),
            employment=131_000.0,
        ),
    ])
    monkeypatch.setattr(_cap, "_fetch_ces_releases", lambda: fetched)

    first = capture_ces_print(date(2026, 2, 6), store_path=tmp_path)
    second = capture_ces_print(date(2026, 2, 6), store_path=tmp_path)

    assert first.appended == 1
    assert second.appended == 0
    assert second.skipped == 1


def test_capture_ces_flags_corrected_level(tmp_path, monkeypatch):
    monkeypatch.setenv("BLS_API_KEY", "dummy-key")

    base = _combined_row(
        industry_type="domain", industry_code="05",
        ref_date=date(2026, 1, 1), vintage_date=date(2026, 2, 6),
        employment=131_000.0,
    )
    monkeypatch.setattr(_cap, "_fetch_ces_releases", lambda: _combined_frame([base]))
    capture_ces_print(date(2026, 2, 6), store_path=tmp_path)

    # Re-capture the same ukey with a corrected level (a later vintage_date so it
    # is still censored in, but the same (ref,rev,bmr) ukey already present).
    corrected = dict(base)
    corrected["employment"] = 131_500.0
    corrected["vintage_date"] = date(2026, 2, 6)
    monkeypatch.setattr(
        _cap, "_fetch_ces_releases", lambda: _combined_frame([corrected])
    )
    result = capture_ces_print(date(2026, 2, 6), store_path=tmp_path)

    assert result.appended == 0
    assert len(result.corrected) == 1
    assert result.corrected[0].stored_employment == 131_000.0
    assert result.corrected[0].incoming_employment == 131_500.0


@pytest.mark.network
def test_capture_ces_live_fetch(tmp_path):
    import os

    if not os.environ.get("BLS_API_KEY"):
        pytest.skip("BLS_API_KEY not set")

    result = capture_ces_print(date.today(), store_path=tmp_path)

    assert isinstance(result, CaptureResult)
    assert result.appended >= 0
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::test_capture_ces_appends_and_censors -q --no-cov`
  Expected: FAIL with `ImportError: cannot import name 'capture_ces_print'` (function
  not yet defined; the file errors on collection until Step 3).

- [ ] **Step 3: Implement** — add `capture_ces_print` to `capture.py`. The
  `_fetch_ces_releases` import is module-level (it is pure-python, deferred network is
  inside it); the BLS-key guard fires before it is called.

```python
import os

from nfp_ingest.releases import _fetch_ces_releases
from nfp_ingest.vintage_store import append_to_vintage_store, compact_partition


def capture_ces_print(
    as_of: date,
    *,
    store_path: Path = VINTAGE_STORE_PATH,
) -> CaptureResult:
    """Capture the current CES print knowable as of ``as_of`` and append it.

    Spec §5.1: fetch the JSON-API current print (tagged + IND-IMD-1-dropped by
    :func:`nfp_ingest.releases._fetch_ces_releases`), remap to
    ``VINTAGE_STORE_SCHEMA``, censor ``vintage_date <= as_of``, flag corrected
    levels (§5.1.4), then ``append_to_vintage_store`` → ``compact_partition`` on
    each touched ``(ces, sa)`` partition.

    ``BLS_API_KEY`` is a **hard prerequisite** (§5.1.1, §13): without it the
    upstream fetch returns an empty frame, which would be a silent empty capture.

    Parameters
    ----------
    as_of : date
        Knowability cutoff. No row with ``vintage_date > as_of`` is appended.
    store_path : Path
        Root of the Hive-partitioned vintage store.

    Returns
    -------
    CaptureResult
        Rows appended, corrected-level records, and rows skipped (already stored).
    """
    if not os.environ.get("BLS_API_KEY"):
        raise RuntimeError(
            "BLS_API_KEY is required for CES capture (update --as-of). "
            "The BLS JSON API current-print fetch needs a key; without it the "
            "fetch returns an empty frame (silent data loss). Set BLS_API_KEY."
        )

    fetched = _fetch_ces_releases()
    if fetched.is_empty():
        logger.warning("CES fetch returned no rows for as_of=%s", as_of)
        return CaptureResult(appended=0, corrected=[], skipped=0)

    remapped = _remap_ces_to_store_schema(fetched)
    censored = remapped.filter(pl.col("vintage_date") <= as_of)
    if censored.is_empty():
        logger.warning("CES capture: no rows with vintage_date <= %s", as_of)
        return CaptureResult(appended=0, corrected=[], skipped=0)

    corrected: list[CorrectedLevel] = []
    touched: list[bool] = []
    for sa in censored["seasonally_adjusted"].unique().to_list():
        part = censored.filter(pl.col("seasonally_adjusted") == sa)
        cl = _detect_corrected_levels(
            part, store_path, source="ces", seasonally_adjusted=sa
        )
        for c in cl:
            logger.warning(
                "CORRECTED-LEVEL ces sa=%s ref=%s code=%s rev=%s bmr=%s "
                "stored=%s incoming=%s",
                sa, c.ref_date, c.industry_code, c.revision,
                c.benchmark_revision, c.stored_employment, c.incoming_employment,
            )
        corrected.extend(cl)
        touched.append(sa)

    appended = append_to_vintage_store(censored, store_path)
    for sa in touched:
        compact_partition(store_path, source="ces", seasonally_adjusted=sa)

    return CaptureResult(
        appended=appended,
        corrected=corrected,
        skipped=censored.height - appended,
    )
```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py -q --no-cov -m "not network"`
  Expected: PASS (all unit tests; the `network` test self-skips/deselects). Then the
  full ingest suite + lint:
  `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/ -q --no-cov -m "not network"`
  and `uv run ruff check packages/nfp-ingest/src/nfp_ingest/`.

- [ ] **Step 5: Commit** —
  `git add packages/nfp-ingest/src/nfp_ingest/capture.py packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py`
  then `git commit -m "feat(ingest): capture_ces_print month-T CES capture adapter (spec §5.1)"`


---

## Phase 5 — `update` command, `_run_*` helpers, snapshot day-12 fix, and the guardrail suite

**Spec:** §5, §5.0, §5.3, §6.2, §4a, §7. **Depends on:** Phase 3 (`advance_release_calendar`),
Phase 4 (`capture_ces_print`/`CaptureResult`), Phase 6 (`capture_qcew_quarter`). **Provides for
Phase 8:** the plain `_run_update` / `_run_snapshot` helpers `watch` calls (so a Typer-decorated
command is never invoked directly).

This phase has two arms: **5.1–5.3** build the `update` command + `_run_*` helpers + the
`snapshot` day-12 fix; **5.4–5.7** build the `test_update_guardrail.py` suite (§7). All new code
lives in `packages/nfp-vintages/src/nfp_vintages/__main__.py`; tests in
`packages/nfp-vintages/src/nfp_vintages/tests/`. Every import inside a command body is
**deferred** (the Typer callback runs `load_dotenv()` before `VINTAGE_STORE_PATH` resolves at
`nfp_lookups.paths` import — `paths.py:155`).

---

### Task 5.1: Extract plain `_run_snapshot` helper + fix the snapshot day-12 convention (§4a)

The current `snapshot` command (`__main__.py:291-324`) validates `start.day == 12` **only when
`--grid-end` is None** (`__main__.py:308-311`) and the grid loop seeds `_date(y, m, 12)`
ignoring `start.day` (`__main__.py:316-320`) — so `snapshot --as-of 2026-03-05 --grid-end …`
silently snapshots `2026-03-12`. Move the real work into a plain `_run_snapshot` helper that
validates **both** paths, and make the command a thin wrapper.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py` (new)

- [ ] **Step 1: Write the failing test**

```python
"""CLI tests for the production surface (snapshot day-12, update orchestration).

Phase 5 of specs/cli_production_workflow.md. Uses Typer's CliRunner with
deferred-import command bodies monkeypatched so no network/store/key is touched.
"""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from nfp_vintages.__main__ import app

runner = CliRunner()


class TestSnapshotDay12:
    def test_grid_mode_rejects_non_12th_as_of(self):
        # Today this silently snapshots 2026-03-12; it must be rejected.
        result = runner.invoke(
            app, ["snapshot", "--as-of", "2026-03-05", "--grid-end", "2026-06-12"]
        )
        assert result.exit_code != 0
        assert "12th" in result.output or "day-12" in result.output

    def test_single_mode_rejects_non_12th_as_of(self):
        result = runner.invoke(app, ["snapshot", "--as-of", "2026-03-05"])
        assert result.exit_code != 0
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestSnapshotDay12 -q --no-cov`
Expected: FAIL — `test_grid_mode_rejects_non_12th_as_of` returns exit_code 0 (the grid path
accepts the non-12th `--as-of` today).

- [ ] **Step 3: Implement** — add `_run_snapshot` and rewrite the `snapshot` command. Add
  `from datetime import date` to the module imports (top of file) if not present.

```python
def _run_snapshot(as_of: date, grid_end: date | None = None) -> None:
    """Write hash-pinned ModelData snapshot(s); plain helper (no Typer types)."""
    from nfp_ingest.snapshots import snapshot_model_data

    if as_of.day != 12:
        raise ValueError("--as-of must fall on the 12th (day-12 convention)")

    if grid_end is None:
        dates = [as_of]
    else:
        dates = []
        y, m = as_of.year, as_of.month
        while date(y, m, 12) <= grid_end:
            dates.append(date(y, m, 12))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    for d in dates:
        path, digest = snapshot_model_data(d)
        print(f"  {d}: {path} (hash {digest[:12]})")


@app.command()
def snapshot(
    as_of: str = typer.Option(..., "--as-of", help="Knowledge cutoff, YYYY-MM-DD (day-12)."),
    grid_end: str | None = typer.Option(
        None, "--grid-end", help="If set, snapshot every month's 12th from --as-of through here."
    ),
) -> None:
    """Write hash-pinned ModelData snapshot(s) for the given as-of date(s)."""
    from datetime import date as _date

    end = _date.fromisoformat(grid_end) if grid_end is not None else None
    try:
        _run_snapshot(_date.fromisoformat(as_of), end)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--as-of") from exc
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestSnapshotDay12 -q --no-cov`
Expected: PASS. Then `uv run ruff check packages/nfp-vintages` (clean).

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py
git commit -m "fix(cli): enforce day-12 on snapshot --as-of in both single and grid paths"
```

---

### Task 5.2: The `update` command + `_run_update` plain helper

`_run_update` orchestrates §5.0 (calendar advance) → §5.1 (CES capture) → §5.2 (QCEW, gated) →
§5.3 (indicators refresh), with all imports deferred. The `update` command is a thin wrapper.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py`

- [ ] **Step 1: Write the failing test**

```python
class TestUpdateOrchestration:
    def test_update_runs_calendar_then_ces_then_indicators(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar",
            lambda: calls.append("calendar"),
        )

        class _Res:
            appended, corrected, skipped = 3, [], 0

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: calls.append("ces") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter",
            lambda as_of, *, store_path=None: calls.append("qcew") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.indicators.download_indicators",
            lambda: calls.append("indicators") or {},
        )

        result = runner.invoke(app, ["update", "--as-of", "2026-06-12"])
        assert result.exit_code == 0, result.output
        assert calls == ["calendar", "ces", "qcew", "indicators"]

    def test_only_ces_skips_indicators_and_qcew(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: calls.append("calendar")
        )

        class _Res:
            appended, corrected, skipped = 1, [], 0

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: calls.append("ces") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.indicators.download_indicators",
            lambda: calls.append("indicators"),
        )
        result = runner.invoke(app, ["update", "--as-of", "2026-06-12", "--only", "ces"])
        assert result.exit_code == 0, result.output
        assert calls == ["calendar", "ces"]

    def test_no_refresh_calendar_skips_scrape(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: calls.append("calendar")
        )

        class _Res:
            appended, corrected, skipped = 0, [], 1

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter",
            lambda as_of, *, store_path=None: _Res(),
        )
        monkeypatch.setattr("nfp_ingest.indicators.download_indicators", lambda: {})
        result = runner.invoke(
            app, ["update", "--as-of", "2026-06-12", "--no-refresh-calendar"]
        )
        assert result.exit_code == 0, result.output
        assert "calendar" not in calls
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestUpdateOrchestration -q --no-cov`
Expected: FAIL — `No such command 'update'` (the command does not exist yet).

- [ ] **Step 3: Implement** — add `_run_update` and the `update` command. `--only` accepts
  `ces|qcew|indicators` (None ⇒ all). The CES/QCEW captures already append+compact their own
  touched partitions internally (Phase 4/6); `_run_update` reports the `CaptureResult` and
  surfaces any corrected-level warnings.

```python
def _run_update(
    as_of: date,
    *,
    only: str | None = None,
    refresh_calendar: bool = True,
    store_path=None,
) -> None:
    """Capture month-T prints into the store; plain helper (no Typer types)."""
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    store_path = store_path if store_path is not None else VINTAGE_STORE_PATH

    if refresh_calendar:
        from nfp_vintages.calendar import advance_release_calendar

        advance_release_calendar()

    if only in (None, "ces"):
        from nfp_ingest.capture import capture_ces_print

        res = capture_ces_print(as_of, store_path=store_path)
        print(f"  CES: appended {res.appended}, skipped {res.skipped}")
        for c in res.corrected:
            print(f"  CORRECTED-LEVEL ces {c.ref_date} {c.industry_code} "
                  f"rev{c.revision}/bmr{c.benchmark_revision}: "
                  f"{c.stored_employment} -> {c.incoming_employment}")

    if only in (None, "qcew"):
        from nfp_ingest.capture import capture_qcew_quarter

        res = capture_qcew_quarter(as_of, store_path=store_path)
        print(f"  QCEW: appended {res.appended}, skipped {res.skipped}")
        for c in res.corrected:
            print(f"  CORRECTED-LEVEL qcew {c.ref_date} {c.industry_code} "
                  f"rev{c.revision}/bmr{c.benchmark_revision}: "
                  f"{c.stored_employment} -> {c.incoming_employment}")

    if only in (None, "indicators"):
        from nfp_ingest.indicators import download_indicators

        results = download_indicators()
        total = sum(results.values()) if results else 0
        print(f"  Indicators: {total} rows across {len(results or {})} series")


@app.command()
def update(
    as_of: str = typer.Option(..., "--as-of", help="Knowledge cutoff, YYYY-MM-DD."),
    only: str | None = typer.Option(
        None, "--only", help="Limit to one source: ces | qcew | indicators."
    ),
    no_refresh_calendar: bool = typer.Option(
        False, "--no-refresh-calendar", help="Skip the release-calendar scrape (assume current)."
    ),
) -> None:
    """Advance the calendar, capture month-T prints, and append them to the store."""
    from datetime import date as _date

    if only is not None and only not in ("ces", "qcew", "indicators"):
        raise typer.BadParameter("must be ces, qcew, or indicators", param_hint="--only")
    _run_update(
        _date.fromisoformat(as_of), only=only, refresh_calendar=not no_refresh_calendar
    )
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py -q --no-cov`
Expected: PASS (all `TestUpdateOrchestration` + `TestSnapshotDay12`). Then
`uv run ruff check packages/nfp-vintages`.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py
git commit -m "feat(cli): add update command (calendar advance + CES/QCEW capture + indicators)"
```

---

### Task 5.3: `_run_update` self-healing compaction of fragmented partitions (§6.2)

If a prior `update` crashed between `append_to_vintage_store` and `compact_partition`, a
partition is left with >1 fragment (order-sensitive, read-amplifying). `_run_update` must
compact any touched-source partition that has more than one parquet file, regardless of whether
this run appended — cheap and idempotent.

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py`

- [ ] **Step 1: Write the failing test**

```python
class TestUpdateSelfHealingCompaction:
    def test_update_compacts_pre_existing_fragments(self, tmp_path, monkeypatch):
        import polars as pl
        from nfp_ingest.vintage_store import append_to_vintage_store

        store = tmp_path / "store"
        # Two disjoint appends → two fragment files in the same (ces, True) partition.
        from nfp_vintages.tests._fixtures import make_ces_rows  # Task 5.7 helper

        append_to_vintage_store(make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06"), store)
        append_to_vintage_store(make_ces_rows(ref_month="2026-02-12", vintage="2026-03-06"), store)
        part = store / "source=ces" / "seasonally_adjusted=true"
        assert len(list(part.glob("*.parquet"))) == 2

        # update with everything stubbed except the heal pass; capture appends nothing.
        monkeypatch.setattr("nfp_vintages.calendar.advance_release_calendar", lambda: None)

        class _Res:
            appended, corrected, skipped = 0, [], 1

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print", lambda a, *, store_path=None: _Res()
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter", lambda a, *, store_path=None: _Res()
        )
        monkeypatch.setattr("nfp_ingest.indicators.download_indicators", lambda: {})

        from nfp_vintages.__main__ import _run_update

        _run_update(date(2026, 6, 12), store_path=store)
        assert len(list(part.glob("*.parquet"))) == 1
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestUpdateSelfHealingCompaction -q --no-cov`
Expected: FAIL — two fragment files remain (no heal pass yet).

- [ ] **Step 3: Implement** — append a heal pass at the end of `_run_update` that compacts any
  `(source, seasonally_adjusted)` partition holding >1 parquet file.

```python
    # self-healing: compact any partition left fragmented by a prior crashed run
    from nfp_ingest.vintage_store import compact_partition
    from nfp_lookups.paths import is_remote

    if not is_remote(store_path):
        for source_dir in sorted((store_path).glob("source=*")):
            source = source_dir.name.split("=", 1)[1]
            for sa_dir in sorted(source_dir.glob("seasonally_adjusted=*")):
                if len(list(sa_dir.glob("*.parquet"))) > 1:
                    sa = sa_dir.name.split("=", 1)[1] == "true"
                    compact_partition(store_path, source, sa)
```

(Remote stores enumerate partitions via `UPath`; the implementer threads `storage_options_for`
through the glob — `compact_partition` already handles remote deletes. The local guard keeps the
test hermetic.)

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py -q --no-cov`
Expected: PASS. Then `uv run ruff check packages/nfp-vintages`.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py
git commit -m "feat(cli): self-heal fragmented partitions in update (compact >1-file partitions)"
```

---

### Task 5.4: Guardrail — idempotence (`test_update_guardrail.py`)

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py` (shared synthetic rows)
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py`

- [ ] **Step 1: Write the failing test** — the fixtures module does not exist yet, so the import
  fails first.

```python
"""§7 guardrail: a month-T capture must be idempotent and must not perturb existing months."""

from __future__ import annotations

import polars as pl

from nfp_ingest.vintage_store import (
    append_to_vintage_store,
    compact_partition,
    read_vintage_store,
)
from nfp_vintages.tests._fixtures import make_ces_rows


def _relation(store) -> dict:
    """Map the dedup ukey -> employment for the (ces, True) partition."""
    lf = read_vintage_store(store, source="ces", seasonally_adjusted=True)
    df = lf.collect()
    key_cols = [
        "ref_date", "industry_type", "industry_code", "geographic_type",
        "geographic_code", "revision", "benchmark_revision", "ownership",
        "size_class_type", "size_class_code",
    ]
    return {
        tuple(r[c] for c in key_cols): r["employment"]
        for r in df.iter_rows(named=True)
    }


class TestIdempotence:
    def test_capture_append_compact_twice_same_relation(self, tmp_path):
        store = tmp_path / "store"
        rows = make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06")

        append_to_vintage_store(rows, store)
        compact_partition(store, "ces", True)
        first = _relation(store)

        # second run: re-append identical rows must add 0, compact must be a no-op
        added = append_to_vintage_store(rows, store)
        compact_partition(store, "ces", True)
        second = _relation(store)

        assert added == 0
        assert first == second

    def test_same_ukey_later_vintage_does_not_change_min_vintage_level(self, tmp_path):
        store = tmp_path / "store"
        early = make_ces_rows(
            ref_month="2026-01-12", vintage="2026-02-06", employment=150_000.0
        )
        late = make_ces_rows(
            ref_month="2026-01-12", vintage="2026-03-06", employment=151_000.0
        )
        append_to_vintage_store(early, store)
        append_to_vintage_store(late, store)
        compact_partition(store, "ces", True)
        rel = _relation(store)
        # compact keeps MIN(vintage_date) per ukey → the early (first real-time) level wins
        assert set(rel.values()) == {150_000.0}
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestIdempotence -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: nfp_vintages.tests._fixtures` (created in Task 5.7).

- [ ] **Step 3: Implement** — this task's *code* is the test; the fixture builder lands in Task
  5.7. To make 5.4 pass in isolation, add a minimal `make_ces_rows` to `_fixtures.py` now
  (Task 5.7 extends it):

```python
"""Synthetic VINTAGE_STORE_SCHEMA rows for guardrail tests (no store/network)."""

from __future__ import annotations

from datetime import date

import polars as pl

from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


def make_ces_rows(
    *,
    ref_month: str,
    vintage: str,
    employment: float = 150_000.0,
    industry_code: str = "05",
    revision: int = 0,
    benchmark_revision: int = 0,
    seasonally_adjusted: bool = True,
) -> pl.DataFrame:
    """One CES headline row in the rebuilt store schema."""
    ownership = "private" if industry_code == "05" else "total"
    row = {
        "source": "ces",
        "ref_date": date.fromisoformat(ref_month),
        "vintage_date": date.fromisoformat(vintage),
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "industry_type": "total",
        "industry_code": industry_code,
        "ownership": ownership,
        "size_class_type": None,
        "size_class_code": None,
        "geographic_type": "national",
        "geographic_code": "00",
        "seasonally_adjusted": seasonally_adjusted,
    }
    cols = list(VINTAGE_STORE_SCHEMA.keys())
    return pl.DataFrame([{c: row.get(c) for c in cols}], schema=VINTAGE_STORE_SCHEMA)
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestIdempotence -q --no-cov`
Expected: PASS. Then `uv run ruff check packages/nfp-vintages`.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py
git commit -m "test(cli): guardrail idempotence (append/compact twice == same ukey relation)"
```

---

### Task 5.5: Guardrail — first-print-unchanged

A capture must not perturb `first_print_changes()` / `wedge_first_print_changes()` for
already-present months. **Documented limitation:** this proves a capture is non-destructive but
**cannot** catch a dropped same-revision correction (§6.3) — that is the runtime `CORRECTED-LEVEL`
warning's job (Phase 4 Task 4.2).

**Files:**
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py`

- [ ] **Step 1: Write the failing test** (skips cleanly if the store helpers can't build a
  first-print window from the synthetic fixture — keep it self-contained with two consecutive
  months so `first_print_changes` has a prior partner).

```python
class TestFirstPrintUnchanged:
    def test_capture_does_not_move_existing_first_print(self, tmp_path):
        from nfp_ingest.first_print import first_print_changes
        from nfp_vintages.tests._fixtures import make_first_print_window

        store = tmp_path / "store"
        make_first_print_window(store)  # two months of rev0/rev1 rows, private '05'
        before = first_print_changes(store_path=store, industry_code="05")

        # a NEW, later month's capture must not change earlier months' first prints
        append_to_vintage_store(
            make_ces_rows(ref_month="2026-03-12", vintage="2026-04-03",
                          revision=0, employment=152_000.0),
            store,
        )
        compact_partition(store, "ces", True)
        after = first_print_changes(store_path=store, industry_code="05")

        common = before.join(after, on="period", how="inner", suffix="_after")
        assert (common["change_k"] == common["change_k_after"]).all()
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestFirstPrintUnchanged -q --no-cov`
Expected: FAIL — `make_first_print_window` not in `_fixtures` yet (added in Task 5.7); or a
column-name mismatch the implementer reconciles against `first_print.py:53-156` (the change
column may be `change_k` or `first_print_change_k` — verify and pin in the test).

- [ ] **Step 3: Implement** — no production code; the fixture `make_first_print_window` is added
  in Task 5.7. Reconcile the change-column name against `first_print.py` and the
  `first_print_changes` signature (`store_path=`, `industry_code=`) before pinning the assert.

- [ ] **Step 4: Run, verify pass** — after Task 5.7's fixture exists.

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestFirstPrintUnchanged -q --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py
git commit -m "test(cli): guardrail first-print-unchanged across a later-month capture"
```

---

### Task 5.6: Guardrail — calendar-not-advanced ⇒ loud failure

If the calendar does not cover `T`, the CES tag join returns null and (without a guard) every
row is censored out — a silent empty capture. `update` must fail loudly instead. (The raise
lives in `capture_ces_print`, Phase 4; this test pins the end-to-end behaviour through `update`.)

**Files:**
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py`

- [ ] **Step 1: Write the failing test**

```python
class TestCalendarNotAdvancedLoudFailure:
    def test_update_errors_when_calendar_missing_target(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from nfp_vintages.__main__ import app

        # advance_release_calendar is a no-op (simulating a stale/missing calendar),
        # and capture_ces_print raises because the tag join is empty for T.
        monkeypatch.setattr("nfp_vintages.calendar.advance_release_calendar", lambda: None)

        def _raise(as_of, *, store_path=None):
            raise RuntimeError(
                f"no vintage calendar rows for {as_of}; run with calendar advanced"
            )

        monkeypatch.setattr("nfp_ingest.capture.capture_ces_print", _raise)
        result = CliRunner().invoke(
            app, ["update", "--as-of", "2026-06-12", "--only", "ces", "--no-refresh-calendar"]
        )
        assert result.exit_code != 0
        assert "calendar" in (result.output + str(result.exception)).lower()
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestCalendarNotAdvancedLoudFailure -q --no-cov`
Expected: FAIL — `update` currently swallows or never raises (depends on Task 5.2's error
propagation). Confirm `_run_update` does **not** catch the capture exception.

- [ ] **Step 3: Implement** — ensure `_run_update` lets the capture exception propagate (do not
  wrap CES capture in a `try/except` that prints-and-continues). No new code if Task 5.2 already
  propagates; otherwise remove any swallowing. The matching raise in `capture_ces_print` is a
  Phase 4 cross-reference — add a note in Phase 4 Task 4.3 that an empty tag join for `T` must
  `raise RuntimeError` (not return an empty `CaptureResult`).

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestCalendarNotAdvancedLoudFailure -q --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py \
        packages/nfp-vintages/src/nfp_vintages/__main__.py
git commit -m "test(cli): guardrail loud-failure when the release calendar lacks the target month"
```

---

### Task 5.7: Guardrail fixtures — benchmark double-row, shutdown sentinel, overlap divergence

Extend `_fixtures.py` with the dangerous-edge builders and add the overlap-divergence diagnostic
(compares rev0/bmr0 + rev1/bmr0 **levels** between a synthetic bootstrap fixture and a capture,
**excluding** sentinel rows — flag, not assert-zero).

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py`

- [ ] **Step 1: Write the failing test**

```python
class TestDangerEdges:
    def test_benchmark_double_row_keeps_both_tracks(self, tmp_path):
        from nfp_vintages.tests._fixtures import make_benchmark_double_row

        store = tmp_path / "store"
        append_to_vintage_store(make_benchmark_double_row(ref_month="2025-12-12"), store)
        compact_partition(store, "ces", True)
        df = read_vintage_store(store, source="ces", seasonally_adjusted=True).collect()
        keys = set(zip(df["revision"].to_list(), df["benchmark_revision"].to_list()))
        assert (1, 0) in keys and (2, 1) in keys  # both coherent tracks survive

    def test_overlap_divergence_excludes_shutdown_sentinel(self, tmp_path):
        from nfp_vintages.tests._fixtures import make_shutdown_sentinel_row, make_ces_rows

        store = tmp_path / "store"
        append_to_vintage_store(make_shutdown_sentinel_row(ref_month="2025-10-12"), store)
        append_to_vintage_store(make_ces_rows(ref_month="2025-11-12", vintage="2025-12-05"), store)
        compact_partition(store, "ces", True)
        df = read_vintage_store(store, source="ces", seasonally_adjusted=True).collect()
        scored = df.filter(pl.col("employment") > 0)  # sentinel excluded
        assert (-1.0 not in scored["employment"].to_list())
        assert scored.height == 1  # only the real Nov row remains in scope
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py::TestDangerEdges -q --no-cov`
Expected: FAIL — `make_benchmark_double_row` / `make_shutdown_sentinel_row` not in `_fixtures`.

- [ ] **Step 3: Implement** — extend `_fixtures.py`.

```python
def make_benchmark_double_row(*, ref_month: str) -> pl.DataFrame:
    """One ref_date published as BOTH (rev1,bmr0) and (rev2,bmr1) on a Feb benchmark."""
    a = make_ces_rows(ref_month=ref_month, vintage="2026-02-06",
                      revision=1, benchmark_revision=0, employment=149_500.0)
    b = make_ces_rows(ref_month=ref_month, vintage="2026-02-06",
                      revision=2, benchmark_revision=1, employment=149_900.0)
    return pl.concat([a, b])


def make_shutdown_sentinel_row(*, ref_month: str) -> pl.DataFrame:
    """The -1.0 'no print' sentinel the rebuilt store writes for shutdown-skipped slots."""
    return make_ces_rows(
        ref_month=ref_month, vintage="2025-11-12", revision=0,
        benchmark_revision=0, employment=-1.0,
    )


def make_first_print_window(store) -> None:
    """Two consecutive months with a rev0 print + a prior-month rev1 partner."""
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06", revision=0,
                      employment=150_000.0), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-03-06", revision=1,
                      employment=150_300.0), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-02-12", vintage="2026-03-06", revision=0,
                      employment=150_800.0), store)
    compact_partition(store, "ces", True)
```

(Add `from nfp_ingest.vintage_store import append_to_vintage_store, compact_partition` to the top
of `_fixtures.py`.)

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py -q --no-cov`
Expected: PASS (all guardrail classes). Then `uv run ruff check packages/nfp-vintages` and the
package suite `uv run pytest packages/nfp-vintages -m "not network and not slow" --no-cov -q`.

- [ ] **Step 5: Commit**

```bash
git add packages/nfp-vintages/src/nfp_vintages/tests/_fixtures.py \
        packages/nfp-vintages/src/nfp_vintages/tests/test_update_guardrail.py
git commit -m "test(cli): guardrail fixtures (benchmark double-row, shutdown sentinel, overlap)"
```

---


## Phase 6 — QCEW conditional capture wired into `update`

> **Spec:** `specs/cli_production_workflow.md` §5.2 (QCEW conditional capture), §5.0 (calendar advance dependency), §6.1 (ukey now keys size_class), §6.3 (corrected-level warning). Build order item 6.
>
> **Prerequisites already landed in earlier phases (do not re-implement here):**
> - Phase 1 — `packages/nfp-ingest/src/nfp_ingest/qcew_acquire.py` exposes the **public** `acquire_qcew_levels(start_year, end_year=None)` and `acquire_qcew_size_native(start_year, end_year=None)` (verbatim relocation of `rebuild_store._acquire_qcew_levels` / `_acquire_qcew_size_native`). Both loop over **all quarters of every year** in `[start_year, end_year]` and tag `revision=0`.
> - Phase 2 — `append_to_vintage_store` / `compact_partition` ukey lists now include `"ownership"`, `"size_class_type"`, `"size_class_code"` (`vintage_store.py` §6.1), so distinct QCEW Q1 size buckets no longer collapse on append.
> - Phase 4 — `packages/nfp-ingest/src/nfp_ingest/capture.py` already defines `@dataclass CaptureResult(appended, corrected, skipped)`, `@dataclass CorrectedLevel(ref_date, industry_code, revision, benchmark_revision, stored_employment, incoming_employment)`, `capture_ces_print(...)`, and `_detect_corrected_levels(new_rows, store_path, source, seasonally_adjusted)`. **Phase 6 adds `capture_qcew_quarter` and `_knowable_qcew_quarter` to this same file and reuses those existing symbols — it does not redefine them.**
> - Phase 5 — `packages/nfp-vintages/src/nfp_vintages/__main__.py` already defines the `update` Typer command (calendar advance → CES capture → indicators → compact) with the `--only` option. **Phase 6 adds the QCEW leg to that existing command body.**
>
> **Key code facts grounding the implementation (verified read-only):**
> - `get_qcew_vintage_date(ref_quarter, ref_year, revision, calendar=None)` takes `ref_quarter` as the **string `'Q1'..'Q4'`** (`revision_schedules.py:299`). With the §5.0 calendar advanced it returns the exact `vintage_date`; without it, the lag fallback (`revision_schedules.py:342-365`) assigns a day-1 approximation.
> - `build_qcew_panel(raw)` returns 14-of-16 store columns but its final `.select(...)` **omits `size_class_type`/`size_class_code`** (`qcew_crosswalk.py:263-276`) — `capture_qcew_quarter` must null-fill those two before append. It stamps `source='qcew'`, `seasonally_adjusted=False`, `benchmark_revision=0`, `revision` from the raw rows.
> - `build_size_class_panel(native)` returns the full 14-column store schema **with non-null** `size_class_type`/`size_class_code` (`size_class.py:96-116`); Q1-only.
> - `acquire_qcew_levels` / `acquire_qcew_size_native` fetch **whole years**, so the single-quarter wrapper fetches the containing year then **filters** to the one quarter (`year`/`qtr` survive into the raw frame the crosswalk consumes via `_QCEW_LEVELS_REQUIRED`; the size endpoint URL is Q1-only by path).
> - `append_to_vintage_store` is anti-join idempotent and `compact_partition` keeps `MIN(vintage_date)` per ukey (`vintage_store.py:678-815`).

---

### Task 6.1: `capture_qcew_quarter` — knowable-quarter selection + single-quarter capture

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/capture.py`
- Test: `packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py` (extend; created in Phase 4)

This task adds two functions to the existing `capture.py`: `_knowable_qcew_quarter(as_of)` (the §5.2 knowable test) and `capture_qcew_quarter(as_of, *, store_path)` (fetch containing year → filter to one quarter → crosswalk → null-fill size cols → corrected-level check → append → compact). It reuses the Phase-4 `CaptureResult`, `CorrectedLevel`, and `_detect_corrected_levels` symbols already in the module.

- [ ] **Step 1: Write the failing test** — `_knowable_qcew_quarter` picks the most recent quarter whose rev-0 `vintage_date ≤ as_of`, and a no-new-quarter `as_of` returns `None`.

  Append to `packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py`:

  ```python
  from datetime import date

  import polars as pl
  import pytest

  from nfp_ingest import capture as _cap
  from nfp_ingest.capture import (
      CaptureResult,
      _knowable_qcew_quarter,
      capture_qcew_quarter,
  )
  from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


  # --- helpers -------------------------------------------------------------

  def _qcew_store_row(
      ref_date: date,
      vintage_date: date,
      *,
      industry_code: str = "05",
      industry_type: str = "total",
      ownership: str = "private",
      revision: int = 0,
      employment: float = 130_000.0,
      size_class_type: str | None = None,
      size_class_code: str | None = None,
  ) -> dict:
      """One VINTAGE_STORE_SCHEMA-conformant QCEW row (NSA)."""
      return {
          "geographic_type": "national",
          "geographic_code": "00",
          "ownership": ownership,
          "industry_type": industry_type,
          "industry_code": industry_code,
          "ref_date": ref_date,
          "vintage_date": vintage_date,
          "revision": revision,
          "benchmark_revision": 0,
          "employment": employment,
          "size_class_type": size_class_type,
          "size_class_code": size_class_code,
          "source": "qcew",
          "seasonally_adjusted": False,
      }


  def _write_qcew_partition(rows: list[dict], store_path) -> None:
      df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
      pdir = store_path / "source=qcew" / "seasonally_adjusted=false"
      pdir.mkdir(parents=True, exist_ok=True)
      df.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "data.parquet")


  def _qcew_panel_rows(ref_year: int, qtr: int, employment: float) -> pl.DataFrame:
      """Stand-in for build_qcew_panel output: 14-of-16 cols, NO size_class_*.

      Mirrors qcew_crosswalk.build_qcew_panel's final .select (which omits
      size_class_type/size_class_code) so capture_qcew_quarter's null-fill is
      exercised. vintage_date is whatever the real schedule assigns; the test
      monkeypatches the schedule lookup, so any placeholder date is fine here.
      """
      ref_month = (qtr - 1) * 3 + 1
      ref = date(ref_year, ref_month, 1)
      cols = [c for c in VINTAGE_STORE_SCHEMA if c not in ("size_class_type", "size_class_code")]
      return pl.DataFrame(
          {
              "geographic_type": ["national"],
              "geographic_code": ["00"],
              "ownership": ["private"],
              "industry_type": ["total"],
              "industry_code": ["05"],
              "ref_date": [ref],
              "vintage_date": [date(ref_year, ref_month + 5, 1)],
              "revision": [0],
              "benchmark_revision": [0],
              "employment": [employment],
              "source": ["qcew"],
              "seasonally_adjusted": [False],
          }
      ).select(cols)


  # --- _knowable_qcew_quarter ---------------------------------------------

  class TestKnowableQcewQuarter:
      def test_picks_most_recent_knowable_quarter(self, monkeypatch):
          # Q1-2024 rev0 published 2024-05-01; Q2-2024 rev0 published 2024-08-01.
          def fake_vdate(ref_quarter, ref_year, revision):
              table = {
                  ("Q1", 2024): date(2024, 5, 1),
                  ("Q2", 2024): date(2024, 8, 1),
                  ("Q3", 2024): date(2024, 11, 1),
              }
              return table.get((ref_quarter, ref_year), date(2099, 1, 1))

          monkeypatch.setattr(_cap, "get_qcew_vintage_date", fake_vdate)
          # As of 2024-06-01: Q1-2024 is knowable, Q2-2024 is not yet.
          assert _knowable_qcew_quarter(date(2024, 6, 1)) == ("Q1", 2024)

      def test_returns_none_when_no_quarter_knowable(self, monkeypatch):
          # Every candidate publishes in the far future ⇒ nothing knowable.
          monkeypatch.setattr(
              _cap, "get_qcew_vintage_date",
              lambda ref_quarter, ref_year, revision: date(2099, 1, 1),
          )
          assert _knowable_qcew_quarter(date(2024, 6, 1)) is None
  ```

  ```python
  # --- capture_qcew_quarter -----------------------------------------------

  class TestCaptureQcewQuarter:
      def test_no_new_quarter_returns_skipped_no_append(self, tmp_path, monkeypatch):
          # Store already holds Q1-2024; as-of makes Q1-2024 the newest knowable.
          _write_qcew_partition(
              [_qcew_store_row(date(2024, 1, 1), date(2024, 5, 1))], tmp_path
          )

          monkeypatch.setattr(
              _cap, "get_qcew_vintage_date",
              lambda ref_quarter, ref_year, revision: (
                  date(2024, 5, 1) if (ref_quarter, ref_year) == ("Q1", 2024)
                  else date(2099, 1, 1)
              ),
          )

          def _boom(*a, **k):  # acquire must NOT be called on a no-op
              raise AssertionError("acquire_qcew_levels called on a no-op month")

          monkeypatch.setattr(_cap, "acquire_qcew_levels", _boom)

          result = capture_qcew_quarter(date(2024, 6, 1), store_path=tmp_path)

          assert isinstance(result, CaptureResult)
          assert result.appended == 0
          assert result.skipped == 1
          assert result.corrected == []

      def test_knowable_new_quarter_appends_rev0(self, tmp_path, monkeypatch):
          # Empty store; Q1-2024 becomes knowable as of 2024-06-01.
          monkeypatch.setattr(
              _cap, "get_qcew_vintage_date",
              lambda ref_quarter, ref_year, revision: (
                  date(2024, 5, 1) if (ref_quarter, ref_year) == ("Q1", 2024)
                  else date(2099, 1, 1)
              ),
          )
          # acquire returns a raw frame; build_qcew_panel/build_size_class_panel
          # are monkeypatched to the test panel (no real crosswalk/network).
          monkeypatch.setattr(
              _cap, "acquire_qcew_levels",
              lambda start_year, end_year=None: pl.DataFrame({"year": [2024], "qtr": [1]}),
          )
          monkeypatch.setattr(
              _cap, "acquire_qcew_size_native",
              lambda start_year, end_year=None: pl.DataFrame({"year": [2024]}),
          )
          monkeypatch.setattr(
              _cap, "build_qcew_panel",
              lambda raw: _qcew_panel_rows(2024, 1, 130_000.0),
          )
          # Size leg disabled for this test (return an empty Q1 size frame).
          empty_size = pl.DataFrame(schema=VINTAGE_STORE_SCHEMA).filter(pl.lit(False))
          monkeypatch.setattr(_cap, "build_size_class_panel", lambda native: empty_size)

          result = capture_qcew_quarter(date(2024, 6, 1), store_path=tmp_path)

          assert result.skipped == 0
          assert result.appended == 1
          assert result.corrected == []

          stored = pl.read_parquet(
              tmp_path / "source=qcew" / "seasonally_adjusted=false" / "*.parquet"
          )
          assert stored.height == 1
          assert stored["revision"].to_list() == [0]
          assert stored["industry_code"].to_list() == ["05"]
          # null-fill of the missing size cols held:
          assert stored["size_class_type"].to_list() == [None]
          assert stored["size_class_code"].to_list() == [None]
  ```

- [ ] **Step 2: Run the test, verify it fails**
  - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::TestKnowableQcewQuarter -q --no-cov`
  - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::TestCaptureQcewQuarter -q --no-cov`
  - Expected: **FAIL** with `ImportError: cannot import name '_knowable_qcew_quarter'` / `'capture_qcew_quarter'` from `nfp_ingest.capture` (neither function exists yet).

- [ ] **Step 3: Implement** — add the two functions plus their module-level imports to `capture.py`.

  Add these imports near the top of `packages/nfp-ingest/src/nfp_ingest/capture.py` (module level, so the tests can `monkeypatch.setattr(_cap, ...)` each name):

  ```python
  from nfp_ingest.qcew_acquire import acquire_qcew_levels, acquire_qcew_size_native
  from nfp_ingest.qcew_crosswalk import build_qcew_panel
  from nfp_ingest.size_class import build_size_class_panel
  from nfp_ingest.vintage_store import (
      append_to_vintage_store,
      compact_partition,
  )
  from nfp_lookups.revision_schedules import get_qcew_vintage_date
  from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
  ```

  > If Phase 4 already imports any of `append_to_vintage_store` / `compact_partition` / `VINTAGE_STORE_SCHEMA` at module level, do **not** duplicate them — add only the names not already present.

  Append the functions to `capture.py`:

  ```python
  # ---------------------------------------------------------------------------
  # QCEW conditional quarter capture (spec §5.2)
  # ---------------------------------------------------------------------------

  # How far back to scan for a knowable quarter. QCEW rev-0 lags the reference
  # quarter by ~5 months, so 8 candidate quarters (2 years) always covers the
  # newest-knowable quarter for any monthly as-of.
  _QCEW_CANDIDATE_QUARTERS = 8


  def _knowable_qcew_quarter(as_of: date) -> tuple[str, int] | None:
      """Most recent QCEW quarter whose rev-0 vintage_date is <= ``as_of``.

      Iterates candidate ``(ref_quarter, ref_year)`` pairs newest-first and
      returns the first whose ``get_qcew_vintage_date(..., revision=0)`` is on
      or before ``as_of``. Returns ``None`` when no candidate is knowable yet
      (the steady-state monthly no-op — QCEW is quarterly, §5.2).

      Requires the §5.0 calendar to be advanced so the schedule returns real
      release dates rather than the day-1 lag fallback
      (``revision_schedules.py:342-365``).
      """
      # The quarter containing ``as_of`` cannot have been published yet, so start
      # from the previous quarter and walk back.
      q = (as_of.month - 1) // 3 + 1
      year = as_of.year
      # Step back one quarter to the newest possibly-published quarter.
      q -= 1
      if q == 0:
          q = 4
          year -= 1

      for _ in range(_QCEW_CANDIDATE_QUARTERS):
          ref_quarter = f"Q{q}"
          rev0_vdate = get_qcew_vintage_date(ref_quarter, year, 0)
          if rev0_vdate <= as_of:
              return (ref_quarter, year)
          q -= 1
          if q == 0:
              q = 4
              year -= 1
      return None


  def capture_qcew_quarter(
      as_of: date,
      *,
      store_path: Path = VINTAGE_STORE_PATH,
  ) -> CaptureResult:
      """Capture the newest knowable QCEW quarter and append it to the store.

      Most months this is a **no-op** (QCEW is quarterly): if no new quarter is
      knowable as of ``as_of``, or the newest knowable quarter is already in the
      store, returns ``CaptureResult(appended=0, corrected=[], skipped=1)``.

      Otherwise fetches the containing **year** via the relocated public
      acquire helpers (§5.2), filters to the single knowable quarter, runs the
      crosswalk (``build_qcew_panel`` for levels, ``build_size_class_panel`` for
      the Q1 size cross-product), null-fills the size columns the levels builder
      omits, censors ``vintage_date <= as_of``, runs the §6.3 corrected-level
      comparison, then appends and compacts the ``(qcew, seasonally_adjusted=
      False)`` partition. QCEW is NSA-only, so every row is tagged
      ``revision=0``/``seasonally_adjusted=False`` by the builders.

      Parameters
      ----------
      as_of : date
          Knowability cutoff. No row with ``vintage_date > as_of`` is appended.
      store_path : Path
          Root of the Hive-partitioned vintage store.

      Returns
      -------
      CaptureResult
          ``skipped=1`` (and ``appended=0``) when there is no new quarter to
          capture; otherwise the append count and any corrected-level warnings.
      """
      knowable = _knowable_qcew_quarter(as_of)
      if knowable is None:
          logger.info("QCEW: no knowable quarter as of %s — skipping", as_of)
          return CaptureResult(appended=0, corrected=[], skipped=1)

      ref_quarter, ref_year = knowable
      qtr = int(ref_quarter[1])

      # Fetch the containing YEAR (the helpers loop over full years), then filter
      # to the one knowable quarter. The levels endpoint carries year+qtr; the
      # size endpoint is Q1-only by URL path, so the size leg only runs for Q1.
      raw_levels = acquire_qcew_levels(ref_year, ref_year)
      raw_levels_q = raw_levels.filter(
          (pl.col("year").cast(pl.Int64) == ref_year)
          & (pl.col("qtr").cast(pl.Int64) == qtr)
      )
      levels = build_qcew_panel(raw_levels_q)
      # build_qcew_panel's .select omits size_class_* (qcew_crosswalk.py:263-276);
      # the store schema requires them, so null-fill before append.
      levels = levels.with_columns(
          size_class_type=pl.lit(None, pl.Utf8),
          size_class_code=pl.lit(None, pl.Utf8),
      )

      parts: list[pl.DataFrame] = [levels]
      if qtr == 1:
          raw_size = acquire_qcew_size_native(ref_year, ref_year)
          size = build_size_class_panel(raw_size)
          if size.height:
              parts.append(size)

      new_rows = (
          pl.concat(parts, how="diagonal_relaxed")
          .select(list(VINTAGE_STORE_SCHEMA.keys()))
          .cast(dict(VINTAGE_STORE_SCHEMA))
      )

      # Censor to the knowability cutoff.
      new_rows = new_rows.filter(pl.col("vintage_date") <= as_of)
      if new_rows.height == 0:
          logger.info(
              "QCEW: %s %d knowable but no rows survive vintage_date <= %s",
              ref_quarter, ref_year, as_of,
          )
          return CaptureResult(appended=0, corrected=[], skipped=1)

      # §6.3 corrected-level comparison BEFORE the append anti-join.
      corrected = _detect_corrected_levels(
          new_rows, store_path, source="qcew", seasonally_adjusted=False
      )
      for c in corrected:
          logger.warning(
              "CORRECTED-LEVEL qcew %s rev=%d bmr=%d: stored=%.1f incoming=%.1f",
              c.ref_date, c.revision, c.benchmark_revision,
              c.stored_employment, c.incoming_employment,
          )

      appended = append_to_vintage_store(new_rows, store_path)
      compact_partition(store_path, source="qcew", seasonally_adjusted=False)

      skipped = 0 if appended else 1
      logger.info(
          "QCEW: captured %s %d — appended %d rows (%d corrected)",
          ref_quarter, ref_year, appended, len(corrected),
      )
      return CaptureResult(appended=appended, corrected=corrected, skipped=skipped)
  ```

  > `VINTAGE_STORE_PATH`, `Path`, `logger`, `CaptureResult`, `CorrectedLevel`, and `_detect_corrected_levels` are already defined/imported in `capture.py` by Phase 4 — reuse them.

- [ ] **Step 4: Run, verify pass**
  - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::TestKnowableQcewQuarter -q --no-cov` → Expected: **PASS** (2 tests).
  - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py::TestCaptureQcewQuarter -q --no-cov` → Expected: **PASS** (2 tests).
  - Run the package suite + lint:
    - `uv run pytest packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py -q --no-cov` → Expected: **PASS** (Phase-4 CES tests + Phase-6 QCEW tests).
    - `uv run ruff check packages/nfp-ingest/src/nfp_ingest/capture.py packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py` → Expected: clean.

- [ ] **Step 5: Commit**
  ```bash
  git add packages/nfp-ingest/src/nfp_ingest/capture.py \
          packages/nfp-ingest/src/nfp_ingest/tests/test_capture.py
  git commit -m "feat(ingest): add capture_qcew_quarter conditional quarter capture

Iterate candidate quarters, pick the most recent whose rev-0 vintage_date
<= as_of; fetch the containing year via acquire_qcew_levels then filter to
the one quarter (Q1-only for the size cross-product). Null-fill the
size_class_* columns build_qcew_panel omits, run the §6.3 corrected-level
check, then append + compact. Returns skipped when no new quarter is
knowable (the steady-state monthly no-op). Spec §5.2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

### Task 6.2: Wire QCEW capture into the `update` command (`--only qcew` + the all path)

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py` (extend the Phase-5 `update` command body)
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py` (extend; created in Phase 5)

Phase 5 built `update` with the calendar-advance → CES → indicators → compact flow and the `--only` gate. Phase 6 adds the QCEW leg: `capture_qcew_quarter(as_of, store_path=...)` runs when `--only` is `None` (the all path) or `qcew`. All imports stay **deferred inside the command body** (per the CONTRACT: `load_dotenv` runs in the app callback before `VINTAGE_STORE_PATH` resolves at import).

- [ ] **Step 1: Write the failing test** — `update --only qcew` invokes `capture_qcew_quarter` with the parsed as-of and does **not** invoke the CES/indicator legs; the all path invokes QCEW too.

  Append to `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py`:

  ```python
  from datetime import date

  from typer.testing import CliRunner

  from nfp_vintages.__main__ import app

  runner = CliRunner()


  class TestUpdateQcewLeg:
      def test_only_qcew_calls_qcew_capture_not_ces(self, monkeypatch):
          calls: dict[str, object] = {}

          import nfp_ingest.capture as _cap
          import nfp_vintages.calendar as _cal

          monkeypatch.setattr(_cal, "advance_release_calendar", lambda: None)

          def fake_qcew(as_of, *, store_path):
              calls["qcew_as_of"] = as_of
              from nfp_ingest.capture import CaptureResult
              return CaptureResult(appended=0, corrected=[], skipped=1)

          def fake_ces(as_of, *, store_path):
              calls["ces_called"] = True
              from nfp_ingest.capture import CaptureResult
              return CaptureResult(appended=0, corrected=[], skipped=0)

          monkeypatch.setattr(_cap, "capture_qcew_quarter", fake_qcew)
          monkeypatch.setattr(_cap, "capture_ces_print", fake_ces)

          result = runner.invoke(
              app, ["update", "--as-of", "2024-06-12", "--only", "qcew"]
          )

          assert result.exit_code == 0, result.output
          assert calls["qcew_as_of"] == date(2024, 6, 12)
          assert "ces_called" not in calls  # --only qcew skips the CES leg

      def test_all_path_calls_qcew_capture(self, monkeypatch):
          calls: dict[str, object] = {}

          import nfp_ingest.capture as _cap
          import nfp_ingest.indicators as _ind
          import nfp_vintages.calendar as _cal

          monkeypatch.setattr(_cal, "advance_release_calendar", lambda: None)
          monkeypatch.setattr(_ind, "download_indicators", lambda: None)

          from nfp_ingest.capture import CaptureResult

          monkeypatch.setattr(
              _cap, "capture_ces_print",
              lambda as_of, *, store_path: CaptureResult(0, [], 0),
          )

          def fake_qcew(as_of, *, store_path):
              calls["qcew_called"] = True
              return CaptureResult(appended=0, corrected=[], skipped=1)

          monkeypatch.setattr(_cap, "capture_qcew_quarter", fake_qcew)

          result = runner.invoke(
              app, ["update", "--as-of", "2024-06-12", "--no-refresh-calendar"]
          )

          assert result.exit_code == 0, result.output
          assert calls.get("qcew_called") is True
  ```

  > The `--no-refresh-calendar` flag in the all-path test sidesteps the network calendar scrape; the QCEW leg still runs because the all path is gated only on `--only`, not on the calendar flag. The `_cal`/`_ind` monkeypatches keep the test fully offline.

- [ ] **Step 2: Run the test, verify it fails**
  - `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestUpdateQcewLeg -q --no-cov`
  - Expected: **FAIL** — `KeyError: 'qcew_as_of'` (the Phase-5 `update` body has no QCEW leg yet, so `capture_qcew_quarter` is never called and `calls` stays empty).

- [ ] **Step 3: Implement** — add the QCEW leg to the existing `update` command body in `__main__.py`.

  In the Phase-5 `update` command, locate the source-capture block (the CES leg gated by `--only`). Add the QCEW leg alongside it. The deferred-import + `--only` pattern looks like this (showing the relevant slice of the command body; keep the calendar-advance, CES, indicators, and compact steps Phase 5 already wrote):

  ```python
  @app.command()
  def update(
      as_of: str = typer.Option(..., "--as-of", help="Knowability cutoff, YYYY-MM-DD."),
      only: str | None = typer.Option(
          None, "--only", help="Restrict to one source: ces | qcew | indicators."
      ),
      no_refresh_calendar: bool = typer.Option(
          False, "--no-refresh-calendar", help="Skip the §5.0 calendar scrape."
      ),
  ) -> None:
      """Advance the calendar, capture knowable prints for ``as_of``, append."""
      from datetime import date as _date

      from nfp_ingest import capture as _capture
      from nfp_ingest import indicators as _indicators
      from nfp_lookups.paths import VINTAGE_STORE_PATH
      from nfp_vintages import calendar as _calendar

      cutoff = _date.fromisoformat(as_of)

      # §5.0 — advance the vintage calendar to ``as_of`` before any capture.
      if not no_refresh_calendar:
          _calendar.advance_release_calendar()

      # CES leg (Phase 5) — runs on the all path or with --only ces.
      if only in (None, "ces"):
          ces_result = _capture.capture_ces_print(cutoff, store_path=VINTAGE_STORE_PATH)
          typer.echo(
              f"CES: appended {ces_result.appended}, "
              f"skipped {ces_result.skipped}, "
              f"corrected {len(ces_result.corrected)}"
          )

      # QCEW leg (Phase 6) — runs on the all path or with --only qcew.
      # Most months this is a no-op (QCEW is quarterly): capture_qcew_quarter
      # returns skipped=1 when no new quarter is knowable (§5.2).
      if only in (None, "qcew"):
          qcew_result = _capture.capture_qcew_quarter(
              cutoff, store_path=VINTAGE_STORE_PATH
          )
          typer.echo(
              f"QCEW: appended {qcew_result.appended}, "
              f"skipped {qcew_result.skipped}, "
              f"corrected {len(qcew_result.corrected)}"
          )

      # Indicators leg (Phase 5) — a full FRED refresh, not an append (§5.3).
      if only in (None, "indicators"):
          _indicators.download_indicators()
          typer.echo("Indicators: refreshed")
  ```

  > **Only add the QCEW `if only in (None, "qcew"):` block** if Phase 5's body does not already contain it. Do not duplicate the calendar/CES/indicator steps — they are shown only for placement context. The corrected-level rows surface in `qcew_result.corrected` for the §6.3 warning (already logged inside `capture_qcew_quarter`); the `typer.echo` line reports the count to the operator.
  >
  > Phase 5 owns the post-capture `compact_partition` sweep for touched partitions; `capture_qcew_quarter` already compacts the `(qcew, False)` partition itself (Task 6.1), so no extra QCEW compaction is needed here.

- [ ] **Step 4: Run, verify pass**
  - `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py::TestUpdateQcewLeg -q --no-cov` → Expected: **PASS** (2 tests).
  - Run the full update-CLI test file + lint:
    - `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py -q --no-cov` → Expected: **PASS** (Phase-5 CES/indicator tests + Phase-6 QCEW tests).
    - `uv run ruff check packages/nfp-vintages/src/nfp_vintages/__main__.py packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py` → Expected: clean.

- [ ] **Step 5: Commit**
  ```bash
  git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
          packages/nfp-vintages/src/nfp_vintages/tests/test_cli_update.py
  git commit -m "feat(cli): wire QCEW conditional capture into update

Add the QCEW leg to alt-nfp update: capture_qcew_quarter runs on the all
path and with --only qcew (a no-op most months, since QCEW is quarterly).
Imports stay deferred inside the command body so load_dotenv resolves
VINTAGE_STORE_PATH first. Spec §5.2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

**Phase 6 done-when:** `capture_qcew_quarter` selects the newest knowable quarter (rev-0 `vintage_date ≤ as_of`), fetches the containing year and filters to that one quarter, null-fills the size columns the levels builder omits, runs the §6.3 corrected-level comparison, and appends+compacts the `(qcew, False)` partition — returning `skipped=1` on the steady-state monthly no-op. The `update` command invokes it on the all path and under `--only qcew`, with all imports deferred. No firewall path (`transform_to_panel`, `build_model_data`, A1/A2/A3 goldens, `nfp-model/*`) is touched.

---


## Phase 7 — `status` command + `store_status.py` (spec §8)

A cheap, read-only health + knowability report built **only** on `read_vintage_store`
(`packages/nfp-ingest/src/nfp_ingest/vintage_store.py:336-406`) — never `transform_to_panel`
(the expensive growth/censoring path) and never `views.py`. Three deliverables:

1. **Task 7.1** — `PartitionCoverage` + `StoreStatus` dataclasses and `compute_status` per-`(source,
   seasonally_adjusted)` coverage (raw row presence, no `employment > 0` filter so the Oct-2025
   `-1` sentinel counts as present — the rationale at `first_print.py:79-84`).
2. **Task 7.2** — the forward **UNCAPTURED** alarm + **missing-month** list (headline series only)
   folded into `compute_status`, plus `format_status` rendering incl. the resolved-URI header and
   the `.env` LOCAL-FALLBACK warning.
3. **Task 7.3** — the `status` Typer command in `__main__.py` with all imports **deferred** inside
   the body (so `load_dotenv()` in the app callback runs before `VINTAGE_STORE_PATH` resolves —
   `paths.py:155` binds at import).

Firewall: this phase never imports or calls `transform_to_panel`, `build_model_data`,
`first_print.py`, or any A1/A2/A3 golden path. Store-writing is **not** done here; all tests build a
synthetic `tmp_path` store with `pl.DataFrame.write_parquet` into the Hive layout and read it back —
never a real MinIO store.

---

### Task 7.1: `compute_status` per-`(source, seasonally_adjusted)` coverage via `read_vintage_store`

**Files:**
- Create: `packages/nfp-vintages/src/nfp_vintages/store_status.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py`

- [ ] **Step 1: Write the failing test** — assert `compute_status` reports one `PartitionCoverage`
  per `(source, seasonally_adjusted)` partition, with raw row presence (the Oct-2025 `-1` sentinel
  counts toward coverage; no `employment > 0` filter), computed straight off `read_vintage_store`.

  ```python
  # packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  """Tests for the read-only `status` report (spec §8).

  All store I/O is against a synthetic Hive-partitioned tmp_path store built by
  ``_write_store_rows`` below — NEVER a real MinIO store (conftest auto-loads prod
  creds). ``compute_status`` reads via ``read_vintage_store`` and must never call
  ``transform_to_panel``.
  """

  from __future__ import annotations

  from datetime import date

  import polars as pl
  from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
  from nfp_vintages.store_status import (
      PartitionCoverage,
      StoreStatus,
      compute_status,
  )


  def _row(
      *,
      source: str,
      sa: bool,
      ref_date: date,
      vintage_date: date,
      revision: int = 0,
      benchmark_revision: int = 0,
      employment: float = 100.0,
      industry_code: str = "00",
      geographic_code: str = "00",
  ) -> dict:
      """One VINTAGE_STORE_SCHEMA row as a dict (defaults = national total headline)."""
      return {
          "geographic_type": "national",
          "geographic_code": geographic_code,
          "ownership": "total",
          "industry_type": "total",
          "industry_code": industry_code,
          "ref_date": ref_date,
          "vintage_date": vintage_date,
          "revision": revision,
          "benchmark_revision": benchmark_revision,
          "employment": employment,
          "size_class_type": None,
          "size_class_code": None,
          "source": source,
          "seasonally_adjusted": sa,
      }


  def _write_store_rows(store_path, rows: list[dict]) -> None:
      """Write rows into the Hive layout the store reader expects.

      Partitions on (source, seasonally_adjusted); the partition columns are
      encoded in the directory names (Hive), so they are dropped from the file.
      """
      df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
      for (source, sa), part in df.group_by(["source", "seasonally_adjusted"]):
          part_dir = (
              store_path
              / f"source={source}"
              / f"seasonally_adjusted={str(sa).lower()}"
          )
          part_dir.mkdir(parents=True, exist_ok=True)
          part.drop("source", "seasonally_adjusted").write_parquet(
              part_dir / "part-0.parquet"
          )


  def test_compute_status_partition_coverage(tmp_path):
      """One PartitionCoverage per (source, sa); raw row presence, sentinel counts."""
      rows = [
          # CES SA: two months, the second is the Oct-2025 -1 sentinel slot.
          _row(source="ces", sa=True, ref_date=date(2025, 9, 1),
               vintage_date=date(2025, 11, 20), employment=159000.0),
          _row(source="ces", sa=True, ref_date=date(2025, 10, 1),
               vintage_date=date(2025, 12, 16), employment=-1.0),
          # QCEW NSA: one quarter.
          _row(source="qcew", sa=False, ref_date=date(2025, 1, 1),
               vintage_date=date(2025, 9, 1), employment=140000.0),
      ]
      _write_store_rows(tmp_path, rows)

      status = compute_status(tmp_path, as_of=date(2026, 1, 12))

      assert isinstance(status, StoreStatus)
      parts = {(p.source, p.seasonally_adjusted): p for p in status.per_partition}
      assert set(parts) == {("ces", True), ("qcew", False)}

      ces = parts[("ces", True)]
      assert isinstance(ces, PartitionCoverage)
      # Both CES rows counted — the -1 sentinel is NOT filtered out.
      assert ces.row_count == 2
      assert ces.earliest_ref == date(2025, 9, 1)
      assert ces.latest_ref == date(2025, 10, 1)
      assert ces.last_capture == date(2025, 12, 16)
      assert ces.distinct_vintages == 2

      qcew = parts[("qcew", False)]
      assert qcew.row_count == 1
      assert qcew.latest_ref == date(2025, 1, 1)
      assert qcew.last_capture == date(2025, 9, 1)
  ```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py::test_compute_status_partition_coverage -q --no-cov`
  Expected: FAIL with `ModuleNotFoundError: No module named 'nfp_vintages.store_status'`
  (the module does not exist yet).

- [ ] **Step 3: Implement** — create `store_status.py` with the two dataclasses and a
  `compute_status` that, for this task, only fills `per_partition` (the `uncaptured` /
  `missing_months` / `corrected` lists are populated in Task 7.2; seed them empty here). Coverage is
  one `read_vintage_store` lazy scan per `(source, sa)` partition, aggregated with Polars — **no**
  `transform_to_panel`, **no** `employment > 0` filter.

  ```python
  # packages/nfp-vintages/src/nfp_vintages/store_status.py
  """Read-only store health + knowability report (spec §8).

  Built on ``read_vintage_store`` (partition-prune + projection pushdown,
  LazyFrame) ONLY — never ``transform_to_panel`` (the expensive growth/censoring
  path) and never ``views.py`` (panel-grain, post-transform). Coverage is raw
  row presence (no ``employment > 0`` filter) so the Oct-2025 ``-1`` "no print"
  sentinel (``first_print.py:79-84``) counts as present.
  """

  from __future__ import annotations

  from dataclasses import dataclass, field
  from datetime import date

  import polars as pl
  from nfp_ingest.vintage_store import read_vintage_store
  from nfp_lookups.paths import VINTAGE_STORE_PATH, is_canonical_store, is_remote

  # (source, seasonally_adjusted) partitions present in the rebuilt store.
  _PARTITIONS: tuple[tuple[str, bool], ...] = (
      ("ces", True),
      ("ces", False),
      ("qcew", False),
  )


  @dataclass(frozen=True)
  class PartitionCoverage:
      """Coverage of one ``(source, seasonally_adjusted)`` store partition."""

      source: str
      seasonally_adjusted: bool
      earliest_ref: date | None
      latest_ref: date | None
      row_count: int
      last_capture: date | None
      distinct_vintages: int


  @dataclass(frozen=True)
  class StoreStatus:
      """The full ``status`` report — header flags, coverage, and alarms."""

      store_uri: str
      is_remote: bool
      is_canonical: bool
      per_partition: list[PartitionCoverage] = field(default_factory=list)
      uncaptured: list[str] = field(default_factory=list)
      missing_months: list[str] = field(default_factory=list)
      corrected: list[str] = field(default_factory=list)


  def _partition_coverage(store_path, source: str, sa: bool) -> PartitionCoverage | None:
      """Aggregate one partition via ``read_vintage_store``; None if empty/absent."""
      lf = read_vintage_store(store_path, source=source, seasonally_adjusted=sa)
      agg = lf.select(
          pl.len().alias("row_count"),
          pl.col("ref_date").min().alias("earliest_ref"),
          pl.col("ref_date").max().alias("latest_ref"),
          pl.col("vintage_date").max().alias("last_capture"),
          pl.col("vintage_date").n_unique().alias("distinct_vintages"),
      ).collect()
      row_count = int(agg.item(0, "row_count"))
      if row_count == 0:
          return None
      return PartitionCoverage(
          source=source,
          seasonally_adjusted=sa,
          earliest_ref=agg.item(0, "earliest_ref"),
          latest_ref=agg.item(0, "latest_ref"),
          row_count=row_count,
          last_capture=agg.item(0, "last_capture"),
          distinct_vintages=int(agg.item(0, "distinct_vintages")),
      )


  def compute_status(
      store_path=VINTAGE_STORE_PATH,
      as_of: date | None = None,
  ) -> StoreStatus:
      """Read-only coverage + knowability report for the vintage store.

      Reads via ``read_vintage_store`` only. ``as_of`` (default: today) bounds the
      forward UNCAPTURED alarm (Task 7.2). Never calls ``transform_to_panel``.
      """
      if as_of is None:
          as_of = date.today()

      per_partition: list[PartitionCoverage] = []
      for source, sa in _PARTITIONS:
          cov = _partition_coverage(store_path, source, sa)
          if cov is not None:
              per_partition.append(cov)

      return StoreStatus(
          store_uri=str(store_path),
          is_remote=is_remote(store_path),
          is_canonical=is_canonical_store(store_path),
          per_partition=per_partition,
          uncaptured=[],
          missing_months=[],
          corrected=[],
      )
  ```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py::test_compute_status_partition_coverage -q --no-cov`
  Expected: PASS. Then run the file suite and lint:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py -q --no-cov` and
  `uv run ruff check packages/nfp-vintages/src/nfp_vintages/store_status.py packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py`.

- [ ] **Step 5: Commit** — stage only the two new files:
  ```bash
  git add packages/nfp-vintages/src/nfp_vintages/store_status.py \
          packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  git commit -m "feat(status): per-partition store coverage via read_vintage_store"
  ```

---

### Task 7.2: forward UNCAPTURED alarm + missing-month list + `format_status`

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/store_status.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py` (extend)

- [ ] **Step 1: Write the failing test** — two new tests. (a) An **uncaptured** CES month: the
  store's latest CES ref is Aug-2025 but as-of is 2026-01-12, so the calendar says Sep/Oct/Nov-2025
  rev0 were already published → those ref-months must appear in `status.uncaptured`. (b) The
  **missing-month / sentinel** case from Task 7.1's store: the Oct-2025 `-1` sentinel row counts as
  **present**, so Oct-2025 must NOT be flagged missing in `status.missing_months` (raw row presence,
  no `employment > 0`). Also assert `format_status` renders the LOCAL-FALLBACK warning for a local
  path and lists each partition.

  ```python
  # append to packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  from nfp_vintages.store_status import format_status


  def test_compute_status_flags_uncaptured_ces_month(tmp_path):
      """Store lags the calendar: published-but-absent CES ref-months are flagged."""
      # Store stops at Aug-2025; as-of 2026-01-12 → Sep/Oct/Nov rev0 are out by then.
      rows = [
          _row(source="ces", sa=True, ref_date=date(2025, 7, 1),
               vintage_date=date(2025, 8, 1), employment=158000.0),
          _row(source="ces", sa=True, ref_date=date(2025, 8, 1),
               vintage_date=date(2025, 9, 5), employment=158200.0),
      ]
      _write_store_rows(tmp_path, rows)

      status = compute_status(tmp_path, as_of=date(2026, 1, 12))

      joined = " ".join(status.uncaptured)
      assert "ces" in joined
      # At least Sep-2025 should be reported uncaptured (rev0 published Oct-2025).
      assert "2025-09" in joined


  def test_oct_2025_sentinel_not_flagged_missing(tmp_path):
      """A -1 sentinel row at Oct-2025 counts as present (raw row presence)."""
      rows = [
          _row(source="ces", sa=True, ref_date=date(2025, 9, 1),
               vintage_date=date(2025, 11, 20), employment=159000.0),
          # The shutdown "no print" sentinel: literal -1.0 at the Oct-2025 slot.
          _row(source="ces", sa=True, ref_date=date(2025, 10, 1),
               vintage_date=date(2025, 12, 16), employment=-1.0),
          _row(source="ces", sa=True, ref_date=date(2025, 11, 1),
               vintage_date=date(2025, 12, 16), employment=159100.0),
      ]
      _write_store_rows(tmp_path, rows)

      status = compute_status(tmp_path, as_of=date(2026, 1, 12))

      # Oct-2025 has a (sentinel) row → NOT an interior hole.
      assert "2025-10" not in " ".join(status.missing_months)


  def test_format_status_local_fallback_warning(tmp_path):
      """A local (non-remote) store renders the .env LOCAL-FALLBACK warning."""
      rows = [
          _row(source="ces", sa=True, ref_date=date(2025, 9, 1),
               vintage_date=date(2025, 11, 20), employment=159000.0),
      ]
      _write_store_rows(tmp_path, rows)

      text = format_status(compute_status(tmp_path, as_of=date(2025, 12, 12)))

      assert "LOCAL FALLBACK" in text
      assert "NFP_STORE_URI" in text
      assert "ces" in text
  ```

- [ ] **Step 2: Run the tests, verify they fail** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py -q --no-cov -k "uncaptured or sentinel or format"`
  Expected: FAIL — `test_compute_status_flags_uncaptured_ces_month` fails because `uncaptured`
  is still empty; `test_format_status_local_fallback_warning` fails with `ImportError: cannot import
  name 'format_status'` (not yet defined).

- [ ] **Step 3: Implement** — add the forward-alarm + missing-month helpers and `format_status`.
  The UNCAPTURED scan walks forward from each source's latest captured ref-month and asks the
  calendar (`get_ces_vintage_date(ref_month, 0)` / `get_qcew_vintage_date(ref_quarter, ref_year, 0)`
  — `revision_schedules.py:299,368`) whether rev0 was published `<= as_of`; if so but the store
  lacks that ref, it is uncaptured. Missing-month is over the headline series (`geo 00`, industry
  `00`/`05`) on **raw row presence** (no `employment > 0`), so the sentinel counts present.

  ```python
  # add these imports at the top of store_status.py
  from nfp_lookups.revision_schedules import (
      get_ces_vintage_date,
      get_qcew_vintage_date,
  )
  ```

  ```python
  # add to store_status.py (module level)

  def _next_month(d: date) -> date:
      """First of the month after *d*."""
      return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


  def _ces_uncaptured(latest_ref: date | None, as_of: date) -> list[str]:
      """CES ref-months whose rev0 was published <= as_of but are not in the store."""
      if latest_ref is None:
          return []
      out: list[str] = []
      candidate = _next_month(latest_ref)
      # Bound the walk: stop once the calendar puts rev0 past as_of.
      while candidate <= as_of:
          try:
              v0 = get_ces_vintage_date(candidate, 0)
          except ValueError:
              break
          if v0 <= as_of:
              out.append(f"ces {candidate:%Y-%m}")
          candidate = _next_month(candidate)
      return out


  def _qcew_uncaptured(latest_ref: date | None, as_of: date) -> list[str]:
      """QCEW quarters whose rev0 was published <= as_of but are not in the store."""
      if latest_ref is None:
          return []
      out: list[str] = []
      # Step quarter-by-quarter from the quarter after the latest captured ref.
      q_start_month = ((latest_ref.month - 1) // 3) * 3 + 1
      year, month = latest_ref.year, q_start_month
      # Advance to the next quarter.
      month += 3
      if month > 12:
          year, month = year + 1, month - 12
      while date(year, month, 1) <= as_of:
          ref_quarter = f"Q{(month - 1) // 3 + 1}"
          try:
              v0 = get_qcew_vintage_date(ref_quarter, year, 0)
          except ValueError:
              break
          if v0 <= as_of:
              out.append(f"qcew {ref_quarter}-{year}")
          month += 3
          if month > 12:
              year, month = year + 1, month - 12
      return out


  def _missing_headline_months(store_path) -> list[str]:
      """Interior CES-SA ref-month gaps over the headline series (geo 00, ind 00/05).

      Raw row presence (no ``employment > 0`` filter) so a -1 sentinel row counts
      as present and Oct-2025 is not flagged a hole. A month is "present" if either
      the total (``00``) or private (``05``) headline row exists for it.
      """
      lf = read_vintage_store(
          store_path,
          source="ces",
          seasonally_adjusted=True,
          geographic_type="national",
          geographic_code="00",
      )
      present = (
          lf.filter(pl.col("industry_code").is_in(["00", "05"]))
          .select(pl.col("ref_date").dt.truncate("1mo"))
          .unique()
          .collect()
          .get_column("ref_date")
          .sort()
          .to_list()
      )
      if len(present) < 2:
          return []
      have = set(present)
      out: list[str] = []
      cursor = present[0]
      last = present[-1]
      while cursor < last:
          cursor = _next_month(cursor)
          if cursor not in have:
              out.append(f"{cursor:%Y-%m}")
      return out


  def format_status(status: StoreStatus) -> str:
      """Render a StoreStatus as a human-readable multi-line report."""
      lines: list[str] = []
      flags = []
      if status.is_remote:
          flags.append("REMOTE")
      else:
          flags.append("LOCAL")
      if status.is_canonical:
          flags.append("CANONICAL")
      lines.append(f"store: {status.store_uri}  [{'/'.join(flags)}]")
      if not status.is_remote:
          lines.append(
              "  WARNING: LOCAL FALLBACK — NFP_STORE_URI unset; reading the "
              "local data/store, not the canonical S3 store."
          )

      lines.append("")
      lines.append("coverage (source, seasonally_adjusted):")
      for p in status.per_partition:
          lines.append(
              f"  {p.source:<5} sa={str(p.seasonally_adjusted):<5} "
              f"rows={p.row_count:>8,} "
              f"ref=[{p.earliest_ref}..{p.latest_ref}] "
              f"last_capture={p.last_capture} vintages={p.distinct_vintages}"
          )

      if status.uncaptured:
          lines.append("")
          lines.append("UNCAPTURED (published per calendar, absent from store):")
          lines.extend(f"  {u}" for u in status.uncaptured)

      if status.missing_months:
          lines.append("")
          lines.append("missing headline months (interior gaps):")
          lines.extend(f"  {m}" for m in status.missing_months)

      if status.corrected:
          lines.append("")
          lines.append("CORRECTED-LEVEL (incoming != stored employment):")
          lines.extend(f"  {c}" for c in status.corrected)

      return "\n".join(lines)
  ```

  Then wire the new helpers into `compute_status`. Replace the `return StoreStatus(...)` block from
  Task 7.1 with one that computes the alarms from the per-partition coverage:

  ```python
  # in compute_status, replace the empty-list return with:
      by_key = {(p.source, p.seasonally_adjusted): p for p in per_partition}
      ces_sa = by_key.get(("ces", True))
      qcew = by_key.get(("qcew", False))

      uncaptured: list[str] = []
      uncaptured.extend(
          _ces_uncaptured(ces_sa.latest_ref if ces_sa else None, as_of)
      )
      uncaptured.extend(
          _qcew_uncaptured(qcew.latest_ref if qcew else None, as_of)
      )

      missing_months = _missing_headline_months(store_path)

      return StoreStatus(
          store_uri=str(store_path),
          is_remote=is_remote(store_path),
          is_canonical=is_canonical_store(store_path),
          per_partition=per_partition,
          uncaptured=uncaptured,
          missing_months=missing_months,
          corrected=[],
      )
  ```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py -q --no-cov`
  Expected: PASS (all five tests). Then lint:
  `uv run ruff check packages/nfp-vintages/src/nfp_vintages/store_status.py packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py`.

- [ ] **Step 5: Commit** — stage only the modified module + test:
  ```bash
  git add packages/nfp-vintages/src/nfp_vintages/store_status.py \
          packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  git commit -m "feat(status): forward UNCAPTURED alarm, missing-month list, format_status"
  ```

---

### Task 7.3: `status` Typer command (deferred imports)

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py` (extend)

- [ ] **Step 1: Write the failing test** — invoke the Typer command against a synthetic `tmp_path`
  store passed through `--store` and assert the rendered report appears in stdout. (The `--store`
  override lets the test point at `tmp_path` without touching real env.) Mirror the CliRunner pattern
  from `test_cli_snapshot.py:10-18`.

  ```python
  # append to packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  from nfp_vintages.__main__ import app
  from typer.testing import CliRunner


  def test_status_command_renders_report(tmp_path):
      """`alt-nfp status --store <tmp> --as-of D` prints the coverage report."""
      rows = [
          _row(source="ces", sa=True, ref_date=date(2025, 9, 1),
               vintage_date=date(2025, 11, 20), employment=159000.0),
          _row(source="qcew", sa=False, ref_date=date(2025, 1, 1),
               vintage_date=date(2025, 9, 1), employment=140000.0),
      ]
      _write_store_rows(tmp_path, rows)

      result = CliRunner().invoke(
          app,
          ["status", "--store", str(tmp_path), "--as-of", "2025-12-12"],
      )

      assert result.exit_code == 0, result.output
      assert "coverage" in result.output
      assert "ces" in result.output
      assert "qcew" in result.output
  ```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py::test_status_command_renders_report -q --no-cov`
  Expected: FAIL — Typer exits non-zero with `No such command 'status'` (the command is not yet
  registered on `app`).

- [ ] **Step 3: Implement** — add the `status` command to `__main__.py`. All imports are
  **deferred inside the body** so `load_dotenv()` (in the `main` callback,
  `__main__.py:24-33`) runs before `VINTAGE_STORE_PATH` resolves. When `--store` is given, build a
  store path with the same `NFP_STORE_URI` resolution the lookups layer uses; otherwise fall back to
  the import-time `VINTAGE_STORE_PATH`. Insert this command immediately before the existing
  `snapshot` command (`__main__.py:291`).

  ```python
  @app.command()
  def status(
      as_of: str | None = typer.Option(
          None, "--as-of", help="Knowability cutoff for the UNCAPTURED alarm (YYYY-MM-DD)."
      ),
      store: str | None = typer.Option(
          None, "--store", help="Override the store URI/path (default: VINTAGE_STORE_PATH)."
      ),
  ) -> None:
      """Read-only store coverage + 'what's uncaptured' report (spec §8)."""
      from datetime import date as _date

      from nfp_lookups.paths import VINTAGE_STORE_PATH
      from nfp_vintages.store_status import compute_status, format_status

      if store is not None:
          if store.startswith(("s3://", "s3a://")):
              from upath import UPath

              store_path = UPath(store)
          else:
              store_path = Path(store)
      else:
          store_path = VINTAGE_STORE_PATH

      as_of_date = _date.fromisoformat(as_of) if as_of is not None else None
      report = compute_status(store_path, as_of=as_of_date)
      print(format_status(report))
  ```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py::test_status_command_renders_report -q --no-cov`
  Expected: PASS. Then run the full file + the existing CLI test + lint:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py packages/nfp-vintages/src/nfp_vintages/tests/test_cli_snapshot.py -q --no-cov` and
  `uv run ruff check packages/nfp-vintages/src/nfp_vintages/__main__.py`.

- [ ] **Step 5: Commit** — stage only the CLI module + test:
  ```bash
  git add packages/nfp-vintages/src/nfp_vintages/__main__.py \
          packages/nfp-vintages/src/nfp_vintages/tests/test_store_status.py
  git commit -m "feat(cli): add alt-nfp status command (deferred imports)"
  ```


---


## Phase 8 — `watch` command + `feed.py` (Spec §9)

Feed-driven automation: a thin trigger on top of `update`, for a daily cron. Two
deliverables: (a) `nfp_download/release_dates/feed.py` — pure `parse_feed` + networked
`fetch_feed` reusing the curl_cffi Chrome-impersonating session; (b) the `alt-nfp watch`
CLI command — fetch the BLS RSS feed per `--source`, decide "is this new?" from store
coverage (`compute_status`), trigger `update --as-of <pubDate>` for new releases, and
(with `--snapshot`) run `snapshot` at the **day-12 anchor of the captured ref-month**
(NOT raw `pubDate`).

**Cross-phase contract this phase depends on (declared, not authored here):**
- Phase 5 splits the legacy bare-run into plain helpers `_run_update(as_of, only=None,
  no_refresh_calendar=False)` and `_run_snapshot(as_of, grid_end=None)` in `__main__.py`.
  `watch` calls **these plain helpers**, never the Typer-decorated `update`/`snapshot`
  commands (calling a Typer command in-process leaves `Option(...)` defaults as
  `OptionInfo` objects). The watch test monkeypatches `_run_update`/`_run_snapshot`.
- Phase 7 provides `nfp_vintages.store_status.compute_status(store_path, as_of)` returning
  `StoreStatus` with field `uncaptured: list[str]`, where each element is
  `"<source>:<ref_iso>"` — `source ∈ {ces, qcew}`, `ref_iso` is the ISO date of the
  uncaptured ref-month (CES, first of month e.g. `"ces:2025-05-01"`) or
  `"<source>:<year>-Q<n>"` for QCEW (e.g. `"qcew:2025-Q1"`). `watch` reads only the
  `ces:`/`qcew:` rows and the embedded ref date; it derives the ref-month from
  **`compute_status`, never from the RSS title** (spec principle 3 — the store is the
  source of truth; the feed only says "a release is out *now*" and supplies `pubDate`).

---

### Task 8.1: `feed.py` — pure `parse_feed(xml) -> list[FeedItem]`

**Files:**
- Create: `packages/nfp-download/src/nfp_download/release_dates/feed.py`
- Test: `packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py`

`parse_feed` is the pure, no-network half: parse a BLS RSS 2.0 feed string into
`FeedItem`s. `pubDate` is RFC-822 (e.g. `Fri, 06 Jun 2025 08:30:00 -0400`), parsed with
`email.utils.parsedate_to_datetime(...).date()` — **not** a hand-rolled month regex. The
fixture is a realistic captured-RSS string (we could not live-fetch — `www.bls.gov` 403s a
plain GET, the exact Akamai block that forces the curl_cffi session in Task 8.2; this is
standard RSS 2.0 with `<item><title><pubDate><guid>`).

- [ ] **Step 1: Write the failing test** — pure parse over a captured RSS fixture.

```python
"""Unit tests for feed.parse_feed (pure, no network).

Fixture is standard RSS 2.0 (BLS empsit/cewqtr feeds publish this shape):
each <item> carries <title>, an RFC-822 <pubDate>, and a <guid>. We could not
live-capture in red phase — www.bls.gov 403s a plain GET (the Akamai block that
forces fetch_feed's curl_cffi session); pubDate format pinned to RFC-822.
"""

from __future__ import annotations

from datetime import date

from nfp_download.release_dates.feed import FeedItem, parse_feed

EMPSIT_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Employment Situation</title>
    <link>https://www.bls.gov/news.release/empsit.htm</link>
    <item>
      <title>Employment Situation Summary</title>
      <link>https://www.bls.gov/news.release/archives/empsit_06062025.htm</link>
      <pubDate>Fri, 06 Jun 2025 08:30:00 -0400</pubDate>
      <guid>https://www.bls.gov/news.release/archives/empsit_06062025.htm</guid>
    </item>
    <item>
      <title>Employment Situation Summary</title>
      <link>https://www.bls.gov/news.release/archives/empsit_05022025.htm</link>
      <pubDate>Fri, 02 May 2025 08:30:00 -0400</pubDate>
      <guid>https://www.bls.gov/news.release/archives/empsit_05022025.htm</guid>
    </item>
  </channel>
</rss>
"""


class TestParseFeed:
    def test_returns_feed_items(self):
        items = parse_feed(EMPSIT_RSS)
        assert len(items) == 2
        assert all(isinstance(it, FeedItem) for it in items)

    def test_first_item_fields(self):
        items = parse_feed(EMPSIT_RSS)
        first = items[0]
        assert first.title == "Employment Situation Summary"
        assert first.pub_date == date(2025, 6, 6)
        assert first.guid == "https://www.bls.gov/news.release/archives/empsit_06062025.htm"

    def test_pubdate_parsed_as_date(self):
        items = parse_feed(EMPSIT_RSS)
        assert all(isinstance(it.pub_date, date) for it in items)
        assert items[1].pub_date == date(2025, 5, 2)

    def test_empty_feed_returns_empty_list(self):
        empty = '<?xml version="1.0"?><rss version="2.0"><channel/></rss>'
        assert parse_feed(empty) == []

    def test_item_missing_pubdate_is_skipped(self):
        no_date = """<?xml version="1.0"?>
        <rss version="2.0"><channel>
          <item><title>No date</title><guid>g1</guid></item>
        </channel></rss>"""
        assert parse_feed(no_date) == []
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py::TestParseFeed -q --no-cov`
  Expected: **FAIL** with `ModuleNotFoundError: No module named 'nfp_download.release_dates.feed'`
  (the module does not exist yet).

- [ ] **Step 3: Implement** `parse_feed` + the module scaffold (`fetch_feed` lands in Task 8.2).

```python
"""BLS release RSS feed — fetch + parse.

The feed answers the production question the calendar can only predict and
shutdowns can delay: "is the release out *now*?". `parse_feed` is pure (no
network); `fetch_feed` reuses the scraper's curl_cffi Chrome-impersonating
session (www.bls.gov/Akamai 403s a plain httpx GET — memory
`bls-akamai-blocking-intermittent`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime

EMPSIT_FEED_URL = "https://www.bls.gov/feed/empsit.rss"
CEWQTR_FEED_URL = "https://www.bls.gov/feed/cewqtr.rss"


@dataclass
class FeedItem:
    """One RSS <item>: release title, publication date, and unique id."""

    title: str
    pub_date: date
    guid: str


def _text(item: ET.Element, tag: str) -> str | None:
    """Return the stripped text of the first child `tag`, or None."""
    child = item.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_feed(xml: str) -> list[FeedItem]:
    """Parse a BLS RSS 2.0 feed into FeedItems (pure, no network).

    `pubDate` is RFC-822 (e.g. ``Fri, 06 Jun 2025 08:30:00 -0400``); we take its
    calendar date. Items missing a title, a parseable pubDate, or a guid are
    skipped rather than raising — a malformed item should not sink the poll.

    Parameters
    ----------
    xml : str
        Raw RSS feed body.

    Returns
    -------
    list[FeedItem]
        One per well-formed <item>, in feed order (BLS lists newest first).
    """
    root = ET.fromstring(xml)
    items: list[FeedItem] = []
    for item in root.iter("item"):
        title = _text(item, "title")
        raw_date = _text(item, "pubDate")
        guid = _text(item, "guid")
        if title is None or raw_date is None or guid is None:
            continue
        try:
            pub = parsedate_to_datetime(raw_date).date()
        except (TypeError, ValueError):
            continue
        items.append(FeedItem(title=title, pub_date=pub, guid=guid))
    return items
```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py::TestParseFeed -q --no-cov`
  Expected: **PASS** (5 tests). Then run the package suite and lint:
  `uv run pytest packages/nfp-download/src/nfp_download/tests/release_dates/ -q --no-cov` and
  `uv run ruff check packages/nfp-download/src/nfp_download/release_dates/feed.py packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py`.

- [ ] **Step 5: Commit** —
  `git add packages/nfp-download/src/nfp_download/release_dates/feed.py packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py`
  then `git commit -m "feat(download): add release-feed parse_feed + FeedItem (RSS 2.0, RFC-822 pubDate)"`.

---

### Task 8.2: `feed.py` — networked `fetch_feed(url, *, session=None)`

**Files:**
- Modify: `packages/nfp-download/src/nfp_download/release_dates/feed.py`
- Test: `packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py` (extend)

`fetch_feed` is the network half. `create_session` (`scraper.py:191-199`) returns a
curl_cffi **`AsyncSession`** used as `async with … await session.get(...)`. The contract's
`fetch_feed` is **sync** (`-> list[FeedItem]`), so it wraps an async helper in
`asyncio.run` — mirroring `_build_release_calendar`'s `asyncio.run(...)`
(`__main__.py:129`). It is network-marked. The pure-parse path is already covered by
Task 8.1, so the network test asserts only the transport contract (returns a list of
`FeedItem`).

- [ ] **Step 1: Write the failing test** — network-marked smoke for `fetch_feed`.

```python
import pytest

from nfp_download.release_dates.feed import EMPSIT_FEED_URL, FeedItem, fetch_feed


@pytest.mark.network
class TestFetchFeed:
    def test_fetch_empsit_returns_feed_items(self):
        items = fetch_feed(EMPSIT_FEED_URL)
        assert isinstance(items, list)
        assert items, "empsit feed should publish at least one item"
        assert all(isinstance(it, FeedItem) for it in items)
        # Newest-first: the top item should be the most recent release.
        assert items[0].pub_date >= items[-1].pub_date
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest "packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py::TestFetchFeed" -q --no-cov -m network`
  Expected: **FAIL** with `ImportError: cannot import name 'fetch_feed'` (only `parse_feed`
  exists after Task 8.1). (If run without `-m network` the class is deselected; this step
  must pass `-m network`.)

- [ ] **Step 3: Implement** — add `fetch_feed` + its async helper to `feed.py`.

```python
import asyncio

from nfp_download.release_dates.scraper import create_session


async def _fetch_feed_async(url: str, session=None) -> list[FeedItem]:
    """Fetch + parse one feed; reuse `session` if given, else open one."""
    if session is not None:
        resp = await session.get(url)
        resp.raise_for_status()
        return parse_feed(resp.text)
    async with create_session() as owned:
        resp = await owned.get(url)
        resp.raise_for_status()
        return parse_feed(resp.text)


def fetch_feed(url: str, *, session=None) -> list[FeedItem]:
    """Fetch and parse a BLS RSS feed (network).

    Transport is the scraper's curl_cffi Chrome-impersonating
    :func:`nfp_download.release_dates.scraper.create_session` — www.bls.gov sits
    behind Akamai TLS fingerprinting, so a plain httpx GET 403s. The session is
    an async curl_cffi ``AsyncSession``; this sync wrapper drives it via
    ``asyncio.run`` (mirroring ``_build_release_calendar``).

    Parameters
    ----------
    url : str
        Feed URL (``EMPSIT_FEED_URL`` or ``CEWQTR_FEED_URL``).
    session : curl_cffi AsyncSession, optional
        Reuse an open async session (e.g. polling both feeds in one run);
        when None, a session is opened and closed for this call.

    Returns
    -------
    list[FeedItem]
    """
    return asyncio.run(_fetch_feed_async(url, session=session))
```

- [ ] **Step 4: Run, verify pass** — network test (requires connectivity):
  `uv run pytest "packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py::TestFetchFeed" -q --no-cov -m network`
  Expected: **PASS**. Then confirm the non-network suite still passes and lint is clean:
  `uv run pytest packages/nfp-download/src/nfp_download/tests/release_dates/ -q --no-cov -m "not network"` and
  `uv run ruff check packages/nfp-download/src/nfp_download/release_dates/feed.py packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py`.

- [ ] **Step 5: Commit** —
  `git add packages/nfp-download/src/nfp_download/release_dates/feed.py packages/nfp-download/src/nfp_download/tests/release_dates/test_feed.py`
  then `git commit -m "feat(download): add fetch_feed reusing scraper curl_cffi session (network)"`.

---

### Task 8.3: `alt-nfp watch` command — trigger-on-new + day-12 snapshot anchor

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py`

`watch` polls the feed per `--source`, and for each source asks **`compute_status`** (not
the RSS title) which ref-month/quarter is uncaptured. If the source has an `uncaptured`
entry, it triggers `_run_update(as_of=<pubDate ISO>, only=<source>)`; if `--snapshot`, it
also runs `_run_snapshot(as_of=<refmonth>-12)` — the **day-12 anchor of the captured
ref-month**, because `snapshot --as-of` enforces day-12 (§4a) and would reject the raw
`pubDate`. On a day with nothing uncaptured it is a clean no-op.

`source ∈ {all, ces, qcew}` maps to feed URLs: `all → [empsit, cewqtr]`,
`ces → [empsit]`, `qcew → [cewqtr]`. The matching `update --only` value is `ces`/`qcew`.

The watch test monkeypatches `feed.fetch_feed` and the `_run_update`/`_run_snapshot`
helpers (Phase 5), but lets **`compute_status` run for real** against `tmp_path` — so the
fixture seeds both store rows (CES partition) **and** `vintage_dates.parquet` (the forward
"uncaptured" alarm reads the calendar). Present-vs-absent ref-month is what flips
trigger-vs-no-op. All store/path env points at `tmp_path` (no real MinIO — conftest
auto-loads prod creds).

- [ ] **Step 1: Write the failing test** — trigger-on-new, no-op-when-present, day-12 anchor.

```python
"""CLI tests for `alt-nfp watch` (feed-driven trigger).

Monkeypatches the feed and the _run_update/_run_snapshot helpers; lets
compute_status run for real against a tmp store + vintage_dates.parquet so the
present/absent ref-month decides trigger-vs-no-op. Store-write-free: we only
seed rows and read them. NEVER points at real MinIO (conftest loads prod creds).
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from typer.testing import CliRunner

from nfp_download.release_dates.feed import FeedItem

runner = CliRunner()


def _seed_store(store_root, *, ref_dates, vintage_date):
    """Write a minimal CES SA partition with the given headline ref_dates."""
    part = store_root / "source=ces" / "seasonally_adjusted=true"
    part.mkdir(parents=True, exist_ok=True)
    rows = {
        "ref_date": list(ref_dates),
        "industry_type": ["total"] * len(ref_dates),
        "industry_code": ["00"] * len(ref_dates),
        "ownership": ["total"] * len(ref_dates),
        "size_class_type": [None] * len(ref_dates),
        "size_class_code": [None] * len(ref_dates),
        "geographic_type": ["national"] * len(ref_dates),
        "geographic_code": ["00"] * len(ref_dates),
        "revision": [0] * len(ref_dates),
        "benchmark_revision": [0] * len(ref_dates),
        "vintage_date": [vintage_date] * len(ref_dates),
        "employment": [150000.0 + i for i in range(len(ref_dates))],
    }
    pl.DataFrame(rows).write_parquet(part / "part-0.parquet")


def _seed_calendar(intermediate_dir, rows):
    """Write a minimal vintage_dates.parquet (publication/ref/vintage/rev)."""
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "publication": pl.Utf8,
            "ref_date": pl.Date,
            "vintage_date": pl.Date,
            "revision": pl.Int64,
            "benchmark_revision": pl.Int64,
        },
        orient="row",
    ).write_parquet(intermediate_dir / "vintage_dates.parquet")


@pytest.fixture
def watch_env(tmp_path, monkeypatch):
    """Point store + intermediate dirs at tmp; return (store, intermediate)."""
    store_root = tmp_path / "store"
    intermediate = tmp_path / "intermediate"
    monkeypatch.setenv("NFP_BASE_DIR", str(tmp_path))
    monkeypatch.delenv("NFP_STORE_URI", raising=False)
    return store_root, intermediate


def _patch_feed(monkeypatch, pub_date):
    """Make fetch_feed return one empsit item published on `pub_date`."""
    item = FeedItem(
        title="Employment Situation Summary",
        pub_date=pub_date,
        guid=f"empsit_{pub_date}",
    )
    import nfp_download.release_dates.feed as feed_mod

    monkeypatch.setattr(feed_mod, "fetch_feed", lambda url, **kw: [item])


def test_triggers_update_when_refmonth_uncaptured(watch_env, monkeypatch):
    """A feed release whose ref-month is NOT in the store triggers update."""
    store_root, intermediate = watch_env
    # Store has CES through 2025-04; calendar says 2025-05 rev0 published 2025-06-06.
    _seed_store(
        store_root,
        ref_dates=[date(2025, 3, 1), date(2025, 4, 1)],
        vintage_date=date(2025, 5, 2),
    )
    _seed_calendar(
        intermediate,
        rows=[
            ("ces", date(2025, 4, 1), date(2025, 5, 2), 0, 0),
            ("ces", date(2025, 5, 1), date(2025, 6, 6), 0, 0),
        ],
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    calls = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda **kw: calls.append(kw))
    monkeypatch.setattr(cli, "_run_snapshot", lambda **kw: calls.append(("snap", kw)))

    result = runner.invoke(cli.app, ["watch", "--source", "ces"])
    assert result.exit_code == 0, result.output
    update_calls = [c for c in calls if isinstance(c, dict)]
    assert len(update_calls) == 1
    assert update_calls[0]["as_of"] == "2025-06-06"  # pubDate is the update as-of
    assert update_calls[0]["only"] == "ces"


def test_no_op_when_refmonth_already_present(watch_env, monkeypatch):
    """A feed release whose ref-month IS captured triggers nothing."""
    store_root, intermediate = watch_env
    _seed_store(
        store_root,
        ref_dates=[date(2025, 4, 1), date(2025, 5, 1)],
        vintage_date=date(2025, 6, 6),
    )
    _seed_calendar(
        intermediate,
        rows=[
            ("ces", date(2025, 4, 1), date(2025, 5, 2), 0, 0),
            ("ces", date(2025, 5, 1), date(2025, 6, 6), 0, 0),
        ],
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    calls = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda **kw: calls.append(kw))
    monkeypatch.setattr(cli, "_run_snapshot", lambda **kw: calls.append(("snap", kw)))

    result = runner.invoke(cli.app, ["watch", "--source", "ces"])
    assert result.exit_code == 0, result.output
    assert calls == []  # nothing uncaptured → clean no-op


def test_snapshot_uses_day12_anchor_not_pubdate(watch_env, monkeypatch):
    """With --snapshot, snapshot as-of is <refmonth>-12, not the pubDate."""
    store_root, intermediate = watch_env
    _seed_store(
        store_root,
        ref_dates=[date(2025, 3, 1), date(2025, 4, 1)],
        vintage_date=date(2025, 5, 2),
    )
    _seed_calendar(
        intermediate,
        rows=[
            ("ces", date(2025, 4, 1), date(2025, 5, 2), 0, 0),
            ("ces", date(2025, 5, 1), date(2025, 6, 6), 0, 0),
        ],
    )
    _patch_feed(monkeypatch, date(2025, 6, 6))

    snaps = []
    import nfp_vintages.__main__ as cli

    monkeypatch.setattr(cli, "_run_update", lambda **kw: None)
    monkeypatch.setattr(cli, "_run_snapshot", lambda **kw: snaps.append(kw))

    result = runner.invoke(cli.app, ["watch", "--source", "ces", "--snapshot"])
    assert result.exit_code == 0, result.output
    assert len(snaps) == 1
    # Captured ref-month is 2025-05 → anchor 2025-05-12, NOT the pubDate 2025-06-06.
    assert snaps[0]["as_of"] == "2025-05-12"
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py -q --no-cov`
  Expected: **FAIL** — `watch` is not a registered command yet, so
  `runner.invoke(cli.app, ["watch", ...])` exits non-zero (`No such command 'watch'`),
  failing the `exit_code == 0` assertions.

- [ ] **Step 3: Implement** — add `watch` to `__main__.py`. All imports deferred in the
  command body (the app callback runs `load_dotenv()` before `VINTAGE_STORE_PATH` resolves).

```python
@app.command()
def watch(
    source: str = typer.Option(
        "all", "--source", help="Which feed(s) to poll: all | ces | qcew."
    ),
    snapshot_after: bool = typer.Option(
        False, "--snapshot", help="Also bake a ModelData snapshot for each new release."
    ),
) -> None:
    """Poll the BLS release feed; trigger `update` on a newly-published release.

    Designed for a daily cron. The feed answers only "a release is out *now*"
    and supplies the release day (``pubDate``); the **store** (via
    ``compute_status``) is the source of truth for which ref-month/quarter is
    still uncaptured. A clean no-op on days with nothing new.
    """
    from datetime import date as _date

    from nfp_download.release_dates.feed import (
        CEWQTR_FEED_URL,
        EMPSIT_FEED_URL,
        fetch_feed,
    )

    from nfp_vintages.store_status import compute_status

    feeds = {"ces": EMPSIT_FEED_URL, "qcew": CEWQTR_FEED_URL}
    if source == "all":
        wanted = ["ces", "qcew"]
    elif source in feeds:
        wanted = [source]
    else:
        raise typer.BadParameter(
            "must be one of: all, ces, qcew", param_hint="--source"
        )

    for src in wanted:
        items = fetch_feed(feeds[src])
        if not items:
            print(f"  {src}: feed empty — skipping")
            continue
        # BLS lists newest-first; the top item is the latest release.
        latest = items[0]
        pub = latest.pub_date

        # The store decides whether this release's ref-period is captured.
        status = compute_status(as_of=pub)
        uncaptured = [u for u in status.uncaptured if u.startswith(f"{src}:")]
        if not uncaptured:
            print(f"  {src}: latest release ({pub}) already captured — no-op")
            continue

        ref_token = uncaptured[0].split(":", 1)[1]  # ISO date or YYYY-Qn
        print(f"  {src}: NEW release {pub} (uncaptured {ref_token}) — updating")
        _run_update(as_of=pub.isoformat(), only=src)

        if snapshot_after:
            anchor = _snapshot_anchor(ref_token)
            print(f"  {src}: snapshot at day-12 anchor {anchor}")
            _run_snapshot(as_of=anchor)


def _snapshot_anchor(ref_token: str) -> str:
    """Day-12 anchor (YYYY-MM-12) for an uncaptured ref token.

    `ref_token` is either an ISO ref date (CES, ``2025-05-01``) or a QCEW
    ``YYYY-Qn`` token; in both cases the snapshot cutoff is the 12th of the
    ref-month (QCEW quarter end month), the convention ``snapshot --as-of``
    enforces (§4a). Never the raw pubDate.
    """
    from datetime import date as _date

    if "-Q" in ref_token:
        year_str, q_str = ref_token.split("-Q")
        month = int(q_str) * 3  # Q1->Mar, Q2->Jun, Q3->Sep, Q4->Dec
        return _date(int(year_str), month, 12).isoformat()
    ref = _date.fromisoformat(ref_token)
    return _date(ref.year, ref.month, 12).isoformat()
```

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py -q --no-cov`
  Expected: **PASS** (3 tests). Then run the package suite + lint:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/ -q --no-cov -m "not network and not slow"` and
  `uv run ruff check packages/nfp-vintages/src/nfp_vintages/__main__.py packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py`.
  Note: this task depends on Phase 5's `_run_update`/`_run_snapshot` helpers and Phase 7's
  `compute_status`/`StoreStatus.uncaptured` already being in place; if running this phase in
  isolation, those two cross-phase symbols are the only external prerequisites.

- [ ] **Step 5: Commit** —
  `git add packages/nfp-vintages/src/nfp_vintages/__main__.py packages/nfp-vintages/src/nfp_vintages/tests/test_cli_watch.py`
  then `git commit -m "feat(cli): add alt-nfp watch — feed-driven update trigger + day-12 snapshot anchor"`.


---


## Phase 9 — `scripts/bootstrap_store.py` + legacy retirement + docs

> Spec: `specs/cli_production_workflow.md` §10 (bootstrap script + legacy retirement),
> §11 (file map), §14 step 9. This phase is the LAST in the build order: it depends on
> Phase 1 (the relocated `nfp_ingest.qcew_acquire.acquire_qcew_levels` /
> `acquire_qcew_size_native`), Phase 3 (the lifted `nfp_vintages.calendar.advance_release_calendar`),
> and Phases 5–8 (the `update`/`status`/`watch` commands). It (a) writes the one-time
> bootstrap script generalizing the `_t8_promote.py` cutover, (b) deletes the retired legacy
> commands from `__main__.py` while keeping `advance_release_calendar` callable, and (c)
> updates the five CLAUDE.md command banners/maps. Retirement MUST keep the suite green (no
> orphaned imports).

---

### Task 9.1: `scripts/bootstrap_store.py` — one-time rebuild + promote

The bootstrap script lifts the **rebuild** lineage (not the legacy `build_store` one), ordered
exactly per §10: `download_ces()` → `build_ces_panel()` → `acquire_qcew_levels`/
`acquire_qcew_size_native` → `build_qcew_panel`/`build_size_class_panel` →
`compose_rebuild_panel` → `write_rebuild_store(allow_canonical=False)` to a **scratch** prefix
→ **promote** via a generalized copy-then-delete cutover (the `_t8_promote.py:cutover` flow,
`scripts/_t8_promote.py:117-139`). The promote keeps the `is_canonical_store` refusal: a write
or mirror straight to `…/store` is the exact hazard CLAUDE.md warns about (filenames encode
vintage ranges, so an overwrite leaves stale fragments). `scripts/mirror_store.py` is
**not** reused — it is overwrite-only (`scripts/mirror_store.py:68-71`, `fs.put_file` with no
orphan delete).

> **Container-safety (Tier C — absorbs plans/15 Phase 2 Tasks 7 & 9).** Because this is a
> single-process script, it is the one place a run-scoped temp dir actually works — the retired
> multi-subcommand `download`/`process`/`build` lineage could not share one tempdir across
> separate processes, which is exactly why plans/15 deferred its Tasks 7 (rebuild tempdir) and 9
> (scraped-HTML temp) to here. Wrap the rebuild in
> `with tempfile.TemporaryDirectory(prefix="altnfp-bootstrap-") as tmp:` and thread `Path(tmp)`
> as the scratch root for every byproduct that previously landed under `./data`: the raw
> `download_ces()` cesvinall files, any scraped release HTML, and the intermediate revision
> parquets. Nothing the script writes may persist under `./data` on Bloomberg's container — only
> the rebuilt **store** (S3 via `NFP_STORE_URI`) survives the run. (See plans/15 Phase 2;
> Tasks 8 + 10 of that plan — the HTTP-cache and SAE-checkpoint tempfile defaults — are already
> done and need nothing here.)

It is a **script**, not a CLI command (§4 table row, §10): invoked as
`uv run python scripts/bootstrap_store.py …`, `argparse` for flags.

**Files:**
- Create: `/Users/lowell/Projects/alt-nfp/scripts/bootstrap_store.py`
- Test: `/Users/lowell/Projects/alt-nfp/packages/nfp-vintages/src/nfp_vintages/tests/test_bootstrap_store.py`

> Why the test lives under `packages/nfp-vintages/.../tests/`: `pyproject.toml` sets
> `testpaths = ["packages"]` (line 90) — pytest does **not** collect from `scripts/`. The test
> imports the script by file path via `importlib.util.spec_from_file_location`, anchored at
> `nfp_lookups.paths.BASE_DIR / "scripts" / "bootstrap_store.py"`.

- [ ] **Step 1: Write the failing test** — a no-network, no-real-store smoke test. It
  monkeypatches the four acquisition seams (`download_ces`, `acquire_qcew_levels`,
  `acquire_qcew_size_native`, and the `advance_release_calendar` calendar scrape) to return
  tiny synthetic inputs, points the scratch + canonical at two `tmp_path` local prefixes
  (both `is_canonical_store(...) == False`, so the guard is satisfied), runs `main()`, and
  asserts the canonical scratch dir ends up with the composed partitions.

```python
"""Smoke test for scripts/bootstrap_store.py (no network, no real store)."""
from __future__ import annotations

import importlib.util
from datetime import date

import polars as pl
import pytest
from nfp_lookups.paths import BASE_DIR


def _load_bootstrap():
    """Import scripts/bootstrap_store.py by path (scripts/ is not on testpaths)."""
    path = BASE_DIR / "scripts" / "bootstrap_store.py"
    spec = importlib.util.spec_from_file_location("bootstrap_store", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_ces_panel() -> pl.DataFrame:
    """One CES row in VINTAGE_STORE_SCHEMA (already remapped/total taxonomy)."""
    from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

    row = {
        "ref_date": date(2024, 1, 1),
        "industry_type": "total",
        "industry_code": "00",
        "ownership": "total",
        "geographic_type": "national",
        "geographic_code": "00",
        "size_class_type": None,
        "size_class_code": None,
        "revision": 0,
        "benchmark_revision": 0,
        "employment": 158000.0,
        "vintage_date": date(2024, 2, 2),
        "source": "ces",
        "seasonally_adjusted": True,
    }
    return pl.DataFrame([row], schema=dict(VINTAGE_STORE_SCHEMA))


def _tiny_qcew_levels_raw() -> pl.DataFrame:
    """Raw area-slice columns build_qcew_panel consumes (own_code '0' anchor)."""
    return pl.DataFrame(
        {
            "area_fips": ["US000"],
            "own_code": ["0"],
            "industry_code": ["10"],
            "agglvl_code": ["10"],
            "year": ["2024"],
            "qtr": ["1"],
            "month1_emplvl": [158000000],
            "month2_emplvl": [158100000],
            "month3_emplvl": [158200000],
            "revision": [0],
        }
    )


def test_bootstrap_builds_scratch_then_promotes_to_canonical(monkeypatch, tmp_path):
    boot = _load_bootstrap()

    scratch = tmp_path / "store-rebuild"
    canonical = tmp_path / "store"

    # Acquisition seams → synthetic, zero network.
    monkeypatch.setattr(boot, "download_ces", lambda *a, **k: None)
    monkeypatch.setattr(boot, "advance_release_calendar", lambda *a, **k: None)
    monkeypatch.setattr(boot, "build_ces_panel", lambda *a, **k: _tiny_ces_panel())
    monkeypatch.setattr(
        boot, "acquire_qcew_levels", lambda *a, **k: _tiny_qcew_levels_raw()
    )
    monkeypatch.setattr(
        boot, "acquire_qcew_size_native", lambda *a, **k: pl.DataFrame()
    )

    boot.main(
        argv=[
            "--scratch", str(scratch),
            "--canonical", str(canonical),
            "--start-year", "2024",
            "--end-year", "2024",
        ]
    )

    # Promotion left the canonical prefix populated and scratch drained of orphans.
    canon_files = sorted(canonical.glob("**/*.parquet"))
    assert canon_files, "canonical store has no parquet partitions after bootstrap"
    ces_part = canonical / "source=ces" / "seasonally_adjusted=true"
    assert ces_part.exists(), "expected source=ces/seasonally_adjusted=true partition"
    df = pl.read_parquet(canon_files[0])
    assert df.height >= 1


def test_bootstrap_refuses_canonical_uri_as_scratch(monkeypatch, tmp_path):
    """--scratch must not be the canonical store (is_canonical_store guard)."""
    boot = _load_bootstrap()
    monkeypatch.setattr(boot, "download_ces", lambda *a, **k: None)
    monkeypatch.setattr(boot, "advance_release_calendar", lambda *a, **k: None)
    monkeypatch.setattr(boot, "build_ces_panel", lambda *a, **k: _tiny_ces_panel())
    monkeypatch.setattr(
        boot, "acquire_qcew_levels", lambda *a, **k: _tiny_qcew_levels_raw()
    )
    monkeypatch.setattr(
        boot, "acquire_qcew_size_native", lambda *a, **k: pl.DataFrame()
    )
    with pytest.raises(SystemExit):
        boot.main(
            argv=[
                "--scratch", "s3://alt-nfp/store",
                "--canonical", str(tmp_path / "store"),
                "--start-year", "2024",
                "--end-year", "2024",
            ]
        )
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_bootstrap_store.py -q --no-cov`
  Expected: **FAIL** — `scripts/bootstrap_store.py` does not exist yet, so
  `spec_from_file_location` returns a loader that raises `FileNotFoundError` (the `_load_bootstrap`
  helper fails before any assertion).

- [ ] **Step 3: Implement** — `scripts/bootstrap_store.py`. The promote step generalizes
  `_t8_promote.py:cutover` (`scripts/_t8_promote.py:117-139`): copy every scratch file into the
  canonical prefix under its rebuilt name, then delete any canonical orphan not in the new set,
  then verify the keyset and sizes. Module-level `download_ces` / `build_ces_panel` /
  `acquire_qcew_levels` / `acquire_qcew_size_native` / `advance_release_calendar` bindings are
  imported at module scope so the test can monkeypatch them. Local `Path` prefixes use plain
  filesystem ops; `s3://` prefixes use `s3fs` (the `_t8_promote.py` `_fs()` pattern). The
  `is_canonical_store` guard is checked on `--scratch` before any work.

```python
#!/usr/bin/env python3
"""One-time historical store rebuild + promote (NOT a CLI command).

Lifts the **rebuild** lineage (spec cli_production_workflow.md §10), ordered:

    download_ces()                        # extract cesvinall/ triangular CSVs
    advance_release_calendar()            # vintage_dates.parquet present for overlap parity
    build_ces_panel()                     # CES NSA+SA store-schema rows
    acquire_qcew_levels(...)  -> build_qcew_panel(...)
    acquire_qcew_size_native(...) -> build_size_class_panel(...)   # Q1-only
    compose_rebuild_panel(...)
    write_rebuild_store(scratch, allow_canonical=False)            # scratch prefix
    promote(scratch -> canonical)         # copy-then-delete cutover (_t8_promote flow)

Usage::

    NFP_STORE_URI=s3://alt-nfp/store-rebuild \\
      uv run python scripts/bootstrap_store.py \\
      --scratch s3://alt-nfp/store-rebuild --canonical s3://alt-nfp/store

Scope is national-only, 2017+ (the intended canonical scope). QCEW is fetched
live from the CEW API (not the bulk ZIPs), so only `download_ces` is wired.

The promote step copies rebuild files into the canonical prefix then deletes the
old orphans (filenames encode vintage ranges, so a plain overwrite-mirror would
leave both files and corrupt the store — the exact hazard CLAUDE.md warns about).
`scripts/mirror_store.py` is overwrite-only and is deliberately NOT used here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- .env MUST load before any nfp_* import: nfp_lookups.paths reads
#     NFP_STORE_URI at import time. ---
from dotenv import load_dotenv

load_dotenv(".env")

from nfp_download.bls.bulk import download_ces  # noqa: E402
from nfp_ingest.ces_builder import build_ces_panel  # noqa: E402
from nfp_ingest.qcew_acquire import (  # noqa: E402
    acquire_qcew_levels,
    acquire_qcew_size_native,
)
from nfp_ingest.qcew_crosswalk import build_qcew_panel  # noqa: E402
from nfp_ingest.size_class import build_size_class_panel  # noqa: E402
from nfp_lookups.paths import is_canonical_store  # noqa: E402

from nfp_vintages.calendar import advance_release_calendar  # noqa: E402
from nfp_vintages.rebuild_store import (  # noqa: E402
    compose_rebuild_panel,
    write_rebuild_store,
)


def _is_remote(uri: str) -> bool:
    return uri.startswith(("s3://", "s3a://"))


def _s3fs():
    import os

    import s3fs

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    return s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint} if endpoint else {},
    )


def _store_path(uri: str):
    """A pathlib-compatible handle for a scratch/canonical store location."""
    if _is_remote(uri):
        from upath import UPath

        import os

        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        client_kwargs = {"endpoint_url": endpoint} if endpoint else {}
        return UPath(
            uri,
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            client_kwargs=client_kwargs,
        )
    return Path(uri)


# ---------------------------------------------------------------------------
# Promote: generalized _t8_promote.py:cutover (copy-then-delete per partition)
# ---------------------------------------------------------------------------


def _local_keys(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.glob("**/*.parquet"))


def _s3_keys(fs, prefix: str) -> list[str]:
    """Genuine children of *prefix* only (store vs store-rebuild share a head)."""
    return sorted(k for k in fs.find(prefix) if k.startswith(prefix + "/"))


def _promote_local(scratch: Path, canonical: Path) -> None:
    rel_keys = _local_keys(scratch)
    if not rel_keys:
        sys.exit(f"FATAL: scratch store {scratch} is empty — refusing promote.")
    canonical.mkdir(parents=True, exist_ok=True)
    # 1) copy rebuild files in (under their rebuilt names).
    for rel in rel_keys:
        dst = canonical / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((scratch / rel).read_bytes())
    # 2) delete old-named orphans (anything canonical-side not in the new set).
    new_set = set(rel_keys)
    for p in sorted(canonical.glob("**/*.parquet")):
        if p.relative_to(canonical).as_posix() not in new_set:
            p.unlink()
    # 3) verify: canonical == exactly the rebuild set.
    final = set(_local_keys(canonical))
    if final != new_set:
        sys.exit(f"FATAL: post-promote keyset mismatch under {canonical}.")
    print(f"promote (local): +{len(new_set)} files; canonical == rebuild set, verified")


def _promote_remote(scratch_uri: str, canonical_uri: str) -> None:
    fs = _s3fs()
    src = scratch_uri.removeprefix("s3://").rstrip("/")
    dst = canonical_uri.removeprefix("s3://").rstrip("/")
    src_keys = _s3_keys(fs, src)
    if not src_keys:
        sys.exit(f"FATAL: scratch store {scratch_uri} is empty — refusing promote.")
    new_dst = {k.replace(src, dst, 1): k for k in src_keys}  # dst -> src
    # 1) copy rebuild files in (new names).
    for dst_key, src_key in new_dst.items():
        fs.pipe_file(dst_key, fs.cat_file(src_key))
    # 2) delete old-named orphans.
    for k in _s3_keys(fs, dst):
        if k not in new_dst:
            fs.rm(k)
    # 3) verify keyset.
    final = _s3_keys(fs, dst)
    if final != sorted(new_dst):
        sys.exit(f"FATAL: post-promote keyset mismatch under {canonical_uri}.")
    print(f"promote (s3): +{len(new_dst)} files; canonical == rebuild set, verified")


def _promote_scratch_to_canonical(scratch_uri: str, canonical_uri: str) -> None:
    """Copy-then-delete cutover from *scratch* to *canonical* (no overwrite-mirror)."""
    if _is_remote(canonical_uri):
        _promote_remote(scratch_uri, canonical_uri)
    else:
        _promote_local(Path(scratch_uri), Path(canonical_uri))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-time store rebuild + promote.")
    parser.add_argument(
        "--scratch",
        required=True,
        help="Scratch store URI/path (e.g. s3://alt-nfp/store-rebuild). "
        "Must NOT be the canonical store.",
    )
    parser.add_argument(
        "--canonical",
        required=True,
        help="Canonical store URI/path to promote into (e.g. s3://alt-nfp/store).",
    )
    parser.add_argument(
        "--start-year", type=int, default=2017, help="First QCEW reference year."
    )
    parser.add_argument(
        "--end-year", type=int, default=None, help="Last QCEW reference year (inclusive)."
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Build the scratch store but skip the canonical promote.",
    )
    args = parser.parse_args(argv)

    # Guard FIRST — refuse the canonical store as a scratch target.
    if is_canonical_store(args.scratch):
        sys.exit(
            f"refusing to bootstrap straight to the canonical store ({args.scratch}); "
            "target a scratch prefix (e.g. s3://alt-nfp/store-rebuild)."
        )

    print("=== Bootstrap: download CES triangular CSVs ===")
    download_ces()

    print("=== Bootstrap: advance release calendar (overlap parity) ===")
    advance_release_calendar()

    print("=== Bootstrap: build CES panel (NSA + SA) ===")
    ces = build_ces_panel()
    print(f"  CES: {ces.height:,} rows")

    print(f"=== Bootstrap: acquire QCEW levels ({args.start_year}-{args.end_year}) ===")
    raw_qcew = acquire_qcew_levels(start_year=args.start_year, end_year=args.end_year)
    qcew_levels = build_qcew_panel(raw_qcew)
    print(f"  QCEW levels: {qcew_levels.height:,} rows")

    print(f"=== Bootstrap: acquire QCEW size native ({args.start_year}-{args.end_year}) ===")
    size_native = acquire_qcew_size_native(
        start_year=args.start_year, end_year=args.end_year
    )
    size = build_size_class_panel(size_native) if size_native.height else None
    if size is not None:
        print(f"  QCEW size: {size.height:,} rows")
    else:
        print("  QCEW size: 0 rows (skipped)")

    print("=== Bootstrap: compose panels ===")
    panel = compose_rebuild_panel(ces, qcew_levels, size)
    print(f"  Combined: {panel.height:,} rows")

    print(f"=== Bootstrap: write scratch store ({args.scratch}) ===")
    write_rebuild_store(_store_path(args.scratch), panel=panel)

    if args.no_promote:
        print("Done (scratch only; --no-promote set).")
        return

    print(f"=== Bootstrap: promote {args.scratch} -> {args.canonical} ===")
    _promote_scratch_to_canonical(args.scratch, args.canonical)
    print("Done.")


if __name__ == "__main__":
    main()
```

> **Signature note (`write_rebuild_store`):** the real signature is
> `write_rebuild_store(panel, store_path=None, *, allow_canonical=False)`
> (`rebuild_store.py:579-584`) — `panel` is positional-first, `store_path` second. The call
> above passes the store path positionally and `panel=` as a keyword, which binds correctly
> (`store_path` is the second positional). If a reviewer prefers explicitness, write it as
> `write_rebuild_store(panel, _store_path(args.scratch), allow_canonical=False)`. Either form
> hits the same `is_canonical_store(out_path) and not allow_canonical` guard
> (`rebuild_store.py:612`), which already refuses a canonical scratch path even though `main`
> guards it earlier.

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_bootstrap_store.py -q --no-cov`
  Expected: **PASS** (both tests). Then run the package suite and lint:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests -q -m "not network" --no-cov`
  and `uv run ruff check scripts/bootstrap_store.py packages/nfp-vintages/`.

- [ ] **Step 5: Commit** —
  `git add scripts/bootstrap_store.py packages/nfp-vintages/src/nfp_vintages/tests/test_bootstrap_store.py`
  then
  `git commit -m "feat(bootstrap): one-time store rebuild+promote script (cli_production_workflow §10)"`

---

### Task 9.2: Retire the legacy CLI commands + bare-run from `__main__.py`

Delete the retired commands from the everyday surface (§4, §10): `download`,
`download-indicators`, `process`, `current`, `build`, `build-rebuild`, and the
`invoke_without_command=True` bare-run chain. Their reusable bodies have already moved
(QCEW acquire → `nfp_ingest.qcew_acquire` in Phase 1; the calendar scrape →
`nfp_vintages.calendar.advance_release_calendar` in Phase 3; the rebuild compose/write into
`scripts/bootstrap_store.py`, Task 9.1). The new `update`/`status`/`watch` commands (Phases
5–8) and `snapshot` (with the §4a day-12 fix) remain. The callback keeps only `load_dotenv()`
(no fallthrough run). Retirement MUST keep the suite green — no orphaned imports, no test that
invokes a deleted command.

**Files:**
- Modify: `/Users/lowell/Projects/alt-nfp/packages/nfp-vintages/src/nfp_vintages/__main__.py`
- Test: `/Users/lowell/Projects/alt-nfp/packages/nfp-vintages/src/nfp_vintages/tests/test_cli_legacy_retired.py`

- [ ] **Step 1: Write the failing test** — assert the retired command names are gone from the
  Typer app, that the surviving production commands are present, and that the bare invocation
  (no subcommand) no longer triggers a build (it should error/exit, not run a pipeline).

```python
"""Legacy CLI retirement: retired commands gone, production surface intact."""
from __future__ import annotations

from typer.testing import CliRunner

from nfp_vintages.__main__ import app

runner = CliRunner()

_RETIRED = {"download", "download-indicators", "process", "current", "build", "build-rebuild"}
_KEPT = {"update", "status", "watch", "snapshot"}


def _registered_command_names() -> set[str]:
    names: set[str] = set()
    for cmd in app.registered_commands:
        # Typer derives the CLI name from the function name (underscores -> hyphens)
        # unless an explicit name was passed to @app.command(...).
        names.add(cmd.name or cmd.callback.__name__.replace("_", "-"))
    return names


def test_legacy_commands_are_gone():
    registered = _registered_command_names()
    assert _RETIRED.isdisjoint(registered), f"retired commands still present: {registered}"


def test_production_commands_present():
    registered = _registered_command_names()
    missing = _KEPT - registered
    assert not missing, f"expected production commands missing: {missing}"


def test_retired_command_invocation_errors():
    result = runner.invoke(app, ["build"])
    assert result.exit_code != 0, "`alt-nfp build` should no longer be a command"


def test_bare_invocation_does_not_run_a_build():
    # No subcommand: the old behavior chained download->...->build. After retirement
    # the bare run must NOT silently rebuild the store; it should show help/usage and
    # exit non-zero (no_args_is_help) rather than execute a pipeline.
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "build" not in result.stdout.lower() or "Usage" in result.stdout
```

- [ ] **Step 2: Run the test, verify it fails** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_legacy_retired.py -q --no-cov`
  Expected: **FAIL** — `test_legacy_commands_are_gone` fails because `download`/`process`/`build`/
  etc. are still registered (`__main__.py:36-288`), and `test_bare_invocation_does_not_run_a_build`
  fails because the `invoke_without_command=True` callback (`__main__.py:24-33`) still chains a
  build.

- [ ] **Step 3: Implement** — rewrite `__main__.py` to drop the legacy commands and the bare-run.
  Only the production surface remains. The callback no longer takes `invoke_without_command=True`
  and only loads `.env`. `no_args_is_help=True` on the app makes a bare `alt-nfp` print usage and
  exit non-zero instead of running a pipeline. Shown here is the **header through the callback**
  plus the retired block replaced by a one-line pointer comment; the `update`/`status`/`watch`
  commands authored in Phases 5–8 and `snapshot` (Phase 5 day-12 fix) are unchanged and remain
  below this header (do not duplicate them — only the legacy block is deleted).

```python
"""Production CLI for the alt-nfp vintage store.

Usage::

    alt-nfp update --as-of T [--only ces|qcew|indicators]  # capture knowable prints for T
    alt-nfp status [--as-of T] [--store URI]               # store coverage + uncaptured alarm
    alt-nfp watch [--source ces|qcew|all] [--snapshot]     # feed-driven trigger (cron)
    alt-nfp snapshot --as-of T [--grid-end E]              # hash-pinned ModelData (day-12)

One-time historical load is a SCRIPT, not a command:

    uv run python scripts/bootstrap_store.py --scratch s3://alt-nfp/store-rebuild \\
        --canonical s3://alt-nfp/store

The legacy stage pipeline (download / download-indicators / process / current /
build / build-rebuild and the bare-run chain) was retired in the production-workflow
reshape (specs/cli_production_workflow.md §10). The calendar scrape it used now lives in
nfp_vintages.calendar.advance_release_calendar (invoked by `update`); the rebuild compose/
write moved to scripts/bootstrap_store.py.
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

app = typer.Typer(help="Production vintage-store CLI for alt-nfp.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Load environment config before any command resolves store paths."""
    load_dotenv()


# --- production commands: update / status / watch (Phases 5–8) + snapshot below ---
```

> **Deletion checklist (exact spans in the pre-Phase-9 file, before the Phases 5–8 edits land
> on top):** remove the `from pathlib import Path` import (`__main__.py:16`, only used by the
> retired `build`), the `invoke_without_command=True` callback body chain (`__main__.py:24-33`),
> `download` (`:36-47`), `download-indicators` (`:50-58`), `_build_release_calendar` (`:61-168`
> — already lifted to `nfp_vintages.calendar` in Phase 3; delete the in-file copy), `process`
> (`:171-186`), `current` (`:189-195`), `build` (`:198-215`), and `build-rebuild` (`:218-288`,
> whose only unique imports were `_acquire_qcew_*` from `rebuild_store` — those private names no
> longer exist after Phase 1, so leaving this command would orphan-import and break collection).
> Keep `snapshot` (`:291-324`) with the Phase 5 day-12 fix and the `if __name__ == '__main__': app()`
> tail.

- [ ] **Step 4: Run, verify pass** —
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests/test_cli_legacy_retired.py -q --no-cov`
  Expected: **PASS**. Then confirm **no orphaned imports / green suite**:
  `uv run pytest packages/nfp-vintages/src/nfp_vintages/tests -q -m "not network" --no-cov`
  (in particular `test_cli_update.py` and `test_update_guardrail.py` from Phases 5–6 still pass —
  they exercise the kept commands, not the retired ones), and
  `uv run ruff check packages/nfp-vintages/` (catches the now-unused `Path`/`typer.Option`
  imports if any slipped through).

- [ ] **Step 5: Commit** —
  `git add packages/nfp-vintages/src/nfp_vintages/__main__.py packages/nfp-vintages/src/nfp_vintages/tests/test_cli_legacy_retired.py`
  then
  `git commit -m "refactor(cli): retire legacy download/process/build commands + bare-run (§10)"`

---

### Task 9.3: Update CLAUDE.md command banners + package maps

§14 step 9 and §11 require the docs to reflect the new surface: the production commands
(`update`/`status`/`watch`/`snapshot`), the bootstrap **script** replacing the legacy
build chain, and the new modules (`nfp_ingest/qcew_acquire.py`, `nfp_ingest/capture.py`,
`nfp_vintages/calendar.py`, `nfp_vintages/store_status.py`,
`nfp_download/release_dates/feed.py`). This is a docs-only task (no code, no test).

**Files:**
- Modify: `/Users/lowell/Projects/alt-nfp/CLAUDE.md`
- Modify: `/Users/lowell/Projects/alt-nfp/packages/nfp-vintages/CLAUDE.md`
- Modify: `/Users/lowell/Projects/alt-nfp/packages/nfp-ingest/CLAUDE.md`
- Modify: `/Users/lowell/Projects/alt-nfp/packages/nfp-download/CLAUDE.md`

- [ ] **Step 1: (no failing test — docs only).** Instead, grep to confirm the stale banners that
  must change are present before editing:
  `grep -rn "alt-nfp download\|alt-nfp process\|alt-nfp build\b\|build-rebuild\|invoke_without_command\|bare alt-nfp" CLAUDE.md packages/*/CLAUDE.md`
  Expected: matches in root `CLAUDE.md` (the `uv run alt-nfp --help` command banner region) and
  `packages/nfp-vintages/CLAUDE.md` (the "Key Commands" + "CLI" map blocks). These are the lines
  to rewrite.

- [ ] **Step 2: Edit root `CLAUDE.md`.** In the `## Commands` block, replace the single
  `uv run alt-nfp --help` line's surrounding context so the production surface is documented
  and the bootstrap is a script. Concretely, append after the existing
  `uv run alt-nfp --help` line:

```markdown
uv run alt-nfp update --as-of 2026-01-12       # capture knowable month-T prints, append to store
uv run alt-nfp status                          # store coverage + uncaptured/corrected alarm
uv run alt-nfp watch --source all              # BLS-feed-driven trigger (cron)
uv run python scripts/bootstrap_store.py \      # one-time historical rebuild + promote (NOT a command)
    --scratch s3://alt-nfp/store-rebuild --canonical s3://alt-nfp/store
```

  And, in the **Hard rules → "Rebuild to scratch; promote deliberately"** bullet, update the
  "never `alt-nfp build` straight to `…/store`" sentence to reference the script: the everyday
  CLI no longer has a `build` command; the one-time rebuild path is
  `scripts/bootstrap_store.py` (scratch prefix → deliberate copy-then-delete promote), with the
  `is_canonical_store` guard still refusing a canonical scratch target.

- [ ] **Step 3: Edit `packages/nfp-vintages/CLAUDE.md`.** Replace the entire "Key Commands"
  fenced block and the `__main__.py` map line so they describe the production surface and the
  script, and add `calendar.py` + `store_status.py` to the package structure map:

```markdown
## Key Commands

# Production month-T workflow (specs/cli_production_workflow.md)
uv run alt-nfp update --as-of 2026-01-12 [--only ces|qcew|indicators]  # capture + append
uv run alt-nfp status [--as-of 2026-01-12] [--store URI]               # coverage report
uv run alt-nfp watch [--source ces|qcew|all] [--snapshot]              # feed-driven (cron)
uv run alt-nfp snapshot --as-of 2026-01-12 [--grid-end 2026-06-12]     # hash-pinned ModelData

# One-time historical rebuild + promote — a SCRIPT, not a CLI command:
uv run python scripts/bootstrap_store.py --scratch s3://alt-nfp/store-rebuild \
    --canonical s3://alt-nfp/store

# Run vintage tests / lint
pytest src/nfp_vintages/tests/
ruff check src/nfp_vintages/
```

  Update the package-structure comment for `__main__.py` to
  `# CLI entry point (update/status/watch/snapshot; legacy build chain retired §10)` and add
  two map lines:
  `├── calendar.py             # advance_release_calendar() — release-calendar scrape (lifted from __main__)`
  and
  `├── store_status.py         # compute_status()/format_status() — read-only coverage report (status)`.
  Also update the "CLI (`__main__.py`): typer app with subcommands…" bullet under **Key
  Patterns** to list `update`/`status`/`watch`/`snapshot` and note the legacy stage commands
  were retired (bootstrap is now `scripts/bootstrap_store.py`).

- [ ] **Step 4: Edit `packages/nfp-ingest/CLAUDE.md` and `packages/nfp-download/CLAUDE.md`.**
  In `nfp-ingest`'s package-structure map, add:
  `├── qcew_acquire.py         # acquire_qcew_levels()/acquire_qcew_size_native() — CEW API slices (was private in nfp-vintages)`
  and
  `├── capture.py              # capture_ces_print()/capture_qcew_quarter() — month-T current-print → store (update)`.
  In `nfp-download`'s `release_dates/` map, add:
  `│   ├── feed.py              # parse_feed()/fetch_feed() — BLS empsit/cewqtr RSS (curl_cffi impersonation, for watch)`.
  Then verify no banner still advertises a retired command:
  `grep -rn "alt-nfp download\|alt-nfp process\|alt-nfp build\b\|build-rebuild" CLAUDE.md packages/*/CLAUDE.md`
  Expected: **no matches** (every legacy-command reference has been rewritten to the production
  surface or the script).

- [ ] **Step 5: Commit** —
  `git add CLAUDE.md packages/nfp-vintages/CLAUDE.md packages/nfp-ingest/CLAUDE.md packages/nfp-download/CLAUDE.md`
  then
  `git commit -m "docs(cli): update CLAUDE.md banners/maps for production workflow + bootstrap script (§14.9)"`

---

**Phase 9 done-check (run before declaring complete):**
- `uv run pytest -m "not network and not slow" --no-cov` — full fast suite green (no orphaned
  imports from the legacy deletion; the bootstrap smoke + legacy-retired tests pass).
- `uv run ruff check .` — clean.
- Manual: `uv run python scripts/bootstrap_store.py --help` prints the argparse usage (script is
  importable and not a Typer command); `uv run alt-nfp --help` lists only
  `update`/`status`/`watch`/`snapshot`.
