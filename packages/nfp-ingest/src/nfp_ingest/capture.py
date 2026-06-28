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
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from nfp_lookups.industry import ownership_for
from nfp_lookups.paths import VINTAGE_DATES_PATH, VINTAGE_STORE_PATH, storage_options_for
from nfp_lookups.revision_schedules import get_qcew_vintage_date
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

from nfp_ingest.ces_alfred import build_ces_alfred_window
from nfp_ingest.qcew_acquire import acquire_qcew_levels, acquire_qcew_size_native
from nfp_ingest.qcew_crosswalk import build_qcew_panel
from nfp_ingest.releases import _fetch_ces_releases
from nfp_ingest.size_class import build_size_class_panel
from nfp_ingest.vintage_store import (
    append_to_vintage_store,
    compact_partition,
    read_vintage_store,
)

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


# ---------------------------------------------------------------------------
# QCEW conditional quarter capture (spec §5.2)
# ---------------------------------------------------------------------------

# How far back to scan for a knowable quarter. QCEW rev-0 lags the reference
# quarter by ~5 months, so 8 candidate quarters (2 years) always covers the
# newest-knowable quarter for any monthly as-of.
_QCEW_CANDIDATE_QUARTERS = 8


def _knowable_qcew_quarter(as_of: date) -> tuple[str, int] | None:
    """Most recent QCEW quarter whose rev-0 ``vintage_date`` is ``<= as_of``.

    Iterates candidate ``(ref_quarter, ref_year)`` pairs newest-first and returns
    the first whose ``get_qcew_vintage_date(..., revision=0)`` is on or before
    ``as_of``. Returns ``None`` when no candidate is knowable yet (the steady-state
    monthly no-op — QCEW is quarterly, §5.2).

    Requires the §5.0 calendar to be advanced so the schedule returns real release
    dates rather than the day-1 lag fallback (``revision_schedules.py:358-365``).

    Parameters
    ----------
    as_of : date
        Knowability cutoff.

    Returns
    -------
    tuple[str, int] | None
        ``(ref_quarter, ref_year)`` for the newest knowable quarter (e.g.
        ``("Q1", 2024)``), or ``None`` when none is knowable as of ``as_of``.
    """
    # The quarter containing ``as_of`` cannot have been published yet, so start
    # from the previous quarter and walk back.
    q = (as_of.month - 1) // 3 + 1
    year = as_of.year
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

    Otherwise (spec §5.2) fetches the containing **year** via the relocated public
    acquire helpers, filters to the single knowable quarter, runs the crosswalk
    (``build_qcew_panel`` for levels, ``build_size_class_panel`` for the Q1 size
    cross-product), null-fills the ``size_class_*`` columns the levels builder
    omits, censors ``vintage_date <= as_of``, runs the §6.3 corrected-level
    comparison, then appends + compacts the ``(qcew, seasonally_adjusted=False)``
    partition. QCEW is NSA-only, so every row is tagged ``revision=0`` /
    ``seasonally_adjusted=False`` by the builders.

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

    # Already-stored short-circuit (spec §5.2): the docstring's "or the newest
    # knowable quarter is already in the store" no-op — return BEFORE the network
    # fetch so the steady-state monthly run does no work. Read-only; mirrors
    # _detect_corrected_levels' existence guard, so it is container-safe and uses
    # the passed store_path (hermetic under pytest — no wipe risk). All QCEW in
    # this store is rev-0 (acquire tags revision=0), so a rev-0 row in any month
    # of the quarter means the quarter is captured.
    partition_dir = store_path / "source=qcew" / "seasonally_adjusted=false"
    if partition_dir.exists():
        quarter_months = [date(ref_year, (qtr - 1) * 3 + m, 1) for m in (1, 2, 3)]
        already = (
            read_vintage_store(store_path, source="qcew", seasonally_adjusted=False)
            .filter(pl.col("ref_date").is_in(quarter_months) & (pl.col("revision") == 0))
            .select("ref_date")
            .head(1)
            .collect()
        )
        if already.height:
            logger.info("QCEW: %s %d already stored — skipping", ref_quarter, ref_year)
            return CaptureResult(appended=0, corrected=[], skipped=1)

    # Fetch the containing YEAR (the helpers loop over full years), then filter to
    # the one knowable quarter. The levels endpoint carries year+qtr; the size
    # endpoint is Q1-only by URL path, so the size leg only runs for Q1.
    raw_levels = acquire_qcew_levels(ref_year, ref_year)
    raw_levels_q = raw_levels.filter(
        (pl.col("year").cast(pl.Int64) == ref_year)
        & (pl.col("qtr").cast(pl.Int64) == qtr)
    )
    levels = build_qcew_panel(raw_levels_q)
    # build_qcew_panel's .select omits size_class_* (qcew_crosswalk.py); the store
    # schema requires them, so null-fill before append.
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
        .select(list(VINTAGE_STORE_SCHEMA))
        .cast(VINTAGE_STORE_SCHEMA)
    )

    # Censor to the knowability cutoff.
    new_rows = new_rows.filter(pl.col("vintage_date") <= as_of)
    if new_rows.height == 0:
        logger.info(
            "QCEW: %s %d knowable but no rows survive vintage_date <= %s",
            ref_quarter,
            ref_year,
            as_of,
        )
        return CaptureResult(appended=0, corrected=[], skipped=1)

    # §6.3 corrected-level comparison BEFORE the append anti-join.
    corrected = _detect_corrected_levels(
        new_rows, store_path, source="qcew", seasonally_adjusted=False
    )
    for c in corrected:
        logger.warning(
            "CORRECTED-LEVEL qcew %s rev=%d bmr=%d: stored=%.1f incoming=%.1f",
            c.ref_date,
            c.revision,
            c.benchmark_revision,
            c.stored_employment,
            c.incoming_employment,
        )

    appended = append_to_vintage_store(new_rows, store_path)
    compact_partition(store_path, source="qcew", seasonally_adjusted=False)

    skipped = 0 if appended else 1
    logger.info(
        "QCEW: captured %s %d — appended %d rows (%d corrected)",
        ref_quarter,
        ref_year,
        appended,
        len(corrected),
    )
    return CaptureResult(appended=appended, corrected=corrected, skipped=skipped)


# ---------------------------------------------------------------------------
# CES ALFRED window capture (spec §5 / §6 extended)
# ---------------------------------------------------------------------------


def _ces_store_frontier(store_path: Path) -> date:
    """Max CES ``vintage_date`` currently in the store (or a low sentinel if empty).

    Computes the newest ``vintage_date`` across all CES rows in the store to
    determine the frontier for ALFRED window capture. Returns a low sentinel
    (``1900-01-01``) when the store partition does not exist or is empty,
    allowing the caller to decide whether to backfill from the beginning or
    append from a known frontier.

    Parameters
    ----------
    store_path : Path
        Root of the Hive-partitioned vintage store.

    Returns
    -------
    date
        Maximum ``vintage_date`` currently in the store, or ``date(1900, 1, 1)``
        if the CES partition is absent or empty.
    """
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
                "CORRECTED-LEVEL ces sa=%s ref=%s code=%s rev=%s "
                "stored=%s incoming=%s",
                sa,
                c.ref_date,
                c.industry_code,
                c.revision,
                c.stored_employment,
                c.incoming_employment,
            )
        corrected.extend(cl)

    if dry_run:
        return CaptureResult(appended=0, corrected=corrected, skipped=rows.height)

    appended = append_to_vintage_store(rows, store_path)
    for sa in sas:
        compact_partition(store_path, source="ces", seasonally_adjusted=sa)
    return CaptureResult(appended=appended, corrected=corrected, skipped=rows.height - appended)
