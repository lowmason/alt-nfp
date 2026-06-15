"""Store-rebuild orchestration: compose panels + guarded scratch write (store_rebuild T5).

Composes the three source panels (CES, QCEW levels, QCEW size) into one
``VINTAGE_STORE_SCHEMA`` frame and writes it to a **scratch** store,
refusing the canonical ``s3://alt-nfp/store``.

The acquire layer (QCEW API-slice fetchers + size NAICS→CES crosswalk) fetches
public BLS API slices over plain httpx (``data.bls.gov`` needs no impersonation;
only www.bls.gov is Akamai-fingerprinted) and transforms them into frames ready
for :func:`~nfp_ingest.qcew_crosswalk.build_qcew_panel`
and :func:`~nfp_ingest.size_class.build_size_class_panel`.

Usage::

    uv run alt-nfp build-rebuild [--allow-canonical]
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any

import httpx
import polars as pl
from nfp_lookups.industry import QCEW_AREA_NATIONAL
from nfp_lookups.paths import (
    VINTAGE_STORE_PATH,
    is_canonical_store,
    is_remote,
    storage_options_for,
)
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

logger = logging.getLogger(__name__)

# First year of the rebuild scope.  The rebuild covers 2017-present so that
# QCEW revisions pre-2017 (not needed by the model) aren't fetched.
_REBUILD_START_YEAR: int = 2017

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

    Filters to private establishments (``own_code == '5'``), selects the
    columns required by ``build_qcew_panel``, casts the three
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
        only, ``month*_emplvl`` as ``Int64``, ``revision`` as ``Int64``.
    """
    # No disclosure filter here (unlike the size path): the area endpoint's
    # all-sizes national aggregates at agglvl 13–16 are large cells BLS does not
    # suppress. Suppression only bites the finer size×industry cells (see
    # _size_raw_to_native step 2).
    return (
        df.filter(pl.col("own_code") == "5")
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


def _acquire_qcew_levels(
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
                logger.info("  %d rows after private filter", prepped.height)
                slices.append(prepped)

    if not slices:
        raise RuntimeError(
            "_acquire_qcew_levels: no QCEW area slices fetched — "
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


def _acquire_qcew_size_native(
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
            "_acquire_qcew_size_native: no QCEW size slices fetched — "
            "check network access and BLS API availability"
        )

    raw_size = pl.concat(slices, how="diagonal_relaxed")
    return _size_raw_to_native(raw_size)


# ---------------------------------------------------------------------------
# Core compose function
# ---------------------------------------------------------------------------


def compose_rebuild_panel(
    ces: pl.DataFrame,
    qcew_levels: pl.DataFrame,
    size: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compose CES + QCEW-levels + optional QCEW-size into one store-schema frame.

    Parameters
    ----------
    ces : pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant rows from
        :func:`nfp_ingest.ces_builder.build_ces_panel`.
        Has ``size_class_type``/``size_class_code`` (both null).
    qcew_levels : pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant rows from
        :func:`nfp_ingest.qcew_crosswalk.build_qcew_panel`.
        **Omits** ``size_class_type``/``size_class_code`` entirely (the builder's
        ``.select(...)`` ends before those cols); ``diagonal_relaxed`` null-fills them.
    size : pl.DataFrame or None
        ``VINTAGE_STORE_SCHEMA``-conformant rows from
        :func:`nfp_ingest.size_class.build_size_class_panel` with non-null
        ``size_class_type``/``size_class_code``.  When provided, QCEW level rows
        whose series-month has size coverage are replaced by the size frame's
        ``total``/``'0'`` (all-sizes) + bucket rows — preventing double-counting
        under the §7 ``IS NULL OR size_class_code='0'`` selector.

    Returns
    -------
    pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant frame (all 14 cols including
        ``source``/``seasonally_adjusted``) sorted deterministically.

    Notes
    -----
    §7 anti-join: when *size* is provided, a ``qcew_levels`` row is dropped
    **only** when the size frame contains a ``total``/``'0'`` row for the
    same ``(geographic_type, geographic_code, ownership, industry_type,
    industry_code, ref_date)`` — i.e. row existence on a 6-col key, not a
    month/quarter filter.  Industry-months without size coverage keep their
    null-size level row (partial coverage).
    """
    if size is not None:
        # Derive the "has size coverage" key set from size total/'0' rows only.
        # These are exactly the rows that would double-count against a null-size
        # qcew_levels row under the §7 all-sizes predicate.
        coverage_keys = (
            size.filter(pl.col("size_class_code") == "0")
            .select(_SERIES_IDENTITY_KEY)
            .unique()
        )
        # Anti-join: drop qcew_levels rows that have size coverage.
        # Done BEFORE concat so the coverage keys never collide with the CES
        # or size frame rows during the join.
        qcew_to_union = qcew_levels.join(coverage_keys, on=_SERIES_IDENTITY_KEY, how="anti")
    else:
        qcew_to_union = qcew_levels

    # Build the parts list for diagonal_relaxed concat.
    # diagonal_relaxed null-fills qcew_to_union's missing size_class_* columns.
    parts: list[pl.DataFrame] = [ces, qcew_to_union]
    if size is not None:
        parts.append(size)

    combined = pl.concat(parts, how="diagonal_relaxed")

    # Pin canonical column order AND every dtype. ``diagonal_relaxed`` is the
    # *relaxed* concat: when the three input frames disagree on a column's dtype
    # it coerces to a common supertype (e.g. a builder emitting ``revision`` as
    # i64, or the all-null ``size_class_*`` columns landing as Null instead of
    # Utf8). The explicit cast makes the "VINTAGE_STORE_SCHEMA-conformant"
    # contract enforced here, not dependent on the builders happening to agree.
    combined = combined.select(list(VINTAGE_STORE_SCHEMA.keys())).cast(
        dict(VINTAGE_STORE_SCHEMA)
    )

    return combined.sort(
        "source",
        "industry_type",
        "industry_code",
        "ref_date",
        "size_class_type",
        "size_class_code",
        "vintage_date",
        "revision",
    )


# ---------------------------------------------------------------------------
# Guarded Hive-partition write
# ---------------------------------------------------------------------------


def write_rebuild_store(
    panel: pl.DataFrame,
    store_path: Any = None,
    *,
    allow_canonical: bool = False,
) -> None:
    """Write *panel* as a Hive-partitioned parquet store, targeting a scratch prefix.

    Mirrors the write half of :func:`nfp_vintages.build_store.build_store`.
    The canonical guard is the **first** statement — no I/O happens before it fires.

    Parameters
    ----------
    panel : pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant frame including ``source`` and
        ``seasonally_adjusted`` partition columns.
    store_path : Path-like or None
        Output store root.  Defaults to :data:`nfp_lookups.paths.VINTAGE_STORE_PATH`.
        The intended target for rebuilds is the scratch prefix
        ``s3://alt-nfp/store-rebuild`` (set via ``NFP_STORE_URI``).
    allow_canonical : bool
        Permit writing to the canonical store.  Defaults to ``False``.
        Passing ``True`` is dangerous — see root ``CLAUDE.md``.

    Raises
    ------
    RuntimeError
        If *store_path* is the canonical store (``s3://alt-nfp/store`` or
        equivalent) and *allow_canonical* is ``False``.
    """
    out_path = store_path if store_path is not None else VINTAGE_STORE_PATH

    # Guard is first — no I/O before this check.
    if is_canonical_store(out_path) and not allow_canonical:
        raise RuntimeError(
            "refusing to write the canonical store in place "
            f"({out_path}); target a scratch prefix (e.g. s3://alt-nfp/store-rebuild) "
            "or pass allow_canonical=True. "
            "See CLAUDE.md 'Never rebuild the canonical store in place'."
        )

    if not is_remote(out_path):
        out_path.mkdir(parents=True, exist_ok=True)

    for (source, sa), partition_df in panel.group_by(
        ["source", "seasonally_adjusted"], maintain_order=True,
    ):
        sa_str = str(sa).lower()
        partition_dir = out_path / f"source={source}" / f"seasonally_adjusted={sa_str}"

        if not is_remote(out_path):
            partition_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing parquet files in this partition before writing.
        if partition_dir.exists():
            for f in partition_dir.glob("*.parquet"):
                f.unlink()

        write_df = partition_df.drop(["source", "seasonally_adjusted"])
        vmin = write_df["vintage_date"].min()
        vmax = write_df["vintage_date"].max()
        # Polars aggregates skip nulls; an all-null vintage_date would silently
        # name the file ``v_None_None.parquet``. Fail loud — the rebuild path
        # pulls from API slices where a missing vintage_date signals bad data.
        if vmin is None or vmax is None:
            raise ValueError(
                f"partition (source={source}, seasonally_adjusted={sa_str}) has "
                "null vintage_date values; cannot name the output file"
            )
        fname = f"v_{vmin}_{vmax}.parquet"

        write_df.write_parquet(
            str(partition_dir / fname),
            storage_options=storage_options_for(out_path),
        )
        print(f"  {partition_dir.name}: {write_df.height:,} rows → {fname}")

    print(f"Wrote rebuild store to {out_path}")
