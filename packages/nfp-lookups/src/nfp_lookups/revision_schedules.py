"""QCEW, CES, and SAE revision schedules.

Encodes the structural lag patterns and noise multipliers for each vintage
of QCEW and CES data. Provides helper functions to compute publication
dates (lag-based) and lookup noise multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class PublicationCalendar:
    """Actual BLS publication dates for exact vintage-date lookups.

    When available, ``get_qcew_vintage_date`` and ``get_ces_vintage_date``
    prefer exact dates from these DataFrames before falling back to the
    lag-based approximation in ``RevisionSpec``.
    """

    ces_release_dates: pl.DataFrame  # ref_month, revision_number, pub_date
    qcew_release_dates: pl.DataFrame  # ref_quarter, ref_year, revision_number, pub_date


_DEFAULT_CALENDAR: PublicationCalendar | None = None


def _load_vintage_dates() -> pl.DataFrame | None:
    """Try to load vintage_dates.parquet from the intermediate directory."""
    from .paths import INTERMEDIATE_DIR

    path = INTERMEDIATE_DIR / "vintage_dates.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def get_default_calendar() -> PublicationCalendar:
    """Return the default PublicationCalendar, lazily built on first call.

    Loads exact dates from ``vintage_dates.parquet`` when available.
    Falls back to empty DataFrames (all lookups use lag approximation).
    """
    global _DEFAULT_CALENDAR
    if _DEFAULT_CALENDAR is not None:
        return _DEFAULT_CALENDAR

    vdf = _load_vintage_dates()

    if vdf is not None and vdf.height > 0:
        ces_rows = (
            vdf.filter(pl.col("publication") == "ces")
            .rename({"ref_date": "ref_month", "vintage_date": "pub_date", "revision": "revision_number"})
            .select("ref_month", "revision_number", "pub_date")
        )
        qcew_rows = (
            vdf.filter(pl.col("publication") == "qcew")
            .with_columns(
                pl.col("ref_date").dt.month().map_elements(
                    lambda m: f"Q{(m - 1) // 3 + 1}", return_dtype=pl.Utf8
                ).alias("ref_quarter"),
                pl.col("ref_date").dt.year().alias("ref_year"),
            )
            .rename({"vintage_date": "pub_date", "revision": "revision_number"})
            .select("ref_quarter", "ref_year", "revision_number", "pub_date")
        )
    else:
        ces_rows = pl.DataFrame(
            schema={"ref_month": pl.Date, "revision_number": pl.Int32, "pub_date": pl.Date}
        )
        qcew_rows = pl.DataFrame(
            schema={
                "ref_quarter": pl.Utf8, "ref_year": pl.Int32,
                "revision_number": pl.Int32, "pub_date": pl.Date,
            }
        )

    _DEFAULT_CALENDAR = PublicationCalendar(
        ces_release_dates=ces_rows,
        qcew_release_dates=qcew_rows,
    )
    return _DEFAULT_CALENDAR


@dataclass(frozen=True)
class RevisionSpec:
    """Specification for a single revision vintage.

    Parameters
    ----------
    revision_number : int
        0 = initial publication, 1+ = subsequent revisions, -1 = benchmark.
    lag_months : int
        Months from end of reference period to publication.
    noise_multiplier : float
        Relative noise scaling vs final vintage (final = 1.0).
    """

    revision_number: int
    lag_months: int
    noise_multiplier: float


# QCEW revision schedule by reference quarter.
# Q1 is revised 4 times (5 total vintages: initial + 4 revisions)
# Q2 is revised 3 times (4 total vintages)
# Q3 is revised 2 times (3 total vintages)
# Q4 is revised 1 time  (2 total vintages)
#
# This asymmetry arises from the annual file structure: each annual
# QCEW release revises all quarters of its reference year, so earlier
# quarters accumulate more revision opportunities.
#
# noise_multiplier: empirical recalibration (2017+, excl 2020-2021).
# Q1 (January-heavy) has large revision RMSE through rev 3; Q2-Q4 follow M3 ratios.
QCEW_REVISIONS: dict[str, list[RevisionSpec]] = {
    'Q1': [
        RevisionSpec(revision_number=0, lag_months=5, noise_multiplier=25.0),
        RevisionSpec(revision_number=1, lag_months=8, noise_multiplier=20.0),
        RevisionSpec(revision_number=2, lag_months=14, noise_multiplier=15.0),
        RevisionSpec(revision_number=3, lag_months=20, noise_multiplier=5.0),
        RevisionSpec(revision_number=4, lag_months=26, noise_multiplier=1.0),
    ],
    'Q2': [
        RevisionSpec(revision_number=0, lag_months=5, noise_multiplier=17.0),
        RevisionSpec(revision_number=1, lag_months=11, noise_multiplier=4.0),
        RevisionSpec(revision_number=2, lag_months=17, noise_multiplier=1.2),
        RevisionSpec(revision_number=3, lag_months=23, noise_multiplier=1.0),
    ],
    'Q3': [
        RevisionSpec(revision_number=0, lag_months=5, noise_multiplier=17.0),
        RevisionSpec(revision_number=1, lag_months=14, noise_multiplier=4.0),
        RevisionSpec(revision_number=2, lag_months=20, noise_multiplier=1.0),
    ],
    'Q4': [
        RevisionSpec(revision_number=0, lag_months=8, noise_multiplier=17.0),
        RevisionSpec(revision_number=1, lag_months=17, noise_multiplier=1.0),
    ],
}


# CES revision schedule (same for all reference months).
# v0 (first print): published ~1 month after reference, noisiest
# v1 (second estimate): ~2 months after, revised
# v2 (third estimate): ~3 months after, revised again
# v-1 (benchmark): ~13 months after, annual benchmark revision
CES_REVISIONS: list[RevisionSpec] = [
    RevisionSpec(revision_number=0, lag_months=1, noise_multiplier=3.0),
    RevisionSpec(revision_number=1, lag_months=2, noise_multiplier=2.0),
    RevisionSpec(revision_number=2, lag_months=3, noise_multiplier=1.5),
    RevisionSpec(revision_number=-1, lag_months=13, noise_multiplier=1.0),
]


# ---------------------------------------------------------------------------
# Private helpers for building revision rows from hard-coded dicts
# ---------------------------------------------------------------------------


def _offset_month(d: date, months: int) -> date:
    """Add *months* to a date, returning the 1st of the resulting month."""
    total = d.month + months
    year = d.year + (total - 1) // 12
    month = ((total - 1) % 12) + 1
    return date(year, month, 1)


def _build_monthly_revision_rows(
    release_dates: dict[date, date],
    revision_specs: list[RevisionSpec] | None,
) -> list[dict]:
    """Build revision rows for a monthly program (CES or SAE).

    For revision 0, uses the exact date from *release_dates*.
    For revision N > 0, looks up the release date for
    ``ref_month + N months`` — the next program release that carries the
    revised estimate.  Benchmark (revision -1) is skipped.

    Parameters
    ----------
    release_dates : dict[date, date]
        Mapping ref_month (1st of month) → first-print pub_date.
    revision_specs : list[RevisionSpec] or None
        If provided, determines which non-negative revision numbers to
        generate.  If *None*, only revision 0 is generated.

    Returns
    -------
    list[dict]
        Rows with keys: ref_month, revision_number, pub_date.
    """
    if revision_specs is not None:
        rev_numbers = sorted(
            s.revision_number for s in revision_specs if s.revision_number >= 0
        )
    else:
        rev_numbers = [0]

    rows: list[dict] = []
    for ref in sorted(release_dates):
        for rev_num in rev_numbers:
            if rev_num == 0:
                rows.append({
                    'ref_month': ref,
                    'revision_number': rev_num,
                    'pub_date': release_dates[ref],
                })
            else:
                forward_month = _offset_month(ref, rev_num)
                if forward_month in release_dates:
                    rows.append({
                        'ref_month': ref,
                        'revision_number': rev_num,
                        'pub_date': release_dates[forward_month],
                    })
    return rows


def _build_qcew_revision_rows(
    release_dates: dict[tuple[int, int], date],
) -> list[dict]:
    """Build QCEW revision rows (revision 0 only from hard-coded dates).

    Revision > 0 is not populated here; the lag-based approximation
    in :func:`get_qcew_vintage_date` handles those.

    Returns
    -------
    list[dict]
        Rows with keys: ref_quarter, ref_year, revision_number, pub_date.
    """
    rows: list[dict] = []
    for (yr, q), pub_date in sorted(release_dates.items()):
        rows.append({
            'ref_quarter': f'Q{q}',
            'ref_year': yr,
            'revision_number': 0,
            'pub_date': pub_date,
        })
    return rows


# ---------------------------------------------------------------------------
# Private lookup helpers for exact date retrieval from DataFrames
# ---------------------------------------------------------------------------


def _lookup_monthly_date(
    df: pl.DataFrame, ref_month: date, revision: int,
) -> date | None:
    """Look up an exact date in a CES/SAE-style DataFrame.

    Returns
    -------
    date or None
        Exact publication date, or *None* if not found.
    """
    result = df.filter(
        (pl.col('ref_month') == ref_month)
        & (pl.col('revision_number') == revision)
    )
    if result.height == 0:
        return None
    return result.item(0, 'pub_date')


def _lookup_qcew_date(
    df: pl.DataFrame,
    ref_quarter: str,
    ref_year: int,
    revision: int,
) -> date | None:
    """Look up an exact date in a QCEW-style DataFrame.

    Returns
    -------
    date or None
        Exact publication date, or *None* if not found.
    """
    result = df.filter(
        (pl.col('ref_quarter') == ref_quarter)
        & (pl.col('ref_year') == ref_year)
        & (pl.col('revision_number') == revision)
    )
    if result.height == 0:
        return None
    return result.item(0, 'pub_date')


# ---------------------------------------------------------------------------
# Public vintage-date functions
# ---------------------------------------------------------------------------


def get_qcew_vintage_date(
    ref_quarter: str,
    ref_year: int,
    revision: int,
    calendar: PublicationCalendar | None = None,
) -> date:
    """Compute the publication date for a specific QCEW vintage.

    When a *calendar* is provided (or the default calendar contains an
    exact date), returns the exact date.  Otherwise falls back to the
    lag-based approximation.

    Parameters
    ----------
    ref_quarter : str
        Reference quarter ('Q1' through 'Q4').
    ref_year : int
        Reference year.
    revision : int
        Revision number (0 = initial, 1+ = subsequent).
    calendar : PublicationCalendar or None
        If provided, prefer exact dates.  Defaults to the module-level
        calendar built from hard-coded dates.

    Returns
    -------
    date
        Publication date (exact if available, approximate otherwise).

    Raises
    ------
    ValueError
        If ref_quarter is invalid or revision number exceeds available vintages.
    """
    if calendar is None:
        calendar = get_default_calendar()

    exact = _lookup_qcew_date(
        calendar.qcew_release_dates, ref_quarter, ref_year, revision,
    )
    if exact is not None:
        return exact

    # Fall back to lag-based approximation
    if ref_quarter not in QCEW_REVISIONS:
        raise ValueError(f'Invalid ref_quarter: {ref_quarter}. Must be Q1-Q4.')

    specs = QCEW_REVISIONS[ref_quarter]
    matching = [s for s in specs if s.revision_number == revision]
    if not matching:
        max_rev = max(s.revision_number for s in specs)
        raise ValueError(
            f'Revision {revision} not available for {ref_quarter}. '
            f'Max revision is {max_rev}.'
        )

    lag = matching[0].lag_months

    # End of reference quarter
    quarter_num = int(ref_quarter[1])
    end_month = quarter_num * 3
    # Approximate: add lag_months to end of reference quarter
    pub_month = end_month + lag
    pub_year = ref_year + (pub_month - 1) // 12
    pub_month = ((pub_month - 1) % 12) + 1

    return date(pub_year, pub_month, 1)


def get_ces_vintage_date(
    ref_month: date,
    revision: int,
    calendar: PublicationCalendar | None = None,
) -> date:
    """Compute the publication date for a specific CES vintage.

    When a *calendar* is provided (or the default calendar contains an
    exact date), returns the exact date.  Otherwise falls back to the
    lag-based approximation.

    Parameters
    ----------
    ref_month : date
        First of the reference month.
    revision : int
        Revision number (0 = first print, 1 = second, 2 = third,
        -1 = benchmark).
    calendar : PublicationCalendar or None
        If provided, prefer exact dates.  Defaults to the module-level
        calendar built from hard-coded dates.

    Returns
    -------
    date
        Publication date (exact if available, approximate otherwise).

    Raises
    ------
    ValueError
        If revision number is not in the CES revision schedule.
    """
    if calendar is None:
        calendar = get_default_calendar()

    exact = _lookup_monthly_date(calendar.ces_release_dates, ref_month, revision)
    if exact is not None:
        return exact

    # Fall back to lag-based approximation
    matching = [s for s in CES_REVISIONS if s.revision_number == revision]
    if not matching:
        valid = sorted({s.revision_number for s in CES_REVISIONS})
        raise ValueError(
            f'Revision {revision} not in CES schedule. Valid: {valid}.'
        )

    lag = matching[0].lag_months
    total_months = ref_month.month + lag
    pub_year = ref_month.year + (total_months - 1) // 12
    pub_month = ((total_months - 1) % 12) + 1

    return date(pub_year, pub_month, 1)


# def get_sae_vintage_date(
#     ref_month: date,
#     revision: int,
#     calendar: PublicationCalendar | None = None,
# ) -> date:
#     """Compute the publication date for a specific SAE vintage."""
#     if calendar is None:
#         calendar = get_default_calendar()
#
#     exact = _lookup_monthly_date(calendar.sae_release_dates, ref_month, revision)
#     if exact is not None:
#         return exact
#
#     lag = 2 + revision
#     total_months = ref_month.month + lag
#     pub_year = ref_month.year + (total_months - 1) // 12
#     pub_month = ((total_months - 1) % 12) + 1
#
#     return date(pub_year, pub_month, 1)


def get_noise_multiplier(source: str, revision_number: int) -> float:
    """Lookup noise multiplier for a given source and revision number.

    Parameters
    ----------
    source : str
        Source identifier. For QCEW, use 'qcew_Q1' through 'qcew_Q4'.
        For CES, use 'ces'. For payroll providers, returns 1.0.
    revision_number : int
        Revision number to look up.

    Returns
    -------
    float
        Noise multiplier (1.0 = final vintage noise level).
    """
    if source.startswith('qcew_Q'):
        quarter = source.replace('qcew_', '')
        if quarter in QCEW_REVISIONS:
            for spec in QCEW_REVISIONS[quarter]:
                if spec.revision_number == revision_number:
                    return spec.noise_multiplier
        return 1.0
    elif source.startswith('ces'):
        for spec in CES_REVISIONS:
            if spec.revision_number == revision_number:
                return spec.noise_multiplier
        return 1.0
    else:
        # Payroll providers are not revised
        return 1.0
