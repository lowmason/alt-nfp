"""Build vintage_dates dataset from release_dates.parquet with revision codes.

Revision semantics (publication-specific):

- **CES**: revisions 0, 1, 2 + benchmark rows (revision=2, benchmark_revision=1)
  for all months of each benchmarked year. Benchmark triggered when Jan Y+1 is
  published (typically in February).
- **SAE**: revisions 0, 1 + two benchmarks (benchmark_revision=1 and 2). Benchmark
  triggered by March Y+1 and March Y+2 SAE releases.
- **QCEW**: revisions 0-4 depending on quarter. Q1 gets 4 revisions, Q2 gets 3,
  Q3 gets 2, Q4 gets 1.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
from nfp_download.release_dates.config import RELEASE_DATES_PATH, VINTAGE_DATES_PATH

CES_MONTHLY_REVISIONS = [0, 1, 2]
# SAE_MONTHLY_REVISIONS = [0, 1]

# Years covered by the BLS archive scrape.  Ref_dates before this are filled
# with first-Friday estimates via _generate_ces_pre_scrape_dates().
_CES_SCRAPE_START_YEAR = 2008

# Earliest QCEW ref_date present in the BLS archive scrape (Q3 2007).
# Quarters before this are filled via _generate_qcew_pre_scrape_dates().
_QCEW_SCRAPE_START = date(2007, 9, 12)

# QCEW initial-publication lag (months after quarter-end month).
# BLS accelerated the QCEW schedule around 2016; before that,
# publication lagged ~7 months after the reference quarter.
_QCEW_MODERN_PUBLICATION_LAG: dict[int, int] = {
    3: 5,   # Q1 — modern schedule (2016+)
    6: 5,   # Q2
    9: 5,   # Q3
    12: 6,  # Q4 + annual
}
_QCEW_HISTORICAL_PUBLICATION_LAG = 7  # months, pre-2016

# CES Oct 2025: released with Nov 2025 (government shutdown).
CES_OCT_2025_RELEASED_WITH_NOV_REF = date(2025, 10, 12)
CES_NOV_2025_REF = date(2025, 11, 12)
CES_OCT_2025_VINTAGE_FALLBACK = date(2025, 12, 16)

# # SAE Sept 2013: released with Oct 2013 (Oct 2013 government shutdown).
# SAE_SEPT_2013_RELEASED_WITH_OCT_REF = date(2013, 9, 12)
# SAE_OCT_2013_REF = date(2013, 10, 12)
# SAE_SEPT_2013_VINTAGE_FALLBACK = date(2013, 11, 22)
#
# # SAE Oct 2025: released with Nov 2025 (government shutdown).
# SAE_OCT_2025_RELEASED_WITH_NOV_REF = date(2025, 10, 12)
# SAE_NOV_2025_REF = date(2025, 11, 12)
# SAE_OCT_2025_VINTAGE_FALLBACK = date(2025, 12, 16)

# Supplemental release dates for ref_dates missing from the scrape (early 2010).
SUPPLEMENTAL_RELEASE_DATES = [
    ('ces', date(2010, 1, 12), date(2010, 2, 5)),
    ('ces', date(2010, 2, 12), date(2010, 3, 5)),
    ('ces', date(2010, 3, 12), date(2010, 4, 2)),
    # ('sae', date(2010, 1, 12), date(2010, 3, 14)),
    # ('sae', date(2010, 2, 12), date(2010, 3, 25)),
    # ('sae', date(2010, 3, 12), date(2010, 4, 15)),
]

# US federal holidays that can land on the first Friday of a month.
_NEW_YEARS_DAY = 1  # Jan 1
_INDEPENDENCE_DAY_MONTH = 7  # Jul 4


def _first_friday(year: int, month: int) -> date:
    """Return the first Friday of a given month/year."""
    d = date(year, month, 1)
    days_ahead = (4 - d.weekday()) % 7  # 4 = Friday
    return d + timedelta(days=days_ahead)


def _ces_publication_date(ref_year: int, ref_month: int) -> date:
    """Estimate the CES publication date for a reference month.

    CES releases on the first Friday of the month *after* the reference
    month.  When that Friday falls on Jan 1 or Jul 4, the release shifts
    to the second Friday.
    """
    pub_month = ref_month + 1
    pub_year = ref_year
    if pub_month > 12:
        pub_month = 1
        pub_year += 1

    friday = _first_friday(pub_year, pub_month)

    if (friday.month == 1 and friday.day == _NEW_YEARS_DAY) or (
        friday.month == _INDEPENDENCE_DAY_MONTH and friday.day == 4
    ):
        friday += timedelta(days=7)

    return friday


def _generate_ces_pre_scrape_dates() -> list[tuple[str, date, date]]:
    """Generate (publication, ref_date, vintage_date) for CES months before
    the BLS archive scrape coverage (2003-01 through 2007-12).
    """
    rows: list[tuple[str, date, date]] = []
    for year in range(2003, _CES_SCRAPE_START_YEAR):
        for month in range(1, 13):
            ref = date(year, month, 12)
            pub = _ces_publication_date(year, month)
            rows.append(('ces', ref, pub))
    return rows


def _qcew_publication_date(ref_year: int, quarter_end_month: int) -> date:
    """Estimate the QCEW initial publication date for a reference quarter.

    Returns the 1st of the month that is *lag* months after the
    quarter-end month, using the modern (post-2016) schedule.
    """
    lag = _QCEW_MODERN_PUBLICATION_LAG[quarter_end_month]
    total = quarter_end_month + lag
    pub_year = ref_year + (total - 1) // 12
    pub_month = ((total - 1) % 12) + 1
    return date(pub_year, pub_month, 1)


def _generate_qcew_pre_scrape_dates() -> list[tuple[str, date, date]]:
    """Generate (publication, ref_date, vintage_date) for QCEW quarters before
    the BLS archive scrape coverage (2003-Q1 through 2007-Q2).

    Uses the historical 7-month publication lag (BLS did not accelerate
    the QCEW schedule until ~2016).
    """
    rows: list[tuple[str, date, date]] = []
    lag = _QCEW_HISTORICAL_PUBLICATION_LAG
    for year in range(2003, _QCEW_SCRAPE_START.year + 1):
        for qm in (3, 6, 9, 12):
            ref = date(year, qm, 12)
            if ref >= _QCEW_SCRAPE_START:
                break
            total = qm + lag
            pub_year = year + (total - 1) // 12
            pub_month = ((total - 1) % 12) + 1
            pub = date(pub_year, pub_month, 1)
            rows.append(('qcew', ref, pub))
    return rows


def _add_ces_revisions(df: pl.DataFrame) -> pl.DataFrame:
    """Expand CES release_dates into revision rows (0, 1, 2)."""
    parts = []
    for n in CES_MONTHLY_REVISIONS:
        if n == 0:
            parts.append(
                df.filter(pl.col('publication') == 'ces').with_columns(
                    pl.lit(0).alias('revision'),
                    pl.lit(0).alias('benchmark_revision'),
                )
            )
        else:
            parts.append(
                df.filter(pl.col('publication') == 'ces').with_columns(
                    pl.col('vintage_date').dt.offset_by(f'{n}mo').alias('vintage_date'),
                    pl.lit(n).alias('revision'),
                    pl.lit(0).alias('benchmark_revision'),
                )
            )
    return pl.concat(parts)


# def _add_sae_revisions(df: pl.DataFrame) -> pl.DataFrame:
#     """Expand SAE release_dates into revision rows (0, 1)."""
#     parts = []
#     for n in SAE_MONTHLY_REVISIONS:
#         if n == 0:
#             parts.append(
#                 df.filter(pl.col('publication') == 'sae').with_columns(
#                     pl.lit(0).alias('revision'),
#                     pl.lit(0).alias('benchmark_revision'),
#                 )
#             )
#         else:
#             parts.append(
#                 df.filter(pl.col('publication') == 'sae').with_columns(
#                     pl.col('vintage_date').dt.offset_by(f'{n}mo').alias('vintage_date'),
#                     pl.lit(n).alias('revision'),
#                     pl.lit(0).alias('benchmark_revision'),
#                 )
#             )
#     return pl.concat(parts)


def _add_qcew_revisions(df: pl.DataFrame) -> pl.DataFrame:
    """Expand QCEW release_dates into revision rows (0..max_rev by quarter)."""
    qcew = df.filter(pl.col('publication') == 'qcew').with_columns(
        pl.col('ref_date').dt.month().alias('month'),
    )
    max_rev = (
        pl.when(pl.col('month').is_between(1, 3))
        .then(4)
        .when(pl.col('month').is_between(4, 6))
        .then(3)
        .when(pl.col('month').is_between(7, 9))
        .then(2)
        .otherwise(1)
        .alias('max_rev')
    )
    qcew = qcew.with_columns(max_rev)
    parts = []
    for n in range(5):
        if n == 0:
            parts.append(
                qcew.with_columns(
                    pl.lit(0).alias('revision'),
                    pl.lit(0).alias('benchmark_revision'),
                ).select('publication', 'ref_date', 'vintage_date', 'revision', 'benchmark_revision')
            )
        else:
            subset = qcew.filter(pl.col('max_rev') >= n)
            parts.append(
                subset.with_columns(
                    pl.col('vintage_date').dt.offset_by(f'{n}mo').alias('vintage_date'),
                    pl.lit(n).alias('revision'),
                    pl.lit(0).alias('benchmark_revision'),
                ).select('publication', 'ref_date', 'vintage_date', 'revision', 'benchmark_revision')
            )
    return pl.concat(parts)


def _ces_benchmark_vintage_dates(release_df: pl.DataFrame) -> pl.DataFrame:
    """When Jan Y+1 is published, all year Y ref_dates get benchmark_revision=1."""
    ces = release_df.filter(pl.col('publication') == 'ces')
    jan_releases = ces.filter(pl.col('ref_date').dt.month() == 1).select(
        pl.col('ref_date').dt.year().alias('benchmark_year'),
        pl.col('vintage_date').alias('benchmark_vintage'),
    ).unique()
    ces_refs = ces.select('publication', 'ref_date').unique().with_columns(
        pl.col('ref_date').dt.year().alias('ref_year'),
        (pl.col('ref_date').dt.year() + 1).alias('benchmark_year'),
    )
    return ces_refs.join(
        jan_releases,
        on='benchmark_year',
        how='inner',
    ).select(
        pl.col('publication'),
        pl.col('ref_date'),
        pl.col('benchmark_vintage').alias('vintage_date'),
        pl.lit(max(CES_MONTHLY_REVISIONS)).alias('revision'),
        pl.lit(1).alias('benchmark_revision'),
    )


# def _sae_benchmark_vintage_dates(release_df: pl.DataFrame) -> pl.DataFrame:
#     """Two benchmarks per ref_date (revision=1, benchmark_revision=1 or 2)."""
#     sae = release_df.filter(pl.col('publication') == 'sae')
#     march_releases = sae.filter(pl.col('ref_date').dt.month() == 3).select(
#         pl.col('ref_date').dt.year().alias('year'),
#         pl.col('vintage_date').alias('benchmark_vintage'),
#     )
#     sae_refs = sae.select('publication', 'ref_date').unique().with_columns(
#         pl.col('ref_date').dt.year().alias('ref_year'),
#     )
#     first = sae_refs.with_columns(
#         (pl.col('ref_year') + 1).alias('benchmark_year'),
#     ).join(
#         march_releases,
#         left_on='benchmark_year',
#         right_on='year',
#         how='inner',
#     ).select(
#         pl.col('publication'),
#         pl.col('ref_date'),
#         pl.col('benchmark_vintage').alias('vintage_date'),
#         pl.lit(max(SAE_MONTHLY_REVISIONS)).alias('revision'),
#         pl.lit(1).alias('benchmark_revision'),
#     )
#     second = sae_refs.with_columns(
#         (pl.col('ref_year') + 2).alias('benchmark_year'),
#     ).join(
#         march_releases,
#         left_on='benchmark_year',
#         right_on='year',
#         how='inner',
#     ).select(
#         pl.col('publication'),
#         pl.col('ref_date'),
#         pl.col('benchmark_vintage').alias('vintage_date'),
#         pl.lit(max(SAE_MONTHLY_REVISIONS)).alias('revision'),
#         pl.lit(2).alias('benchmark_revision'),
#     )
#     return pl.concat([first, second])


def _apply_shutdown_override(
    df: pl.DataFrame,
    publication: str,
    affected_ref: date,
    companion_ref: date,
    fallback_vintage: date,
) -> pl.DataFrame:
    """Override vintage_date for a ref_date released alongside another (shutdown)."""
    companion = df.filter(
        (pl.col('publication') == publication)
        & (pl.col('ref_date') == companion_ref)
    )
    vint = (
        companion.item(0, 'vintage_date')
        if companion.height > 0
        else fallback_vintage
    )
    df = df.filter(
        ~((pl.col('publication') == publication) & (pl.col('ref_date') == affected_ref))
    )
    df = pl.concat([
        df,
        pl.DataFrame({
            'publication': [publication],
            'ref_date': [affected_ref],
            'vintage_date': [vint],
        }),
    ]).unique(subset=['publication', 'ref_date'])
    return df


def build_vintage_dates(release_dates_path: Path | None = None) -> pl.DataFrame:
    """Build vintage_dates DataFrame from release_dates parquet.

    Applies publication-specific revision logic, government shutdown special
    cases, and supplemental release dates for early-2010 gaps. Filters to
    vintage_date <= today.

    Parameters
    ----------
    release_dates_path : Path or None
        Path to release_dates.parquet. Defaults to the configured path.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: publication, ref_date, vintage_date, revision,
        benchmark_revision. Sorted by (publication, ref_date, vintage_date,
        revision, benchmark_revision).
    """
    path = release_dates_path or RELEASE_DATES_PATH
    df = pl.read_parquet(path)

    # Merge supplemental + pre-scrape release dates for gaps
    all_supplemental = (
        SUPPLEMENTAL_RELEASE_DATES
        + _generate_ces_pre_scrape_dates()
        + _generate_qcew_pre_scrape_dates()
    )
    supplemental = pl.DataFrame(
        [
            {'publication': p, 'ref_date': ref, 'vintage_date': vint}
            for p, ref, vint in all_supplemental
        ],
        schema={'publication': pl.Utf8, 'ref_date': pl.Date, 'vintage_date': pl.Date},
    )
    existing_keys = df.select('publication', 'ref_date').unique()
    supplemental = supplemental.join(
        existing_keys, on=['publication', 'ref_date'], how='anti',
    )
    if supplemental.height > 0:
        df = pl.concat([df, supplemental]).unique(subset=['publication', 'ref_date'])

    # Government shutdown overrides
    df = _apply_shutdown_override(
        df, 'ces', CES_OCT_2025_RELEASED_WITH_NOV_REF,
        CES_NOV_2025_REF, CES_OCT_2025_VINTAGE_FALLBACK,
    )
    # df = _apply_shutdown_override(
    #     df, 'sae', SAE_SEPT_2013_RELEASED_WITH_OCT_REF,
    #     SAE_OCT_2013_REF, SAE_SEPT_2013_VINTAGE_FALLBACK,
    # )
    # df = _apply_shutdown_override(
    #     df, 'sae', SAE_OCT_2025_RELEASED_WITH_NOV_REF,
    #     SAE_NOV_2025_REF, SAE_OCT_2025_VINTAGE_FALLBACK,
    # )

    # Build revision rows for each publication
    with_revisions = pl.concat([
        _add_ces_revisions(df),
        # _add_sae_revisions(df),
        _add_qcew_revisions(df),
    ])

    # Build benchmark rows
    ces_bench = _ces_benchmark_vintage_dates(df)
    # sae_bench = _sae_benchmark_vintage_dates(df)
    benchmark_rows = ces_bench  # pl.concat([ces_bench, sae_bench])

    out = (
        pl.concat([with_revisions, benchmark_rows])
        .filter(pl.col('vintage_date') <= pl.lit(date.today()))
        .sort(['publication', 'ref_date', 'vintage_date', 'revision', 'benchmark_revision'])
    )
    return out


def build_and_save(release_dates_path: Path | None = None) -> pl.DataFrame:
    """Build vintage_dates and write to parquet.

    Parameters
    ----------
    release_dates_path : Path or None
        Path to release_dates.parquet.

    Returns
    -------
    pl.DataFrame
        The built vintage_dates DataFrame.
    """
    df = build_vintage_dates(release_dates_path)
    VINTAGE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(VINTAGE_DATES_PATH)
    return df
