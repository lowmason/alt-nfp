"""Combine CES and QCEW revision Parquets into ``revisions.parquet``.

Reads the source-specific files, vertically concatenates them, and adds
region- and division-level aggregates by summing state employment within each
geographic grouping.

Note: SAE support is commented out.
"""

from __future__ import annotations

import polars as pl
from nfp_lookups.geography import FIPS_TO_DIVISION, FIPS_TO_REGION
from nfp_lookups.paths import INTERMEDIATE_DIR

OUTPUT_PATH = INTERMEDIATE_DIR / 'revisions.parquet'

GROUP_COLS = [
    'source',
    'seasonally_adjusted',
    'geographic_type',
    'geographic_code',
    'industry_type',
    'industry_code',
    'ref_date',
    'vintage_date',
    'revision',
    'benchmark_revision',
]


def build_revisions(*, save: bool = True) -> pl.DataFrame:
    """Combine QCEW and CES revisions and aggregate to region/division.

    Parameters
    ----------
    save : bool
        If True (default), write the result to ``revisions.parquet``.

    Returns
    -------
    pl.DataFrame
        The combined revisions dataset with national, state, region, and
        division geographic levels.
    """
    qcew = pl.read_parquet(INTERMEDIATE_DIR / 'qcew_revisions.parquet')
    ces = pl.read_parquet(INTERMEDIATE_DIR / 'ces_revisions.parquet').with_columns(
        revision=pl.col('revision').cast(pl.UInt8),
        benchmark_revision=pl.col('benchmark_revision').cast(pl.UInt8),
    )
    # sae = pl.read_parquet(INTERMEDIATE_DIR / 'sae_revisions.parquet')

    revisions_all = pl.concat([qcew, ces])  # sae removed

    revisions_national = revisions_all.filter(
        pl.col('geographic_type').eq('national'),
    )
    revisions_state = revisions_all.filter(
        pl.col('geographic_type').eq('state'),
        pl.col('geographic_code').ne('00'),
    )
    assert revisions_state.height + revisions_national.height == revisions_all.height

    # Aggregate state -> region
    revisions_region = (
        revisions_state.with_columns(
            geographic_type=pl.lit('region', pl.Utf8),
            geographic_code=pl.col('geographic_code').replace_strict(
                FIPS_TO_REGION, default=None,
            ),
        )
        .group_by(GROUP_COLS)
        .agg(employment=pl.col('employment').sum())
    )

    # Aggregate state -> division
    revisions_division = (
        revisions_state.with_columns(
            geographic_type=pl.lit('division', pl.Utf8),
            geographic_code=pl.col('geographic_code').replace_strict(
                FIPS_TO_DIVISION, default=None,
            ),
        )
        .group_by(GROUP_COLS)
        .agg(employment=pl.col('employment').sum())
    )

    revisions_df = (
        pl.concat([
            revisions_national,
            revisions_state,
            revisions_region,
            revisions_division,
        ])
        .sort(*GROUP_COLS)
    )

    # Uniqueness assertion
    dedup = revisions_df.unique(subset=GROUP_COLS)
    assert revisions_df.height == dedup.height, (
        f'Duplicate rows: {revisions_df.height} total vs {dedup.height} unique'
    )

    print(f'Revisions: {revisions_df.height:,} rows')

    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        revisions_df.write_parquet(OUTPUT_PATH)
        print(f'Wrote {OUTPUT_PATH}')

    return revisions_df


def main() -> None:
    """Entry point: build the combined revisions table and write it to ``OUTPUT_PATH``."""
    build_revisions(save=True)


if __name__ == '__main__':
    main()
