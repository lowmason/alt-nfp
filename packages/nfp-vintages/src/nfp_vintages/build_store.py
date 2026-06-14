"""Build the Hive-partitioned vintage store from revisions + releases.

Merges historical revisions (from :mod:`~nfp_vintages.processing.combine`)
with current estimates (``releases.parquet`` from ``bls-estimates``), normalizes
``industry_type``, deduplicates, and writes the vintage store partitioned by
``(source, seasonally_adjusted)``.

Invoked via the ``alt-nfp build`` CLI (see :mod:`nfp_vintages.__main__`).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from nfp_lookups.paths import (
    DATA_DIR,
    INTERMEDIATE_DIR,
    VINTAGE_STORE_PATH,
    is_canonical_store,
    is_remote,
    storage_options_for,
)

REVISIONS_PATH = INTERMEDIATE_DIR / 'revisions.parquet'
RELEASES_PATH = DATA_DIR / 'releases.parquet'


def build_store(
    revisions_path: Path | None = None,
    releases_path: Path | None = None,
    store_path: Path | None = None,
    allow_canonical: bool = False,
) -> None:
    """Build the vintage store from revisions and current releases.

    Parameters
    ----------
    revisions_path : Path or None
        Path to ``revisions.parquet``. Defaults to ``INTERMEDIATE_DIR/revisions.parquet``.
    releases_path : Path or None
        Path to ``releases.parquet`` (from ``bls-estimates``).
        Defaults to ``DATA_DIR/releases.parquet``.
    store_path : Path or None
        Output vintage store root. Defaults to ``VINTAGE_STORE_PATH``.
    allow_canonical : bool
        Permit rebuilding the canonical remote store in place. Defaults to ``False``.
        See root ``CLAUDE.md`` "Never rebuild the canonical store in place".
    """
    rev_path = revisions_path or REVISIONS_PATH
    rel_path = releases_path or RELEASES_PATH
    out_path = store_path or VINTAGE_STORE_PATH

    if is_canonical_store(out_path) and not allow_canonical:
        raise RuntimeError(
            'refusing to rebuild the canonical store in place '
            f'({out_path}); write to a scratch prefix (e.g. .../store-rebuild) '
            "or pass allow_canonical=True. See CLAUDE.md 'Never rebuild the "
            "canonical store in place'."
        )

    revisions = pl.read_parquet(rev_path).with_columns(
        current=pl.lit(0, pl.UInt8),
    )

    if rel_path.exists():
        releases = pl.read_parquet(rel_path).with_columns(
            current=pl.lit(1, pl.UInt8),
        )
        combined = pl.concat([revisions, releases], how='diagonal_relaxed')
    else:
        print(f'  Note: {rel_path} not found — building store from revisions only')
        combined = revisions

    # Normalize industry_type: set to 'national' where industry_code == '00'
    combined = combined.with_columns(
        pl.when(pl.col('industry_code').eq('00'))
        .then(pl.lit('national'))
        .otherwise(pl.col('industry_type'))
        .alias('industry_type'),
    )

    # Deduplicate: when a revision and a current release overlap, the current
    # release wins (sort by current ascending, group_by, take last).
    key_cols = [
        'source', 'seasonally_adjusted',
        'geographic_type', 'geographic_code',
        'industry_type', 'industry_code',
        'ref_date', 'revision', 'benchmark_revision',
    ]
    combined = (
        combined.sort('current')
        .unique(subset=key_cols, keep='last')
        .drop('current')
    )

    print(f'Combined store: {combined.height:,} rows')

    # Write Hive-partitioned parquet (local dir or object storage)
    if not is_remote(out_path):
        out_path.mkdir(parents=True, exist_ok=True)

    for (source, sa), partition_df in combined.group_by(
        ['source', 'seasonally_adjusted'], maintain_order=True,
    ):
        sa_str = str(sa).lower()
        partition_dir = out_path / f'source={source}' / f'seasonally_adjusted={sa_str}'
        if not is_remote(out_path):
            partition_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing files in partition
        if partition_dir.exists():
            for f in partition_dir.glob('*.parquet'):
                f.unlink()

        write_df = partition_df.drop(['source', 'seasonally_adjusted'])
        vmin = write_df['vintage_date'].min()
        vmax = write_df['vintage_date'].max()
        fname = f'v_{vmin}_{vmax}.parquet'
        write_df.write_parquet(
            str(partition_dir / fname),
            storage_options=storage_options_for(out_path),
        )
        print(f'  {partition_dir.name}: {write_df.height:,} rows -> {fname}')

    print(f'Wrote vintage store to {out_path}')
