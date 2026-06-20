"""Process QCEW data into ``qcew_revisions.parquet``.

Two data sources are merged:

1. **Bulk quarterly files** (2003-present): employment from multiple ownership
   and aggregation-level streams, processed from
   :func:`~nfp_download.bls.bulk.download_qcew_bulk`.

   Streams:
   - Total all-ownership (``own_code='0'``) → ``(national, '00')``
   - Private 2-digit sectors (``own_code='5'``, ``agglvl`` 14/54)
   - Government by ownership (``own_code`` 1/2/3, ``agglvl`` 11/51) →
     sectors 91/92/93
   - Manufacturing 3-digit NAICS (``own_code='5'``, ``agglvl`` 15/55) →
     durable (31) / nondurable (32) split

   Sector data is aggregated through the industry hierarchy (sectors →
   supersectors → domains).  Single-sector supersectors (20, 50, 80) are
   duplicated as sector rows (23, 51, 81).  All rows are ``revision=0``.

2. **Revisions CSV** (2017-present): total employment with revision history
   (``revision`` 0-4), from :func:`~nfp_download.bls.bulk.download_qcew`.
   Where both sources overlap on ``(geographic_code, industry_code, ref_date,
   revision)``, the revisions CSV is preferred (it has exact vintage dates).
"""

from __future__ import annotations

from datetime import date

import polars as pl
from nfp_lookups.industry import (
    GOVT_OWNERSHIP_TO_SECTOR,
    NAICS3_TO_MFG_SECTOR,
    get_domain_supersectors,
    get_supersector_components,
    qcew_to_sector,
)
from nfp_lookups.paths import (
    DOWNLOADS_DIR,
    INTERMEDIATE_DIR,
    VINTAGE_DATES_PATH,
    storage_options_for,
)
from nfp_lookups.revision_schedules import get_qcew_vintage_date

OUTPUT_PATH = INTERMEDIATE_DIR / 'qcew_revisions.parquet'
QCEW_CSV_PATH = DOWNLOADS_DIR / 'qcew' / 'qcew-revisions.csv'
BULK_PATH = DOWNLOADS_DIR / 'qcew' / 'qcew_bulk.parquet'

OUTPUT_COLUMNS = [
    'source', 'seasonally_adjusted',
    'geographic_type', 'geographic_code',
    'industry_type', 'industry_code',
    'ref_date', 'vintage_date',
    'revision', 'benchmark_revision',
    'employment',
]

# NAICS 2-digit codes that appear in the bulk files for private sectors.
# Manufacturing codes ('1022', '31-33', '31') are excluded — those are
# handled via the 3-digit durable/nondurable split instead.
_NAICS_TO_SECTOR: dict[str, str] = {
    k: v
    for k, v in {**qcew_to_sector(), '44-45': '44', '48-49': '48'}.items()
    if v != '31'
}

# Sector → supersector lookup built from get_supersector_components() so it
# includes government (91→90, 92→90, 93→90) and nondurable mfg (32→30).
_SECTOR_TO_SS: dict[str, str] = {}
for _ss, _sectors in get_supersector_components().items():
    for _sec in _sectors:
        _SECTOR_TO_SS[_sec] = _ss
_SECTOR_TO_SS['32'] = '30'

# Domain definitions: supersector codes composing each domain.
_DOMAIN_SPECS: dict[str, list[str]] = {
    '05': get_domain_supersectors('05'),
    '06': get_domain_supersectors('06'),
    '07': get_domain_supersectors('07'),
    '08': get_domain_supersectors('08'),
}

# State name -> FIPS mapping for the revisions CSV.
_STATE_FIPS_TO_NAME: dict[str, str] = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
    '06': 'California', '08': 'Colorado', '09': 'Connecticut',
    '10': 'Delaware', '11': 'District of Columbia', '12': 'Florida',
    '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois',
    '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky',
    '22': 'Louisiana', '23': 'Maine', '24': 'Maryland',
    '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana',
    '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire',
    '34': 'New Jersey', '35': 'New Mexico', '36': 'New York',
    '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania',
    '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota',
    '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont',
    '51': 'Virginia', '53': 'Washington', '54': 'West Virginia',
    '55': 'Wisconsin', '56': 'Wyoming', '72': 'Puerto Rico',
}
STATES_NAME_TO_FIPS: dict[str, str] = {
    'United States': '00',
    **{name: fips for fips, name in _STATE_FIPS_TO_NAME.items()},
}


# ---------------------------------------------------------------------------
# Bulk data processing
# ---------------------------------------------------------------------------


def _compute_vintage_date(ref_date: date) -> date:
    """Approximate vintage_date for an initial (revision=0) QCEW observation."""
    qtr = (ref_date.month - 1) // 3 + 1
    return get_qcew_vintage_date(f'Q{qtr}', ref_date.year, revision=0)


def _process_bulk() -> pl.DataFrame:
    """Read ``qcew_bulk.parquet`` and produce sector + aggregated rows.

    Four input streams are extracted from the bulk data:

    1. **Total** (``own_code='0'``) → ``(national, '00')``
    2. **Private 2-digit sectors** (``own_code='5'``, ``agglvl`` 14/54) —
       all sectors except manufacturing (handled separately)
    3. **Government by ownership** (``own_code`` 1/2/3, ``agglvl`` 11/51) →
       sectors 91/92/93
    4. **Manufacturing 3-digit** (``own_code='5'``, ``agglvl`` 15/55) →
       durable (31) / nondurable (32)

    Sectors are then aggregated → supersectors → domains.  Single-sector
    supersectors (20, 50, 80) are duplicated as sector rows (23, 51, 81).
    """
    if not BULK_PATH.exists():
        print(f'  bulk file not found: {BULK_PATH}')
        return pl.DataFrame(schema=dict.fromkeys(OUTPUT_COLUMNS, pl.Utf8))

    df = pl.read_parquet(BULK_PATH)

    # --- Unpivot monthly columns into long format ---
    df = df.with_columns(
        pl.col('year').cast(pl.Int32),
        pl.col('qtr').cast(pl.Int32),
    )
    monthly = df.unpivot(
        ['month1_emplvl', 'month2_emplvl', 'month3_emplvl'],
        index=['area_fips', 'own_code', 'industry_code', 'agglvl_code', 'year', 'qtr'],
        variable_name='month_col',
        value_name='employment',
    ).with_columns(
        month_offset=pl.col('month_col').str.extract(r'month(\d)').cast(pl.Int32) - 1,
        employment=pl.col('employment').cast(pl.Float64),
    ).filter(
        pl.col('employment').is_not_null() & (pl.col('employment') > 0)
    ).with_columns(
        month=((pl.col('qtr') - 1) * 3 + 1 + pl.col('month_offset')).cast(pl.Int32),
    ).with_columns(
        ref_date=pl.date(pl.col('year'), pl.col('month'), 12),
    ).drop('month_col', 'month_offset', 'month')

    # --- Map area_fips to geographic_type / geographic_code ---
    monthly = monthly.with_columns(
        geographic_type=pl.when(pl.col('area_fips') == 'US000')
        .then(pl.lit('national'))
        .otherwise(pl.lit('state')),
        geographic_code=pl.when(pl.col('area_fips') == 'US000')
        .then(pl.lit('00'))
        .otherwise(pl.col('area_fips').str.slice(0, 2)),
    )

    geo_group = ['geographic_type', 'geographic_code', 'ref_date']
    sector_group = [*geo_group, 'industry_type', 'industry_code']

    # === Stream 1: total (own_code=0) ===
    total_rows = monthly.filter(
        (pl.col('own_code') == '0')
        & pl.col('agglvl_code').is_in({'10', '50'})
    ).with_columns(
        industry_type=pl.lit('national'),
        industry_code=pl.lit('00'),
    )

    # === Stream 2: private 2-digit sectors (excl manufacturing) ===
    private_raw = monthly.filter(
        (pl.col('own_code') == '5')
        & pl.col('agglvl_code').is_in({'14', '54'})
    )
    naics_map = pl.DataFrame({
        'industry_code': list(_NAICS_TO_SECTOR.keys()),
        'sector_code': list(_NAICS_TO_SECTOR.values()),
    })
    private_sectors = (
        private_raw.join(naics_map, on='industry_code', how='inner')
        .drop('industry_code')
        .rename({'sector_code': 'industry_code'})
        .with_columns(industry_type=pl.lit('sector'))
        .group_by(sector_group)
        .agg(employment=pl.col('employment').sum())
    )

    # === Stream 3: government by ownership (own_code 1/2/3) ===
    govt_raw = monthly.filter(
        pl.col('own_code').is_in({'1', '2', '3'})
        & pl.col('agglvl_code').is_in({'11', '51'})
        & (pl.col('industry_code') == '10')
    )
    govt_map = pl.DataFrame({
        'own_code': list(GOVT_OWNERSHIP_TO_SECTOR.keys()),
        'sector_code': list(GOVT_OWNERSHIP_TO_SECTOR.values()),
    })
    govt_sectors = (
        govt_raw.join(govt_map, on='own_code', how='inner')
        .with_columns(
            industry_type=pl.lit('sector'),
            industry_code=pl.col('sector_code'),
        )
        .group_by(sector_group)
        .agg(employment=pl.col('employment').sum())
    )

    # === Stream 4: manufacturing 3-digit → durable (31) / nondurable (32) ===
    mfg_raw = monthly.filter(
        (pl.col('own_code') == '5')
        & pl.col('agglvl_code').is_in({'15', '55'})
    )
    mfg_map = pl.DataFrame({
        'industry_code': list(NAICS3_TO_MFG_SECTOR.keys()),
        'sector_code': list(NAICS3_TO_MFG_SECTOR.values()),
    })
    mfg_sectors = (
        mfg_raw.join(mfg_map, on='industry_code', how='inner')
        .drop('industry_code')
        .rename({'sector_code': 'industry_code'})
        .with_columns(industry_type=pl.lit('sector'))
        .group_by(sector_group)
        .agg(employment=pl.col('employment').sum())
    )

    # === Combine all sector rows ===
    keep_cols = [
        'geographic_type', 'geographic_code', 'industry_type',
        'industry_code', 'ref_date', 'employment',
    ]
    all_sectors = pl.concat([
        private_sectors.select(keep_cols),
        govt_sectors.select(keep_cols),
        mfg_sectors.select(keep_cols),
    ])

    # === Aggregate sectors → supersectors ===
    ss_map = pl.DataFrame({
        'industry_code': list(_SECTOR_TO_SS.keys()),
        'supersector_code': list(_SECTOR_TO_SS.values()),
    })
    supersector_rows = (
        all_sectors.join(ss_map, on='industry_code', how='inner')
        .group_by([*geo_group, 'supersector_code'])
        .agg(employment=pl.col('employment').sum())
        .rename({'supersector_code': 'industry_code'})
        .with_columns(industry_type=pl.lit('supersector'))
    )

    # === Aggregate supersectors → domains ===
    domain_frames: list[pl.DataFrame] = []
    for domain_code, ss_list in _DOMAIN_SPECS.items():
        domain_df = (
            supersector_rows
            .filter(pl.col('industry_code').is_in(ss_list))
            .group_by(geo_group)
            .agg(employment=pl.col('employment').sum())
            .with_columns(
                industry_code=pl.lit(domain_code),
                industry_type=pl.lit('domain'),
            )
        )
        domain_frames.append(domain_df)

    # === Combine all hierarchy levels ===
    # Sectors 23, 51, 81 come from the private NAICS mapping (they are the
    # sole sector in their respective supersectors 20, 50, 80 — see
    # SINGLE_SECTOR_SUPERSECTORS).  No separate duplication step needed here
    # because the raw NAICS data naturally provides them.
    combined = pl.concat([
        total_rows.select(keep_cols),
        all_sectors.select(keep_cols),
        supersector_rows.select(keep_cols),
        pl.concat(domain_frames).select(keep_cols),
    ])

    # --- Add vintage metadata ---
    vintage_dates = combined.select('ref_date').unique().to_series().to_list()
    vdate_map = {d: _compute_vintage_date(d) for d in vintage_dates}
    vdate_df = pl.DataFrame({
        'ref_date': list(vdate_map.keys()),
        'vintage_date': list(vdate_map.values()),
    })

    combined = (
        combined.join(vdate_df, on='ref_date', how='left')
        .with_columns(
            source=pl.lit('qcew'),
            seasonally_adjusted=pl.lit(False),
            revision=pl.lit(0, pl.UInt8),
            benchmark_revision=pl.lit(0, pl.UInt8),
            employment=pl.col('employment') / 1000.0,
        )
        .select(OUTPUT_COLUMNS)
    )

    print(
        f'  bulk: {combined.height:,} rows across '
        f'{combined["industry_code"].n_unique()} industries'
    )
    return combined


# ---------------------------------------------------------------------------
# Revisions CSV processing (unchanged logic from original)
# ---------------------------------------------------------------------------


def _process_revisions_csv() -> pl.DataFrame:
    """Process the QCEW revisions CSV (2017+) with revision history."""
    if not QCEW_CSV_PATH.exists():
        print(f'  revisions CSV not found: {QCEW_CSV_PATH}')
        return pl.DataFrame(schema=dict.fromkeys(OUTPUT_COLUMNS, pl.Utf8))

    qcew_1 = (
        pl.read_csv(
            QCEW_CSV_PATH,
            schema_overrides={
                'Year': pl.Utf8,
                'Quarter': pl.Int64,
                'Area': pl.Utf8,
                'Field': pl.Utf8,
                'Initial Value': pl.Utf8,
                'First Revised Value': pl.Utf8,
                'Second Revised Value': pl.Utf8,
                'Third Revised Value': pl.Utf8,
                'Fourth Revised Value': pl.Utf8,
                'Final Value': pl.Utf8,
            },
        )
        .filter(
            pl.col('Area').is_in(STATES_NAME_TO_FIPS.keys()),
            pl.col('Field').str.contains('Employment'),
        )
        .with_columns(
            qtr_date=pl.concat_str(
                pl.lit('12'),
                pl.col('Quarter').mul(3),
                pl.col('Year'),
                separator=' ',
            ).str.to_date(format='%d %m %Y'),
            ref_date=pl.concat_str(
                pl.lit('12'),
                pl.col('Field').str.replace(' Employment', ''),
                pl.col('Year'),
                separator=' ',
            ).str.to_date(format='%d %B %Y'),
            geographic_code=pl.col('Area').replace_strict(
                STATES_NAME_TO_FIPS, default=None,
            ),
        )
        .select(
            qtr_date=pl.col('qtr_date'),
            geographic_type=pl.when(pl.col('geographic_code').eq('00'))
            .then(pl.lit('national'))
            .otherwise(pl.lit('state')),
            geographic_code=pl.col('geographic_code'),
            industry_type=pl.lit('national'),
            industry_code=pl.lit('00'),
            ref_date=pl.col('ref_date'),
            emp_0=pl.col('Initial Value'),
            emp_1=pl.col('First Revised Value'),
            emp_2=pl.col('Second Revised Value'),
            emp_3=pl.col('Third Revised Value'),
            emp_4=pl.col('Fourth Revised Value'),
        )
        .unpivot(
            ['emp_0', 'emp_1', 'emp_2', 'emp_3', 'emp_4'],
            index=[
                'qtr_date', 'geographic_type', 'geographic_code',
                'industry_type', 'industry_code', 'ref_date',
            ],
            value_name='employment',
            variable_name='revision',
        )
        .filter(
            ~pl.col('employment').is_in(['Not yet published', 'Not applicable'])
        )
        .with_columns(
            revision=pl.col('revision')
            .str.replace('emp_', '')
            .cast(pl.UInt8),
            employment=pl.col('employment').cast(pl.Float64).truediv(1000),
        )
        .sort('geographic_code', 'ref_date', 'revision')
    )

    # Join vintage dates
    if VINTAGE_DATES_PATH.exists():
        vintage_dates = (
            pl.read_parquet(
                str(VINTAGE_DATES_PATH), storage_options=storage_options_for(VINTAGE_DATES_PATH)
            )
            .filter(pl.col('publication').eq('qcew'))
            .select(
                qtr_date=pl.col('ref_date'),
                revision=pl.col('revision').cast(pl.UInt8),
                benchmark_revision=pl.col('benchmark_revision').cast(pl.UInt8),
                vintage_date=pl.col('vintage_date'),
            )
        )
        qcew_1 = qcew_1.join(vintage_dates, on=['qtr_date', 'revision'], how='left')
    else:
        qcew_1 = qcew_1.with_columns(
            benchmark_revision=pl.lit(0, pl.UInt8),
            vintage_date=pl.lit(None, pl.Date),
        )

    output = (
        qcew_1.with_columns(
            source=pl.lit('qcew'),
            seasonally_adjusted=pl.lit(False),
        ).select(OUTPUT_COLUMNS)
    )

    print(f'  revisions CSV: {output.height:,} rows')
    return output


# ---------------------------------------------------------------------------
# Main: merge both sources
# ---------------------------------------------------------------------------


def main() -> None:
    """Process QCEW bulk + revisions CSV data and write ``qcew_revisions.parquet``."""
    print('=== Processing QCEW bulk data ===')
    bulk = _process_bulk()

    print('=== Processing QCEW revisions CSV ===')
    revisions = _process_revisions_csv()

    # Merge: revisions CSV wins on overlap (has exact vintage dates + revision history)
    dedup_key = [
        'geographic_type', 'geographic_code',
        'industry_type', 'industry_code',
        'ref_date', 'revision',
    ]

    if revisions.height > 0 and bulk.height > 0:
        # Tag source priority: revisions=1 (keep), bulk=0 (fallback)
        revisions_tagged = revisions.with_columns(_priority=pl.lit(1, pl.Int8))
        bulk_tagged = bulk.with_columns(_priority=pl.lit(0, pl.Int8))
        combined = (
            pl.concat([revisions_tagged, bulk_tagged])
            .sort('_priority', descending=True)
            .unique(subset=dedup_key, keep='first')
            .drop('_priority')
        )
    elif revisions.height > 0:
        combined = revisions
    else:
        combined = bulk

    combined = combined.sort(
        'geographic_code', 'industry_code', 'ref_date', 'revision',
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUTPUT_PATH)
    print(f'Wrote {combined.height:,} rows to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
