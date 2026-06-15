"""Unified observation panel schema and validation.

Defines PANEL_SCHEMA for the vintage-tracked observation panel and provides
validation to ensure data integrity before model consumption.
"""

from __future__ import annotations

import polars as pl

# Unified observation panel schema: each row is one observation of employment
# growth for one industry unit in one period from one source at one vintage.
PANEL_SCHEMA: dict[str, pl.DataType] = {
    'period': pl.Date,
    'geographic_type': pl.Utf8,
    'geographic_code': pl.Utf8,
    'industry_code': pl.Utf8,
    'industry_level': pl.Utf8,
    'source': pl.Utf8,
    'source_type': pl.Utf8,
    'growth': pl.Float64,
    'employment_level': pl.Float64,
    'is_seasonally_adjusted': pl.Boolean,
    'vintage_date': pl.Date,
    'revision_number': pl.Int32,
    'is_final': pl.Boolean,
    'publication_lag_months': pl.Int32,
    'coverage_ratio': pl.Float64,
}

# Columns that must be non-null
_REQUIRED_NON_NULL = {
    'period', 'geographic_type', 'geographic_code',
    'industry_code', 'industry_level', 'source', 'source_type',
}

# Valid values for categorical columns
_VALID_INDUSTRY_LEVELS = {'domain', 'supersector', 'sector'}
_VALID_SOURCE_TYPES = {'official_sa', 'official_nsa', 'census', 'payroll'}
_VALID_GEOGRAPHIC_TYPES = {'national', 'state', 'region', 'division'}


def validate_panel(df: pl.DataFrame) -> pl.DataFrame:
    """Validate that a DataFrame conforms to PANEL_SCHEMA.

    Parameters
    ----------
    df : pl.DataFrame
        Observation panel to validate.

    Returns
    -------
    pl.DataFrame
        The input DataFrame (unchanged) if valid.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    # Check all required columns are present
    missing = set(PANEL_SCHEMA.keys()) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    # Check dtypes
    for col, expected_dtype in PANEL_SCHEMA.items():
        actual_dtype = df.schema[col]
        if actual_dtype != expected_dtype:
            raise ValueError(
                f'Column {col!r} has dtype {actual_dtype}, expected {expected_dtype}'
            )

    # Check no duplicates on the full observation key
    _dup_cols = [
        'period', 'geographic_type', 'geographic_code',
        'source', 'industry_level', 'industry_code', 'revision_number',
    ]
    dup_check = df.group_by(_dup_cols).len()
    dups = dup_check.filter(pl.col('len') > 1)
    if len(dups) > 0:
        n_dups = len(dups)
        sample = dups.head(3).to_dicts()
        raise ValueError(
            f'{n_dups} duplicate ({", ".join(_dup_cols)}) '
            f'combinations found. Examples: {sample}'
        )

    # Check growth values are finite where not null
    growth_col = df['growth']
    non_null_growth = growth_col.drop_nulls()
    if len(non_null_growth) > 0:
        inf_count = non_null_growth.filter(
            non_null_growth.is_infinite() | non_null_growth.is_nan()
        ).len()
        if inf_count > 0:
            raise ValueError(f'{inf_count} non-finite growth values found (inf or NaN)')

    # Check revision_number consistency with source_type
    # Payroll providers should only have revision_number=0
    payroll_rows = df.filter(pl.col('source_type') == 'payroll')
    if len(payroll_rows) > 0:
        bad_revisions = payroll_rows.filter(
            (pl.col('revision_number') != 0) & pl.col('revision_number').is_not_null()
        )
        if len(bad_revisions) > 0:
            raise ValueError(
                f'{len(bad_revisions)} payroll observations have revision_number != 0'
            )

    return df


def empty_panel() -> pl.DataFrame:
    """Return an empty DataFrame conforming to PANEL_SCHEMA."""
    return pl.DataFrame(schema=PANEL_SCHEMA)


# Hive-partitioned vintage store schema (source × seasonally_adjusted).
#
# ``ownership`` (store_rebuild §3/§7) splits the axis the single ``industry_code``
# column conflated: ``private`` (the modeled hierarchy) / ``total`` (the ``00``
# anchor). ``government`` is reserved (deferred, §11). ``industry_type`` retires
# the legacy ``'national'`` value in favour of ``total/domain/supersector/sector``.
VINTAGE_STORE_SCHEMA: dict[str, pl.DataType] = {
    "geographic_type": pl.Utf8,
    "geographic_code": pl.Utf8,
    "ownership": pl.Utf8,
    "industry_type": pl.Utf8,
    "industry_code": pl.Utf8,
    "ref_date": pl.Date,
    "vintage_date": pl.Date,
    "revision": pl.UInt8,
    "benchmark_revision": pl.UInt8,
    "employment": pl.Float64,
    # Size-class dimension (store_rebuild §8): populated only for QCEW Q1
    # (ref-month ∈ {01,02,03}); null for CES and QCEW Q2/Q3/Q4. ``size_class_code``
    # is singular. All-sizes selection = ``size_class_type IS NULL OR
    # size_class_code = '0'`` (§7).
    "size_class_type": pl.Utf8,
    "size_class_code": pl.Utf8,
    "source": pl.Utf8,
    "seasonally_adjusted": pl.Boolean,
}


# Parquet schemas for historical revision data

QCEW_VINTAGE_SCHEMA: dict[str, pl.DataType] = {
    'ref_year': pl.Int32,
    'ref_quarter': pl.Int32,
    'ref_month': pl.Int32,
    'area_fips': pl.Utf8,
    'industry_code': pl.Utf8,
    'own_code': pl.Int32,
    'employment': pl.Int64,
    'revision_number': pl.Int32,
    'vintage_date': pl.Date,
}

CES_VINTAGE_SCHEMA: dict[str, pl.DataType] = {
    'ref_date': pl.Date,
    'supersector_code': pl.Utf8,
    'seasonal_adjustment': pl.Utf8,
    'employment': pl.Float64,
    'revision_number': pl.Int32,
    'vintage_date': pl.Date,
    'bls_series_id': pl.Utf8,
}

PUBLICATION_CALENDAR_SCHEMA: dict[str, pl.DataType] = {
    'source': pl.Utf8,
    'ref_period': pl.Utf8,
    'revision_number': pl.Int32,
    'publication_date': pl.Date,
}
