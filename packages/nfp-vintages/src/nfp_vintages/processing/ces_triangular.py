"""Process CES triangular-revision CSVs into ``ces_revisions.parquet``.

Reads every ``tri_{code}_{SA|NSA}.csv`` file from ``cesvinall/``, extracts
the initial and revised diagonals (revisions 0, 1, 2), joins vintage dates,
and writes the result to ``data/intermediate/ces_revisions.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from nfp_lookups.industry import (
    CES_SECTOR_TO_NAICS,
    INDUSTRY_MAP,
    SINGLE_SECTOR_SUPERSECTORS,
)
from nfp_lookups.paths import (
    DOWNLOADS_DIR,
    INTERMEDIATE_DIR,
    VINTAGE_DATES_PATH,
    storage_options_for,
)
from polars import selectors as cs

CES_DIR = DOWNLOADS_DIR / 'ces' / 'cesvinall'
OUTPUT_PATH = INTERMEDIATE_DIR / 'ces_revisions.parquet'


def _build_schema(path: Path) -> tuple[dict[str, pl.DataType], list[str], dict[str, str]]:
    """Inspect a sample CSV to discover ``Mon_YY`` vintage columns.

    Returns
    -------
    tuple
        ``(schema, selected_columns, rename_mapping)``
    """
    rows = pl.read_csv(path / 'tri_050000_SA.csv')
    columns = rows.columns
    available = set(columns)

    schema: dict[str, pl.DataType] = {'year': pl.UInt16, 'month': pl.UInt8}
    schema.update({col: pl.Float64 for col in columns if col not in schema})

    years = list(range(2003, 2030))
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ]

    selected: list[str] = ['year', 'month']
    renamed: dict[str, str] = {}
    for yr in years:
        for m, mo in enumerate(month_names):
            y = str(yr)[2:]
            col_name = f'{mo}_{y}'
            if col_name in available:
                selected.append(col_name)
                renamed[col_name] = f'emp_{yr}_{m + 1}'

    return schema, selected, renamed


def read_triangular_ces(
    path: Path,
    file: str,
    industry_type: str,
    industry_code: str,
    schema: dict[str, pl.DataType],
    selected: list[str],
    renamed: dict[str, str],
) -> pl.DataFrame:
    """Read one triangular CSV and extract the k=0,1,2 revision diagonals.

    Parameters
    ----------
    path : Path
        Directory containing the CSV files.
    file : str
        Stem of the CSV file (without ``.csv``).
    industry_type : str
        Industry level label (e.g. ``'domain'``, ``'supersector'``).
    industry_code : str
        Two-digit industry code (e.g. ``'00'``).
    schema : dict
        Column-type overrides for :func:`polars.read_csv`.
    selected : list[str]
        Ordered column names to select after reading.
    renamed : dict[str, str]
        Mapping from ``Mon_YY`` names to ``emp_YYYY_M`` names.

    Returns
    -------
    pl.DataFrame
        Tidy DataFrame with ref_date, revision, geographic/industry metadata,
        and employment.
    """
    tri_df = (
        pl.read_csv(f'{path}/{file}.csv', schema_overrides=schema)
        .select(selected)
        .rename(renamed)
        .with_columns(
            ref_date=pl.date(pl.col('year'), pl.col('month'), pl.lit(12, pl.UInt8)),
            ref_year=pl.col('year'),
            ref_month=pl.col('month'),
        )
        .filter(pl.col('ref_date').gt(pl.date(2002, 12, 12)))
        .select(cs.starts_with('ref_'), cs.starts_with('emp_'))
    )

    emp_cols = [c for c in tri_df.columns if c.startswith('emp_')]
    n_cols = len(emp_cols)
    n_rows = len(tri_df)

    first_row_year = int(tri_df[0, 'ref_year'])
    first_row_month = int(tri_df[0, 'ref_month'])
    first_col_parts = emp_cols[0].split('_')
    first_col_year, first_col_month = int(first_col_parts[1]), int(first_col_parts[2])
    col_offset = (first_row_year - first_col_year) * 12 + (first_row_month - first_col_month)

    revisions = []
    for k in range(3):
        n = min(n_cols - col_offset, n_rows - k)
        col_years: list[int] = []
        col_months: list[int] = []
        diag_values: list[float | None] = []

        for j in range(n):
            ci = j + col_offset
            parts = emp_cols[ci].split('_')
            col_years.append(int(parts[1]))
            col_months.append(int(parts[2]))
            diag_values.append(tri_df[j + k, emp_cols[ci]])

        revisions.append(
            pl.DataFrame(
                {
                    'year': col_years,
                    'month': col_months,
                    'revision': k,
                    'employment': diag_values,
                }
            )
        )

    return (
        pl.concat(revisions)
        .with_columns(
            ref_date=pl.date(
                pl.col('year'), pl.col('month'), pl.lit(12, pl.UInt8),
            ),
            employment=pl.col('employment').cast(pl.Float64),
        )
        .sort('ref_date', 'revision')
        .select(
            ref_date=pl.col('ref_date'),
            ref_year=pl.col('year'),
            ref_month=pl.col('month'),
            revision=pl.col('revision'),
            geographic_type=pl.lit('national', pl.Utf8),
            geographic_code=pl.lit('00', pl.Utf8),
            industry_type=pl.lit(industry_type, pl.Utf8),
            industry_code=pl.lit(industry_code, pl.Utf8),
            employment=pl.col('employment'),
        )
    )


_SECTOR_RECODE = {
    ces: naics
    for ces, naics in CES_SECTOR_TO_NAICS.items()
    if ces != naics
}


def _recode_to_naics(df: pl.DataFrame) -> pl.DataFrame:
    """Recode CES industry codes to NAICS equivalents and add sector aliases.

    1. Remap CES sector codes (41→42, 42→44, 43→48) to match QCEW NAICS codes.
    2. Duplicate single-sector supersectors (20, 50, 80) as sector rows (23, 51, 81).
    """
    if _SECTOR_RECODE:
        df = df.with_columns(
            pl.when(
                (pl.col('industry_type') == 'sector')
                & pl.col('industry_code').is_in(list(_SECTOR_RECODE.keys()))
            )
            .then(pl.col('industry_code').replace(_SECTOR_RECODE))
            .otherwise(pl.col('industry_code'))
            .alias('industry_code')
        )

    sector_aliases = df.filter(
        (pl.col('industry_type') == 'supersector')
        & pl.col('industry_code').is_in(list(SINGLE_SECTOR_SUPERSECTORS.keys()))
    ).with_columns(
        industry_type=pl.lit('sector'),
        industry_code=pl.col('industry_code').replace(SINGLE_SECTOR_SUPERSECTORS),
    )

    return pl.concat([df, sector_aliases])


def main(ces_dir: Path | None = None) -> None:
    """Read all CES triangular CSVs, join vintage dates, and write Parquet.

    Parameters
    ----------
    ces_dir : Path or None
        Override for the cesvinall directory. Defaults to ``CES_DIR``.
    """
    path = ces_dir or CES_DIR
    ces_files = {p.stem for p in path.iterdir()}
    print(f'Number of CES files: {len(ces_files)}')

    schema, selected, renamed = _build_schema(path)

    # Build file list from INDUSTRY_MAP
    codes: list[tuple[str, str, str, str]] = []
    for adj in ['NSA', 'SA']:
        for entry in INDUSTRY_MAP:
            stem = f'tri_{entry.ces_code}_{adj}'
            industry_type = 'national' if entry.industry_code == '00' else entry.industry_type
            codes.append((stem, entry.industry_code, entry.industry_name, industry_type))

    realized_codes = [
        (file, itype, icode, iname)
        for file, icode, iname, itype in codes
        if file in ces_files
    ]
    print(f'Number of realized industry codes: {len(realized_codes)}')

    nsa_parts: list[pl.DataFrame] = []
    sa_parts: list[pl.DataFrame] = []
    for file, industry_type, industry_code, _name in realized_codes:
        tri_df = read_triangular_ces(
            path, file, industry_type, industry_code, schema, selected, renamed,
        )
        if 'NSA' in file:
            nsa_parts.append(tri_df)
        else:
            sa_parts.append(tri_df)

    print(f'Number of NSA revisions: {len(nsa_parts)}')
    print(f'Number of SA revisions: {len(sa_parts)}')

    nsa_df = pl.concat(nsa_parts).with_columns(
        source=pl.lit('ces', pl.Utf8),
        seasonally_adjusted=pl.lit(False, pl.Boolean),
    )
    sa_df = pl.concat(sa_parts).with_columns(
        source=pl.lit('ces', pl.Utf8),
        seasonally_adjusted=pl.lit(True, pl.Boolean),
    )

    vintage_dates = (
        pl.read_parquet(
            str(VINTAGE_DATES_PATH), storage_options=storage_options_for(VINTAGE_DATES_PATH)
        )
        .filter(pl.col('publication') == 'ces')
        .drop('publication')
    )

    ces_df = (
        pl.concat([nsa_df, sa_df])
        .join(vintage_dates, on=['ref_date', 'revision'], how='left')
        .select(
            'source', 'seasonally_adjusted',
            'geographic_type', 'geographic_code',
            'industry_type', 'industry_code',
            'ref_date', 'vintage_date',
            'revision', 'benchmark_revision',
            'employment',
        )
    )

    ces_df = _recode_to_naics(ces_df)

    print(f'Number of CES revision observations: {ces_df.height:,}')

    # Uniqueness assertion
    dedup = ces_df.unique(subset=[
        'source', 'seasonally_adjusted',
        'geographic_type', 'geographic_code',
        'industry_type', 'industry_code',
        'ref_date', 'vintage_date',
        'revision', 'benchmark_revision',
    ])
    assert ces_df.height == dedup.height, (
        f'Duplicate rows: {ces_df.height} total vs {dedup.height} unique'
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ces_df.write_parquet(OUTPUT_PATH)
    print(f'Wrote {ces_df.height:,} rows to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
