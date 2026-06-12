"""Download QCEW data from BLS.

Two download functions:

- ``download_qcew``: the revisions CSV (2017-present), for revision history.
- ``download_qcew_bulk``: quarterly singlefile ZIPs (2003-present), for
  sector-level employment by state.  Each ~280 MB ZIP is downloaded, filtered
  to the needed rows, and discarded — only the compact filtered parquet is kept.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import httpx
import polars as pl
from nfp_download.client import create_client, get_with_retry
from nfp_lookups.geography import STATES
from nfp_lookups.paths import DATA_DIR

QCEW_CSV_URL = 'https://www.bls.gov/cew/revisions/qcew-revisions.csv'
QCEW_FILENAME = 'qcew-revisions.csv'

BULK_BASE_URL = 'https://data.bls.gov/cew/data/files'
BULK_OUTPUT_FILENAME = 'qcew_bulk.parquet'

_STATE_AREA_FIPS: frozenset[str] = frozenset(f'{s}000' for s in STATES)
_WANTED_AREAS: frozenset[str] = _STATE_AREA_FIPS | {'US000'}

# Aggregation levels kept from bulk singlefiles:
#   10/50 = national/state total
#   11/51 = national/state by ownership (government extraction)
#   14/54 = national/state by NAICS 2-digit sector
#   15/55 = national/state by NAICS 3-digit subsector (mfg durable/nondurable)
_WANTED_AGGLVL: frozenset[str] = frozenset(
    {'10', '11', '14', '15', '50', '51', '54', '55'}
)

_KEEP_COLUMNS: list[str] = [
    'area_fips', 'own_code', 'industry_code', 'agglvl_code',
    'year', 'qtr',
    'month1_emplvl', 'month2_emplvl', 'month3_emplvl',
]


def download_qcew(
    data_dir: Path | None = None,
    *,
    client: httpx.Client | None = None,
) -> None:
    """Download the QCEW revisions CSV.

    Parameters
    ----------
    data_dir : Path or None
        Root data directory. Defaults to ``DATA_DIR``.
    client : httpx.Client or None
        Optional pre-built client. A new one is created if not provided.
    """
    base = (data_dir or DATA_DIR) / 'downloads'
    qcew_dir = base / 'qcew'
    qcew_dir.mkdir(parents=True, exist_ok=True)
    out_path = qcew_dir / QCEW_FILENAME

    own_client = client is None
    if client is None:
        client = create_client()

    try:
        r = get_with_retry(client, QCEW_CSV_URL)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        print(f'  saved {QCEW_FILENAME}')
    finally:
        if own_client:
            client.close()


def _filter_bulk_csv(csv_bytes: bytes) -> pl.DataFrame:
    """Read a QCEW quarterly singlefile CSV and return the filtered subset.

    Keeps rows matching:
    - ``area_fips`` in national + state FIPS set
    - ``agglvl_code`` in :data:`_WANTED_AGGLVL`
    - ``own_code`` in ``{'0', '1', '2', '3', '5'}``
      (total, federal, state, local government, private)
    """
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        infer_schema_length=0,
        n_threads=1,
    )
    df = df.filter(
        pl.col('area_fips').is_in(_WANTED_AREAS)
        & pl.col('agglvl_code').is_in(_WANTED_AGGLVL)
        & pl.col('own_code').is_in({'0', '1', '2', '3', '5'})
    )
    present = [c for c in _KEEP_COLUMNS if c in df.columns]
    return df.select(present)


def download_qcew_bulk(
    start_year: int = 2003,
    end_year: int = 2025,
    data_dir: Path | None = None,
    *,
    client: httpx.Client | None = None,
) -> Path:
    """Download QCEW quarterly singlefile ZIPs and extract filtered data.

    For each year, downloads the ~280 MB ZIP, extracts the CSV, filters to
    national + state rows for total and private-sector industries, then
    discards the ZIP.  The compact filtered result is saved as a single
    parquet file.

    Parameters
    ----------
    start_year : int
        First year to download (default 2003).
    end_year : int
        Last year to download inclusive (default 2025).
    data_dir : Path or None
        Root data directory. Defaults to ``DATA_DIR``.
    client : httpx.Client or None
        Optional pre-built client. A new one is created if not provided.

    Returns
    -------
    Path
        Path to the output parquet file.
    """
    base = (data_dir or DATA_DIR) / 'downloads'
    qcew_dir = base / 'qcew'
    qcew_dir.mkdir(parents=True, exist_ok=True)
    out_path = qcew_dir / BULK_OUTPUT_FILENAME

    own_client = client is None
    if client is None:
        client = create_client()

    frames: list[pl.DataFrame] = []
    try:
        for year in range(start_year, end_year + 1):
            url = (
                f'{BULK_BASE_URL}/{year}/csv/{year}_qtrly_singlefile.zip'
            )
            print(f'  downloading {year} quarterly singlefile ...', flush=True)
            r = get_with_retry(client, url, timeout=300.0)
            r.raise_for_status()

            with tempfile.TemporaryDirectory() as tmp:
                zip_path = Path(tmp) / f'{year}.zip'
                zip_path.write_bytes(r.content)

                with zipfile.ZipFile(zip_path) as zf:
                    csv_names = [
                        n for n in zf.namelist() if n.endswith('.csv')
                    ]
                    if not csv_names:
                        print(f'    WARNING: no CSV in {year} ZIP', flush=True)
                        continue
                    csv_bytes = zf.read(csv_names[0])

                filtered = _filter_bulk_csv(csv_bytes)
                frames.append(filtered)
                print(
                    f'    {year}: kept {filtered.height:,} rows '
                    f'({len(r.content) / 1024 / 1024:.0f} MB downloaded)',
                    flush=True,
                )
    finally:
        if own_client:
            client.close()

    if not frames:
        print('  WARNING: no data collected', flush=True)
        return out_path

    combined = pl.concat(frames, how='diagonal_relaxed')
    combined.write_parquet(out_path)
    print(f'  wrote {out_path} ({combined.height:,} rows)', flush=True)
    return out_path
