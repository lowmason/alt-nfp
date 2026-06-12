"""Bulk file downloads from BLS: CES triangular vintages + QCEW files.

Three download functions (moved here from ``nfp_vintages.download`` — they
are acquisition, not vintage processing):

- ``download_ces``: scrapes the CES vintage-data page for ``cesvinall.zip``
  (triangular revision CSVs) and extracts it into
  ``{data_dir}/downloads/ces/cesvinall/``. Only the zip is needed; the xlsx
  workbooks on the same page contain the same data.
- ``download_qcew``: the QCEW revisions CSV (2017-present).
- ``download_qcew_bulk``: quarterly singlefile ZIPs (2003-present) for
  sector-level employment by state. Each ~280 MB ZIP is downloaded,
  filtered to the needed rows, and discarded — only the compact filtered
  parquet is kept.

The CES page and the revisions CSV live on www.bls.gov, where Akamai bot
management fingerprints the TLS handshake (plain httpx gets 403 regardless
of headers), so transport is the Chrome-impersonating session from
:func:`nfp_download.client.create_impersonating_session`. The bulk files
live on data.bls.gov, which plain httpx fetches fine.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import httpx
import polars as pl
from bs4 import BeautifulSoup
from curl_cffi.requests import Session
from nfp_lookups.geography import STATES
from nfp_lookups.paths import DATA_DIR

from nfp_download.client import (
    create_client,
    create_impersonating_session,
    get_with_retry,
)

# ---------------------------------------------------------------------------
# CES triangular vintage files (cesvinall.zip)
# ---------------------------------------------------------------------------

CES_INDEX_URL = 'https://www.bls.gov/web/empsit/cesvindata.htm'
CES_BASE_URL = 'https://www.bls.gov/web/empsit/'


def _find_zip_url(html: str) -> str:
    """Locate the ``cesvinall.zip`` link on the CES vintage-data page."""
    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if 'cesvinall.zip' in href.lower():
            return urljoin(CES_BASE_URL, href)
    raise RuntimeError('cesvinall.zip link not found on CES index page')


def download_ces(
    data_dir: Path | None = None,
    *,
    session: Session | None = None,
) -> None:
    """Download and extract ``cesvinall.zip`` from the BLS CES vintage page.

    The zip is extracted into ``{data_dir}/downloads/ces/cesvinall/``.

    Parameters
    ----------
    data_dir : Path or None
        Root data directory. Defaults to ``DATA_DIR``.
    session : curl_cffi.requests.Session or None
        Optional pre-built impersonating session. A new one is created if
        not provided.
    """
    ces_dir = (data_dir or DATA_DIR) / 'downloads' / 'ces'
    ces_dir.mkdir(parents=True, exist_ok=True)

    own_session = session is None
    if session is None:
        session = create_impersonating_session()

    try:
        r = get_with_retry(session, CES_INDEX_URL)
        r.raise_for_status()
        zip_url = _find_zip_url(r.text)

        r = get_with_retry(session, zip_url)
        r.raise_for_status()

        extract_to = ces_dir / 'cesvinall'
        extract_to.mkdir(parents=True, exist_ok=True)
        zip_path = ces_dir / 'cesvinall.zip'
        zip_path.write_bytes(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        zip_path.unlink()
        print(f'  extracted cesvinall.zip -> {extract_to}/')
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# QCEW revisions CSV + quarterly bulk singlefiles
# ---------------------------------------------------------------------------

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
    session: Session | None = None,
) -> None:
    """Download the QCEW revisions CSV.

    Parameters
    ----------
    data_dir : Path or None
        Root data directory. Defaults to ``DATA_DIR``.
    session : curl_cffi.requests.Session or None
        Optional pre-built impersonating session. A new one is created if
        not provided.
    """
    base = (data_dir or DATA_DIR) / 'downloads'
    qcew_dir = base / 'qcew'
    qcew_dir.mkdir(parents=True, exist_ok=True)
    out_path = qcew_dir / QCEW_FILENAME

    own_session = session is None
    if session is None:
        session = create_impersonating_session()

    try:
        r = get_with_retry(session, QCEW_CSV_URL)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        print(f'  saved {QCEW_FILENAME}')
    finally:
        if own_session:
            session.close()


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
