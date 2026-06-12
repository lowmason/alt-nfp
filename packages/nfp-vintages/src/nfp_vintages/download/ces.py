"""Download CES vintage (triangular revision) data from BLS.

Scrapes the CES vintage-data page for the ``cesvinall.zip`` bundle and
extracts it into ``DATA_DIR / 'downloads' / 'ces' / 'cesvinall'``.
Only the zip is needed; the individual xlsx workbooks on the same page
contain the same data and are not used downstream.

Both URLs live on www.bls.gov, where Akamai bot management fingerprints
the TLS handshake (plain httpx gets 403 regardless of headers), so
transport is the Chrome-impersonating session from
:func:`nfp_download.client.create_impersonating_session`.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from curl_cffi.requests import Session
from nfp_download.client import create_impersonating_session, get_with_retry
from nfp_lookups.paths import DATA_DIR

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
