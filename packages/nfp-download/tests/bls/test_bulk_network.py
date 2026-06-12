"""Live tests for the www.bls.gov download transports.

www.bls.gov (Akamai) fingerprints the TLS handshake and 403s non-browser
clients, so these verify the Chrome-impersonating session actually passes
bot detection on the CES vintage-data page and the QCEW revisions CSV —
not just that the code runs. The cesvinall.zip itself is not downloaded
here to keep the suite polite. Marked network; deselected in CI with
``-m 'not network'``.
"""

import pytest
from nfp_download.bls.bulk import CES_INDEX_URL, QCEW_FILENAME, _find_zip_url, download_qcew
from nfp_download.client import create_impersonating_session, get_with_retry

pytestmark = pytest.mark.network


def test_ces_index_fetch_and_zip_link():
    """CES vintage index returns real HTML containing the cesvinall.zip link."""
    with create_impersonating_session() as session:
        r = get_with_retry(session, CES_INDEX_URL)
    assert r.status_code == 200
    assert 'Access Denied' not in r.text
    zip_url = _find_zip_url(r.text)
    assert zip_url.startswith('https://www.bls.gov/')
    assert zip_url.lower().endswith('cesvinall.zip')


def test_download_qcew_revisions_csv(tmp_path):
    """The QCEW revisions CSV downloads end-to-end with real content."""
    download_qcew(tmp_path)
    out = tmp_path / 'downloads' / 'qcew' / QCEW_FILENAME
    assert out.exists()
    data = out.read_bytes()
    assert len(data) > 100_000
    assert b'Access Denied' not in data[:2000]
