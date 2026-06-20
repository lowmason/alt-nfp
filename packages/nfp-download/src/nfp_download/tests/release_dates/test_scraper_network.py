"""Live tests for the BLS release-calendar scraper transport.

www.bls.gov (Akamai) fingerprints the TLS handshake, so these tests verify
the Chrome-impersonating session actually passes bot detection — not just
that parsing works. Marked network; deselected in CI with ``-m 'not network'``.
"""

import asyncio

import pytest
from nfp_download.release_dates.config import PUBLICATIONS
from nfp_download.release_dates.scraper import (
    create_session,
    download_one,
    fetch_index,
    parse_index_page,
)

pytestmark = pytest.mark.network

# Conservative floors: empsit is monthly since 2003 (~270 releases),
# cewqtr quarterly (~80) — well above these even if BLS trims old years.
MIN_ENTRIES = {'ces': 200, 'qcew': 60}


def test_fetch_and_parse_all_publication_indexes():
    """Both archive index pages return 200 and parse into release entries."""

    async def _run() -> dict[str, int]:
        counts: dict[str, int] = {}
        async with create_session() as session:
            for pub in PUBLICATIONS:
                html = await fetch_index(session, pub.index_url)
                entries = parse_index_page(html, pub.name, pub.series, pub.frequency)
                counts[pub.name] = len(entries)
        return counts

    counts = asyncio.run(_run())
    for name, floor in MIN_ENTRIES.items():
        assert counts[name] >= floor, f'{name}: {counts[name]} entries < {floor}'


def test_download_one_release_page(tmp_path):
    """A single archived release page downloads to disk with real content."""

    async def _run():
        async with create_session() as session:
            pub = next(p for p in PUBLICATIONS if p.name == 'ces')
            html = await fetch_index(session, pub.index_url)
            entries = parse_index_page(html, pub.name, pub.series, pub.frequency)
            assert entries, 'index parsed to zero entries'
            semaphore = asyncio.Semaphore(1)
            return await download_one(session, semaphore, entries[0], pub.name, tmp_path)

    path = asyncio.run(_run())
    assert path is not None and path.exists()
    text = path.read_text(encoding='utf-8')
    assert len(text) > 10_000
    assert 'Access Denied' not in text
