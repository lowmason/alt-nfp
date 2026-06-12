"""Fetch BLS archive index pages and download individual release HTMLs.

Fetches index page, parses release links, downloads each release HTML to
data/downloads/releases/{pub}/.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

from nfp_download.release_dates.config import (
    BASE_URL,
    PUBLICATIONS,
    RELEASES_DIR,
    START_YEAR,
)

MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]
MONTH_TO_NUM = {name: i for i, name in enumerate(MONTH_NAMES, 1)}
QUARTER_TO_MONTH = {'First': 3, 'Second': 6, 'Third': 9, 'Fourth': 12}

MONTH_YEAR_RE = re.compile(
    r'(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+(\d{4})',
    re.IGNORECASE,
)
QUARTER_RE = re.compile(
    r'(First|Second|Third|Fourth)\s+Quarter',
    re.IGNORECASE,
)
YEAR_RE = re.compile(r'\b(20\d{2})\b')


def archive_href_re(series: str) -> re.Pattern:
    """Build a regex that matches archive hrefs for the given BLS series."""
    return re.compile(rf'/news\.release/archives/{re.escape(series)}_\d{{8}}\.htm')


@dataclass
class ReleaseEntry:
    """A single release: reference year, month, and archive URL."""

    ref_year: int
    ref_month: int
    url: str


def _find_next_ul(element: Tag) -> Tag | None:
    """Find the next <ul> sibling after the given element."""
    sibling = element.find_next_sibling()
    while sibling:
        if sibling.name == 'ul':
            return sibling
        if sibling.name and sibling.name.startswith('h'):
            break
        sibling = sibling.find_next_sibling()
    return None


def _resolve_url(url: str) -> str:
    """Turn a possibly relative URL into an absolute URL."""
    if url.startswith('http'):
        return url
    base = BASE_URL.rstrip('/')
    path = url if url.startswith('/') else f'/{url}'
    return f'{base}{path}'


def parse_index_page(
    html: str,
    publication_name: str,
    series: str,
    frequency: str,
) -> list[ReleaseEntry]:
    """Parse an archive index page into release entries (years >= START_YEAR).

    Parameters
    ----------
    html : str
        Raw HTML of the BLS archive index page.
    publication_name : str
        Publication name (e.g. ``'ces'``).
    series : str
        BLS series code (e.g. ``'empsit'``).
    frequency : str
        ``'monthly'`` or ``'quarterly'``.

    Returns
    -------
    list[ReleaseEntry]
        Parsed release entries with ref_year, ref_month, and archive URL.
    """
    soup = BeautifulSoup(html, 'lxml')
    href_re = archive_href_re(series)
    entries: list[ReleaseEntry] = []

    for h4 in soup.find_all('h4'):
        year_match = YEAR_RE.search(h4.get_text())
        if not year_match:
            continue
        year = int(year_match.group(1))
        if year < START_YEAR:
            continue

        ul = _find_next_ul(h4)
        if not ul:
            continue

        for li in ul.find_all('li', recursive=False):
            li_text = li.get_text()
            anchor = None
            for a in li.find_all('a', href=True):
                if href_re.search(a.get('href', '')):
                    anchor = a
                    break
            if not anchor:
                continue

            href = anchor.get('href', '')
            if not href_re.search(href):
                continue
            url = _resolve_url(href)

            if frequency == 'monthly':
                month_match = (
                    MONTH_YEAR_RE.search(li_text)
                    or MONTH_YEAR_RE.search(anchor.get_text() or '')
                )
                if not month_match:
                    continue
                month_name, year_str = month_match.group(1), month_match.group(2)
                ref_year = int(year_str)
                ref_month = MONTH_TO_NUM.get(month_name)
                if ref_month is None:
                    continue
            else:
                quarter_match = QUARTER_RE.search(li_text)
                if not quarter_match:
                    continue
                quarter_name = quarter_match.group(1)
                ref_year = year
                ref_month = QUARTER_TO_MONTH.get(quarter_name)
                if ref_month is None:
                    continue

            entries.append(ReleaseEntry(
                ref_year=ref_year, ref_month=ref_month, url=url,
            ))

    return entries


def _request_headers() -> dict[str, str]:
    """Browser-like headers for BLS requests."""
    return {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.bls.gov/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }


async def fetch_index(client: httpx.AsyncClient, url: str) -> str:
    """Fetch index page HTML.

    Parameters
    ----------
    client : httpx.AsyncClient
        HTTP client.
    url : str
        Index page URL.

    Returns
    -------
    str
        Raw HTML string.
    """
    r = await client.get(url, headers=_request_headers())
    r.raise_for_status()
    return r.text


async def download_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    entry: ReleaseEntry,
    publication_name: str,
    out_dir: Path,
) -> Path | None:
    """Download one release HTML to out_dir/{pub}_{yyyy}_{mm}.htm."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mm = f'{entry.ref_month:02d}'
    path = out_dir / f'{publication_name}_{entry.ref_year}_{mm}.htm'
    if path.exists():
        return path

    async with semaphore:
        r = await client.get(entry.url)
        r.raise_for_status()
        path.write_text(r.text, encoding='utf-8')
        return path


async def download_all(
    entries: list[ReleaseEntry],
    publication_name: str,
    concurrency: int = 5,
) -> list[Path]:
    """Download all release HTMLs for a publication; skip existing files.

    Parameters
    ----------
    entries : list[ReleaseEntry]
        Release entries to download.
    publication_name : str
        Publication name (used for output directory and filenames).
    concurrency : int
        Maximum concurrent downloads.

    Returns
    -------
    list[Path]
        Paths to downloaded or already-existing release HTML files.
    """
    out_dir = RELEASES_DIR / publication_name
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        http2=True,
        base_url=BASE_URL,
        follow_redirects=True,
        timeout=30.0,
        headers=_request_headers(),
    ) as client:
        tasks = [
            download_one(client, semaphore, e, publication_name, out_dir)
            for e in entries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    paths: list[Path] = []
    for r in results:
        if isinstance(r, Exception):
            raise r
        if r is not None:
            paths.append(r)
    return paths


async def scrape_publication(publication_name: str | None = None) -> None:
    """Scrape release HTMLs for one or all publications.

    Parameters
    ----------
    publication_name : str or None
        If given, scrape only that publication. Otherwise scrape all.
    """
    pubs = PUBLICATIONS
    if publication_name:
        pubs = [p for p in PUBLICATIONS if p.name == publication_name]
        if not pubs:
            raise ValueError(f'Unknown publication: {publication_name!r}')

    async with httpx.AsyncClient(
        http2=True,
        follow_redirects=True,
        timeout=30.0,
        headers=_request_headers(),
    ) as client:
        for pub in pubs:
            html = await fetch_index(client, pub.index_url)
            entries = parse_index_page(html, pub.name, pub.series, pub.frequency)
            if not entries:
                continue
            await download_all(entries, pub.name)
