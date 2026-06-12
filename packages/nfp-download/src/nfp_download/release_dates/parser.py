"""Extract release (vintage) dates from downloaded BLS release HTML files.

Each BLS news release contains an embargo line with the release date; this
module parses ref_date from the filename and vintage_date from the HTML.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

VINTAGE_DATE_RE = re.compile(
    r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+'
    r'(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+'
    r'(\d{1,2}),\s+(\d{4})',
    re.IGNORECASE,
)

MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]
MONTH_TO_NUM = {name: i for i, name in enumerate(MONTH_NAMES, 1)}


def parse_vintage_date(html_content: str) -> date | None:
    """Extract release (vintage) date from embargo line in HTML.

    Parameters
    ----------
    html_content : str
        Raw HTML content of a BLS release page.

    Returns
    -------
    date or None
        Parsed release date, or ``None`` if not found.
    """
    match = VINTAGE_DATE_RE.search(html_content)
    if not match:
        return None
    month_name, day_str, year_str = match.group(1), match.group(2), match.group(3)
    month = MONTH_TO_NUM.get(month_name)
    if month is None:
        return None
    try:
        day = int(day_str)
        year = int(year_str)
        return date(year, month, day)
    except (ValueError, TypeError):
        return None


def parse_ref_from_path(path: Path) -> tuple[int, int] | None:
    """Parse reference year and month from a release filename.

    Expected format: ``{pub}_{yyyy}_{mm}.htm`` (e.g. ``ces_2010_03.htm``).

    Parameters
    ----------
    path : Path
        Path to the release HTML file.

    Returns
    -------
    tuple[int, int] or None
        ``(year, month)`` or ``None`` if the filename doesn't match.
    """
    stem = path.stem
    parts = stem.split('_')
    if len(parts) != 3:
        return None
    try:
        yyyy, mm = int(parts[1]), int(parts[2])
        if 1 <= mm <= 12 and 2000 <= yyyy <= 2100:
            return (yyyy, mm)
    except ValueError:
        pass
    return None


def ref_date_from_year_month(year: int, month: int) -> date:
    """Return the reference date (12th of the month).

    Parameters
    ----------
    year : int
        Reference year.
    month : int
        Reference month.

    Returns
    -------
    date
        The 12th of the given month/year.
    """
    return date(year, month, 12)


def parse_release_file(
    path: Path, publication_name: str,
) -> tuple[str, date, date] | None:
    """Read a release HTML file and extract publication, ref_date, and vintage_date.

    Parameters
    ----------
    path : Path
        Path to a downloaded release HTML file.
    publication_name : str
        Publication name (e.g. ``'ces'``).

    Returns
    -------
    tuple[str, date, date] or None
        ``(publication_name, ref_date, vintage_date)`` or ``None`` if parsing fails.
    """
    ref = parse_ref_from_path(path)
    if ref is None:
        return None
    ref_year, ref_month = ref
    ref_d = ref_date_from_year_month(ref_year, ref_month)

    try:
        content = path.read_text(encoding='utf-8')
    except OSError:
        return None

    vintage_d = parse_vintage_date(content)
    if vintage_d is None:
        return None

    return (publication_name, ref_d, vintage_d)


def collect_release_dates(
    publication_name: str, releases_dir: Path,
) -> Iterator[tuple[str, date, date]]:
    """Walk a publication's release directory and yield parsed release rows.

    Parameters
    ----------
    publication_name : str
        Publication name (e.g. ``'ces'``).
    releases_dir : Path
        Directory containing downloaded release HTML files.

    Yields
    ------
    tuple[str, date, date]
        ``(publication_name, ref_date, vintage_date)`` for each successfully parsed file.
    """
    pattern = f'{publication_name}_*.htm'
    for path in sorted(releases_dir.glob(pattern)):
        row = parse_release_file(path, publication_name)
        if row is None:
            log.warning('Could not parse release date from %s', path)
            continue
        yield row
