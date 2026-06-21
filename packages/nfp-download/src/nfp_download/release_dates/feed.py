"""BLS release RSS feed — fetch + parse.

The feed answers the production question the calendar can only predict and
shutdowns can delay: "is the release out *now*?". ``parse_feed`` is pure (no
network); ``fetch_feed`` reuses the scraper's curl_cffi Chrome-impersonating
session (www.bls.gov/Akamai 403s a plain httpx GET — memory
``bls-akamai-blocking-intermittent``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime

EMPSIT_FEED_URL = "https://www.bls.gov/feed/empsit.rss"
CEWQTR_FEED_URL = "https://www.bls.gov/feed/cewqtr.rss"


@dataclass
class FeedItem:
    """One RSS <item>: release title, publication date, and unique id."""

    title: str
    pub_date: date
    guid: str


def _text(item: ET.Element, tag: str) -> str | None:
    """Return the stripped text of the first child ``tag``, or None."""
    child = item.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_feed(xml: str) -> list[FeedItem]:
    """Parse a BLS RSS 2.0 feed into FeedItems (pure, no network).

    ``pubDate`` is RFC-822 (e.g. ``Fri, 06 Jun 2025 08:30:00 -0400``); the
    calendar date is extracted. Items missing a title, a parseable pubDate, or
    a guid are skipped — a malformed item should not sink the poll.

    Parameters
    ----------
    xml : str
        Raw RSS feed body.

    Returns
    -------
    list[FeedItem]
        One per well-formed <item>, in feed order (BLS lists newest first).
    """
    root = ET.fromstring(xml)
    items: list[FeedItem] = []
    for item in root.iter("item"):
        title = _text(item, "title")
        raw_date = _text(item, "pubDate")
        guid = _text(item, "guid")
        if title is None or raw_date is None or guid is None:
            continue
        try:
            pub = parsedate_to_datetime(raw_date).date()
        except (TypeError, ValueError):
            continue
        items.append(FeedItem(title=title, pub_date=pub, guid=guid))
    return items
