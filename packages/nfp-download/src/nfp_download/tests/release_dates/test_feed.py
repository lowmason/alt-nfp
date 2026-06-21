"""Unit tests for feed.parse_feed (pure, no network).

Fixture is standard RSS 2.0 matching the BLS empsit/cewqtr feed shape: each
<item> carries <title>, an RFC-822 <pubDate>, and a <guid>. We could not
live-capture in red phase — www.bls.gov intermittently 403s a plain GET (the
Akamai TLS block that forces fetch_feed's curl_cffi session; Task 8.2).
pubDate format is pinned to RFC-822.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nfp_download.release_dates.feed import (
    EMPSIT_FEED_URL,
    FeedItem,
    fetch_feed,
    parse_feed,
)

EMPSIT_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Employment Situation</title>
    <link>https://www.bls.gov/news.release/empsit.htm</link>
    <item>
      <title>Employment Situation Summary</title>
      <link>https://www.bls.gov/news.release/archives/empsit_06062025.htm</link>
      <pubDate>Fri, 06 Jun 2025 08:30:00 -0400</pubDate>
      <guid>https://www.bls.gov/news.release/archives/empsit_06062025.htm</guid>
    </item>
    <item>
      <title>Employment Situation Summary</title>
      <link>https://www.bls.gov/news.release/archives/empsit_05022025.htm</link>
      <pubDate>Fri, 02 May 2025 08:30:00 -0400</pubDate>
      <guid>https://www.bls.gov/news.release/archives/empsit_05022025.htm</guid>
    </item>
  </channel>
</rss>
"""


class TestParseFeed:
    def test_returns_feed_items(self):
        items = parse_feed(EMPSIT_RSS)
        assert len(items) == 2
        assert all(isinstance(it, FeedItem) for it in items)

    def test_first_item_fields(self):
        items = parse_feed(EMPSIT_RSS)
        first = items[0]
        assert first.title == "Employment Situation Summary"
        assert first.pub_date == date(2025, 6, 6)
        assert first.guid == "https://www.bls.gov/news.release/archives/empsit_06062025.htm"

    def test_pubdate_parsed_as_date_object(self):
        items = parse_feed(EMPSIT_RSS)
        assert all(isinstance(it.pub_date, date) for it in items)
        assert items[1].pub_date == date(2025, 5, 2)

    def test_items_in_feed_order_newest_first(self):
        items = parse_feed(EMPSIT_RSS)
        assert items[0].pub_date >= items[1].pub_date

    def test_empty_channel_returns_empty_list(self):
        empty = '<?xml version="1.0"?><rss version="2.0"><channel/></rss>'
        assert parse_feed(empty) == []

    def test_item_missing_pubdate_is_skipped(self):
        no_date = """\
<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>No date</title><guid>g1</guid></item>
</channel></rss>"""
        assert parse_feed(no_date) == []

    def test_item_missing_guid_is_skipped(self):
        no_guid = """\
<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>No guid</title>
    <pubDate>Fri, 06 Jun 2025 08:30:00 -0400</pubDate>
  </item>
</channel></rss>"""
        assert parse_feed(no_guid) == []

    def test_item_missing_title_is_skipped(self):
        no_title = """\
<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <pubDate>Fri, 06 Jun 2025 08:30:00 -0400</pubDate>
    <guid>g1</guid>
  </item>
</channel></rss>"""
        assert parse_feed(no_title) == []


# ── network smoke test ──────────────────────────────────────────────────────

@pytest.mark.network
class TestFetchFeedNetwork:
    def test_fetch_empsit_returns_feed_items(self):
        items = fetch_feed(EMPSIT_FEED_URL)
        assert isinstance(items, list)
        assert items, "empsit feed should publish at least one item"
        assert all(isinstance(it, FeedItem) for it in items)
        # BLS lists newest first.
        assert items[0].pub_date >= items[-1].pub_date


# ── monkeypatched unit test (no network) ────────────────────────────────────

_MINIMAL_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Employment Situation Summary</title>
      <pubDate>Fri, 06 Jun 2025 08:30:00 -0400</pubDate>
      <guid>https://www.bls.gov/news.release/archives/empsit_06062025.htm</guid>
    </item>
  </channel>
</rss>
"""


class TestFetchFeedUnit:
    def test_fetch_feed_calls_create_session(self):
        """fetch_feed drives an async session; monkeypatch create_session."""
        fake_resp = MagicMock()
        fake_resp.text = _MINIMAL_RSS
        fake_resp.raise_for_status = MagicMock()

        fake_session = AsyncMock()
        fake_session.get = AsyncMock(return_value=fake_resp)
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "nfp_download.release_dates.feed.create_session",
            return_value=fake_session,
        ):
            items = fetch_feed(EMPSIT_FEED_URL)

        assert len(items) == 1
        assert items[0].pub_date == date(2025, 6, 6)
        fake_session.get.assert_awaited_once_with(EMPSIT_FEED_URL)
