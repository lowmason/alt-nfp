"""Unit tests for parse_index_page (no network, no I/O).

Covers:
- ParseError raised on zero-entry parse (page structure drift).
- Positive path: well-formed HTML with matching anchors returns entries.
"""

from __future__ import annotations

import pytest
from nfp_download.release_dates.scraper import ParseError, ReleaseEntry, parse_index_page

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

# Structurally valid BLS-style page — has a year h4 + ul/li — but the anchor
# hrefs don't match the empsit archive pattern, simulating page drift.
DRIFTED_HTML = """
<html><body>
<h4>2020</h4>
<ul>
  <li><a href="/news.release/otherseries_20200103.htm">January 2020</a></li>
  <li><a href="/news.release/otherseries_20200207.htm">February 2020</a></li>
</ul>
<h4>2019</h4>
<ul>
  <li><a href="/news.release/otherseries_20190104.htm">January 2019</a></li>
</ul>
</body></html>
"""

# Completely empty page — no h4 headings at all.
EMPTY_HTML = "<html><body><p>No content here.</p></body></html>"

# Well-formed page with two valid empsit release anchors.
VALID_HTML = """
<html><body>
<h4>2020</h4>
<ul>
  <li><a href="/news.release/archives/empsit_20200103.htm">January 2020</a></li>
  <li><a href="/news.release/archives/empsit_20200207.htm">February 2020</a></li>
  <li><a href="/news.release/archives/empsit_20200306.htm">March 2020</a></li>
</ul>
</body></html>
"""

# Quarterly page for cewqtr series.
VALID_QUARTERLY_HTML = """
<html><body>
<h4>2020</h4>
<ul>
  <li>First Quarter 2020 <a href="/news.release/archives/cewqtr_20200617.htm">link</a></li>
  <li>Second Quarter 2020 <a href="/news.release/archives/cewqtr_20200916.htm">link</a></li>
</ul>
</body></html>
"""


# ---------------------------------------------------------------------------
# Negative tests — must raise ParseError
# ---------------------------------------------------------------------------


class TestParseIndexPageRaisesOnDrift:
    """parse_index_page raises ParseError rather than returning an empty list."""

    def test_drifted_anchors_raises(self):
        """Structurally intact page whose hrefs don't match the series raises."""
        with pytest.raises(ParseError, match="ces"):
            parse_index_page(DRIFTED_HTML, "ces", "empsit", "monthly")

    def test_empty_page_raises(self):
        """A page with no h4/ul structure at all raises."""
        with pytest.raises(ParseError, match="ces"):
            parse_index_page(EMPTY_HTML, "ces", "empsit", "monthly")

    def test_error_message_names_publication(self):
        """ParseError message includes the publication name for diagnostics."""
        with pytest.raises(ParseError) as exc_info:
            parse_index_page(DRIFTED_HTML, "qcew", "cewqtr", "quarterly")
        assert "qcew" in str(exc_info.value)

    def test_error_mentions_drift(self):
        """ParseError message indicates the page structure may have changed."""
        with pytest.raises(ParseError) as exc_info:
            parse_index_page(EMPTY_HTML, "ces", "empsit", "monthly")
        msg = str(exc_info.value).lower()
        assert any(word in msg for word in ("drift", "structure", "no release", "zero", "0"))


# ---------------------------------------------------------------------------
# Positive tests — valid HTML returns entries unchanged
# ---------------------------------------------------------------------------


class TestParseIndexPagePositive:
    """parse_index_page returns correct entries for well-formed pages."""

    def test_monthly_returns_entries(self):
        entries = parse_index_page(VALID_HTML, "ces", "empsit", "monthly")
        assert len(entries) == 3
        assert all(isinstance(e, ReleaseEntry) for e in entries)

    def test_monthly_entry_fields(self):
        entries = parse_index_page(VALID_HTML, "ces", "empsit", "monthly")
        jan = next(e for e in entries if e.ref_month == 1)
        assert jan.ref_year == 2020
        assert "empsit_20200103" in jan.url

    def test_monthly_absolute_urls(self):
        """Relative hrefs are resolved to absolute URLs."""
        entries = parse_index_page(VALID_HTML, "ces", "empsit", "monthly")
        assert all(e.url.startswith("http") for e in entries)

    def test_quarterly_returns_entries(self):
        entries = parse_index_page(VALID_QUARTERLY_HTML, "qcew", "cewqtr", "quarterly")
        assert len(entries) == 2

    def test_quarterly_months_correct(self):
        entries = parse_index_page(VALID_QUARTERLY_HTML, "qcew", "cewqtr", "quarterly")
        months = sorted(e.ref_month for e in entries)
        assert months == [3, 6]  # First Quarter → 3, Second Quarter → 6
