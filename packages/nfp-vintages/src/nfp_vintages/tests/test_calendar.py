"""Tests for advance_release_calendar — the §5.0 calendar-advance callable.

The unit test stubs the BLS network (fetch_index raises FetchError, exercising
the graceful-403 fallback) so no real bls.gov hit occurs, redirects every path
constant to tmp_path, and asserts VINTAGE_DATES_PATH is written non-empty. The
live path is marked @pytest.mark.network.
"""

from __future__ import annotations

import polars as pl
import pytest
from nfp_download.release_dates.scraper import FetchError


def _patch_paths(monkeypatch, tmp_path):
    """Redirect both the writer's and reader's path bindings to tmp_path."""
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    releases_dir = tmp_path / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    release_dates_path = intermediate / "release_dates.parquet"
    vintage_dates_path = intermediate / "vintage_dates.parquet"

    # advance_release_calendar imports these in-function → resolved at call time.
    monkeypatch.setattr("nfp_lookups.paths.RELEASES_DIR", releases_dir)
    monkeypatch.setattr("nfp_lookups.paths.RELEASE_DATES_PATH", release_dates_path)
    monkeypatch.setattr("nfp_lookups.paths.VINTAGE_DATES_PATH", vintage_dates_path)
    # build_vintage_dates binds RELEASE_DATES_PATH in ITS module at import — patch too.
    monkeypatch.setattr(
        "nfp_ingest.release_dates.vintage_dates.RELEASE_DATES_PATH",
        release_dates_path,
    )
    return release_dates_path, vintage_dates_path


def test_advance_release_calendar_writes_vintage_dates(monkeypatch, tmp_path):
    """With the scrape stubbed to a graceful-403 fallback, the calendar advance
    still builds and writes vintage_dates.parquet from supplemental/pre-scrape rows."""
    release_dates_path, vintage_dates_path = _patch_paths(monkeypatch, tmp_path)

    class _StubSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    def _stub_create_session(*args, **kwargs):
        return _StubSession()

    async def _stub_fetch_index(session, url):
        # Simulate BLS 403 — drives the cached-pages-only fallback (§5.0).
        raise FetchError("stubbed 403")

    monkeypatch.setattr(
        "nfp_download.release_dates.scraper.create_session", _stub_create_session
    )
    monkeypatch.setattr(
        "nfp_download.release_dates.scraper.fetch_index", _stub_fetch_index
    )

    from nfp_vintages.calendar import advance_release_calendar

    advance_release_calendar()

    assert release_dates_path.exists()
    assert vintage_dates_path.exists()
    vdf = pl.read_parquet(vintage_dates_path)
    assert vdf.height > 0
    assert set(vdf.columns) >= {
        "publication",
        "ref_date",
        "vintage_date",
        "revision",
        "benchmark_revision",
    }


@pytest.mark.network
def test_advance_release_calendar_live(monkeypatch, tmp_path):
    """Live BLS scrape path — redirected to tmp so it never clobbers prod."""
    _, vintage_dates_path = _patch_paths(monkeypatch, tmp_path)

    from nfp_vintages.calendar import advance_release_calendar

    advance_release_calendar()

    assert vintage_dates_path.exists()
    assert pl.read_parquet(vintage_dates_path).height > 0
