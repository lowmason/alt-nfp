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


def test_calendar_refresh_writes_no_scraped_html_under_data(monkeypatch, tmp_path):
    """A SUCCESSFUL scrape must route release HTML to a tempdir, never ./data.

    `update` calls advance_release_calendar() on every run (refresh_calendar=True),
    so a hardcoded RELEASES_DIR (= ./data/downloads/releases) write would violate
    the plans/15 container contract on Bloomberg. This drives the non-403 path
    (download_all is reached) and asserts the per-publication HTML target is a
    tempfile dir, with nothing written under the ./data-rooted RELEASES_DIR.
    """
    from pathlib import Path

    data_root = tmp_path / "data"
    releases_dir = data_root / "downloads" / "releases"  # the ./data path that MUST stay empty
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    release_dates_path = intermediate / "release_dates.parquet"
    vintage_dates_path = intermediate / "vintage_dates.parquet"
    monkeypatch.setattr("nfp_lookups.paths.RELEASES_DIR", releases_dir)
    monkeypatch.setattr("nfp_lookups.paths.RELEASE_DATES_PATH", release_dates_path)
    monkeypatch.setattr("nfp_lookups.paths.VINTAGE_DATES_PATH", vintage_dates_path)
    # the scraper module binds RELEASES_DIR at import — patch its binding too, so a
    # regressed (un-threaded) download_all would write under the redirected ./data.
    monkeypatch.setattr("nfp_download.release_dates.scraper.RELEASES_DIR", releases_dir)
    monkeypatch.setattr(
        "nfp_ingest.release_dates.vintage_dates.RELEASE_DATES_PATH", release_dates_path
    )

    class _StubSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        "nfp_download.release_dates.scraper.create_session", lambda *a, **k: _StubSession()
    )

    async def _ok_index(session, url):
        return "<html><body>index</body></html>"

    monkeypatch.setattr("nfp_download.release_dates.scraper.fetch_index", _ok_index)
    # one fake entry per publication so the real download_all builds its out_dir and
    # dispatches download_one (stubbed below to record where the HTML would land).
    monkeypatch.setattr(
        "nfp_download.release_dates.scraper.parse_index_page", lambda *a, **k: [object()]
    )

    html_targets: list[Path] = []

    async def _record_download_one(session, semaphore, entry, publication_name, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "rec.htm"
        path.write_text("<html/>")
        html_targets.append(path)
        return path

    monkeypatch.setattr(
        "nfp_download.release_dates.scraper.download_one", _record_download_one
    )
    # keep the read-back/parse hermetic — the property under test is WHERE html lands.
    monkeypatch.setattr(
        "nfp_download.release_dates.parser.collect_release_dates", lambda *a, **k: []
    )

    from nfp_vintages.calendar import advance_release_calendar

    advance_release_calendar()

    assert html_targets, "download_one was never reached (the scrape path did not run)"
    # Every scraped-HTML file must live under a tempdir, NOT under the ./data tree.
    for p in html_targets:
        assert data_root not in p.parents, f"scraped HTML wrote under ./data: {p}"
    assert not list(releases_dir.rglob("*.htm")), "release HTML leaked under ./data RELEASES_DIR"


@pytest.mark.network
def test_advance_release_calendar_live(monkeypatch, tmp_path):
    """Live BLS scrape path — redirected to tmp so it never clobbers prod."""
    _, vintage_dates_path = _patch_paths(monkeypatch, tmp_path)

    from nfp_vintages.calendar import advance_release_calendar

    advance_release_calendar()

    assert vintage_dates_path.exists()
    assert pl.read_parquet(vintage_dates_path).height > 0
