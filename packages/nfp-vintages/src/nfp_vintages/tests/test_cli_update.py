"""CLI tests for the production surface (snapshot day-12, update orchestration).

Phase 5 of specs/cli_production_workflow.md. Uses Typer's CliRunner with
deferred-import command bodies monkeypatched so no network/store/key is touched.
"""

from __future__ import annotations

from datetime import date

from nfp_ingest.vintage_store import append_to_vintage_store
from nfp_vintages.__main__ import app
from nfp_vintages.tests._fixtures import make_ces_rows
from typer.testing import CliRunner

runner = CliRunner()


def _no_real_snapshot(monkeypatch):
    """Stub snapshot_model_data so a red-phase grid loop never reads the store
    or writes a snapshot (the validation must reject BEFORE it is reached)."""
    monkeypatch.setattr(
        "nfp_ingest.snapshots.snapshot_model_data",
        lambda d: (f"/tmp/snap-{d}.npz", "deadbeefcafe"),
    )


class TestSnapshotDay12:
    def test_grid_mode_rejects_non_12th_as_of(self, monkeypatch):
        # Today this silently snapshots 2026-03-12; it must be rejected.
        _no_real_snapshot(monkeypatch)
        result = runner.invoke(
            app, ["snapshot", "--as-of", "2026-03-05", "--grid-end", "2026-06-12"]
        )
        assert result.exit_code != 0
        assert "12th" in result.output or "day-12" in result.output

    def test_single_mode_rejects_non_12th_as_of(self, monkeypatch):
        _no_real_snapshot(monkeypatch)
        result = runner.invoke(app, ["snapshot", "--as-of", "2026-03-05"])
        assert result.exit_code != 0


class TestUpdateOrchestration:
    def test_update_runs_calendar_then_ces_then_qcew_then_indicators(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar",
            lambda: calls.append("calendar"),
        )

        class _Res:
            appended, corrected, skipped = 3, [], 0

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: calls.append("ces") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter",
            lambda as_of, *, store_path=None: calls.append("qcew") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.indicators.download_indicators",
            lambda: calls.append("indicators") or {},
        )

        result = runner.invoke(app, ["update", "--as-of", "2026-06-12"])
        assert result.exit_code == 0, result.output
        assert calls == ["calendar", "ces", "qcew", "indicators"]

    def test_only_ces_skips_qcew_and_indicators(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: calls.append("calendar")
        )

        class _Res:
            appended, corrected, skipped = 1, [], 0

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: calls.append("ces") or _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.indicators.download_indicators",
            lambda: calls.append("indicators"),
        )
        result = runner.invoke(app, ["update", "--as-of", "2026-06-12", "--only", "ces"])
        assert result.exit_code == 0, result.output
        assert calls == ["calendar", "ces"]

    def test_no_refresh_calendar_skips_scrape(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: calls.append("calendar")
        )

        class _Res:
            appended, corrected, skipped = 0, [], 1

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print",
            lambda as_of, *, store_path=None: _Res(),
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter",
            lambda as_of, *, store_path=None: _Res(),
        )
        monkeypatch.setattr("nfp_ingest.indicators.download_indicators", lambda: {})
        result = runner.invoke(
            app, ["update", "--as-of", "2026-06-12", "--no-refresh-calendar"]
        )
        assert result.exit_code == 0, result.output
        assert "calendar" not in calls

    def test_invalid_only_rejected(self):
        result = runner.invoke(app, ["update", "--as-of", "2026-06-12", "--only", "bogus"])
        assert result.exit_code != 0


class TestUpdateSelfHealingCompaction:
    def test_update_compacts_pre_existing_fragments(self, tmp_path, monkeypatch):
        store = tmp_path / "store"
        # Two disjoint appends → two fragment files in the same (ces, true) partition.
        append_to_vintage_store(
            make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06"), store
        )
        append_to_vintage_store(
            make_ces_rows(ref_month="2026-02-12", vintage="2026-03-06"), store
        )
        part = store / "source=ces" / "seasonally_adjusted=true"
        assert len(list(part.glob("*.parquet"))) == 2

        # Stub everything except the heal pass; capture appends nothing.
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: None
        )

        class _Res:
            appended, corrected, skipped = 0, [], 1

        monkeypatch.setattr(
            "nfp_ingest.capture.capture_ces_print", lambda a, *, store_path=None: _Res()
        )
        monkeypatch.setattr(
            "nfp_ingest.capture.capture_qcew_quarter",
            lambda a, *, store_path=None: _Res(),
        )
        monkeypatch.setattr("nfp_ingest.indicators.download_indicators", lambda: {})

        from nfp_vintages.__main__ import _run_update

        _run_update(date(2026, 6, 12), store_path=store)
        assert len(list(part.glob("*.parquet"))) == 1


class TestUpdateOnlyQcew:
    def test_only_qcew_drives_real_noop_and_skips_ces(self, monkeypatch):
        """--only qcew runs the REAL capture_qcew_quarter no-op; CES leg gated off.

        No --store flag exists on `update`, so the capture must NOT reach the
        store. Patching get_qcew_vintage_date far-future makes
        _knowable_qcew_quarter return None ⇒ capture_qcew_quarter returns
        skipped=1 before any acquire/append/read (store-safe under pytest, where
        VINTAGE_STORE_PATH is canonical MinIO).
        """
        import nfp_ingest.capture as _cap
        import nfp_vintages.calendar as _cal

        # Offline calendar (defensive; --no-refresh-calendar also skips it).
        monkeypatch.setattr(_cal, "advance_release_calendar", lambda: None)

        # Far-future schedule ⇒ no knowable quarter ⇒ real no-op, no store touch.
        monkeypatch.setattr(
            _cap,
            "get_qcew_vintage_date",
            lambda ref_quarter, ref_year, revision: date(2099, 1, 1),
        )

        # acquire MUST NOT be reached on the no-op path.
        def _boom(*a, **k):
            raise AssertionError("acquire_qcew_levels reached on a no-op")

        monkeypatch.setattr(_cap, "acquire_qcew_levels", _boom)

        # CES leg must be gated OFF by --only qcew; flag if it runs.
        def _ces_boom(*a, **k):
            raise AssertionError("capture_ces_print ran under --only qcew")

        monkeypatch.setattr(_cap, "capture_ces_print", _ces_boom)

        result = runner.invoke(
            app,
            ["update", "--as-of", "2024-06-12", "--only", "qcew", "--no-refresh-calendar"],
        )

        # HARD assertion: the real no-op path ran to completion under --only qcew.
        assert result.exit_code == 0, result.output
        # SOFT assertion: 5.2's _run_update emits a QCEW outcome line reporting
        # the skip ("  QCEW: appended 0, skipped 1"). 5.2 owns the wording — if it
        # ever changes, reconcile THESE strings to 5.2's echo, never __main__.py.
        assert "QCEW" in result.output
        assert "skipped 1" in result.output
