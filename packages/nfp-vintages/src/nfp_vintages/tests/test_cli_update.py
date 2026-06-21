"""CLI tests for the production surface (snapshot day-12, update orchestration).

Phase 5 of specs/cli_production_workflow.md. Uses Typer's CliRunner with
deferred-import command bodies monkeypatched so no network/store/key is touched.
"""

from __future__ import annotations

from nfp_vintages.__main__ import app
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
