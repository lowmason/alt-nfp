"""§7 guardrail: month-T capture is idempotent and tie-breaks via min(vintage_date).

Characterization tests — the 10-col ukey + min-vintage compact already give these
properties. The first run pins them so a future change to the dedup rules trips here.
"""

from __future__ import annotations

import polars as pl
from nfp_ingest.vintage_store import (
    append_to_vintage_store,
    compact_partition,
    read_vintage_store,
)
from nfp_vintages.tests._fixtures import make_ces_rows

_UKEY = [
    "ref_date", "industry_type", "industry_code", "geographic_type",
    "geographic_code", "revision", "benchmark_revision", "ownership",
    "size_class_type", "size_class_code",
]


def _relation(store) -> dict:
    """Map the 10-col dedup ukey -> employment for the (ces, true) partition."""
    df = read_vintage_store(store, source="ces", seasonally_adjusted=True).collect()
    return {
        tuple(r[c] for c in _UKEY): r["employment"]
        for r in df.iter_rows(named=True)
    }


class TestIdempotence:
    def test_capture_append_compact_twice_same_relation(self, tmp_path):
        store = tmp_path / "store"
        rows = make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06")

        append_to_vintage_store(rows, store)
        compact_partition(store, "ces", True)
        first = _relation(store)

        # Second run: identical rows must add 0; compact must be a no-op.
        added = append_to_vintage_store(rows, store)
        compact_partition(store, "ces", True)
        second = _relation(store)

        assert added == 0  # re-append of identical rows is fully deduped
        assert first == second

    def test_same_ukey_later_vintage_keeps_min_vintage_level(self, tmp_path):
        store = tmp_path / "store"
        part = store / "source=ces" / "seasonally_adjusted=true"
        part.mkdir(parents=True)
        early = make_ces_rows(
            ref_month="2026-01-12", vintage="2026-02-06", employment=150_000.0
        )
        late = make_ces_rows(
            ref_month="2026-01-12", vintage="2026-03-06", employment=151_000.0
        )
        # Same 10-col ukey, different vintage_date. Write TWO fragment files
        # directly: append's anti-join would drop `late` (same ukey) before it
        # reached a second file, and compact no-ops on a single file — so a single
        # batch append never exercises the tie-break. Two files is the cross-file
        # state compact's min-vintage rule exists to resolve (§7 landmine).
        early.drop(["source", "seasonally_adjusted"]).write_parquet(part / "a.parquet")
        late.drop(["source", "seasonally_adjusted"]).write_parquet(part / "b.parquet")
        assert len(list(part.glob("*.parquet"))) == 2
        compact_partition(store, "ces", True)
        rel = _relation(store)
        # compact keeps MIN(vintage_date) per ukey → the early real-time level wins
        assert set(rel.values()) == {150_000.0}


class TestFirstPrintUnchanged:
    def test_capture_does_not_move_existing_first_prints(self, tmp_path):
        from nfp_ingest.first_print import first_print_changes
        from nfp_ingest.wedge_data import wedge_first_print_changes
        from nfp_vintages.tests._fixtures import make_first_print_window

        store = tmp_path / "store"
        make_first_print_window(store)  # two months × {00 total, 05 private}, co-released

        # Discriminating guard against a vacuous (empty-frame) pin: the wedge must
        # actually resolve at least one ref_date, else the .all() below is empty.
        wedge_before = wedge_first_print_changes(store_path=store)
        assert wedge_before.height >= 1
        fp05_before = first_print_changes(store_path=store, industry_code="05")
        assert fp05_before.filter(
            pl.col("first_print_change_k").is_not_null()
        ).height >= 1

        # A NEW, later month's capture must not move earlier months' first prints.
        append_to_vintage_store(
            make_ces_rows(
                ref_month="2026-03-12", vintage="2026-04-03",
                revision=0, employment=152_000.0, industry_code="05",
            ),
            store,
        )
        append_to_vintage_store(
            make_ces_rows(
                ref_month="2026-03-12", vintage="2026-04-03",
                revision=0, employment=303_000.0, industry_code="00",
            ),
            store,
        )
        compact_partition(store, "ces", True)

        fp05_after = first_print_changes(store_path=store, industry_code="05")
        wedge_after = wedge_first_print_changes(store_path=store)

        common = fp05_before.join(fp05_after, on="ref_date", suffix="_after", how="inner")
        assert (
            common["first_print_change_k"] == common["first_print_change_k_after"]
        ).all()

        wcommon = wedge_before.join(
            wedge_after, on="ref_date", suffix="_after", how="inner"
        )
        assert (wcommon["wedge_change_k"] == wcommon["wedge_change_k_after"]).all()


class TestCalendarNotAdvancedLoudFailure:
    def test_update_errors_when_calendar_missing_target(self, tmp_path, monkeypatch):
        from nfp_vintages.__main__ import app
        from typer.testing import CliRunner

        # advance_release_calendar is a no-op (stale/missing calendar); capture_ces_print
        # raises because the tag join is empty for T (the Phase 4 loud-failure contract).
        monkeypatch.setattr(
            "nfp_vintages.calendar.advance_release_calendar", lambda: None
        )

        def _raise(as_of, *, store_path=None):
            raise RuntimeError(
                f"no vintage calendar rows for {as_of}; advance the calendar first"
            )

        monkeypatch.setattr("nfp_ingest.capture.capture_ces_print", _raise)

        result = CliRunner().invoke(
            app,
            ["update", "--as-of", "2026-06-12", "--only", "ces",
             "--no-refresh-calendar"],
        )
        # _run_update lets the capture exception propagate → Typer non-zero exit.
        assert result.exit_code != 0
        assert "calendar" in (result.output + str(result.exception)).lower()


class TestOverlapDivergence:
    def test_overlap_diagnostic_excludes_sentinel_and_flags(self, tmp_path):
        from nfp_vintages.tests._fixtures import (
            make_shutdown_sentinel_row,
            overlap_level_divergence,
        )

        store = tmp_path / "store"
        # Bootstrap leg: a real rev0 + a -1 sentinel slot.
        append_to_vintage_store(
            make_ces_rows(ref_month="2025-11-12", vintage="2025-12-05",
                          revision=0, employment=150_800.0, industry_code="05"),
            store,
        )
        append_to_vintage_store(
            make_shutdown_sentinel_row(ref_month="2025-10-12"), store
        )
        compact_partition(store, "ces", True)
        bootstrap = read_vintage_store(
            store, source="ces", seasonally_adjusted=True
        ).collect()

        # Capture leg: the same Nov row at a *diverged* level (replaceable not identical),
        # and crucially NO -1 sentinel (the real path never emits one).
        capture = make_ces_rows(
            ref_month="2025-11-12", vintage="2025-12-05",
            revision=0, employment=151_100.0, industry_code="05",
        )

        report = overlap_level_divergence(bootstrap, capture)

        # (a) the -1 sentinel ref_date is excluded from the scored comparison
        assert -1.0 not in report["bootstrap_employment"].to_list()
        assert -1.0 not in report["capture_employment"].to_list()
        # (b) a divergence record is produced (flag, NOT asserted zero — §7.2)
        assert report.height >= 1
        assert "abs_diff" in report.columns
        assert (report["abs_diff"] >= 0).all()
