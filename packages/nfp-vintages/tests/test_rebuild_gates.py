"""Unit + real-store tests for the four §10 store-rebuild acceptance gates.

Mirrors the idioms of ``packages/nfp-ingest/tests/test_store_coverage.py``:
``_check_*``-style gap collectors, ``assert not gaps, "\\n".join(gaps)``, a
``_store_available()`` self-skip probe, ``@pytest.mark.real_store`` wrappers,
and cached store loads.

The **unit** tests are the CI gate (no network, no store): per gate, a
synthetic GOOD frame must yield ``[]`` and a deliberately-BROKEN frame must
yield a gap naming the right problem.  The broken-frame tests are mandatory —
a gate that cannot fail is useless.

The **real-store** wrappers read the *rebuilt* store via ``read_vintage_store``
and run each gate; they self-skip when the store is unavailable and are
maintainer-run.  No test writes any store; no test hits the network.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from nfp_ingest.vintage_store import read_vintage_store
from nfp_lookups.industry import QCEW_SUPERSECTOR
from nfp_lookups.paths import VINTAGE_STORE_PATH
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_vintages.rebuild_gates import (
    gate_gap_fill,
    gate_history_consistency,
    gate_q1_continuity,
    gate_reconstruction_accuracy,
    gate_vintage_integrity,
)

# ---------------------------------------------------------------------------
# Synthetic-frame builders (schema-conformant so dtype bugs surface)
# ---------------------------------------------------------------------------

_SUPERSECTORS = ["10", "20", "30", "40", "50", "55", "60", "65", "70", "80"]


def _row(
    *,
    industry_type: str,
    industry_code: str,
    ownership: str,
    ref_date: date,
    employment: float,
    revision: int = 2,
    benchmark_revision: int = 0,
    vintage_date: date | None = None,
    source: str = "ces",
    seasonally_adjusted: bool = True,
    size_class_type: str | None = None,
    size_class_code: str | None = None,
    geographic_type: str = "national",
    geographic_code: str = "00",
) -> dict:
    """One VINTAGE_STORE_SCHEMA-conformant row as a dict."""
    return {
        "geographic_type": geographic_type,
        "geographic_code": geographic_code,
        "ownership": ownership,
        "industry_type": industry_type,
        "industry_code": industry_code,
        "ref_date": ref_date,
        "vintage_date": vintage_date or ref_date,
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "size_class_type": size_class_type,
        "size_class_code": size_class_code,
        "source": source,
        "seasonally_adjusted": seasonally_adjusted,
    }


def _frame(rows: list[dict]) -> pl.DataFrame:
    """Build a store-schema frame from row dicts (enforces dtypes)."""
    return pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)


def _empty_store_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=VINTAGE_STORE_SCHEMA)


# ===========================================================================
# Gate 1 — History consistency
# ===========================================================================


def _legacy_total_private_rows(ref: date, emp: float) -> list[dict]:
    """Legacy-axis (domain/05, null ownership) rows for the four cohorts."""
    return [
        _row(
            industry_type="domain", industry_code="05", ownership=None,
            ref_date=ref, employment=emp, revision=r, benchmark_revision=b,
            vintage_date=date(2024, 2, 1),
        )
        for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
    ]


def _rebuilt_total_private_rows(ref: date, emp: float) -> list[dict]:
    """Rebuilt-axis (total/05, private) rows for the four cohorts."""
    return [
        _row(
            industry_type="total", industry_code="05", ownership="private",
            ref_date=ref, employment=emp, revision=r, benchmark_revision=b,
            vintage_date=date(2024, 2, 1),
        )
        for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
    ]


def _rebuilt_anchor_rows(ref: date, emp: float) -> list[dict]:
    """Rebuilt-axis (total/00, total) anchor rows for the four cohorts."""
    return [
        _row(
            industry_type="total", industry_code="00", ownership="total",
            ref_date=ref, employment=emp, revision=r, benchmark_revision=b,
            vintage_date=date(2024, 2, 1),
        )
        for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
    ]


def _legacy_anchor_rows(ref: date, emp: float) -> list[dict]:
    """Legacy-axis (national/00, null ownership) anchor rows for four cohorts."""
    return [
        _row(
            industry_type="national", industry_code="00", ownership=None,
            ref_date=ref, employment=emp, revision=r, benchmark_revision=b,
            vintage_date=date(2024, 2, 1),
        )
        for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
    ]


class TestGateHistoryConsistency:
    """Rebuilt CES reproduces the existing store on the ≤2023 core."""

    _REF = date(2023, 6, 12)

    def _good_pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        rebuilt = _frame(
            _rebuilt_total_private_rows(self._REF, 130_000.0)
            + _rebuilt_anchor_rows(self._REF, 158_000.0)
        )
        existing = _frame(
            _legacy_total_private_rows(self._REF, 130_000.0)
            + _legacy_anchor_rows(self._REF, 158_000.0)
        )
        return rebuilt, existing

    def test_good_frame_passes(self) -> None:
        rebuilt, existing = self._good_pair()
        assert gate_history_consistency(rebuilt, existing) == []

    def test_extra_2_1_benchmark_cohorts_still_pass(self) -> None:
        """Rebuilt has EXTRA per-benchmark (2,1) cohorts the old store lacks.

        The (2,1) comparison is cohort-aligned on ``vintage_date``: rebuilt (2,1)
        cohorts the existing store never held drop out of the join and are not
        flagged.  Here the existing (2,1) is at vintage 2024-02-01 (130_000); the
        rebuilt carries that same cohort plus an older (129_111) and a newer one
        the old store lacks — only the matching cohort is compared.
        """
        rebuilt, existing = self._good_pair()
        # Add an OLDER (2,1) cohort with a stale value + an even newer matching one.
        extra = _frame(
            [
                _row(
                    industry_type="total", industry_code="05", ownership="private",
                    ref_date=self._REF, employment=129_111.0, revision=2,
                    benchmark_revision=1, vintage_date=date(2023, 2, 1),
                ),
                _row(
                    industry_type="total", industry_code="05", ownership="private",
                    ref_date=self._REF, employment=130_000.0, revision=2,
                    benchmark_revision=1, vintage_date=date(2025, 2, 1),
                ),
            ]
        )
        rebuilt = pl.concat([rebuilt, extra])
        assert gate_history_consistency(rebuilt, existing) == []

    def test_newer_2_1_benchmark_revised_history_not_hard_divergence(self) -> None:
        """A strictly-newer (2,1) that revised a ≤2023 level must NOT flag hard.

        Annual benchmarks revise prior history.  The rebuilt captured a 2025-02
        (2,1) that re-stated this ≤2023 ref-month to 130_250, alongside the older
        2024-02 (2,1) at 130_000.  The existing store holds ONLY the 2024-02
        (2,1) at 130_000.  A "latest-on-each-side" comparison would diff
        rebuilt's 130_250 against the old 130_000 and report a spurious hard
        divergence; cohort-alignment on vintage_date compares like-for-like and
        the newer benchmark simply has no counterpart to diverge from.
        """
        rebuilt, existing = self._good_pair()
        # Replace the rebuilt 05 (2,1) row (good_pair stamps vintage 2024-02-01,
        # value 130_000) with TWO cohorts: the matching 2024-02 + a newer revised
        # 2025-02 at a different level.
        rebuilt = rebuilt.filter(
            ~(
                (pl.col("industry_code") == "05")
                & (pl.col("revision") == 2)
                & (pl.col("benchmark_revision") == 1)
            )
        )
        rebuilt = pl.concat(
            [
                rebuilt,
                _frame(
                    [
                        _row(
                            industry_type="total", industry_code="05",
                            ownership="private", ref_date=self._REF,
                            employment=130_000.0, revision=2, benchmark_revision=1,
                            vintage_date=date(2024, 2, 1),
                        ),
                        _row(
                            industry_type="total", industry_code="05",
                            ownership="private", ref_date=self._REF,
                            employment=130_250.0, revision=2, benchmark_revision=1,
                            vintage_date=date(2025, 2, 1),
                        ),
                    ]
                ),
            ]
        )
        # Existing store keeps only the 2024-02 (2,1) at 130_000 (from _good_pair).
        assert gate_history_consistency(rebuilt, existing) == []

    def test_mismatched_value_fails(self) -> None:
        rebuilt, existing = self._good_pair()
        # Corrupt one rebuilt employment value well beyond tolerance.
        rebuilt = rebuilt.with_columns(
            pl.when(
                (pl.col("industry_code") == "05") & (pl.col("revision") == 0)
            )
            .then(pl.lit(99_999.0))
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_history_consistency(rebuilt, existing)
        assert gaps
        assert any("diverge" in g for g in gaps)

    def test_2_1_divergence_on_shared_cohort_fails(self) -> None:
        """The (2,1) comparison still FIRES on a shared-cohort divergence.

        Both stores carry the 05 (2,1) at vintage 2024-02-01 = 130_000.  Corrupt
        the rebuilt (2,1) value: because the cohort-aligned join matches on
        vintage_date, the shared cohort still overlaps and the divergence must
        surface (the (2,1) branch is not vacuous).
        """
        rebuilt, existing = self._good_pair()
        rebuilt = rebuilt.with_columns(
            pl.when(
                (pl.col("industry_code") == "05")
                & (pl.col("revision") == 2)
                & (pl.col("benchmark_revision") == 1)
            )
            .then(pl.lit(99_999.0))
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_history_consistency(rebuilt, existing)
        assert gaps
        assert any("diverge" in g for g in gaps)

    def test_small_realistic_drift_fails(self) -> None:
        """A few-thousand-job reconstruction drift must FAIL (not slip as rounding).

        The history gate is the parity replacement; its tolerance must catch a
        realistic summation-order / partial-aggregation error of a few thousand
        jobs, not only sledgehammer corruptions.  Corrupt the (2,0) cohort of 05
        by +2_000 (thousand) — far outside rounding, well inside the ~13_500 slack
        the old rel_tol=1e-4 admitted.  With rel_tol=1e-6 it must flag.
        """
        rebuilt, existing = self._good_pair()
        rebuilt = rebuilt.with_columns(
            pl.when(
                (pl.col("industry_code") == "05")
                & (pl.col("revision") == 2)
                & (pl.col("benchmark_revision") == 0)
            )
            .then(pl.col("employment") + 2_000.0)
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_history_consistency(rebuilt, existing)
        assert gaps
        assert any("diverge" in g for g in gaps)

    def test_anchor_value_divergence_fails(self) -> None:
        """The total/00 anchor value-match path (spec §10 named requirement).

        Corrupt the rebuilt total/00 (2,0) anchor beyond tolerance; the gate must
        report a divergence naming total/00 — not silently pass it.
        """
        rebuilt, existing = self._good_pair()
        rebuilt = rebuilt.with_columns(
            pl.when(
                (pl.col("industry_code") == "00")
                & (pl.col("revision") == 2)
                & (pl.col("benchmark_revision") == 0)
            )
            .then(pl.col("employment") + 3_000.0)
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_history_consistency(rebuilt, existing)
        assert gaps
        assert any("diverge" in g and "total/00" in g for g in gaps)

    def test_missing_rev_bmr_cohort_fails(self) -> None:
        """Dropping the (2,1) cohort from the rebuilt anchor must flag."""
        rebuilt, existing = self._good_pair()
        rebuilt = rebuilt.filter(
            ~(
                (pl.col("industry_code") == "00")
                & (pl.col("revision") == 2)
                & (pl.col("benchmark_revision") == 1)
            )
        )
        gaps = gate_history_consistency(rebuilt, existing)
        assert gaps
        assert any("cohorts" in g and "00" in g for g in gaps)

    def test_code_55_collision_disambiguated(self) -> None:
        """Supersector 55 and sector 55 are distinct series; the key keeps them so.

        A mismatch on sector-55 must NOT be masked by a correct supersector-55
        (and vice versa) — the gate keys on industry_type.
        """
        ref = self._REF
        rebuilt = _frame(
            _rebuilt_total_private_rows(ref, 130_000.0)
            + _rebuilt_anchor_rows(ref, 158_000.0)
            + [
                _row(industry_type="supersector", industry_code="55",
                     ownership="private", ref_date=ref, employment=9_500.0),
                _row(industry_type="sector", industry_code="55",
                     ownership="private", ref_date=ref, employment=2_400.0),
            ]
        )
        existing = _frame(
            _legacy_total_private_rows(ref, 130_000.0)
            + _legacy_anchor_rows(ref, 158_000.0)
            + [
                # supersector 55 matches; sector 55 DIVERGES.
                _row(industry_type="supersector", industry_code="55",
                     ownership=None, ref_date=ref, employment=9_500.0),
                _row(industry_type="sector", industry_code="55",
                     ownership=None, ref_date=ref, employment=8_888.0),
            ]
        )
        gaps = gate_history_consistency(rebuilt, existing)
        assert gaps
        # The sector-55 mismatch surfaces; supersector-55 did not mask it.
        assert any("sector/55" in g for g in gaps)

    def test_ref_dates_at_or_after_cutoff_ignored(self) -> None:
        """Rows on/after the cutoff are outside the known-good core."""
        rebuilt, existing = self._good_pair()
        late_rebuilt = _frame(
            _rebuilt_total_private_rows(date(2024, 6, 12), 999.0)
            + _rebuilt_anchor_rows(date(2024, 6, 12), 999.0)
        )
        late_existing = _frame(
            _legacy_total_private_rows(date(2024, 6, 12), 111.0)
            + _legacy_anchor_rows(date(2024, 6, 12), 111.0)
        )
        rebuilt = pl.concat([rebuilt, late_rebuilt])
        existing = pl.concat([existing, late_existing])
        # The post-cutoff divergence is excluded by the cutoff filter.
        assert gate_history_consistency(rebuilt, existing) == []


# ===========================================================================
# Gate 2 — Gap fill
# ===========================================================================


class TestGateGapFill:
    _FRONTIER = date(2026, 1, 12)

    def _good_frame(self) -> pl.DataFrame:
        rows: list[dict] = []
        # Frontier currency: 05 + 10 supersectors at the frontier ref_date.
        rows.append(
            _row(industry_type="total", industry_code="05", ownership="private",
                 ref_date=self._FRONTIER, employment=130_000.0)
        )
        for c in _SUPERSECTORS:
            rows.append(
                _row(industry_type="supersector", industry_code=c,
                     ownership="private", ref_date=self._FRONTIER,
                     employment=13_000.0)
            )
        # December (2,1) cohorts for 2024 + 2025, complete for the frontier set.
        for year in (2024, 2025):
            dec = date(year, 12, 12)
            rows.append(
                _row(industry_type="total", industry_code="05",
                     ownership="private", ref_date=dec, employment=129_000.0,
                     revision=2, benchmark_revision=1)
            )
            for c in _SUPERSECTORS:
                rows.append(
                    _row(industry_type="supersector", industry_code=c,
                         ownership="private", ref_date=dec, employment=12_900.0,
                         revision=2, benchmark_revision=1)
                )
        return _frame(rows)

    def test_good_frame_passes(self) -> None:
        gaps = gate_gap_fill(self._good_frame(), frontier_ref_date=self._FRONTIER)
        assert gaps == []

    def test_missing_supersector_at_frontier_fails(self) -> None:
        df = self._good_frame().filter(
            ~(
                (pl.col("industry_code") == "70")
                & (pl.col("ref_date") == self._FRONTIER)
            )
        )
        gaps = gate_gap_fill(df, frontier_ref_date=self._FRONTIER)
        assert gaps
        assert any("frontier" in g and "70" in g for g in gaps)

    def test_incomplete_december_cohort_fails(self) -> None:
        df = self._good_frame().filter(
            ~(
                (pl.col("industry_code") == "40")
                & (pl.col("ref_date") == date(2025, 12, 12))
            )
        )
        gaps = gate_gap_fill(df, frontier_ref_date=self._FRONTIER)
        assert gaps
        assert any("December 2025" in g for g in gaps)

    def test_soft_nesting_violation_prefixed_not_hard(self) -> None:
        """A broken 05 = 06 + 08 identity is reported as SOFT, hard gates pass."""
        df = self._good_frame()
        # Add domain rows that DON'T sum: 05 already 130k at frontier; make
        # 06 + 08 = 100k there.
        extra = _frame(
            [
                _row(industry_type="domain", industry_code="06",
                     ownership="private", ref_date=self._FRONTIER,
                     employment=40_000.0),
                _row(industry_type="domain", industry_code="08",
                     ownership="private", ref_date=self._FRONTIER,
                     employment=60_000.0),
            ]
        )
        df = pl.concat([df, extra])
        gaps = gate_gap_fill(df, frontier_ref_date=self._FRONTIER)
        # Only soft findings — no hard gate fired.
        assert gaps
        assert all(g.startswith("SOFT:") for g in gaps)
        assert any("05 != 06+08" in g for g in gaps)

    def test_missing_sector_month_does_not_block(self) -> None:
        """A nesting group missing a component is skipped, not flagged.

        ``08`` is absent at the frontier, so the ``05 == 06 + 08`` group is
        skipped (a missing component never blocks).  ``06`` is set consistent
        with the goods supersectors (10+20+30 = 39,000) so the *other* soft
        identity (``06 == sum(supersectors)``) genuinely holds — isolating the
        skip behaviour under test.
        """
        df = self._good_frame()
        extra = _frame(
            [
                _row(industry_type="domain", industry_code="06",
                     ownership="private", ref_date=self._FRONTIER,
                     employment=39_000.0),
            ]
        )
        df = pl.concat([df, extra])
        gaps = gate_gap_fill(df, frontier_ref_date=self._FRONTIER)
        assert gaps == []

    def test_q1_size_buckets_do_not_double_count_nesting(self) -> None:
        """Size-bucket rows must be filtered before the nesting sum.

        06 carries a Q1 size cross-product (total/'0' + buckets).  If the gate
        summed buckets too, 06 would be inflated and 05=06+08 would falsely
        fail.  With the all-sizes filter, the identity holds.
        """
        q1 = date(2025, 3, 12)
        rows = [
            _row(industry_type="total", industry_code="05", ownership="private",
                 ref_date=q1, employment=100_000.0, source="qcew"),
            _row(industry_type="domain", industry_code="06", ownership="private",
                 ref_date=q1, employment=40_000.0, source="qcew",
                 size_class_type="size_class", size_class_code="0"),
            # Buckets summing to the same 40k — must be EXCLUDED by the filter.
            _row(industry_type="domain", industry_code="06", ownership="private",
                 ref_date=q1, employment=25_000.0, source="qcew",
                 size_class_type="size_class", size_class_code="1"),
            _row(industry_type="domain", industry_code="06", ownership="private",
                 ref_date=q1, employment=15_000.0, source="qcew",
                 size_class_type="size_class", size_class_code="2"),
            _row(industry_type="domain", industry_code="08", ownership="private",
                 ref_date=q1, employment=60_000.0, source="qcew"),
        ]
        df = _frame(rows)
        gaps = gate_gap_fill(
            df, frontier_ref_date=q1, dec_cohort_years=()
        )
        # Frontier hard gate is incomplete here (only 05 present at q1, no
        # supersectors), so we only assert the NESTING soft check passed.
        assert not any("nesting" in g for g in gaps)

    def test_sectors_sum_to_supersector_passes_and_flags(self) -> None:
        """The third §10 identity: each supersector == sum(its stored sectors).

        Build supersector 30 (Manufacturing) with its two stored sectors 31
        (durable) + 32 (nondurable) summing exactly to it — must NOT flag — then
        corrupt 32 so the sum diverges — must flag as SOFT 'supersector 30 !=
        sum(sectors)'.  This pins QCEW_SUPERSECTOR (30 -> {31,32}) as the parent
        map; get_supersector_components() would mis-map 30 -> {31} only.
        """
        # Guard the assumption the test data encodes: supersector 30's stored
        # sectors are exactly the durable/nondurable split {31, 32}.
        assert {str(s) for s in QCEW_SUPERSECTOR["30"]["sectors"]} == {"31", "32"}
        # Build on a complete-frontier good frame so the HARD gates pass and the
        # sectors-sum identity is isolated.  Supersector 30 is 13_000 at the
        # frontier; sectors 31 + 32 here sum to exactly 13_000 (consistent).
        good = self._good_frame()
        consistent = _frame(
            [
                _row(industry_type="sector", industry_code="31",
                     ownership="private", ref_date=self._FRONTIER,
                     employment=8_000.0, source="qcew"),
                _row(industry_type="sector", industry_code="32",
                     ownership="private", ref_date=self._FRONTIER,
                     employment=5_000.0, source="qcew"),
            ]
        )
        gaps = gate_gap_fill(
            pl.concat([good, consistent]), frontier_ref_date=self._FRONTIER
        )
        assert not any("sum(sectors)" in g for g in gaps)

        # Now corrupt 32 so 31 + 32 != supersector 30 — a SOFT-only finding.
        broken = _frame(
            [
                _row(industry_type="sector", industry_code="31",
                     ownership="private", ref_date=self._FRONTIER,
                     employment=8_000.0, source="qcew"),
                _row(industry_type="sector", industry_code="32",
                     ownership="private", ref_date=self._FRONTIER,
                     employment=9_999.0, source="qcew"),
            ]
        )
        gaps = gate_gap_fill(
            pl.concat([good, broken]), frontier_ref_date=self._FRONTIER
        )
        assert gaps
        assert all(g.startswith("SOFT:") for g in gaps)
        assert any("supersector 30 != sum(sectors)" in g for g in gaps)

    def test_missing_sector_does_not_block_sectors_sum(self) -> None:
        """A supersector with only some of its stored sectors is skipped.

        Supersector 60 stores sectors {54, 55, 56}; supply only 54 + 55 (56
        absent) at a deliberately non-summing parent.  The incomplete group is
        skipped — no 'supersector 60 != sum(sectors)' gap.
        """
        ref = date(2025, 6, 12)
        df = _frame(
            [
                _row(industry_type="supersector", industry_code="60",
                     ownership="private", ref_date=ref, employment=20_000.0,
                     source="qcew"),
                _row(industry_type="sector", industry_code="54",
                     ownership="private", ref_date=ref, employment=8_000.0,
                     source="qcew"),
                _row(industry_type="sector", industry_code="55",
                     ownership="private", ref_date=ref, employment=3_000.0,
                     source="qcew"),
            ]
        )
        gaps = gate_gap_fill(df, frontier_ref_date=ref, dec_cohort_years=())
        assert not any("sum(sectors)" in g for g in gaps)

    def test_2_1_fan_out_not_summed_across_vintages(self) -> None:
        """The (2,1) per-benchmark fan-out is collapsed before nesting sums.

        A December (2,1) ref-month where 05 carries TWO (2,1) benchmark vintages
        but 06/08 carry ONE each, with EACH individual vintage satisfying
        05 = 06 + 08.  Summing across vintages would inflate 05 to 2x and emit a
        spurious 'SOFT: 05 != 06+08'.  Collapsing the fan-out (latest vintage per
        cohort) yields no nesting gap.
        """
        dec = date(2024, 12, 12)
        rows = [
            # 05: two (2,1) benchmark vintages, each = 100_000 (latest wins).
            _row(industry_type="total", industry_code="05", ownership="private",
                 ref_date=dec, employment=100_000.0, revision=2,
                 benchmark_revision=1, vintage_date=date(2025, 2, 1),
                 source="qcew"),
            _row(industry_type="total", industry_code="05", ownership="private",
                 ref_date=dec, employment=100_000.0, revision=2,
                 benchmark_revision=1, vintage_date=date(2026, 2, 1),
                 source="qcew"),
            # 06 + 08 = 100_000 (one vintage each).
            _row(industry_type="domain", industry_code="06", ownership="private",
                 ref_date=dec, employment=40_000.0, revision=2,
                 benchmark_revision=1, vintage_date=date(2026, 2, 1),
                 source="qcew"),
            _row(industry_type="domain", industry_code="08", ownership="private",
                 ref_date=dec, employment=60_000.0, revision=2,
                 benchmark_revision=1, vintage_date=date(2026, 2, 1),
                 source="qcew"),
        ]
        df = _frame(rows)
        gaps = gate_gap_fill(df, frontier_ref_date=dec, dec_cohort_years=())
        assert not any("05 != 06+08" in g for g in gaps)


# ===========================================================================
# Gate 3 — Reconstruction accuracy + Q1 continuity
# ===========================================================================


class TestGateReconstructionAccuracy:
    _REF = date(2023, 3, 12)  # a March benchmark month

    def _good_pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        # QCEW slightly ABOVE CES (8131 inclusion) for each residual code.
        qcew_rows = [
            _row(industry_type="sector", industry_code="81", ownership="private",
                 ref_date=self._REF, employment=6_000.0, source="qcew"),
            _row(industry_type="supersector", industry_code="80",
                 ownership="private", ref_date=self._REF, employment=6_050.0,
                 source="qcew"),
            _row(industry_type="domain", industry_code="08", ownership="private",
                 ref_date=self._REF, employment=106_000.0, source="qcew"),
            _row(industry_type="total", industry_code="05", ownership="private",
                 ref_date=self._REF, employment=131_000.0, source="qcew"),
        ]
        ces_rows = [
            _row(industry_type="sector", industry_code="81", ownership="private",
                 ref_date=self._REF, employment=5_950.0, source="ces"),
            _row(industry_type="supersector", industry_code="80",
                 ownership="private", ref_date=self._REF, employment=6_000.0,
                 source="ces"),
            _row(industry_type="domain", industry_code="08", ownership="private",
                 ref_date=self._REF, employment=105_500.0, source="ces"),
            _row(industry_type="total", industry_code="05", ownership="private",
                 ref_date=self._REF, employment=130_500.0, source="ces"),
        ]
        return _frame(qcew_rows), _frame(ces_rows)

    def test_good_frame_passes(self) -> None:
        qcew, ces = self._good_pair()
        assert gate_reconstruction_accuracy(qcew, ces) == []

    def test_multiple_rev_bmr_cohorts_not_summed(self) -> None:
        """Multiple (rev,bmr) cohorts per series-month must NOT be summed.

        CES carries up to five all-sizes cohorts per (series, ref_date); a sum
        would inflate the level ~5x and produce a spurious huge residual.  The
        gate must compare the best-available single cohort per side.
        """
        qcew, ces = self._good_pair()
        # Replace the single 05 CES row with all four cohorts at ~the same level.
        ces = ces.filter(pl.col("industry_code") != "05")
        ces = pl.concat(
            [
                ces,
                _frame(
                    [
                        _row(industry_type="total", industry_code="05",
                             ownership="private", ref_date=self._REF,
                             employment=130_500.0 + d, revision=r,
                             benchmark_revision=b,
                             vintage_date=date(2024, 2, 1))
                        for (r, b, d) in [
                            (0, 0, 4.0), (1, 0, 2.0), (2, 0, 1.0), (2, 1, 0.0)
                        ]
                    ]
                ),
            ]
        )
        # QCEW 05 stays at 131_000; best-available CES 05 = the (2,1) row 130_500.
        # Residual = +500 (small, non-negative) — passes. A sum would be ~522k.
        assert gate_reconstruction_accuracy(qcew, ces) == []

    def test_negative_residual_fails_hard(self) -> None:
        """QCEW below CES contradicts the 8131 direction — a hard failure."""
        qcew, ces = self._good_pair()
        qcew = qcew.with_columns(
            pl.when(pl.col("industry_code") == "81")
            .then(pl.lit(5_000.0))  # below CES's 5,950
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_reconstruction_accuracy(qcew, ces)
        assert gaps
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("NEGATIVE" in g for g in hard)

    def test_large_positive_residual_is_soft(self) -> None:
        """A residual over the relative bound is reported SOFT, not hard."""
        qcew, ces = self._good_pair()
        qcew = qcew.with_columns(
            pl.when(pl.col("industry_code") == "81")
            .then(pl.lit(9_000.0))  # +51% over CES 5,950 — exceeds 10%
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_reconstruction_accuracy(qcew, ces)
        assert gaps
        assert all(g.startswith("SOFT:") for g in gaps)
        assert any("relative bound" in g for g in gaps)

    def test_size_buckets_excluded_from_residual(self) -> None:
        """Q1 size buckets on the QCEW side must not inflate the residual."""
        qcew, ces = self._good_pair()
        bucket = _frame(
            [
                _row(industry_type="sector", industry_code="81",
                     ownership="private", ref_date=self._REF, employment=3_000.0,
                     source="qcew", size_class_type="size_class",
                     size_class_code="1"),
            ]
        )
        qcew = pl.concat([qcew, bucket])
        # Without the all-sizes filter, 81 would be 6,000+3,000 = 9,000 (soft
        # bound breach). With it, the bucket is dropped and the good frame passes.
        assert gate_reconstruction_accuracy(qcew, ces) == []


class TestGateQ1Continuity:
    """Temporal Q1 continuity for suppression-free supersectors."""

    def _series(self, q1_emp: float, code: str = "20") -> pl.DataFrame:
        """A clean supersector with monthly levels around a Q1 month."""
        rows = [
            # Dec (Q4, null-size area level)
            _row(industry_type="supersector", industry_code=code,
                 ownership="private", ref_date=date(2024, 12, 12),
                 employment=8_000.0, source="qcew"),
            # Jan/Feb/Mar (Q1, size total/'0' all-sizes level)
            _row(industry_type="supersector", industry_code=code,
                 ownership="private", ref_date=date(2025, 1, 12),
                 employment=q1_emp, source="qcew",
                 size_class_type="size_class", size_class_code="0"),
            # Apr (Q2, null-size area level)
            _row(industry_type="supersector", industry_code=code,
                 ownership="private", ref_date=date(2025, 4, 12),
                 employment=8_040.0, source="qcew"),
        ]
        return _frame(rows)

    def test_continuous_q1_passes(self) -> None:
        # Q1 ~ mean(8000, 8040) = 8020.
        assert gate_q1_continuity(self._series(8_020.0)) == []

    def test_discontinuous_q1_flagged_soft(self) -> None:
        gaps = gate_q1_continuity(self._series(5_000.0))
        assert gaps
        assert all(g.startswith("SOFT:") for g in gaps)
        assert any("q1_continuity" in g for g in gaps)

    def test_runs_for_supersector_outside_old_triple(self) -> None:
        """The default set covers all 10 supersectors, incl. 65 (not in old triple).

        Pre-fix the default was ('20','30','55'); 65 was never validated.  A Q1
        discontinuity in 65 must now be caught.
        """
        gaps = gate_q1_continuity(self._series(5_000.0, code="65"))
        assert gaps
        assert any("q1_continuity" in g and "/65" in g for g in gaps)
        # And a continuous 65 series passes.
        assert gate_q1_continuity(self._series(8_020.0, code="65")) == []

    def test_gapped_neighbour_skipped(self) -> None:
        """A non-adjacent (gapped) neighbour is not used as the interpolant.

        Nov present, Dec missing, Jan(Q1) present: the nearest prior non-Q1 month
        is 2 calendar months away, so it is not a valid neighbour.  With Apr also
        absent, the Q1 month has no adjacent non-Q1 neighbour and is skipped —
        even though its level is wildly off the gapped Nov level.
        """
        rows = [
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=date(2024, 11, 12),
                 employment=8_000.0, source="qcew"),
            # Dec absent (gap).
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=date(2025, 1, 12),
                 employment=2_000.0, source="qcew",
                 size_class_type="size_class", size_class_code="0"),
            # Feb/Mar/Apr absent — Jan has no adjacent non-Q1 neighbour.
        ]
        assert gate_q1_continuity(_frame(rows)) == []

    def test_isolated_q1_skipped(self) -> None:
        """A Q1 month with no non-Q1 neighbours is skipped (store edge)."""
        rows = [
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=date(2025, 1, 12),
                 employment=5_000.0, source="qcew",
                 size_class_type="size_class", size_class_code="0"),
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=date(2025, 2, 12),
                 employment=5_000.0, source="qcew",
                 size_class_type="size_class", size_class_code="0"),
        ]
        assert gate_q1_continuity(_frame(rows)) == []


# ===========================================================================
# Gate 4 — Vintage integrity
# ===========================================================================


class TestGateVintageIntegrity:
    def _good_slice(self) -> pl.DataFrame:
        rows = []
        v = date(2025, 2, 1)
        for i, ref in enumerate(
            [date(2024, m, 12) for m in (9, 10, 11, 12)]
        ):
            rows.append(
                _row(industry_type="supersector", industry_code="20",
                     ownership="private", ref_date=ref,
                     employment=8_000.0 + i, vintage_date=v,
                     revision=2 if i < 2 else (1 if i == 2 else 0))
            )
        return _frame(rows)

    def test_good_slice_passes(self) -> None:
        assert gate_vintage_integrity(self._good_slice()) == []

    def test_duplicate_series_month_fails(self) -> None:
        df = self._good_slice()
        dup = _frame(
            [
                _row(industry_type="supersector", industry_code="20",
                     ownership="private", ref_date=date(2024, 12, 12),
                     employment=8_003.0, vintage_date=date(2025, 2, 1),
                     revision=0),
            ]
        )
        df = pl.concat([df, dup])
        gaps = gate_vintage_integrity(df)
        assert gaps
        assert any("duplicate" in g for g in gaps)

    def test_cross_vintage_sum_fails(self) -> None:
        """Two distinct vintage_dates on one (series, ref_date) is a fail."""
        rows = [
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=date(2024, 11, 12),
                 employment=8_000.0, vintage_date=date(2025, 1, 1), revision=0),
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=date(2024, 11, 12),
                 employment=8_000.0, vintage_date=date(2025, 2, 1), revision=1),
        ]
        gaps = gate_vintage_integrity(_frame(rows))
        assert gaps
        assert any("vintage" in g for g in gaps)

    def test_null_or_zero_employment_fails(self) -> None:
        df = self._good_slice()
        bad = _frame(
            [
                _row(industry_type="supersector", industry_code="30",
                     ownership="private", ref_date=date(2024, 12, 12),
                     employment=0.0),
            ]
        )
        df = pl.concat([df, bad])
        gaps = gate_vintage_integrity(df)
        assert gaps
        assert any("null/zero" in g for g in gaps)

    def test_q1_size_buckets_not_treated_as_duplicates(self) -> None:
        """Q1 size-bucket rows share (series, ref_date) but differ in size dims.

        With the size dimension in the series key, they are distinct, not dups.
        """
        q1 = date(2025, 3, 12)
        v = date(2025, 5, 1)
        rows = [
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=q1, employment=8_000.0,
                 vintage_date=v, source="qcew",
                 size_class_type="size_class", size_class_code="0"),
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=q1, employment=5_000.0,
                 vintage_date=v, source="qcew",
                 size_class_type="size_class", size_class_code="1"),
            _row(industry_type="supersector", industry_code="20",
                 ownership="private", ref_date=q1, employment=3_000.0,
                 vintage_date=v, source="qcew",
                 size_class_type="size_class", size_class_code="2"),
        ]
        assert gate_vintage_integrity(_frame(rows)) == []


# ===========================================================================
# Real-store wrappers (maintainer-run; self-skip when unavailable)
# ===========================================================================


def _store_available() -> bool:
    """True if the vintage store (local dir or S3) is reachable and non-empty."""
    try:
        return VINTAGE_STORE_PATH.exists() and (
            next(VINTAGE_STORE_PATH.glob("**/*.parquet"), None) is not None
        )
    except Exception:  # unreachable endpoint / bad credentials → skip, not error
        return False


_REBUILT_CACHE: dict[tuple[str, bool], pl.DataFrame] = {}


def _load_rebuilt(source: str) -> pl.DataFrame:
    """Load the NSA partition of the rebuilt store (the rebuild is NSA, §7).

    The maintainer points ``NFP_STORE_URI`` at the scratch rebuild prefix
    (``s3://alt-nfp/store-rebuild``) before running these.
    """
    key = (source, False)
    if key not in _REBUILT_CACHE:
        _REBUILT_CACHE[key] = read_vintage_store(
            source=source, seasonally_adjusted=False
        ).collect()
    return _REBUILT_CACHE[key]


def _is_rebuilt_schema(df: pl.DataFrame) -> bool:
    """True if *df* carries the rebuilt taxonomy, not the legacy axes.

    The rebuilt store uses ``industry_type`` in {total, domain, supersector,
    sector} with a populated ``ownership`` axis; the legacy store uses
    ``'national'`` for the ``00`` total and leaves ``ownership`` null.  These
    gates are defined for the rebuilt store, so a legacy store must SKIP — not
    fail — these wrappers.
    """
    if df.is_empty():
        return False
    itypes = set(df["industry_type"].drop_nulls().unique().to_list())
    has_rebuilt_types = "total" in itypes and "national" not in itypes
    has_ownership = df["ownership"].drop_nulls().len() > 0
    return has_rebuilt_types and has_ownership


@pytest.mark.real_store
@pytest.mark.skipif(
    not _store_available(),
    reason="Vintage store not available (no local data/store/ and no reachable NFP_STORE_URI)",
)
class TestGatesAgainstRealStore:
    """Run each gate against the rebuilt store (maintainer-run).

    These read-only wrappers exercise the gates on real data.  They:

    * read the **NSA** partitions (the rebuild is NSA, §7);
    * self-skip when the available store is the **legacy** schema (these gates
      are defined for the rebuilt taxonomy);
    * Gate 1 needs BOTH the rebuilt and the existing store — configuring two
      store paths is a maintainer concern (see module deferred-notes), so it is
      left to the maintainer's dual-store harness and not exercised here.

    The reconstruction wrapper hard-asserts the **sign** (non-negative
    residual): spec §10 defers only the residual *magnitude* tolerance to the
    first real run — a negative residual is an explicit, non-deferred gate
    FAILURE.  So a broken QCEW reconstruction surfaces RED (blocking promotion),
    never as a benign skip.  The over-``rel_tol`` magnitude breach is SOFT and
    does not fail.
    """

    def test_reconstruction_accuracy_real(self) -> None:
        qcew = _load_rebuilt("qcew")
        ces = _load_rebuilt("ces")
        if not (_is_rebuilt_schema(qcew) and _is_rebuilt_schema(ces)):
            pytest.skip("rebuilt-schema NSA store not available")
        # Restrict to benchmark months (March) / annual averages per §10.
        qcew_bm = qcew.filter(pl.col("ref_date").dt.month() == 3)
        ces_bm = ces.filter(pl.col("ref_date").dt.month() == 3)
        gaps = gate_reconstruction_accuracy(qcew_bm, ces_bm)
        # Sign (non-negative) is hard per §10; magnitude (SOFT) is deferred.
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)

    def test_q1_continuity_real(self) -> None:
        qcew = _load_rebuilt("qcew")
        if not _is_rebuilt_schema(qcew):
            pytest.skip("rebuilt-schema NSA QCEW partition not available")
        # gate_q1_continuity is diagnostic-only — it emits SOFT findings (the
        # Q1/area carry-over is reported, not promotion-blocking, per T6).  This
        # wrapper exercises it against real data for crash-safety; the SOFT
        # findings are surfaced for the maintainer, not asserted away.
        gaps = gate_q1_continuity(qcew)
        assert all(g.startswith("SOFT:") for g in gaps), "\n".join(gaps)

    def test_vintage_integrity_real(self) -> None:
        ces = _load_rebuilt("ces")
        if not _is_rebuilt_schema(ces):
            pytest.skip("rebuilt-schema NSA CES partition not available")
        # An as-of slice: best-available revision per (series, ref_date),
        # pre-benchmark — keeping the size dims so Q1 buckets stay distinct.
        as_of = (
            ces.filter(pl.col("benchmark_revision") == 0)
            .sort("revision", descending=True)
            .unique(
                subset=[
                    "geographic_type", "geographic_code", "ownership",
                    "industry_type", "industry_code", "ref_date",
                    "size_class_type", "size_class_code",
                ],
                keep="first",
            )
        )
        if as_of.is_empty():
            pytest.skip("no pre-benchmark CES rows to slice")
        gaps = gate_vintage_integrity(as_of)
        assert not gaps, "\n".join(gaps)
