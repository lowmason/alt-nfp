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

import os
from datetime import date

import polars as pl
import pytest
from nfp_ingest.vintage_store import read_vintage_store
from nfp_lookups.industry import QCEW_SUPERSECTOR
from nfp_lookups.paths import VINTAGE_STORE_PATH, storage_options_for
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_vintages.rebuild_gates import (
    _EXPECTED_QCEW_CES_RESIDUAL,
    _QCEW_CES_RESIDUAL_BAND,
    gate_ces_fidelity,
    gate_gap_fill,
    gate_history_consistency,
    gate_q1_continuity,
    gate_qcew_fidelity,
    gate_reconstruction_accuracy,
    gate_vintage_integrity,
)
from upath import UPath

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
    seasonally_adjusted: bool = False,
    size_class_type: str | None = None,
    size_class_code: str | None = None,
    geographic_type: str = "national",
    geographic_code: str = "00",
) -> dict:
    """One VINTAGE_STORE_SCHEMA-conformant row as a dict.

    ``seasonally_adjusted`` defaults to ``False`` (NSA): the §10 gates reproduce
    the NSA private hierarchy + the NSA ``00`` anchor, so an unspecified row is
    NSA and the plans/11 T3 ``_nsa_only`` filter is a no-op on it.  SA-rail tests
    pass ``seasonally_adjusted=True`` explicitly.
    """
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


# The history + fidelity gates operate on the NSA hierarchy/anchor, so these
# helpers stamp ``seasonally_adjusted=False`` explicitly — redundant with the
# ``_row`` default (now ``False``) but kept for local clarity.  The plans/11 T3
# ``_nsa_only`` filter on the history gate must be a no-op on these NSA frames;
# SA-specific coverage lives in dedicated tests.
def _legacy_total_private_rows(ref: date, emp: float) -> list[dict]:
    """Legacy-axis (domain/05, null ownership) NSA rows for the four cohorts."""
    return [
        _row(
            industry_type="domain", industry_code="05", ownership=None,
            ref_date=ref, employment=emp, revision=r, benchmark_revision=b,
            vintage_date=date(2024, 2, 1), seasonally_adjusted=False,
        )
        for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
    ]


def _rebuilt_total_private_rows(ref: date, emp: float) -> list[dict]:
    """Rebuilt-axis (total/05, private) NSA rows for the four cohorts."""
    return [
        _row(
            industry_type="total", industry_code="05", ownership="private",
            ref_date=ref, employment=emp, revision=r, benchmark_revision=b,
            vintage_date=date(2024, 2, 1), seasonally_adjusted=False,
        )
        for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
    ]


def _rebuilt_anchor_rows(ref: date, emp: float) -> list[dict]:
    """Rebuilt-axis (total/00, total) NSA anchor rows for the four cohorts."""
    return [
        _row(
            industry_type="total", industry_code="00", ownership="total",
            ref_date=ref, employment=emp, revision=r, benchmark_revision=b,
            vintage_date=date(2024, 2, 1), seasonally_adjusted=False,
        )
        for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
    ]


def _legacy_anchor_rows(ref: date, emp: float) -> list[dict]:
    """Legacy-axis (national/00, null ownership) NSA anchor rows for four cohorts."""
    return [
        _row(
            industry_type="national", industry_code="00", ownership=None,
            ref_date=ref, employment=emp, revision=r, benchmark_revision=b,
            vintage_date=date(2024, 2, 1), seasonally_adjusted=False,
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
                    seasonally_adjusted=False,
                ),
                _row(
                    industry_type="total", industry_code="05", ownership="private",
                    ref_date=self._REF, employment=130_000.0, revision=2,
                    benchmark_revision=1, vintage_date=date(2025, 2, 1),
                    seasonally_adjusted=False,
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
                            vintage_date=date(2024, 2, 1), seasonally_adjusted=False,
                        ),
                        _row(
                            industry_type="total", industry_code="05",
                            ownership="private", ref_date=self._REF,
                            employment=130_250.0, revision=2, benchmark_revision=1,
                            vintage_date=date(2025, 2, 1), seasonally_adjusted=False,
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

    def test_benchmark_free_print_divergence_fails_hard(self) -> None:
        """A (1,0) second-print divergence from the legacy store is HARD.

        The benchmark-free prints (0,0)/(1,0) carry no annual-benchmark
        ambiguity, so the rebuilt store must reproduce the legacy store's first
        and second prints to rounding (verified 2026-06-16: 0/2520 diverge on
        the real stores).  A drift here is a real reconstruction bug — HARD.
        """
        rebuilt, existing = self._good_pair()
        rebuilt = rebuilt.with_columns(
            pl.when(
                (pl.col("industry_code") == "05")
                & (pl.col("revision") == 1)
                & (pl.col("benchmark_revision") == 0)
            )
            .then(pl.col("employment") + 2_000.0)
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_history_consistency(rebuilt, existing)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert any("diverge" in g for g in hard)

    def test_2_1_divergence_from_legacy_is_soft_not_hard(self) -> None:
        """A (2,1) divergence from the LEGACY store is SOFT, not a hard failure.

        Verified against the real stores (2026-06-16): the legacy benchmark
        splice mis-stamped the latest benchmark value under the earliest
        vintage_date (a lookahead bug), so the rebuilt (2,1) — which reproduces
        the literal cesvinall per-benchmark cells to the unit — *correctly*
        diverges from legacy.  The history gate must therefore treat a
        (2,1)-vs-legacy divergence as SOFT; the HARD accuracy rail on the
        benchmark cohort is :func:`gate_ces_fidelity` (rebuilt vs cesvinall).
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
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard == [], hard
        assert any(g.startswith("SOFT:") and "diverge" in g for g in gaps)

    def test_2_0_drift_from_legacy_is_soft(self) -> None:
        """A (2,0) third-print divergence from the LEGACY store is SOFT.

        The third print lands (for ~3 ref-months/year) in the February
        benchmark release, so the legacy splice deviates from the literal
        cesvinall cell there.  The rebuilt (2,0) reproduces cesvinall (verified
        2026-06-16), so a (2,0)-vs-legacy drift is a documented expected
        divergence — SOFT, with the HARD rail in gate_ces_fidelity.
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
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard == [], hard
        assert any(g.startswith("SOFT:") for g in gaps)

    def test_anchor_benchmark_divergence_is_soft(self) -> None:
        """The total/00 anchor (2,0) divergence from legacy is SOFT, named.

        Spec §10 names the total/00 anchor value-match path; under the
        recalibrated semantics a benchmark-cohort anchor divergence from the
        legacy store is reported SOFT (the legacy splice deviates from
        cesvinall), still naming total/00 so it is auditable.
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
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard == [], hard
        assert any(g.startswith("SOFT:") and "total/00" in g for g in gaps)

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
                     ownership="private", ref_date=ref, employment=9_500.0,
                     seasonally_adjusted=False),
                _row(industry_type="sector", industry_code="55",
                     ownership="private", ref_date=ref, employment=2_400.0,
                     seasonally_adjusted=False),
            ]
        )
        existing = _frame(
            _legacy_total_private_rows(ref, 130_000.0)
            + _legacy_anchor_rows(ref, 158_000.0)
            + [
                # supersector 55 matches; sector 55 DIVERGES.
                _row(industry_type="supersector", industry_code="55",
                     ownership=None, ref_date=ref, employment=9_500.0,
                     seasonally_adjusted=False),
                _row(industry_type="sector", industry_code="55",
                     ownership=None, ref_date=ref, employment=8_888.0,
                     seasonally_adjusted=False),
            ]
        )
        gaps = gate_history_consistency(rebuilt, existing)
        assert gaps
        # The sector-55 mismatch surfaces; supersector-55 did not mask it.
        assert any("sector/55" in g for g in gaps)

    def test_sa_rows_excluded_from_nsa_history(self) -> None:
        """SA rows are filtered before the legacy-store history comparison.

        The legacy-core reproduction is the NSA hierarchy + NSA ``00`` anchor; the
        HARD (0,0)/(1,0) rail was verified on NSA, and SA fidelity is
        ``gate_ces_fidelity``'s job.  Here the rebuilt store also carries SA
        ``00`` first-prints that diverge wildly from the legacy SA value (which
        the rebuilt+legacy frames here share the same NSA values).  Because the
        legacy wrapper loads NSA only but the rebuilt frame may carry SA, an
        unfiltered NSA-rebuilt↔SA-legacy (or SA-rebuilt↔NSA-legacy) pairing on a
        key lacking ``seasonally_adjusted`` would spuriously HARD-fail.  With the
        ``_nsa_only`` filter the SA rows drop out and only the NSA core is
        compared — no HARD gap.
        """
        rebuilt, existing = self._good_pair()
        # Add SA anchor rows to the rebuilt store with a wildly different value.
        sa = _frame(
            [
                _row(industry_type="total", industry_code="00", ownership="total",
                     ref_date=self._REF, employment=99_999.0, revision=r,
                     benchmark_revision=b, vintage_date=date(2024, 2, 1),
                     seasonally_adjusted=True)
                for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
            ]
        )
        rebuilt = pl.concat([rebuilt, sa])
        # Existing (legacy) store also carries SA rows that diverge (the legacy
        # store may load SA+NSA in the wrapper) — both must be filtered.
        existing_sa = _frame(
            [
                _row(industry_type="national", industry_code="00", ownership=None,
                     ref_date=self._REF, employment=42_000.0, revision=r,
                     benchmark_revision=b, vintage_date=date(2024, 2, 1),
                     seasonally_adjusted=True)
                for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]
            ]
        )
        existing = pl.concat([existing, existing_sa])
        gaps = gate_history_consistency(rebuilt, existing)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard == [], hard

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
# Gate 1b — CES fidelity (rebuilt CES vs cesvinall reference)
# ===========================================================================


class TestGateCesFidelity:
    """Rebuilt CES reproduces the cesvinall-derived reference to the unit.

    The TRUE CES reconstruction-accuracy rail (mirrors ``gate_qcew_fidelity``
    for QCEW).  Where ``gate_history_consistency`` compares against the *legacy
    store* — whose benchmark splice deviates from BLS-published cesvinall on the
    (2,0)/(2,1) cohorts — this gate compares against cesvinall itself, so it is
    the HARD accuracy check on the benchmark-bearing cohorts the history gate
    now treats SOFT.
    """

    _REF = date(2023, 6, 12)

    def _good_pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        rows = (
            _rebuilt_total_private_rows(self._REF, 130_000.0)
            + _rebuilt_anchor_rows(self._REF, 158_000.0)
        )
        frame = _frame(rows)
        return frame, frame

    def test_good_frame_passes(self) -> None:
        rebuilt, reference = self._good_pair()
        assert gate_ces_fidelity(rebuilt, reference) == []

    def test_print_value_mismatch_fails_hard(self) -> None:
        rebuilt, reference = self._good_pair()
        rebuilt = rebuilt.with_columns(
            pl.when(
                (pl.col("industry_code") == "05") & (pl.col("revision") == 0)
            )
            .then(pl.col("employment") + 50.0)
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_ces_fidelity(rebuilt, reference)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert any("fidelity" in g for g in hard)

    def test_2_1_value_mismatch_fails_hard(self) -> None:
        """A (2,1) benchmark-cohort divergence from cesvinall is HARD here.

        This is the rail the recalibrated history gate relies on: the
        benchmark cohort is no longer hard-checked against legacy, so a real
        (2,1) builder regression must be caught against the cesvinall
        reference.
        """
        rebuilt, reference = self._good_pair()
        rebuilt = rebuilt.with_columns(
            pl.when(
                (pl.col("industry_code") == "00")
                & (pl.col("revision") == 2)
                & (pl.col("benchmark_revision") == 1)
            )
            .then(pl.col("employment") + 5.0)
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_ces_fidelity(rebuilt, reference)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert any("fidelity" in g and "total/00" in g for g in hard)

    def test_rounding_slack_passes(self) -> None:
        """A sub-rounding (< abs_tol) wobble does not fail the gate."""
        rebuilt, reference = self._good_pair()
        rebuilt = rebuilt.with_columns((pl.col("employment") + 0.4).alias("employment"))
        assert gate_ces_fidelity(rebuilt, reference) == []

    def test_row_present_only_one_side_is_soft(self) -> None:
        rebuilt, reference = self._good_pair()
        # Drop the (2,1) anchor cohort from the reference: rebuilt now carries a
        # row the reference lacks — a coverage difference, reported SOFT.
        reference = reference.filter(
            ~(
                (pl.col("industry_code") == "00")
                & (pl.col("revision") == 2)
                & (pl.col("benchmark_revision") == 1)
            )
        )
        gaps = gate_ces_fidelity(rebuilt, reference)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard == [], hard
        assert any(g.startswith("SOFT:") for g in gaps)

    def _sa_nsa_pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """A rebuilt+reference pair carrying BOTH SA and NSA rows.

        SA and NSA share the same ``vintage_date`` per (series, ref, rev, bmr)
        (Decision A) but differ in value: NSA 158_000, SA 156_000.  Both frames
        carry ``seasonally_adjusted`` by construction.
        """
        rows: list[dict] = []
        for (r, b) in [(0, 0), (1, 0), (2, 0), (2, 1)]:
            rows.append(
                _row(industry_type="total", industry_code="00", ownership="total",
                     ref_date=self._REF, employment=158_000.0, revision=r,
                     benchmark_revision=b, vintage_date=date(2024, 2, 1),
                     seasonally_adjusted=False)
            )
            rows.append(
                _row(industry_type="total", industry_code="00", ownership="total",
                     ref_date=self._REF, employment=156_000.0, revision=r,
                     benchmark_revision=b, vintage_date=date(2024, 2, 1),
                     seasonally_adjusted=True)
            )
        frame = _frame(rows)
        return frame, frame

    def test_sa_and_nsa_match_cohort_for_cohort(self) -> None:
        """SA↔SA and NSA↔NSA match when both carry seasonally_adjusted.

        SA and NSA share ``vintage_date`` (Decision A) but hold different values.
        Without ``seasonally_adjusted`` in the join key an NSA-reference row would
        inner-join BOTH the NSA *and* the SA rebuilt row at the same vintage,
        creating a ~2,000-thousand spurious mismatch (the false HARD fail this
        change locks out).  With the key fix each rail matches itself and the
        gate passes.
        """
        rebuilt, reference = self._sa_nsa_pair()
        assert gate_ces_fidelity(rebuilt, reference) == []

    def test_corrupted_sa_value_named_hard(self) -> None:
        """Corrupting one SA value HARD-fails and the gap names the SA row.

        Proves the SA rail is genuinely checked (not merely de-conflated): the
        NSA values are untouched, only an SA cohort drifts, and the gate must
        catch it.
        """
        rebuilt, reference = self._sa_nsa_pair()
        rebuilt = rebuilt.with_columns(
            pl.when(
                pl.col("seasonally_adjusted")
                & (pl.col("revision") == 0)
            )
            .then(pl.col("employment") + 50.0)
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_ces_fidelity(rebuilt, reference)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("fidelity" in g and "total/00" in g for g in hard)


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

    def test_sa_rows_excluded_from_nesting(self) -> None:
        """SA rows do not enter the additive-nesting sums (plans/11 T3).

        SA does NOT nest additively (``05 != 06 + 08`` in SA).  Build a frame
        whose NSA ``05 == 06 + 08`` holds exactly (130k == 50k + 80k) and a
        PARALLEL SA triple that does NOT nest (128k != 40k + 60k).  No
        supersectors are supplied, so ``06/08 == sum(supersectors)`` is skipped
        (missing components) and only ``05 == 06 + 08`` is checkable — it must
        hold on NSA and must NOT be evaluated on the SA rows.  Without
        ``_nsa_only`` the SA triple would emit a spurious ``SOFT: ... 05 !=
        06+08``; with it, no nesting gap.  (The frontier hard-gate fires for the
        absent supersectors — irrelevant here; we assert only on nesting.)
        """
        ref = date(2025, 6, 12)
        nsa = [
            _row(industry_type="total", industry_code="05", ownership="private",
                 ref_date=ref, employment=130_000.0, seasonally_adjusted=False),
            _row(industry_type="domain", industry_code="06", ownership="private",
                 ref_date=ref, employment=50_000.0, seasonally_adjusted=False),
            _row(industry_type="domain", industry_code="08", ownership="private",
                 ref_date=ref, employment=80_000.0, seasonally_adjusted=False),
        ]
        sa = [
            _row(industry_type="total", industry_code="05", ownership="private",
                 ref_date=ref, employment=128_000.0, seasonally_adjusted=True),
            _row(industry_type="domain", industry_code="06", ownership="private",
                 ref_date=ref, employment=40_000.0, seasonally_adjusted=True),
            _row(industry_type="domain", industry_code="08", ownership="private",
                 ref_date=ref, employment=60_000.0, seasonally_adjusted=True),
        ]
        df = _frame(nsa + sa)
        gaps = gate_gap_fill(df, frontier_ref_date=ref, dec_cohort_years=())
        # The SA non-nesting must NOT produce a nesting gap; NSA nests cleanly.
        assert not any("nesting" in g or "06+08" in g for g in gaps), gaps

    def test_sa_breaks_nesting_when_not_filtered_is_caught(self) -> None:
        """Control: an NSA-only frame whose 05 != 06 + 08 DOES flag SOFT nesting.

        Pairs with ``test_sa_rows_excluded_from_nesting`` to prove the filter is
        what suppresses the gap (not an inert check): the SAME non-nesting triple,
        stamped NSA, must produce the ``05 != 06+08`` SOFT finding.
        """
        ref = date(2025, 6, 12)
        broken = _frame(
            [
                _row(industry_type="total", industry_code="05", ownership="private",
                     ref_date=ref, employment=128_000.0, seasonally_adjusted=False),
                _row(industry_type="domain", industry_code="06", ownership="private",
                     ref_date=ref, employment=40_000.0, seasonally_adjusted=False),
                _row(industry_type="domain", industry_code="08", ownership="private",
                     ref_date=ref, employment=60_000.0, seasonally_adjusted=False),
            ]
        )
        gaps = gate_gap_fill(broken, frontier_ref_date=ref, dec_cohort_years=())
        assert any("05 != 06+08" in g for g in gaps)

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
    """QCEW-vs-CES definitional-residual band gate (recalibrated).

    The rebuilt QCEW is faithful to published QCEW (UI-covered employment); CES
    estimates ALL nonfarm payroll incl. UI-exempt orgs.  So ``QCEW < CES`` is the
    EXPECTED direction and the gate checks each series' MEDIAN residual
    (``qcew/ces - 1``) against a per-series expected band, not a hard sign.
    """

    # Several settled, non-COVID months so the per-series MEDIAN is meaningful.
    _REFS = [date(y, 3, 12) for y in (2022, 2023, 2024)]

    def _pair_at_residuals(
        self, residuals: dict[str, float], refs: list[date] | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build a QCEW/CES pair whose per-series residual is *exactly* given.

        ``residuals`` maps an industry_code in {05, 08, 80, 81} to a fraction
        ``r``; QCEW is stamped at ``ces * (1 + r)``.  CES carries Other Services
        ONLY as supersector/80 (verified 0 CES sector/81 rows), so QCEW sector/81
        is compared to CES supersector/80.
        """
        refs = refs or self._REFS
        # Fixed CES base levels per series (headline thousands).
        ces_base = {"05": 130_000.0, "08": 105_000.0, "80": 6_000.0}
        ces_meta = {
            "05": ("total", "05"),
            "08": ("domain", "08"),
            "80": ("supersector", "80"),
        }
        # QCEW carries the same three PLUS a sector/81 (maps to CES 80).
        qcew_meta = {
            "05": ("total", "05"),
            "08": ("domain", "08"),
            "80": ("supersector", "80"),
            "81": ("sector", "81"),
        }
        ces_rows: list[dict] = []
        qcew_rows: list[dict] = []
        for ref in refs:
            for code, (it, ic) in ces_meta.items():
                ces_rows.append(
                    _row(industry_type=it, industry_code=ic, ownership="private",
                         ref_date=ref, employment=ces_base[code], source="ces")
                )
            for code, (it, ic) in qcew_meta.items():
                # 81 compares to CES 80, so it uses the 80 base.
                base = ces_base["80"] if code == "81" else ces_base[code]
                r = residuals[code]
                qcew_rows.append(
                    _row(industry_type=it, industry_code=ic, ownership="private",
                         ref_date=ref, employment=base * (1.0 + r), source="qcew")
                )
        return _frame(qcew_rows), _frame(ces_rows)

    def _good_pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        # In-band residuals exactly at the verified expectations.
        return self._pair_at_residuals(
            {"05": -0.025, "08": -0.029, "80": -0.225, "81": -0.225}
        )

    def test_good_frame_passes(self) -> None:
        qcew, ces = self._good_pair()
        assert gate_reconstruction_accuracy(qcew, ces) == []

    def test_sa_ces_excluded_from_residual(self) -> None:
        """SA CES rows must not enter the QCEW(NSA)-vs-CES residual (plans/11 T3).

        QCEW is NSA; comparing NSA-QCEW to an SA-CES level would inject a seasonal
        wobble into the residual.  Here the in-band NSA pair passes, but a
        parallel set of SA CES rows at a DIFFERENT level (which, if
        ``_best_available`` picked them, would shift the residual out of band) is
        added.  ``_nsa_only`` on the CES side drops them, so the residual is taken
        on NSA only and the gate still passes.
        """
        qcew, ces = self._good_pair()
        # SA CES at half the NSA level — would wreck the residual if not filtered.
        # Stamp a STRICTLY-LATER vintage_date so that, absent the NSA filter,
        # ``_best_available`` (sorts by vintage_date desc, keys without
        # seasonally_adjusted) would prefer the SA row over the NSA one — making
        # the filter genuinely load-bearing, not order-of-rows luck.
        sa_ces = _frame(
            [
                _row(industry_type=it, industry_code=ic, ownership="private",
                     ref_date=ref, employment=base * 0.5, source="ces",
                     seasonally_adjusted=True,
                     vintage_date=date(ref.year + 1, 2, 1))
                for ref in self._REFS
                for (it, ic, base) in [
                    ("total", "05", 130_000.0),
                    ("domain", "08", 105_000.0),
                    ("supersector", "80", 6_000.0),
                ]
            ]
        )
        ces = pl.concat([ces, sa_ces])
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)

    def _pair_at_residuals_00(
        self, resid: float, refs: list[date] | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """A QCEW ``('total','00','total')`` track + CES ``total/00`` at *resid*.

        CES ``00`` is total nonfarm (NSA); QCEW ``00`` is the total-covered
        track (UI-covered, incl. government).  The provisional expected residual
        is ``_EXPECTED_QCEW_CES_RESIDUAL['00']`` (UI-coverage gap).
        """
        refs = refs or self._REFS
        ces_base_00 = 158_000.0
        ces_rows = [
            _row(industry_type="total", industry_code="00", ownership="total",
                 ref_date=ref, employment=ces_base_00, source="ces")
            for ref in refs
        ]
        qcew_rows = [
            _row(industry_type="total", industry_code="00", ownership="total",
                 ref_date=ref, employment=ces_base_00 * (1.0 + resid),
                 source="qcew")
            for ref in refs
        ]
        return _frame(qcew_rows), _frame(ces_rows)

    def test_00_band_in_band_passes(self) -> None:
        """A QCEW ``00`` total within the provisional band passes."""
        qcew, ces = self._pair_at_residuals_00(_EXPECTED_QCEW_CES_RESIDUAL["00"])
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)

    def test_00_band_out_of_band_fails_hard(self) -> None:
        """A QCEW ``00`` total at 0% (coverage bug erasing the gap) HARD-fails.

        0% is well outside the provisional band around the expected UI-coverage
        gap, and 0.0 is not > pos_margin, so it reaches the out-of-band check and
        fails — naming total/00.
        """
        qcew, ces = self._pair_at_residuals_00(0.0)
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("total/00" in g and "out-of-band" in g for g in hard)

    def test_81_maps_to_ces_80_not_silently_skipped(self) -> None:
        """QCEW sector/81 is compared to CES supersector/80, not dropped.

        Build CES with ONLY supersector/80 (no sector/81 — verified store reality)
        and QCEW with ONLY sector/81.  An in-band 81 must compare and PASS; an
        out-of-band 81 must HARD-fail naming sector/81 — proving the remap fired,
        not a silent skip.
        """
        # In-band 81 -> CES 80 at -22.5%: passes.
        qcew = _frame(
            [
                _row(industry_type="sector", industry_code="81",
                     ownership="private", ref_date=ref,
                     employment=6_000.0 * (1.0 - 0.225), source="qcew")
                for ref in self._REFS
            ]
        )
        ces = _frame(
            [
                _row(industry_type="supersector", industry_code="80",
                     ownership="private", ref_date=ref, employment=6_000.0,
                     source="ces")
                for ref in self._REFS
            ]
        )
        assert gate_reconstruction_accuracy(qcew, ces) == []

        # Out-of-band 81 (-45%) -> hard fail naming sector/81.
        qcew_bad = _frame(
            [
                _row(industry_type="sector", industry_code="81",
                     ownership="private", ref_date=ref,
                     employment=6_000.0 * (1.0 - 0.45), source="qcew")
                for ref in self._REFS
            ]
        )
        gaps = gate_reconstruction_accuracy(qcew_bad, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("sector/81" in g for g in hard)

    def test_residual_too_shallow_fails_hard(self) -> None:
        """80 at -5% (far above its -22.5% band) is a HARD fail."""
        qcew, ces = self._pair_at_residuals(
            {"05": -0.025, "08": -0.029, "80": -0.05, "81": -0.225}
        )
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("supersector/80" in g and "out-of-band" in g for g in hard)

    def test_residual_too_deep_fails_hard(self) -> None:
        """80 at -45% (far below its -22.5% band) is a HARD fail."""
        qcew, ces = self._pair_at_residuals(
            {"05": -0.025, "08": -0.029, "80": -0.45, "81": -0.225}
        )
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("supersector/80" in g and "out-of-band" in g for g in hard)

    def test_positive_residual_fails_hard(self) -> None:
        """A positive residual (QCEW > CES) beyond a small margin is anomalous."""
        qcew, ces = self._pair_at_residuals(
            {"05": 0.03, "08": -0.029, "80": -0.225, "81": -0.225}
        )
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("total/05" in g and "positive" in g for g in hard)

    def test_shallow_05_at_zero_fails_hard(self) -> None:
        """05 at 0% (definitional gap erased) HARD-fails the tight per-series band.

        A uniform 8pp band would admit 0% for the -2.5% series, waving through a
        coverage bug that pulls CES-universe data or includes UI-exempt orgs.  The
        per-series ~2pp band on 05 catches it: |0 - (-0.025)| = 0.025 > 0.02, and
        0.0 is not > pos_margin (0.01) so it reaches the out-of-band check.
        """
        qcew, ces = self._pair_at_residuals(
            {"05": 0.0, "08": -0.029, "80": -0.225, "81": -0.225}
        )
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("total/05" in g and "out-of-band" in g for g in hard)

    def test_shallow_08_at_zero_fails_hard(self) -> None:
        """08 at 0% (definitional gap erased) HARD-fails the tight per-series band.

        |0 - (-0.029)| = 0.029 > 0.02; same named adversarial case as 05.
        """
        qcew, ces = self._pair_at_residuals(
            {"05": -0.025, "08": 0.0, "80": -0.225, "81": -0.225}
        )
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("domain/08" in g and "out-of-band" in g for g in hard)

    def test_shallow_05_just_inside_tight_band_passes(self) -> None:
        """05 at -1.5% (1pp from -2.5%, inside the 2pp band) still PASSES.

        The tight band is not zero-width: a residual within the verified spread of
        the definitional gap must not flag, so the band stays above the claimed
        <1pp p10-p90 spread.
        """
        qcew, ces = self._pair_at_residuals(
            {"05": -0.015, "08": -0.029, "80": -0.225, "81": -0.225}
        )
        gaps = gate_reconstruction_accuracy(qcew, ces)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)

    def test_settled_month_collapse_fails_hard(self) -> None:
        """An implausible collapse in a SETTLED month HARD-fails (not hidden as SOFT).

        The frontier exclusion is date-scoped to the unsettled window, so the SAME
        implausible-collapse signal in settled history is NOT excused as lag — it
        stays in the clean set and trips the per-month implausible-collapse floor as
        a HARD failure (otherwise the median could absorb it).  Here the window is
        pinned to 2025-01-01, so the 2024-03 collapse is settled and must HARD-fail.
        """
        # 80 series: 2022/2023 in-band at -22.5%, 2024-03 collapses to ~-67%.
        # 2024-03 is SETTLED relative to the pinned 2025-01-01 window, so it must
        # NOT be excused as frontier — it must HARD-fail the floor.
        refs = [date(y, 3, 12) for y in (2022, 2023, 2024)]
        ces_rows = [
            _row(industry_type="supersector", industry_code="80",
                 ownership="private", ref_date=ref, employment=6_000.0,
                 source="ces")
            for ref in refs
        ]
        qcew_rows = []
        for ref in refs:
            emp = 2_000.0 if ref.year == 2024 else 6_000.0 * (1.0 - 0.225)
            qcew_rows.append(
                _row(industry_type="supersector", industry_code="80",
                     ownership="private", ref_date=ref, employment=emp,
                     source="qcew")
            )
        gaps = gate_reconstruction_accuracy(
            _frame(qcew_rows), _frame(ces_rows),
            frontier_window_start=date(2025, 1, 1),
        )
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard, "\n".join(gaps)
        assert any("implausible collapse" in g for g in hard)
        # And it must NOT be silently downgraded to a frontier SOFT exclusion.
        assert not any(
            g.startswith("SOFT:") and "frontier" in g for g in gaps
        ), "settled-month collapse must not be excluded as a frontier month"

    def test_all_months_excluded_fails_hard(self) -> None:
        """A series with NO clean months to band-check HARD-fails (not SOFT pass).

        If every comparable month is excluded (here all are COVID years), the
        empty-clean path must be a HARD gap, not a silent SOFT return — otherwise a
        systematic collapse excluded month-by-month could slip through.
        """
        covid_refs = [date(2020, 3, 12), date(2021, 3, 12)]
        ces_rows = [
            _row(industry_type="supersector", industry_code="80",
                 ownership="private", ref_date=ref, employment=6_000.0,
                 source="ces")
            for ref in covid_refs
        ]
        qcew_rows = [
            _row(industry_type="supersector", industry_code="80",
                 ownership="private", ref_date=ref, employment=2_000.0,
                 source="qcew")
            for ref in covid_refs
        ]
        gaps = gate_reconstruction_accuracy(_frame(qcew_rows), _frame(ces_rows))
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert hard and any("no clean" in g for g in hard), "\n".join(gaps)

    def test_covid_and_frontier_months_excluded_and_soft(self) -> None:
        """COVID years and incomplete-frontier months drop from the hard band.

        Add a wild 2020/2021 (COVID) month and a 2025-Q1 incomplete-frontier
        month to an otherwise in-band 80 series.  The median band check must still
        PASS (those months excluded), and the frontier month is surfaced SOFT.
        """
        # Settled in-band 80 across 2022-2024, plus COVID + frontier outliers.
        settled = self._pair_at_residuals(
            {"05": -0.025, "08": -0.029, "80": -0.225, "81": -0.225}
        )
        qcew, ces = settled
        # COVID 2020 month: 80 at -60% (would wreck the median if counted).
        covid_ces = _row(industry_type="supersector", industry_code="80",
                         ownership="private", ref_date=date(2020, 6, 12),
                         employment=6_000.0, source="ces")
        covid_qcew = _row(industry_type="supersector", industry_code="80",
                          ownership="private", ref_date=date(2020, 6, 12),
                          employment=6_000.0 * 0.40, source="qcew")
        # Frontier 2025-Q1 month: 80 at -88% (incomplete size data).
        front_ces = _row(industry_type="supersector", industry_code="80",
                         ownership="private", ref_date=date(2025, 3, 12),
                         employment=6_000.0, source="ces")
        front_qcew = _row(industry_type="supersector", industry_code="80",
                          ownership="private", ref_date=date(2025, 3, 12),
                          employment=6_000.0 * 0.12, source="qcew")
        qcew = pl.concat([qcew, _frame([covid_qcew, front_qcew])])
        ces = pl.concat([ces, _frame([covid_ces, front_ces])])
        gaps = gate_reconstruction_accuracy(qcew, ces)
        # No hard band failure — the outliers were excluded from the median.
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)
        # The frontier month is surfaced SOFT for the maintainer.
        assert any(g.startswith("SOFT:") and "frontier" in g for g in gaps)
        # The COVID month is ALSO surfaced SOFT (not dropped silently) — both
        # classes of exclusion from the hard band are visible to the maintainer.
        assert any(g.startswith("SOFT:") and "COVID" in g for g in gaps)

    def test_size_buckets_excluded_from_residual(self) -> None:
        """Q1 size buckets on the QCEW side must not inflate the residual."""
        qcew, ces = self._good_pair()
        bucket = _frame(
            [
                _row(industry_type="sector", industry_code="81",
                     ownership="private", ref_date=self._REFS[0],
                     employment=3_000.0, source="qcew",
                     size_class_type="size_class", size_class_code="1"),
            ]
        )
        qcew = pl.concat([qcew, bucket])
        # Without the all-sizes filter, 81 at the first ref would be inflated by
        # the bucket and break its band.  With it, the bucket is dropped.
        assert gate_reconstruction_accuracy(qcew, ces) == []

    def test_expected_residual_constant_shape(self) -> None:
        """The seeded expectations cover exactly {00, 05, 08, 80, 81}.

        ``00`` (the QCEW total-covered track) was added in plans/11 T3 with a
        PROVISIONAL expectation that T4 calibrates; all expectations stay
        negative (QCEW sits below CES on the UI-coverage gap).
        """
        assert set(_EXPECTED_QCEW_CES_RESIDUAL) == {"00", "05", "08", "80", "81"}
        assert all(v < 0 for v in _EXPECTED_QCEW_CES_RESIDUAL.values())

    def test_band_constant_covers_every_expected_code(self) -> None:
        """The per-series band dict MUST cover every expected-residual code.

        A code present in _EXPECTED_QCEW_CES_RESIDUAL but absent from the band dict
        would KeyError at band[code]; this pins them in lockstep.  The shallow
        05/08 get a tight band; the deep 80/81 keep the generous one.
        """
        assert set(_QCEW_CES_RESIDUAL_BAND) == set(_EXPECTED_QCEW_CES_RESIDUAL)
        assert _QCEW_CES_RESIDUAL_BAND["05"] < _QCEW_CES_RESIDUAL_BAND["80"]
        assert _QCEW_CES_RESIDUAL_BAND["08"] < _QCEW_CES_RESIDUAL_BAND["81"]


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
# Gate 3b — QCEW reconstruction fidelity (rebuilt vs reference QCEW)
# ===========================================================================


class TestGateQcewFidelity:
    """Rebuilt QCEW must reproduce published/reference QCEW near-exactly.

    Unlike the CES residual gate (a definitional gap), this is the TRUE
    reconstruction-accuracy check: two QCEW frames compared on the full key.
    """

    _REF = date(2024, 6, 12)

    def _reference(self) -> pl.DataFrame:
        return _frame(
            [
                _row(industry_type="supersector", industry_code="80",
                     ownership="private", ref_date=self._REF, employment=4_739.7,
                     source="qcew", seasonally_adjusted=False),
                _row(industry_type="sector", industry_code="81",
                     ownership="private", ref_date=self._REF, employment=4_739.7,
                     source="qcew", seasonally_adjusted=False),
                _row(industry_type="total", industry_code="05",
                     ownership="private", ref_date=self._REF, employment=130_000.0,
                     source="qcew", seasonally_adjusted=False),
            ]
        )

    def test_matching_frames_pass(self) -> None:
        ref = self._reference()
        assert gate_qcew_fidelity(ref.clone(), ref) == []

    def test_perturbed_value_fails(self) -> None:
        ref = self._reference()
        rebuilt = ref.with_columns(
            pl.when(pl.col("industry_code") == "80")
            .then(pl.col("employment") + 50.0)  # well beyond abs_tol + rel_tol
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_qcew_fidelity(rebuilt, ref)
        assert gaps
        assert any("supersector/80" in g and "differ" in g for g in gaps)

    def test_within_tolerance_passes(self) -> None:
        """A rounding-scale diff inside abs_tol does not flag."""
        ref = self._reference()
        rebuilt = ref.with_columns(
            pl.when(pl.col("industry_code") == "80")
            .then(pl.col("employment") + 0.01)  # inside abs_tol=0.05
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        assert gate_qcew_fidelity(rebuilt, ref) == []

    def test_thousand_job_corruption_fails(self) -> None:
        """A +1.0-thousand (1,000-job) corruption at the 05 level HARD-fails.

        With the old ``rel_tol=1e-4`` the effective slack at a 130,000 level was
        ~13,050 jobs, so a few-hundred-to-thousands-job store-write corruption slipped
        through.  This is a SAME-SOURCE to-the-unit reproduction, so ``rel_tol=0``
        and a flat ``abs_tol=0.05`` (50 jobs): +1.0 thousand must flag.
        """
        ref = self._reference()
        rebuilt = ref.with_columns(
            pl.when(pl.col("industry_code") == "05")
            .then(pl.col("employment") + 1.0)  # 1,000 jobs (employment in thousands)
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_qcew_fidelity(rebuilt, ref)
        assert gaps
        assert any("total/05" in g and "differ" in g for g in gaps)

    def test_few_hundred_job_corruption_at_80_fails(self) -> None:
        """A +0.52-thousand (520-job) corruption at the 80 level HARD-fails.

        The old ``rel_tol=1e-4`` admitted ~474 jobs at the 80 level (4,739.7), so a
        +0.52-thousand perturbation passed.  With ``rel_tol=0`` the flat 50-job rail
        flags it.
        """
        ref = self._reference()
        rebuilt = ref.with_columns(
            pl.when(pl.col("industry_code") == "80")
            .then(pl.col("employment") + 0.52)  # 520 jobs
            .otherwise(pl.col("employment"))
            .alias("employment")
        )
        gaps = gate_qcew_fidelity(rebuilt, ref)
        assert gaps
        assert any("supersector/80" in g and "differ" in g for g in gaps)

    def test_missing_row_reported(self) -> None:
        """A row present in reference but absent in rebuilt is reported."""
        ref = self._reference()
        rebuilt = ref.filter(pl.col("industry_code") != "05")
        gaps = gate_qcew_fidelity(rebuilt, ref)
        assert gaps
        assert any("missing" in g and "total/05" in g for g in gaps)

    def test_size_buckets_reduced_before_compare(self) -> None:
        """Q1 size buckets on the rebuilt side are reduced to all-sizes first.

        The stored rebuilt QCEW carries Q1 size buckets; the reference (from
        ``build_qcew_panel``) does not.  Without the all-sizes reduction the join
        would go many-to-one and corrupt the comparison.
        """
        q1 = date(2024, 3, 12)
        ref = _frame(
            [
                _row(industry_type="supersector", industry_code="80",
                     ownership="private", ref_date=q1, employment=4_700.0,
                     source="qcew", seasonally_adjusted=False),
            ]
        )
        rebuilt = _frame(
            [
                _row(industry_type="supersector", industry_code="80",
                     ownership="private", ref_date=q1, employment=4_700.0,
                     source="qcew", seasonally_adjusted=False,
                     size_class_type="size_class", size_class_code="0"),
                # Buckets that must be excluded (else 80 inflates).
                _row(industry_type="supersector", industry_code="80",
                     ownership="private", ref_date=q1, employment=3_000.0,
                     source="qcew", seasonally_adjusted=False,
                     size_class_type="size_class", size_class_code="1"),
                _row(industry_type="supersector", industry_code="80",
                     ownership="private", ref_date=q1, employment=1_700.0,
                     source="qcew", seasonally_adjusted=False,
                     size_class_type="size_class", size_class_code="2"),
            ]
        )
        assert gate_qcew_fidelity(rebuilt, ref) == []


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

    def test_sa_and_nsa_of_one_series_month_not_duplicates(self) -> None:
        """An as-of slice carrying BOTH SA and NSA of one series-month is OK.

        SA and NSA share ``vintage_date`` (Decision A) but are distinct series
        (different employment).  With ``seasonally_adjusted`` in the dup/vintage
        key they are NOT flagged as a duplicate or a cross-vintage sum
        (plans/11 T3 defensive widening).  Note: they also share a vintage_date,
        so without the flag in the key they would trip BOTH check 1 (dup) and
        check 2 (multi-vintage is False here since they share it — so only the
        dup check) — the dup check is the one this pins.
        """
        ref = date(2024, 12, 12)
        v = date(2025, 2, 1)
        rows = [
            _row(industry_type="total", industry_code="00", ownership="total",
                 ref_date=ref, employment=158_000.0, vintage_date=v, source="ces",
                 seasonally_adjusted=False),
            _row(industry_type="total", industry_code="00", ownership="total",
                 ref_date=ref, employment=156_000.0, vintage_date=v, source="ces",
                 seasonally_adjusted=True),
        ]
        assert gate_vintage_integrity(_frame(rows)) == []

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


def _load_rebuilt(source: str, *, seasonally_adjusted: bool = False) -> pl.DataFrame:
    """Load one (source, adjustment) partition of the rebuilt store.

    Defaults to the NSA partition (the §10 NSA-hierarchy gates read NSA only).
    The SA partition (``seasonally_adjusted=True``) is the parallel rail that
    plans/11 T1 adds; ``gate_ces_fidelity`` checks it.  The cache key is
    ``(source, bool)`` so both partitions coexist.  The maintainer points
    ``NFP_STORE_URI`` at the scratch rebuild prefix (``s3://alt-nfp/store-rebuild``)
    before running these.
    """
    key = (source, seasonally_adjusted)
    if key not in _REBUILT_CACHE:
        _REBUILT_CACHE[key] = read_vintage_store(
            source=source, seasonally_adjusted=seasonally_adjusted
        ).collect()
    return _REBUILT_CACHE[key]


def _legacy_store_path() -> UPath | None:
    """The pre-rebuild (legacy) store, for the dual-store history gate.

    Read from ``NFP_LEGACY_STORE_URI`` — the maintainer sets it to the canonical
    store (e.g. ``s3://alt-nfp/store``) while ``NFP_STORE_URI`` points at the
    scratch rebuild (``s3://alt-nfp/store-rebuild``).  Returns ``None`` when
    unset so the history wrapper self-skips (the dual-store config is a
    deliberate maintainer concern, not a default).
    """
    uri = os.environ.get("NFP_LEGACY_STORE_URI")
    if not uri:
        return None
    return UPath(uri, **(storage_options_for(uri) or {}))


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

    The reconstruction wrapper hard-asserts the per-series **median residual
    band**: ``QCEW < CES`` is the expected definitional direction (QCEW counts
    UI-covered employment; CES estimates ALL nonfarm payroll), so the gate now
    checks each series' median ``qcew/ces - 1`` against the verified per-series
    expectations within ``band_tol`` — an out-of-band median (too shallow, too
    deep, or anomalously positive) is a hard FAILURE.  COVID + incomplete-frontier
    months are excluded from the median and surfaced SOFT.

    The fidelity wrapper is the TRUE reconstruction-accuracy check — it compares
    the stored rebuilt QCEW against the reference QCEW it is meant to reproduce.
    That reference comes from the live area endpoint (data.bls.gov), so the
    fidelity real-store test is additionally ``@pytest.mark.network`` and
    self-skips offline.
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
        # The per-series median band is hard; frontier exclusions are SOFT.
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)

    def test_ces_fidelity_real(self) -> None:
        """Rebuilt CES reproduces a fresh build of the local cesvinall triangle.

        The TRUE CES accuracy rail (the CES analogue of
        ``test_qcew_fidelity_real``).  Reference = ``build_ces_panel`` of the
        local ``cesvinall`` CSVs (after plans/11 T1 this emits BOTH SA + NSA);
        the stored rebuild must match it to the unit on every overlapping
        ``(series, ref_date, rev, bmr, vintage_date, seasonally_adjusted)``.
        ``gate_ces_fidelity`` keys on ``seasonally_adjusted`` (plans/11 T3) so
        SA↔SA and NSA↔NSA align cohort-for-cohort.  Self-skips without the local
        triangle (gitignored proprietary input).

        SA-partition tolerance: the SA store partition may be empty/absent on a
        pre-T4 NSA-only rebuild.  We load it best-effort and concat what is
        present; the SA-reference rows then surface as SOFT (present in reference,
        missing in rebuilt), not HARD — so an NSA-only store still passes the hard
        rail.
        """
        from nfp_ingest.ces_builder import CESVINALL_DIR, build_ces_panel

        rebuilt = _load_rebuilt("ces")
        if not _is_rebuilt_schema(rebuilt):
            pytest.skip("rebuilt-schema NSA CES store not available")
        if not CESVINALL_DIR.exists():
            pytest.skip("local cesvinall triangle not available")
        # Add the SA partition if the rebuild carries one (plans/11 T1+T4).
        try:
            sa = _load_rebuilt("ces", seasonally_adjusted=True)
        except Exception:
            sa = None
        if sa is not None and not sa.is_empty():
            rebuilt = pl.concat([rebuilt, sa])
        reference = build_ces_panel(CESVINALL_DIR)
        gaps = gate_ces_fidelity(rebuilt, reference)
        # Value mismatches are hard; coverage/frontier differences (incl. an
        # absent SA partition pre-T4) are SOFT.
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)

    def test_history_consistency_real(self) -> None:
        """Rebuilt CES reproduces the legacy store's benchmark-free core (≤2023).

        Dual-store: reads the rebuilt store (``NFP_STORE_URI``) and the legacy
        store (``NFP_LEGACY_STORE_URI``).  HARD on the benchmark-free
        ``(0,0)``/``(1,0)`` prints + the ``(rev,bmr)`` cohort population; the
        ``(2,0)``/``(2,1)``-vs-legacy divergence is SOFT (the legacy benchmark
        splice deviates from cesvinall — ``gate_ces_fidelity`` is the hard rail
        there).  Self-skips when ``NFP_LEGACY_STORE_URI`` is unset.
        """
        legacy = _legacy_store_path()
        if legacy is None:
            pytest.skip(
                "set NFP_LEGACY_STORE_URI to the canonical store for the "
                "dual-store history gate"
            )
        rebuilt = _load_rebuilt("ces")
        if not _is_rebuilt_schema(rebuilt):
            pytest.skip("rebuilt-schema NSA CES store not available")
        existing = read_vintage_store(
            legacy, source="ces", seasonally_adjusted=False
        ).collect()
        if existing.is_empty() or _is_rebuilt_schema(existing):
            pytest.skip("legacy store unavailable or already rebuilt-schema")
        gaps = gate_history_consistency(rebuilt, existing)
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)

    def test_gap_fill_real(self) -> None:
        """05 + the 10 supersectors are current to the frontier, Dec (2,1) complete.

        HARD: frontier currency at the latest CES ref_date + the two most-recent
        settled December ``(2,1)`` cohorts complete for the frontier set.  The
        additive-nesting identities are SOFT (a missing component never blocks).
        ``frontier_ref_date``/``dec_cohort_years`` are derived from the store so
        the wrapper ages with the data instead of freezing a snapshot.
        """
        ces = _load_rebuilt("ces")
        if not _is_rebuilt_schema(ces):
            pytest.skip("rebuilt-schema NSA CES store not available")
        frontier = ces["ref_date"].max()
        max_year = frontier.year
        dec_years = (
            ces.filter(
                (pl.col("revision") == 2)
                & (pl.col("benchmark_revision") == 1)
                & (pl.col("ref_date").dt.month() == 12)
                & (pl.col("ref_date").dt.year() < max_year)
            )["ref_date"]
            .dt.year()
            .unique()
            .sort()
            .to_list()
        )
        gaps = gate_gap_fill(
            ces, frontier_ref_date=frontier, dec_cohort_years=tuple(dec_years[-2:])
        )
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

    @pytest.mark.network
    def test_qcew_fidelity_real(self) -> None:
        """Rebuilt QCEW reproduces the live area-endpoint reference (opt-in).

        Re-acquires the SAME source the store was built from — the QCEW area
        endpoint ``/api/{y}/{q}/area/US000.csv`` (all 4 quarters, plain httpx) via
        the store's own ``_acquire_qcew_levels`` — crosswalks it with
        ``build_qcew_panel``, and compares to the stored rebuilt QCEW.  This is
        the TRUE reconstruction-accuracy check (the CES residual gate measures a
        definitional gap, not fidelity).  Maintainer-run: ``@network`` +
        ``@real_store``; self-skips offline / without the rebuilt store.

        REFERENCE CHOICE (fixed 2026-06-16).  An earlier draft fetched the
        ``_qtrly_singlefile`` product instead — a DIFFERENT BLS file that
        diverges from the area endpoint on suppression-sensitive small cells
        (Logging 1133 → sector 11 → domain 06, ~768 jobs in 2024), which is a
        product difference, not a store error.  Reproducing from the store's own
        acquire path (area endpoint) makes the comparison apples-to-apples and
        also reuses its tested CSV parsing (no manual ``area_fips`` schema
        coercion) and its ``revision=0`` tagging.
        """
        from nfp_ingest.qcew_crosswalk import build_qcew_panel
        from nfp_vintages.rebuild_store import _acquire_qcew_levels

        qcew = _load_rebuilt("qcew")
        if not _is_rebuilt_schema(qcew):
            pytest.skip("rebuilt-schema NSA QCEW partition not available")

        # The maintainer adjusts the year to one fully present in the store.
        year = 2024
        try:
            raw = _acquire_qcew_levels(start_year=year, end_year=year)
        except Exception as exc:  # network flake / endpoint move → skip, not error
            pytest.skip(f"area endpoint unreachable: {exc}")
        reference = build_qcew_panel(raw)

        # All four quarters, NSA, bmr=0.  The area-levels reference is BLS's real,
        # un-suppressed all-sizes total (agglvl 13–16 is not suppressed) — the
        # CORRECT oracle for the headline.  Q1 was formerly excluded because the
        # §7 compose made the Q1 all-sizes row the size-cross-product total/'0'
        # (buckets summed with disclosure-'N' dropped), which UNDERCOUNT the
        # published area total; the §7 fix (compose_rebuild_panel now overrides
        # the Q1 '0' headline to the area-levels total) closes that, so all four
        # quarters must now reproduce the area endpoint to the unit.  Requires a
        # scratch rebuild with the fix in place.  (The bmr=0 filter drops the
        # stored (2,1) rows the single-benchmark reference lacks.)
        stored = qcew.filter(
            (pl.col("ref_date").dt.year() == year)
            & (pl.col("benchmark_revision") == 0)
        )
        gaps = gate_qcew_fidelity(stored, reference)
        # Value mismatches are hard; missing-row reports are SOFT diagnostics.
        hard = [g for g in gaps if not g.startswith("SOFT:")]
        assert not hard, "\n".join(gaps)
