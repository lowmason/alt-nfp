"""Synthetic VINTAGE_STORE_SCHEMA rows for the update guardrail tests.

These are hand-built store rows (no store I/O, no network) used to exercise the
dangerous edges of append/compact and the first-print consumers. The capture path
itself never produces a -1.0 sentinel; ``make_shutdown_sentinel_row`` fabricates
one to test that the overlap diagnostic excludes it (§7).
"""

from __future__ import annotations

from datetime import date

import polars as pl
from nfp_ingest.vintage_store import append_to_vintage_store, compact_partition
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA


def make_ces_rows(
    *,
    ref_month: str,
    vintage: str,
    employment: float = 150_000.0,
    industry_code: str = "05",
    revision: int = 0,
    benchmark_revision: int = 0,
    seasonally_adjusted: bool = True,
) -> pl.DataFrame:
    """One CES headline row in the rebuilt-store schema.

    Defaults target the modeled private aggregate (``industry_code='05'`` ⇒
    ``ownership='private'``). Pass ``industry_code='00'`` for the total leg
    (⇒ ``ownership='total'``).
    """
    ownership = "private" if industry_code == "05" else "total"
    row = {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": ownership,
        "industry_type": "total",
        "industry_code": industry_code,
        "ref_date": date.fromisoformat(ref_month),
        "vintage_date": date.fromisoformat(vintage),
        "revision": revision,
        "benchmark_revision": benchmark_revision,
        "employment": employment,
        "size_class_type": None,
        "size_class_code": None,
        "source": "ces",
        "seasonally_adjusted": seasonally_adjusted,
    }
    cols = list(VINTAGE_STORE_SCHEMA.keys())
    return pl.DataFrame([{c: row[c] for c in cols}], schema=VINTAGE_STORE_SCHEMA)


def make_benchmark_double_row(*, ref_month: str) -> pl.DataFrame:
    """One ref_date co-published as BOTH (rev1,bmr0) and (rev2,bmr1) on a benchmark.

    The February benchmark restamps a month under the new benchmark revision while
    the pre-benchmark track still exists. Both ukeys are distinct (differ in
    benchmark_revision and revision), so append/compact must keep both rows.
    """
    a = make_ces_rows(
        ref_month=ref_month, vintage="2026-02-06",
        revision=1, benchmark_revision=0, employment=149_500.0,
    )
    b = make_ces_rows(
        ref_month=ref_month, vintage="2026-02-06",
        revision=2, benchmark_revision=1, employment=149_900.0,
    )
    return pl.concat([a, b])


def make_shutdown_sentinel_row(*, ref_month: str) -> pl.DataFrame:
    """The literal ``employment = -1.0`` 'no print' sentinel for a shutdown-skipped slot.

    This is the *value* the rebuilt store writes for a skipped release slot (e.g.
    Oct-2025 rev0); ``first_print_changes`` drops it via ``employment > 0``
    (``first_print.py:84``). Distinct from the *date* quirk
    ``CES_OCT_2025_RELEASED_WITH_NOV_REF``.
    """
    return make_ces_rows(
        ref_month=ref_month, vintage="2025-11-12",
        revision=0, benchmark_revision=0, employment=-1.0,
    )


def make_first_print_window(store) -> None:
    """Seed a two-month first-print window for BOTH the 05 (private) and 00 (total) legs.

    For each industry leg: month-A gets rev0/bmr0 (first print) and rev1/bmr0
    (second print = next month's prior-month partner); month-B gets rev0/bmr0.
    The 00 and 05 legs share vintage stamps so wedge_first_print_changes' same-release
    check passes and it resolves a non-empty frame. Levels differ across legs so the
    wedge is non-trivial. Compacts the partition once at the end.
    """
    # 05 (private) leg
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06", revision=0,
                      employment=150_000.0, industry_code="05"), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-03-06", revision=1,
                      employment=150_300.0, industry_code="05"), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-02-12", vintage="2026-03-06", revision=0,
                      employment=150_800.0, industry_code="05"), store)
    # 00 (total) leg — co-released vintages, larger levels (total > private)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06", revision=0,
                      employment=300_000.0, industry_code="00"), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-01-12", vintage="2026-03-06", revision=1,
                      employment=300_500.0, industry_code="00"), store)
    append_to_vintage_store(
        make_ces_rows(ref_month="2026-02-12", vintage="2026-03-06", revision=0,
                      employment=301_400.0, industry_code="00"), store)
    compact_partition(store, "ces", True)


def overlap_level_divergence(
    bootstrap: pl.DataFrame, capture: pl.DataFrame
) -> pl.DataFrame:
    """Per-row level divergence on score-relevant rows over a bootstrap∩capture window.

    Compares ``employment`` on the rows that drive the A5 score — first print
    (``rev0/bmr0``) and its prior-month partner (``rev1/bmr0``) — between a bootstrap
    reconstruction and a capture. The ``-1.0`` shutdown sentinel is EXCLUDED
    (``employment > 0``) so a real-level capture is not false-flagged against a
    bootstrap ``-1``. Per §7.2 this is a *diagnostic*: it returns the divergence; it
    does not assert it is zero ("replaceable, not identical").

    Returns one row per overlapping score-key with ``bootstrap_employment``,
    ``capture_employment``, ``abs_diff``.
    """
    score_key = [
        "ref_date", "industry_type", "industry_code", "geographic_type",
        "geographic_code", "revision", "benchmark_revision", "ownership",
    ]

    def _scored(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (pl.col("employment") > 0)
            & (pl.col("benchmark_revision") == 0)
            & (pl.col("revision").is_in([0, 1]))
        )

    b = _scored(bootstrap).select([*score_key, pl.col("employment").alias("bootstrap_employment")])
    c = _scored(capture).select([*score_key, pl.col("employment").alias("capture_employment")])
    return (
        b.join(c, on=score_key, how="inner", nulls_equal=True)
        .with_columns(
            (pl.col("capture_employment") - pl.col("bootstrap_employment")).abs().alias("abs_diff")
        )
        .sort("ref_date", "revision")
    )
