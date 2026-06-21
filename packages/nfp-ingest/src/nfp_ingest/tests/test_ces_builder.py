"""Tests for nfp_ingest.ces_builder — CES triangular-revision builder (T2).

Three layers, all offline:

1. *Synthetic* — a hand-built triangle (few vintages × few ref-months) whose
   diagonal extraction is computed by hand, asserting ``(0,0)/(1,0)/(2,0)``
   first-three-prints, the ``(2,1)`` = latest-value rule, day-12 ``ref_date``,
   ``VINTAGE_STORE_SCHEMA`` conformance, and ownership tagging.
2. *Taxonomy* — ``00``→``(total,total)``, ``05``→``(total,private)``, a sector →
   ``(sector,private)``; dropped government/total-services codes are absent.
3. *Local cesvinall cross-check* — reads the read-only cached ``cesvinall`` dir
   and asserts the known ``00`` NSA and SA anchors for ref_date 2023-06-12.
   Self-skips when the local data is unavailable (CI has no ``data/``).
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path as _Path

import polars as pl
import pytest
from nfp_ingest.ces_builder import build_ces_panel
from nfp_lookups.paths import DOWNLOADS_DIR
from nfp_lookups.revision_schedules import get_ces_vintage_date
from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

_CESVINALL = DOWNLOADS_DIR / "ces" / "cesvinall"

# Month name → number, matching the BLS ``Mon_YY`` triangular column headers.
_MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _write_triangle(
    tmp_dir,
    code6: str,
    *,
    vintages: list[tuple[int, int]],
    ref_months: list[tuple[int, int]],
    cells: dict[tuple[int, int], list[float | None]],
    sa: bool = False,
) -> None:
    """Write a synthetic ``tri_{code6}_{NSA|SA}.csv`` triangular frame.

    Parameters
    ----------
    tmp_dir : pathlib.Path
        Directory to write into.
    code6 : str
        Six-digit CES code (file stem ``tri_{code6}_NSA``).
    vintages : list[tuple[int, int]]
        ``(year, month)`` release rows, top-to-bottom.
    ref_months : list[tuple[int, int]]
        ``(year, month)`` reference columns, left-to-right; each becomes a
        ``Mon_YY`` header.
    cells : dict[tuple[int, int], list[float | None]]
        ref-month ``(year, month)`` → one value per vintage row (column read
        top-to-bottom). ``None`` ⇒ blank cell.
    sa : bool
        Write the ``_SA`` variant instead of ``_NSA``.
    """
    data: dict[str, list] = {
        "year": [v[0] for v in vintages],
        "month": [v[1] for v in vintages],
    }
    for (ry, rm) in ref_months:
        header = f"{_MONTHS[rm - 1]}_{ry % 100:02d}"
        data[header] = cells[(ry, rm)]
    suffix = "SA" if sa else "NSA"
    pl.DataFrame(data).write_csv(tmp_dir / f"tri_{code6}_{suffix}.csv")


# --- 1. Synthetic diagonal extraction ------------------------------------


def test_synthetic_diagonals_and_schema(tmp_path):
    """Hand-built triangle: verify the emitted (revision, bmr) values.

    Two annual-benchmark vintage rows ``(Y, 1)`` re-state the 2017-01 ref-month,
    so the column yields the three prints **plus two** ``(2,1)`` rows, one per
    distinct benchmark basis.
    """
    # Vintages (release rows) 2017-01 .. 2019-01; one ref-month column 2017-01.
    # Down the 2017-01 column the print history is:
    #   rev0 (released 2017-01) = 100.0
    #   rev1 (released 2017-02) = 101.0
    #   rev2 (released 2017-03) = 102.0
    # then two annual benchmarks re-state it:
    #   (2018,1) — the 2017 benchmark — = 103.0  -> (2,1) @ Jan-2018 release
    #   (2019,1) — the 2018 benchmark — = 105.0  -> (2,1) @ Jan-2019 release
    vintages = [(2017, m) for m in range(1, 13)] + [(2018, 1), (2019, 1)]
    ref_months = [(2017, 1)]
    col = [100.0, 101.0, 102.0] + [None] * 9 + [103.0, 105.0]
    cells = {(2017, 1): col}
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells)

    out = build_ces_panel(tmp_path, as_of=date(2026, 1, 1))

    # Schema conformance (partition cols carried as plain columns).
    assert set(out.columns) == set(VINTAGE_STORE_SCHEMA)
    for col, dtype in VINTAGE_STORE_SCHEMA.items():
        assert out.schema[col] == dtype, f"{col}: {out.schema[col]} != {dtype}"

    # All rows: NSA, ces source, national geography, null size class.
    assert (out["source"] == "ces").all()
    assert (~out["seasonally_adjusted"]).all()
    assert (out["geographic_type"] == "national").all()
    assert (out["geographic_code"] == "00").all()
    assert out["size_class_type"].is_null().all()
    assert out["size_class_code"].is_null().all()

    # Total nonfarm anchor → (total, total).
    assert (out["industry_type"] == "total").all()
    assert (out["ownership"] == "total").all()

    # ref_date is day-12 of the reference month.
    assert (out["ref_date"] == date(2017, 1, 12)).all()

    # The three first-basis prints are unique-keyed; check them directly.
    prints = {
        (r["revision"], r["benchmark_revision"]): r["employment"]
        for r in out.iter_rows(named=True)
        if r["benchmark_revision"] == 0
    }
    assert prints[(0, 0)] == 100.0
    assert prints[(1, 0)] == 101.0
    assert prints[(2, 0)] == 102.0

    # Per-benchmark (2,1) rows: value ↔ date pairs, one per distinct basis.
    pairs = {
        (r["employment"], r["vintage_date"])
        for r in out.iter_rows(named=True)
        if (r["revision"], r["benchmark_revision"]) == (2, 1)
    }
    assert pairs == {
        (103.0, get_ces_vintage_date(date(2018, 1, 12), 0)),  # 2017 benchmark
        (105.0, get_ces_vintage_date(date(2019, 1, 12), 0)),  # 2018 benchmark
    }

    # Three prints + two per-benchmark (2,1) rows.
    assert out.height == 5

    # First-basis vintage dates match the lookups helper (day-12 keys).
    assert keyed_vintage(out, 0, 0) == get_ces_vintage_date(date(2017, 1, 12), 0)
    assert keyed_vintage(out, 1, 0) == get_ces_vintage_date(date(2017, 1, 12), 1)
    assert keyed_vintage(out, 2, 0) == get_ces_vintage_date(date(2017, 1, 12), 2)


def test_synthetic_unchanged_benchmark_not_duplicated(tmp_path):
    """A later benchmark that leaves the value unchanged emits no extra (2,1)."""
    # Three benchmarks: 2017→103.0, 2018→103.0 (unchanged), 2019→106.0.
    vintages = [(2017, m) for m in range(1, 13)] + [(2018, 1), (2019, 1), (2020, 1)]
    ref_months = [(2017, 1)]
    col = [100.0, 101.0, 102.0] + [None] * 9 + [103.0, 103.0, 106.0]
    cells = {(2017, 1): col}
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells)

    out = build_ces_panel(tmp_path, as_of=date(2026, 1, 1))

    pairs = {
        (r["employment"], r["vintage_date"])
        for r in out.iter_rows(named=True)
        if (r["revision"], r["benchmark_revision"]) == (2, 1)
    }
    # The unchanged 2018 benchmark (Jan-2019 release) is NOT emitted.
    assert pairs == {
        (103.0, get_ces_vintage_date(date(2018, 1, 12), 0)),  # 2017 benchmark
        (106.0, get_ces_vintage_date(date(2020, 1, 12), 0)),  # 2019 benchmark
    }


def test_as_of_excludes_then_includes_benchmark(tmp_path):
    """A fixed ``as_of`` gates (2,1) rows by their benchmark release date.

    Deterministic — no ``date.today()``: before the Jan-2019 release excludes
    the 2018-benchmark (2,1); after it includes it. Prints always retained.
    """
    vintages = [(2017, m) for m in range(1, 13)] + [(2018, 1), (2019, 1)]
    ref_months = [(2017, 1)]
    col = [100.0, 101.0, 102.0] + [None] * 9 + [103.0, 105.0]
    cells = {(2017, 1): col}
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells)

    jan2019_release = get_ces_vintage_date(date(2019, 1, 12), 0)

    # as_of strictly before the Jan-2019 release: only the 2017-benchmark (2,1).
    before = build_ces_panel(tmp_path, as_of=jan2019_release - timedelta(days=1))
    before_pairs = {
        (r["employment"], r["vintage_date"])
        for r in before.iter_rows(named=True)
        if (r["revision"], r["benchmark_revision"]) == (2, 1)
    }
    assert before_pairs == {(103.0, get_ces_vintage_date(date(2018, 1, 12), 0))}
    # Prints survive the as_of gate.
    assert before.filter(pl.col("benchmark_revision") == 0).height == 3

    # as_of on the Jan-2019 release date: both (2,1) rows present.
    after = build_ces_panel(tmp_path, as_of=jan2019_release)
    after_pairs = {
        (r["employment"], r["vintage_date"])
        for r in after.iter_rows(named=True)
        if (r["revision"], r["benchmark_revision"]) == (2, 1)
    }
    assert after_pairs == {
        (103.0, get_ces_vintage_date(date(2018, 1, 12), 0)),
        (105.0, jan2019_release),
    }


def keyed_vintage(df: pl.DataFrame, rev: int, bmr: int) -> date:
    """Return the ``vintage_date`` for one ``(revision, benchmark_revision)``."""
    row = df.filter(
        (pl.col("revision") == rev) & (pl.col("benchmark_revision") == bmr)
    )
    return row.item(0, "vintage_date")


def test_synthetic_builds_sa_and_drops_pre_2017(tmp_path):
    """SA files are built alongside NSA; ref_date < 2017-01-12 is dropped.

    The SA triangle here carries the same cell values as NSA (to keep the
    fixture simple), so both adjustments should produce the same employment
    numbers — the only distinction is the ``seasonally_adjusted`` column.
    """
    # 2016-12 prints + the 2017 benchmark vintage (2018,1) re-stating both.
    vintages = (
        [(2016, m) for m in range(11, 13)]
        + [(2017, m) for m in range(1, 13)]
        + [(2018, 1)]
    )
    n = len(vintages)
    ref_months = [(2016, 12), (2017, 1)]
    # 2016-12 column: rev0/1/2 start at the (2016,12) vintage row (index 1).
    col_2016_12 = [None, 50.0, 51.0, 52.0] + [None] * (n - 5) + [54.0]
    # 2017-01 column: rev0/1/2 start at the (2017,1) vintage row (index 2).
    col_2017_01 = [None, None, 100.0, 101.0, 102.0] + [None] * (n - 6) + [104.0]
    cells = {(2016, 12): col_2016_12, (2017, 1): col_2017_01}
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells)
    _write_triangle(
        tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells, sa=True
    )

    out = build_ces_panel(tmp_path, as_of=date(2026, 1, 1))

    # Only the 2017-01 ref-month survives the 2017+ coverage filter (both SA and NSA).
    assert set(out["ref_date"].unique().to_list()) == {date(2017, 1, 12)}

    # Both SA and NSA are now present.
    assert set(out["seasonally_adjusted"].unique().to_list()) == {True, False}

    # NSA subset matches the expected 2017-01 prints/benchmark.
    nsa = out.filter(~pl.col("seasonally_adjusted"))
    nsa_prints = {
        (r["revision"], r["benchmark_revision"]): r["employment"]
        for r in nsa.iter_rows(named=True)
        if r["benchmark_revision"] == 0
    }
    assert nsa_prints[(0, 0)] == 100.0
    assert nsa_prints[(1, 0)] == 101.0
    assert nsa_prints[(2, 0)] == 102.0
    nsa_pairs = {
        (r["employment"], r["vintage_date"])
        for r in nsa.iter_rows(named=True)
        if (r["revision"], r["benchmark_revision"]) == (2, 1)
    }
    assert nsa_pairs == {(104.0, get_ces_vintage_date(date(2018, 1, 12), 0))}

    # SA subset: same cohort set as NSA (same cells in this fixture).
    sa = out.filter(pl.col("seasonally_adjusted"))
    sa_prints = {
        (r["revision"], r["benchmark_revision"]): r["employment"]
        for r in sa.iter_rows(named=True)
        if r["benchmark_revision"] == 0
    }
    assert sa_prints[(0, 0)] == 100.0
    assert sa_prints[(1, 0)] == 101.0
    assert sa_prints[(2, 0)] == 102.0


def test_sa_and_nsa(tmp_path):
    """build_ces_panel emits both SA and NSA rows; NSA subset unchanged.

    Fixture: a single ``00`` triangle written for both NSA and SA with
    distinguishable employment values (SA values are NSA + 1000).
    Asserts:
    - SA rows have ``seasonally_adjusted=True``; NSA rows False.
    - SA and NSA carry the same ``(revision, benchmark_revision)`` cohorts.
    - SA rows have null ``size_class_type`` / ``size_class_code``.
    - NSA subset is identical to a build from NSA-only dir (regression guard).
    """
    vintages = [(2017, m) for m in range(1, 13)] + [(2018, 1), (2019, 1)]
    ref_months = [(2017, 1)]
    # NSA cell values: 100, 101, 102 prints; 103 and 105 benchmarks.
    nsa_col = [100.0, 101.0, 102.0] + [None] * 9 + [103.0, 105.0]
    # SA cell values: NSA + 1000 so they're easily distinguished.
    sa_col = [1100.0, 1101.0, 1102.0] + [None] * 9 + [1103.0, 1105.0]

    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells={(2017, 1): nsa_col})
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells={(2017, 1): sa_col}, sa=True)

    # Build from combined dir (SA + NSA).
    out = build_ces_panel(tmp_path, as_of=date(2026, 1, 1))

    # Both adjustments present.
    adj_values = set(out["seasonally_adjusted"].unique().to_list())
    assert adj_values == {True, False}

    sa = out.filter(pl.col("seasonally_adjusted"))
    nsa = out.filter(~pl.col("seasonally_adjusted"))

    # SA and NSA carry the same (revision, benchmark_revision) cohort set.
    sa_cohorts = set(sa.select("revision", "benchmark_revision").rows())
    nsa_cohorts = set(nsa.select("revision", "benchmark_revision").rows())
    assert sa_cohorts == nsa_cohorts

    # SA rows have null size class (same as NSA).
    assert sa["size_class_type"].is_null().all()
    assert sa["size_class_code"].is_null().all()

    # SA employment values differ from NSA (the +1000 distinguisher).
    sa_emp = set(sa["employment"].to_list())
    nsa_emp = set(nsa["employment"].to_list())
    assert sa_emp.isdisjoint(nsa_emp)

    # Regression guard: NSA-only dir produces the same NSA rows.
    with tempfile.TemporaryDirectory() as nsa_only_dir:
        nsa_only = _Path(nsa_only_dir)
        # Copy only the NSA file.
        for f in tmp_path.glob("tri_*_NSA.csv"):
            shutil.copy(f, nsa_only / f.name)
        out_nsa_only = build_ces_panel(nsa_only, as_of=date(2026, 1, 1))

    # The NSA subset from the combined build equals the NSA-only build output.
    # Sort both to make comparison order-independent.
    sort_cols = ["industry_type", "industry_code", "ref_date", "revision", "benchmark_revision", "vintage_date", "seasonally_adjusted"]
    nsa_from_combined = nsa.sort(sort_cols)
    nsa_only_sorted = out_nsa_only.sort(sort_cols)
    assert nsa_from_combined.equals(nsa_only_sorted)


def test_frontier_benchmark_with_future_vintage_dropped(tmp_path):
    """A (2,1) row whose benchmark release post-dates ``as_of`` is dropped.

    The first-basis (0,0)/(1,0)/(2,0) prints for that ref-month survive.
    """
    # A 2099 ref-month: its annual benchmark (Jan-2100) post-dates the fixed
    # as_of, so the (2,1) row must be filtered out while the prints survive.
    vintages = [(2099, m) for m in range(1, 13)] + [(2100, 1)]
    n = len(vintages)
    ref_months = [(2099, 1)]
    col = [100.0, 101.0, 102.0] + [None] * (n - 4) + [104.0]
    cells = {(2099, 1): col}
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells)

    out = build_ces_panel(tmp_path, as_of=date(2026, 1, 1))

    keys = {
        (r["revision"], r["benchmark_revision"]) for r in out.iter_rows(named=True)
    }
    assert (0, 0) in keys
    assert (1, 0) in keys
    assert (2, 0) in keys
    assert (2, 1) not in keys  # benchmark release after as_of → dropped


# --- 2. Taxonomy mapping --------------------------------------------------


def _single_ref_triangle(tmp_path, code6: str) -> None:
    """A minimal one-ref-month triangle for taxonomy tests."""
    vintages = [(2017, m) for m in range(1, 6)]
    ref_months = [(2017, 1)]
    cells = {(2017, 1): [100.0, 101.0, 102.0, 103.0, 104.0]}
    _write_triangle(tmp_path, code6, vintages=vintages, ref_months=ref_months, cells=cells)


def test_taxonomy_total_total(tmp_path):
    """``00`` total-nonfarm anchor → (total, total)."""
    _single_ref_triangle(tmp_path, "000000")
    out = build_ces_panel(tmp_path)
    assert (out["industry_type"] == "total").all()
    assert (out["ownership"] == "total").all()
    assert (out["industry_code"] == "00").all()


def test_taxonomy_total_private(tmp_path):
    """``05`` total-private → (total, private)."""
    _single_ref_triangle(tmp_path, "050000")
    out = build_ces_panel(tmp_path)
    assert (out["industry_type"] == "total").all()
    assert (out["ownership"] == "private").all()
    assert (out["industry_code"] == "05").all()


def test_taxonomy_sector_private_with_naics_recode(tmp_path):
    """CES sector 414200 → NAICS sector 42, ownership private."""
    _single_ref_triangle(tmp_path, "414200")  # CES Wholesale (code 41) → NAICS 42
    out = build_ces_panel(tmp_path)
    assert (out["industry_type"] == "sector").all()
    assert (out["ownership"] == "private").all()
    assert (out["industry_code"] == "42").all()


def test_taxonomy_drops_government_and_total_services(tmp_path):
    """Codes 07, 90, 91, 92, 93 are dropped entirely."""
    # One kept + all dropped codes present in the dir.
    _single_ref_triangle(tmp_path, "050000")  # kept → 05
    for ces6 in ("070000", "900000", "909100", "909200", "909300"):
        _single_ref_triangle(tmp_path, ces6)

    out = build_ces_panel(tmp_path)

    assert set(out["industry_code"].unique().to_list()) == {"05"}
    for dropped in ("07", "90", "91", "92", "93"):
        assert dropped not in out["industry_code"].to_list()


# --- 3. Local cesvinall cross-check (offline, read-only) ------------------


@pytest.mark.skipif(
    not (_CESVINALL / "tri_000000_NSA.csv").exists(),
    reason="local cesvinall data not available (CI runs without data/)",
)
def test_local_cesvinall_total_nonfarm_anchors():
    """Known ``00`` NSA anchors for ref_date 2023-06-12 from the real triangle."""
    # Fixed as_of past both benchmark releases so the (2,1) set is deterministic.
    out = build_ces_panel(_CESVINALL, as_of=date(2026, 1, 1))

    jun23 = out.filter(
        (pl.col("industry_code") == "00")
        & (pl.col("ref_date") == date(2023, 6, 12))
        & (~pl.col("seasonally_adjusted"))
    )
    prints = {
        (r["revision"], r["benchmark_revision"]): r["employment"]
        for r in jun23.iter_rows(named=True)
        if r["benchmark_revision"] == 0
    }
    assert prints[(0, 0)] == 156963.0
    assert prints[(1, 0)] == 156945.0
    assert prints[(2, 0)] == 156905.0

    # Per-benchmark (2,1) rows: value ↔ benchmark-release date pairs.
    #   2023 benchmark (Jan-2024 release, 2024-02-02): 156842
    #   2024 benchmark (Jan-2025 release, 2025-02-07): 156701
    pairs = {
        (r["employment"], r["vintage_date"])
        for r in jun23.iter_rows(named=True)
        if (r["revision"], r["benchmark_revision"]) == (2, 1)
    }
    assert pairs == {
        (156842.0, date(2024, 2, 2)),
        (156701.0, date(2025, 2, 7)),
    }

    # First-print vintage date is a real (fixed, past) BLS calendar date.
    assert keyed_vintage(jun23, 0, 0) == date(2023, 7, 7)  # real Friday, not lag-approx

    # Anchor taxonomy + null size class on the real data too.
    assert (jun23["industry_type"] == "total").all()
    assert (jun23["ownership"] == "total").all()
    assert jun23["size_class_type"].is_null().all()


@pytest.mark.skipif(
    not (_CESVINALL / "tri_000000_SA.csv").exists(),
    reason="local cesvinall data not available (CI runs without data/)",
)
def test_local_cesvinall_total_nonfarm_sa_anchors():
    """Known ``00`` SA anchors for ref_date 2023-06-12 from the real triangle.

    SA and NSA share the same ``(revision, benchmark_revision)`` cohort set and
    vintage_date calendar; only the employment values differ.

    Anchors read directly from ``tri_000000_SA.csv`` Jun_23 column:
    - rev0 = 156204 (2023-06 vintage row)
    - rev1 = 156155 (2023-07 vintage row)
    - rev2 = 156075 (2023-08 vintage row)
    - (2,1) bench_year=2024: 156027 at 2024-02-02
    - (2,1) bench_year=2025: 155871 at 2025-02-07
    - (2,1) bench_year=2026: 155880 at 2026-02-11 → filtered by as_of=2026-01-01
    """
    out = build_ces_panel(_CESVINALL, as_of=date(2026, 1, 1))

    jun23_sa = out.filter(
        (pl.col("industry_code") == "00")
        & (pl.col("ref_date") == date(2023, 6, 12))
        & (pl.col("seasonally_adjusted"))
    )

    prints = {
        (r["revision"], r["benchmark_revision"]): r["employment"]
        for r in jun23_sa.iter_rows(named=True)
        if r["benchmark_revision"] == 0
    }
    assert prints[(0, 0)] == 156204.0
    assert prints[(1, 0)] == 156155.0
    assert prints[(2, 0)] == 156075.0

    # Per-benchmark (2,1) rows: bench_year 2026 is filtered by as_of=2026-01-01
    # (its vintage_date 2026-02-11 > 2026-01-01), so only 2 rows remain.
    pairs = {
        (r["employment"], r["vintage_date"])
        for r in jun23_sa.iter_rows(named=True)
        if (r["revision"], r["benchmark_revision"]) == (2, 1)
    }
    assert pairs == {
        (156027.0, date(2024, 2, 2)),
        (155871.0, date(2025, 2, 7)),
    }

    # SA shares the same vintage_date calendar as NSA for first-print.
    assert keyed_vintage(jun23_sa, 0, 0) == date(2023, 7, 7)

    # Taxonomy + null size class.
    assert (jun23_sa["industry_type"] == "total").all()
    assert (jun23_sa["ownership"] == "total").all()
    assert jun23_sa["size_class_type"].is_null().all()
    assert jun23_sa["size_class_code"].is_null().all()
