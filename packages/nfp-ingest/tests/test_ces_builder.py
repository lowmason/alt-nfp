"""Tests for nfp_ingest.ces_builder — CES triangular-revision builder (T2).

Three layers, all offline:

1. *Synthetic* — a hand-built triangle (few vintages × few ref-months) whose
   diagonal extraction is computed by hand, asserting ``(0,0)/(1,0)/(2,0)``
   first-three-prints, the ``(2,1)`` = latest-value rule, day-12 ``ref_date``,
   ``VINTAGE_STORE_SCHEMA`` conformance, and ownership tagging.
2. *Taxonomy* — ``00``→``(total,total)``, ``05``→``(total,private)``, a sector →
   ``(sector,private)``; dropped government/total-services codes are absent.
3. *Local cesvinall cross-check* — reads the read-only cached ``cesvinall`` dir
   and asserts the known ``00`` NSA anchors for ref_date 2023-06-12. Self-skips
   when the local data is unavailable (CI has no ``data/``).
"""

from __future__ import annotations

from datetime import date

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
        Write the ``_SA`` (ignored-by-builder) variant instead of ``_NSA``.
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
    """Hand-built triangle: verify the four emitted (revision, bmr) values."""
    # Vintages (release rows) 2017-01 .. 2017-05; one ref-month column 2017-01.
    # Down the 2017-01 column the print history is:
    #   rev0 (released 2017-01) = 100.0
    #   rev1 (released 2017-02) = 101.0
    #   rev2 (released 2017-03) = 102.0
    #   later                   = 103.0
    #   latest (released 2017-05) = 104.0   <- (2,1) takes this
    vintages = [(2017, m) for m in range(1, 6)]
    ref_months = [(2017, 1)]
    cells = {(2017, 1): [100.0, 101.0, 102.0, 103.0, 104.0]}
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells)

    out = build_ces_panel(tmp_path)

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

    keyed = {
        (r["revision"], r["benchmark_revision"]): r["employment"]
        for r in out.iter_rows(named=True)
    }
    assert keyed[(0, 0)] == 100.0
    assert keyed[(1, 0)] == 101.0
    assert keyed[(2, 0)] == 102.0
    assert keyed[(2, 1)] == 104.0  # latest value down the column

    # Exactly four rows for one ref-month.
    assert out.height == 4

    # Vintage dates match the lookups helper (day-12 keys + Jan-Y+1 benchmark).
    assert keyed_vintage(out, 0, 0) == get_ces_vintage_date(date(2017, 1, 12), 0)
    assert keyed_vintage(out, 1, 0) == get_ces_vintage_date(date(2017, 1, 12), 1)
    assert keyed_vintage(out, 2, 0) == get_ces_vintage_date(date(2017, 1, 12), 2)
    assert keyed_vintage(out, 2, 1) == get_ces_vintage_date(date(2018, 1, 12), 0)


def keyed_vintage(df: pl.DataFrame, rev: int, bmr: int) -> date:
    """Return the ``vintage_date`` for one ``(revision, benchmark_revision)``."""
    row = df.filter(
        (pl.col("revision") == rev) & (pl.col("benchmark_revision") == bmr)
    )
    return row.item(0, "vintage_date")


def test_synthetic_ignores_sa_and_pre_2017(tmp_path):
    """SA files are ignored; ref_date < 2017-01-12 is dropped."""
    vintages = [(2016, m) for m in range(11, 13)] + [(2017, m) for m in range(1, 6)]
    ref_months = [(2016, 12), (2017, 1)]
    cells = {
        (2016, 12): [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
        (2017, 1): [None, None, 100.0, 101.0, 102.0, 103.0, 104.0],
    }
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells)
    # An SA file for the same code must be ignored entirely.
    _write_triangle(
        tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells, sa=True
    )

    out = build_ces_panel(tmp_path)

    # Only the 2017-01 ref-month survives the 2017+ coverage filter.
    assert set(out["ref_date"].unique().to_list()) == {date(2017, 1, 12)}
    keyed = {
        (r["revision"], r["benchmark_revision"]): r["employment"]
        for r in out.iter_rows(named=True)
    }
    assert keyed[(0, 0)] == 100.0
    assert keyed[(1, 0)] == 101.0
    assert keyed[(2, 0)] == 102.0
    assert keyed[(2, 1)] == 104.0


def test_frontier_benchmark_with_future_vintage_dropped(tmp_path):
    """A (2,1) row whose Jan-Y+1 benchmark has not published is dropped.

    The first-basis (0,0)/(1,0)/(2,0) prints for that ref-month survive.
    """
    # A 2099 ref-month: its annual benchmark (Jan-2100) is far in the future,
    # so the (2,1) date is a lag approximation and must be filtered out.
    vintages = [(2099, m) for m in range(1, 6)]
    ref_months = [(2099, 1)]
    cells = {(2099, 1): [100.0, 101.0, 102.0, 103.0, 104.0]}
    _write_triangle(tmp_path, "000000", vintages=vintages, ref_months=ref_months, cells=cells)

    out = build_ces_panel(tmp_path)

    keys = {
        (r["revision"], r["benchmark_revision"]) for r in out.iter_rows(named=True)
    }
    assert (0, 0) in keys
    assert (1, 0) in keys
    assert (2, 0) in keys
    assert (2, 1) not in keys  # future benchmark publication → dropped


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
    out = build_ces_panel(_CESVINALL)

    jun23 = out.filter(
        (pl.col("industry_code") == "00") & (pl.col("ref_date") == date(2023, 6, 12))
    )
    keyed = {
        (r["revision"], r["benchmark_revision"]): r["employment"]
        for r in jun23.iter_rows(named=True)
    }
    assert keyed[(0, 0)] == 156963.0
    assert keyed[(1, 0)] == 156945.0
    assert keyed[(2, 0)] == 156905.0
    assert keyed[(2, 1)] == 156701.0

    # Vintage dates are real (fixed, past) BLS calendar dates — regression-guard
    # the day-12 calendar hit and the Jan-Y+1 (2,1) benchmark derivation.
    assert keyed_vintage(jun23, 0, 0) == date(2023, 7, 7)  # real Friday, not lag-approx
    assert keyed_vintage(jun23, 2, 1) == date(2024, 2, 2)  # Jan-2024 first-print date

    # Anchor taxonomy + null size class on the real data too.
    assert (jun23["industry_type"] == "total").all()
    assert (jun23["ownership"] == "total").all()
    assert jun23["size_class_type"].is_null().all()
