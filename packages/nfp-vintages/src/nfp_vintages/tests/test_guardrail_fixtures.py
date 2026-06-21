"""Shape checks for the synthetic guardrail fixtures (§7 required fixtures)."""

from __future__ import annotations

from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA
from nfp_vintages.tests._fixtures import (
    make_benchmark_double_row,
    make_ces_rows,
    make_shutdown_sentinel_row,
)

_COLS = list(VINTAGE_STORE_SCHEMA.keys())


def test_make_ces_rows_one_schema_row():
    df = make_ces_rows(ref_month="2026-01-12", vintage="2026-02-06")
    assert df.height == 1
    assert df.columns == _COLS
    assert df.schema == VINTAGE_STORE_SCHEMA
    # the headline default targets private '05' (ownership=private)
    assert df["industry_code"].item() == "05"
    assert df["ownership"].item() == "private"


def test_make_benchmark_double_row_two_coherent_tracks():
    df = make_benchmark_double_row(ref_month="2025-12-12")
    assert df.height == 2
    assert df.columns == _COLS
    keys = set(zip(df["revision"].to_list(), df["benchmark_revision"].to_list(), strict=True))
    assert keys == {(1, 0), (2, 1)}
    # both rows are the same ref_date
    assert df["ref_date"].n_unique() == 1


def test_make_shutdown_sentinel_row_literal_minus_one():
    df = make_shutdown_sentinel_row(ref_month="2025-10-12")
    assert df.height == 1
    assert df.columns == _COLS
    assert df["employment"].item() == -1.0
