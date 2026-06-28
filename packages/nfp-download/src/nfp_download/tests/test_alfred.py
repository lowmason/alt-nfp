import os

import polars as pl
import pytest
from nfp_download.alfred import (
    CES_SERIES_SA,
    CES_SERIES_NSA,
    resolve_series_id,
    _matrix_from_observations,
    _title_matches,
)


def test_resolution_tables_cover_all_30_keys():
    expected = {
        ("total", "00"), ("total", "05"), ("domain", "06"), ("domain", "08"),
        *[("supersector", c) for c in
          ("10", "20", "30", "40", "50", "55", "60", "65", "70", "80")],
        *[("sector", c) for c in
          ("21", "22", "31", "32", "42", "44", "48", "52", "53", "54",
           "55", "56", "61", "62", "71", "72")],
    }
    assert set(CES_SERIES_SA) == expected
    assert set(CES_SERIES_NSA) == expected


def test_resolve_sa_aggregates_use_aliases():
    assert resolve_series_id("total", "00", sa=True) == "PAYEMS"
    assert resolve_series_id("supersector", "30", sa=True) == "MANEMP"
    assert resolve_series_id("sector", "42", sa=True) == "USWTRADE"


def test_resolve_nsa_systematic_and_paynsa():
    assert resolve_series_id("total", "00", sa=False) == "PAYNSA"
    assert resolve_series_id("supersector", "30", sa=False) == "CEU3000000001"
    assert resolve_series_id("sector", "42", sa=False) == "CEU4142000001"


def test_resolve_unknown_key_raises():
    with pytest.raises(KeyError):
        resolve_series_id("sector", "99", sa=True)


def test_title_matches_accepts_right_concept_rejects_swap():
    # The USSERV->08 mis-map that title-verify must catch: USSERV is "Other Services".
    assert _title_matches("supersector", "80", "All Employees, Other Services")
    assert not _title_matches("domain", "08", "All Employees, Other Services")
    assert _title_matches("domain", "08", "All Employees, Private Service-Providing")


def test_matrix_parses_sid_yyyymmdd_columns():
    # output_type=2 wide shape: date + {SID}_{YYYYMMDD} columns, "." = missing.
    import datetime
    obs = [
        {"date": "2026-02-01", "PAYEMS_20260306": "159000", "PAYEMS_20260403": "159050"},
        {"date": "2026-03-01", "PAYEMS_20260306": ".", "PAYEMS_20260403": "159200"},
    ]
    long = _matrix_from_observations(obs)
    assert long.columns == ["ref_date", "vintage_date", "value"]
    # 3 non-null cells (the "." dropped)
    assert long.height == 3
    feb = long.filter(pl.col("ref_date") == pl.date(2026, 2, 1)).sort("vintage_date")
    assert feb["vintage_date"].to_list() == [datetime.date(2026, 3, 6), datetime.date(2026, 4, 3)]
    assert feb["value"].to_list() == [159000.0, 159050.0]


@pytest.mark.network
def test_live_payems_resolves_and_fetches():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        pytest.skip("FRED_API_KEY not set")
    import httpx
    from nfp_download.alfred import get_vintage_dates, verify_series_concept
    with httpx.Client() as client:
        title, ok = verify_series_concept(client, "PAYEMS", api_key=key, sa=True)
        assert ok and "Total Nonfarm" in title
        vds = get_vintage_dates(client, "PAYEMS", api_key=key, start="2026-01-01")
        assert vds and all(v >= "2026-01-01" for v in vds)
