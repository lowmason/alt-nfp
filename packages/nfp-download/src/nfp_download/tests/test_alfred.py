import pytest
from nfp_download.alfred import CES_SERIES_SA, CES_SERIES_NSA, resolve_series_id


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
