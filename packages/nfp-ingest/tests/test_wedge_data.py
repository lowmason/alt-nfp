from datetime import date

import polars as pl
import pytest
from nfp_ingest import wedge_data


def _fp(codes):  # build a fake first_print_changes keyed by industry_code
    tbl = {
        "00": pl.DataFrame({"ref_date": [date(2025, 1, 1), date(2025, 2, 1)],
                            "first_print_growth": [0.001, 0.001],
                            "first_print_change_k": [150.0, 160.0],
                            "vintage_date": [date(2025, 2, 6), date(2025, 3, 6)]}),
        "05": pl.DataFrame({"ref_date": [date(2025, 1, 1), date(2025, 2, 1)],
                            "first_print_growth": [0.001, 0.001],
                            "first_print_change_k": [130.0, 145.0],
                            "vintage_date": [date(2025, 2, 6), date(2025, 3, 6)]}),
    }
    def fake(*, store_path=None, geographic_type="national", geographic_code="00",
             industry_type="total", industry_code="00"):
        return tbl[industry_code]
    return fake


def test_wedge_is_00_minus_05(monkeypatch):
    monkeypatch.setattr(wedge_data, "first_print_changes", _fp(("00", "05")))
    df = wedge_data.wedge_first_print_changes()
    assert df["wedge_change_k"].to_list() == [20.0, 15.0]   # 150-130, 160-145


def test_mismatched_vintage_raises(monkeypatch):
    bad = _fp(("00", "05"))
    def fake(*, industry_code="00", **kw):
        d = bad(industry_code=industry_code, **kw)
        if industry_code == "05":  # drift one leg's vintage out of the release window
            d = d.with_columns(vintage_date=pl.Series([date(2025, 5, 1), date(2025, 6, 1)]))
        return d
    monkeypatch.setattr(wedge_data, "first_print_changes", fake)
    with pytest.raises(ValueError, match="vintage"):
        wedge_data.wedge_first_print_changes()
