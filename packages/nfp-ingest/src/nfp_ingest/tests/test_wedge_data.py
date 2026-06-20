from datetime import date
from datetime import date as _d

import polars as pl
import pytest
from nfp_ingest import wedge_data
from nfp_lookups.government import GovIntervention


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


def _wedge_df():
    months = [_d(2025, m, 1) for m in range(1, 5)]            # Jan..Apr 2025
    return pl.DataFrame({"ref_date": months, "chg00": [0.0]*4, "chg05": [0.0]*4,
                         "wedge_change_k": [10.0, 12.0, -40.0, 8.0]})


def test_build_masks_target_and_builds_axis(monkeypatch):
    monkeypatch.setattr(wedge_data, "wedge_first_print_changes", lambda **k: _wedge_df())
    monkeypatch.setattr(wedge_data, "get_known_interventions_as_of", lambda a: [])
    d = wedge_data.build_wedge_model_data(
        as_of=_d(2025, 3, 20), target_month=_d(2025, 4, 1), start=_d(2025, 1, 1))
    assert d["T"] == 4 and d["target_idx"] == 3
    assert d["month_of_year"].tolist() == [1, 2, 3, 4]   # preserved for ALL rows
    assert not d["mask"][3]                               # target month not observed
    assert d["mask"][:3].all()                            # history observed


def test_lookahead_guard_excludes_unannounced_intervention(monkeypatch):
    monkeypatch.setattr(wedge_data, "wedge_first_print_changes", lambda **k: _wedge_df())
    rif = GovIntervention(_d(2025, 3, 1), "rif", "pulse", -40.0, 20.0,
                          announcement_date=_d(2025, 3, 10), source_url="u")
    # as_of BEFORE the announcement → no intervention column
    monkeypatch.setattr(wedge_data, "get_known_interventions_as_of",
                        lambda a: [rif] if a >= _d(2025, 3, 10) else [])
    before = wedge_data.build_wedge_model_data(
        as_of=_d(2025, 3, 5), target_month=_d(2025, 4, 1), start=_d(2025, 1, 1))
    assert before["X_intervention"].shape[1] == 0
    after = wedge_data.build_wedge_model_data(
        as_of=_d(2025, 3, 15), target_month=_d(2025, 4, 1), start=_d(2025, 1, 1))
    assert after["X_intervention"].shape[1] == 1
    assert after["iv_prior_mean"].tolist() == [-40.0]
