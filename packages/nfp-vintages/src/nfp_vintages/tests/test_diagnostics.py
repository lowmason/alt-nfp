# packages/nfp-vintages/tests/test_diagnostics.py
from datetime import date as _d

import numpy as np
import polars as pl
import pytest
from nfp_lookups.paths import VINTAGE_STORE_PATH
from nfp_vintages.diagnostics import OLSResult, _join_revision, build_revision_table, ols
from pytest import approx


def _store_available() -> bool:
    """True if the vintage store is reachable and has SA CES data."""
    try:
        sa_path = VINTAGE_STORE_PATH / "source=ces" / "seasonally_adjusted=true"
        return sa_path.exists() and (
            next(sa_path.glob("**/*.parquet"), None) is not None
        )
    except Exception:
        return False


def test_ols_recovers_known_line():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + rng.normal(scale=1e-6, size=n)
    X = np.column_stack([np.ones(n), x])
    res = ols(X, y)
    assert isinstance(res, OLSResult)
    assert res.coeffs[0] == approx(2.0, abs=1e-3)
    assert res.coeffs[1] == approx(3.0, abs=1e-3)
    assert res.r2 == approx(1.0, abs=1e-6)
    assert res.cov.shape == (2, 2)


def test_ols_r2_zero_for_constant_target_no_slope():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    y = np.full(10, 5.0)
    res = ols(X, y)
    assert res.coeffs[1] == approx(0.0, abs=1e-9)
    assert res.r2 == approx(0.0, abs=1e-9)


def test_join_revision_pure():
    fp = pl.DataFrame({"ref_date": [_d(2023, 1, 1), _d(2023, 2, 1)],
                       "first_print_change_k": [100.0, 150.0]})
    later = pl.DataFrame({"ref_date": [_d(2023, 1, 1), _d(2023, 2, 1)],
                          "later_change_k": [120.0, 140.0]})
    out = _join_revision(fp, later)
    by = {r["ref_date"]: r for r in out.iter_rows(named=True)}
    assert by[_d(2023, 1, 1)]["revision_k"] == pytest.approx(20.0)   # 120 - 100
    assert by[_d(2023, 2, 1)]["revision_k"] == pytest.approx(-10.0)  # 140 - 150


@pytest.mark.real_store
@pytest.mark.skipif(
    not _store_available(),
    reason="Vintage store not available (no SA CES data in store)",
)
def test_build_revision_table_real():
    # Defaults to PRIVATE '05' (Track A): the model nowcasts private NFP.
    tbl = build_revision_table()
    assert {"ref_date", "first_print_change_k", "later_change_k", "revision_k"} <= set(tbl.columns)
    assert tbl.height > 0
    # First-to-third PRIVATE revisions are literature-plausible: tens of k, NOT the
    # ~-223k garbage intercept a cross-gap shift(1) diff produced before the
    # adjacency guard. The pooled |revision| stays well under 100k.
    rev = tbl["revision_k"].drop_nulls().to_numpy()
    assert rev.size > 0
    assert float(np.abs(rev).mean()) < 100.0


@pytest.mark.real_store
@pytest.mark.skipif(
    not _store_available(),
    reason="Vintage store not available (no SA CES data in store)",
)
def test_third_print_changes_gap_safe_adjacent_only():
    # The gap-safe third-print change must NEVER diff across a month gap. Recover the
    # third-print LEVEL months independently, then assert every returned change sits
    # on an adjacent (exactly-one-month-apart) level pair. A naive shift(1) over a
    # store that omits months (e.g. shutdown holes) would diff across a gap and fail.
    from nfp_ingest.vintage_store import read_vintage_store
    from nfp_lookups.paths import VINTAGE_STORE_PATH
    from nfp_vintages.diagnostics import _third_print_changes

    df = _third_print_changes()
    assert {"ref_date", "later_change_k"} <= set(df.columns)
    assert df.height > 0

    level_months = set(
        read_vintage_store(
            VINTAGE_STORE_PATH, source="ces", seasonally_adjusted=True,
            geographic_type="national", geographic_code="00",
            industry_type="total", industry_code="05",
        )
        .collect()
        .filter((pl.col("benchmark_revision") == 0) & (pl.col("employment") > 0))
        .with_columns(pl.col("ref_date").dt.truncate("1mo"))["ref_date"]
        .to_list()
    )
    # Each returned change at month M requires M and its exact prior month to both be
    # real third-print level months — i.e. no gap was bridged.
    for m in df["ref_date"].to_list():
        prev = pl.Series([m]).dt.offset_by("-1mo")[0]
        assert m in level_months
        assert prev in level_months, f"change at {m} bridged a month gap (prev {prev} absent)"


from nfp_vintages.diagnostics import qcew_settled_changes  # noqa: E402


@pytest.mark.real_store
@pytest.mark.skipif(not _store_available(), reason="vintage store unavailable")
def test_qcew_settled_changes_shape():
    df = qcew_settled_changes()
    assert {"ref_date", "qcew_settled_change_k"} <= set(df.columns)
    assert df.height > 0
    # change values are in a sane band (thousands per month); the plan's original
    # threshold of 5000 is exceeded by the COVID Apr-2020 crash (-21,923k), which
    # is real data. Threshold raised to 30000 to accommodate COVID while still
    # catching unit errors (e.g., values in persons instead of thousands).
    vals = df["qcew_settled_change_k"].drop_nulls().to_numpy()
    assert (abs(vals) < 30000).all()


# ---------------------------------------------------------------------------
# Task 8: Aruoba design matrix
# ---------------------------------------------------------------------------
from nfp_vintages.diagnostics import build_aruoba_design  # noqa: E402


def test_build_aruoba_design_skeleton():
    ref = [_d(2023, m, 1) for m in range(1, 7)]
    regressors = {
        "claims_mom": {m: float(i) for i, m in enumerate(ref)},
        "jolts": {m: 9_000.0 + i for i, m in enumerate(ref)},
        "lagged_revision": {m: float(-i) for i, m in enumerate(ref)},
    }
    X, names, used = build_aruoba_design(ref, regressors)
    assert X.shape == (6, len(names))
    assert names[0] == "const"
    assert set(used) == {"claims_mom", "jolts", "lagged_revision"}
    assert np.allclose(X[:, 0], 1.0)  # intercept column


def test_build_aruoba_design_drops_all_nan_regressor():
    ref = [_d(2023, m, 1) for m in range(1, 4)]
    regressors = {
        "claims_mom": dict.fromkeys(ref, 1.0),
        "nfci": dict.fromkeys(ref, float("nan")),  # absent locally → dropped
    }
    X, names, used = build_aruoba_design(ref, regressors)
    assert "nfci" not in names
    assert "nfci" not in used


# ---------------------------------------------------------------------------
# Task 9: Aruoba revision regression
# ---------------------------------------------------------------------------
from nfp_vintages.diagnostics import AruobaResult, aruoba_regression  # noqa: E402


def test_aruoba_recovers_intercept_and_r2():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    # revision = 12 (bias) + 4*x + small noise → R^2 high, intercept ~12.
    rev = 12.0 + 4.0 * x + rng.normal(scale=0.01, size=n)
    X = np.column_stack([np.ones(n), x])
    res = aruoba_regression(rev, X, ["const", "x"])
    assert isinstance(res, AruobaResult)
    assert res.intercept_k == approx(12.0, abs=0.1)
    assert res.r2 > 0.99


def test_aruoba_low_r2_for_pure_noise():
    rng = np.random.default_rng(2)
    n = 300
    rev = rng.normal(scale=20.0, size=n)         # unforecastable
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    res = aruoba_regression(rev, X, ["const", "x"])
    assert res.r2 < 0.1                           # below the gate threshold


# ---------------------------------------------------------------------------
# Task 10: Mincer–Zarnowitz efficiency regression
# ---------------------------------------------------------------------------
from nfp_vintages.diagnostics import MZResult, mincer_zarnowitz  # noqa: E402


def test_mz_efficient_forecast_not_rejected():
    rng = np.random.default_rng(3)
    forecast = rng.normal(100.0, 50.0, 400)
    actual = forecast + rng.normal(0.0, 1.0, 400)   # efficient: alpha~0, beta~1
    res = mincer_zarnowitz(actual, forecast)
    assert isinstance(res, MZResult)
    assert res.alpha == approx(0.0, abs=2.0)
    assert res.beta == approx(1.0, abs=0.05)
    assert res.joint_p > 0.05                         # null not rejected


def test_mz_biased_forecast_rejected():
    rng = np.random.default_rng(4)
    forecast = rng.normal(100.0, 50.0, 400)
    actual = 30.0 + 0.5 * forecast + rng.normal(0.0, 1.0, 400)  # inefficient
    res = mincer_zarnowitz(actual, forecast)
    assert res.joint_p < 0.01                         # null rejected


# ---------------------------------------------------------------------------
# Task 11: Gate decision
# ---------------------------------------------------------------------------
from nfp_vintages.diagnostics import GateConfig, gate_decision  # noqa: E402


def test_gate_diagonal_adequate_when_normal_r2_low():
    r2_by_type = {"normal": 0.04, "turning_point": 0.05, "benchmark_window": 0.06}
    g = gate_decision(r2_by_type, GateConfig())
    assert g["fund_first_release_rebuild"] is False
    assert g["fund_tier3_bd"] is False


def test_gate_funds_bd_when_turning_point_r2_concentrated():
    r2_by_type = {"normal": 0.05, "turning_point": 0.40, "benchmark_window": 0.08}
    g = gate_decision(r2_by_type, GateConfig())
    assert g["fund_tier3_bd"] is True
    assert "turning_point" in g["rationale"]


# ---------------------------------------------------------------------------
# §5A: pooled first-print bias (the post-hoc offset δ)
# ---------------------------------------------------------------------------
from nfp_vintages.diagnostics import pooled_first_print_bias  # noqa: E402


def test_pooled_first_print_bias_median_resists_outlier():
    # Central first-print bias ≈ -8k with one extreme benchmark-month outlier
    # (+871k, the real 2022-11 row). The robust median ignores it; the mean is
    # contaminated. This is exactly the §5A δ-contamination guard.
    rev = pl.DataFrame({
        "ref_date": [_d(2023, m, 1) for m in range(1, 8)],
        "revision_k": [-8.0, -7.0, -9.0, -8.0, -8.0, -7.0, 871.0],
    })
    assert pooled_first_print_bias(rev, method="median") == approx(-8.0)
    assert pooled_first_print_bias(rev) == approx(-8.0)          # median is the default
    assert pooled_first_print_bias(rev, method="mean") > 100.0   # contaminated


def test_pooled_first_print_bias_drops_null_revisions():
    rev = pl.DataFrame({"revision_k": [-8.0, None, -8.0, -8.0]})
    assert pooled_first_print_bias(rev, method="median") == approx(-8.0)
    assert pooled_first_print_bias(rev, method="mean") == approx(-8.0)


# ---------------------------------------------------------------------------
# Task 1: Implied-government consensus
# ---------------------------------------------------------------------------


def test_implied_government_consensus_is_total_minus_private():
    from nfp_vintages.diagnostics import implied_government_consensus

    tbl = pl.DataFrame({
        "ownership": ["total", "private", "total", "private"],
        "industry_type": ["total"] * 4,
        "industry_code": ["00", "05", "00", "05"],
        "ref_date": [_d(2024, 1, 1), _d(2024, 1, 1), _d(2024, 2, 1), _d(2024, 2, 1)],
        "release_date": [_d(2024, 2, 2), _d(2024, 2, 2), _d(2024, 3, 8), _d(2024, 3, 8)],
        "consensus_mean": [180.0, 160.0, 200.0, 175.0],
        "consensus_median": [185.0, 165.0, 210.0, 180.0],
    })
    out = implied_government_consensus(tbl)  # median by default
    assert out.columns == ["ref_date", "release_date", "implied_govt_k"]
    assert out.height == 2
    got = dict(zip(out["ref_date"].to_list(), out["implied_govt_k"].to_list(), strict=True))
    assert got[_d(2024, 1, 1)] == 185.0 - 165.0   # 20.0
    assert got[_d(2024, 2, 1)] == 210.0 - 180.0   # 30.0


# ---------------------------------------------------------------------------
# Task 2: Forecast-encompassing + Bates–Granger weight
# ---------------------------------------------------------------------------


def test_encompassing_returns_none_below_min_obs():
    from nfp_vintages.diagnostics import encompassing
    assert encompassing(np.arange(4.0), np.arange(4.0), np.arange(4.0)) is None


def test_encompassing_model_adds_info_when_consensus_is_noise():
    from nfp_vintages.diagnostics import encompassing
    rng = np.random.default_rng(0)
    actual = rng.normal(150, 50, 60)
    model = actual + rng.normal(0, 5, 60)      # model tracks actual tightly
    consensus = rng.normal(150, 50, 60)        # consensus ~ pure noise
    r = encompassing(actual, model, consensus)
    assert r is not None and r.n == 60
    assert r.p_model_adds_info < 0.05          # model clearly adds info
    assert r.w_model > 0.8                     # weight piles onto the model
    assert r.combo_mae <= min(r.model_mae, r.consensus_mae) + 1e-6


def test_encompassing_consensus_encompasses_model_when_model_is_noise():
    from nfp_vintages.diagnostics import encompassing
    rng = np.random.default_rng(1)
    actual = rng.normal(150, 50, 60)
    consensus = actual + rng.normal(0, 5, 60)  # consensus tracks actual
    model = rng.normal(150, 50, 60)            # model ~ pure noise
    r = encompassing(actual, model, consensus)
    assert r is not None
    assert r.p_model_adds_info > 0.10          # cannot reject b == 0
    assert r.w_model < 0.2                      # weight piles onto consensus


# ---------------------------------------------------------------------------
# Task 3: Per-cell combination gate (month-type × horizon)
# ---------------------------------------------------------------------------


def test_combination_gate_fires_only_where_model_adds_info_and_combo_wins():
    from nfp_vintages.diagnostics import combination_gate
    rng = np.random.default_rng(2)
    # turning_point/t1: BOTH forecasts informative, model better → interior optimum
    # (w_model strictly between 0 and 1) so the blend strictly beats both → fires.
    a_tp = rng.normal(0, 80, 60)
    m_tp = a_tp + rng.normal(0, 20, 60)   # model: good signal
    c_tp = a_tp + rng.normal(0, 35, 60)   # consensus: weaker but real signal (independent noise)
    # normal/t1: consensus tracks, model is pure noise → model adds no info → must not fire.
    a_n = rng.normal(150, 40, 60)
    m_n = rng.normal(150, 40, 60)
    c_n = a_n + rng.normal(0, 6, 60)
    cells = {
        ("turning_point", "t1"): {"actual": a_tp.tolist(), "model": m_tp.tolist(), "consensus": c_tp.tolist()},
        ("normal", "t1"): {"actual": a_n.tolist(), "model": m_n.tolist(), "consensus": c_n.tolist()},
        # turning_point/t7: no consensus (nan) → skipped (insufficient obs)
        ("turning_point", "t7"): {"actual": [1.0, 2.0], "model": [1.0, 2.0], "consensus": [float("nan"), float("nan")]},
    }
    out = combination_gate(cells)
    assert out[("turning_point", "t1")]["fires"] is True
    assert out[("normal", "t1")]["fires"] is False
    assert out[("turning_point", "t7")]["fires"] is False
    assert out[("turning_point", "t7")]["reason"] == "insufficient_paired_obs"


def test_combination_gate_accepts_harness_cell_shape():
    from nfp_vintages.diagnostics import combination_gate

    # Varied values so the encompassing design matrix is full-rank (identical
    # rows → singular X.T@X → LinAlgError in ols; vary each series).
    actual = [100.0, 120.0, 90.0, 140.0, 110.0, 130.0]
    model = [105.0, 115.0, 95.0, 135.0, 108.0, 128.0]
    consensus = [102.0, 118.0, 88.0, 138.0, 112.0, 132.0]
    cells = {("normal", "t1"): {"actual": actual, "model": model, "consensus": consensus}}
    out = combination_gate(cells)
    assert ("normal", "t1") in out and "fires" in out[("normal", "t1")]
