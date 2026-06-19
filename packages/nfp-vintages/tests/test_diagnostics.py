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
    tbl = build_revision_table()
    assert {"ref_date", "first_print_change_k", "later_change_k", "revision_k"} <= set(tbl.columns)
    assert tbl.height > 0


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
