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
