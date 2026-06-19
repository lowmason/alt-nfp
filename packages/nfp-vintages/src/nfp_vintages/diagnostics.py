"""Tier 1 diagnostics — Aruoba revision regression, Mincer-Zarnowitz, gate.

Evaluation-side only; imports no nfp-model code. OLS is hand-rolled on numpy
(no statsmodels in the workspace). See specs/model_improvements.md section 4 and
plans/13-tier01-scoreboard-and-diagnostics.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class OLSResult:
    coeffs: np.ndarray      # (k,)
    cov: np.ndarray         # (k, k) coefficient covariance
    r2: float
    n: int
    residuals: np.ndarray   # (n,)


def ols(X: np.ndarray, y: np.ndarray) -> OLSResult:
    """Ordinary least squares with classical (homoskedastic) covariance."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 0.0 if tss == 0.0 else 1.0 - rss / tss
    dof = max(n - k, 1)
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    return OLSResult(coeffs=beta, cov=sigma2 * xtx_inv, r2=r2, n=n, residuals=resid)


def _latest_ces_changes(store_path=None) -> pl.DataFrame:
    """Latest-vintage CES SA national total over-the-month change (thousands).

    ref_date is truncated to month-start (1st) to align with first_print_changes().
    """
    from nfp_ingest.vintage_store import read_vintage_store
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    store_path = store_path or VINTAGE_STORE_PATH
    lf = read_vintage_store(
        store_path, source="ces", seasonally_adjusted=True,
        geographic_type="national", geographic_code="00",
        industry_type="total", industry_code="00",
    )
    return (
        lf.collect()
        .sort(["ref_date", "vintage_date"])
        .group_by("ref_date").agg(pl.col("employment").last().alias("level"))
        .sort("ref_date")
        .with_columns(
            pl.col("ref_date").dt.truncate("1mo").alias("ref_date"),
            (pl.col("level") - pl.col("level").shift(1)).alias("later_change_k"),
        )
        .select(["ref_date", "later_change_k"])
        .drop_nulls("later_change_k")
    )


def _join_revision(fp: pl.DataFrame, later: pl.DataFrame) -> pl.DataFrame:
    return (
        fp.join(later, on="ref_date", how="inner")
        .with_columns((pl.col("later_change_k") - pl.col("first_print_change_k")).alias("revision_k"))
        .sort("ref_date")
    )


def build_revision_table(store_path=None) -> pl.DataFrame:
    """[ref_date, first_print_change_k, later_change_k, revision_k] over the store."""
    from nfp_ingest.first_print import first_print_changes

    fp = first_print_changes(**({"store_path": store_path} if store_path else {}))
    fp = fp.select(["ref_date", "first_print_change_k"])
    later = _latest_ces_changes(store_path)
    return _join_revision(fp, later)


def build_aruoba_design(
    ref_months: list, regressors: dict[str, dict]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Design matrix X_t with an intercept, plus a name list and the used set.

    ``regressors`` maps name -> {ref_month: value}. A regressor that is entirely
    NaN/missing across ``ref_months`` is dropped (records the skeleton vs full
    regime). Rows with any NaN in a *kept* column are the caller's responsibility
    to drop (see aruoba_regression).
    """
    cols, names, used = [np.ones(len(ref_months))], ["const"], []
    for name, series in regressors.items():
        vals = np.array([series.get(m, np.nan) for m in ref_months], dtype=float)
        if np.all(~np.isfinite(vals)):
            continue
        cols.append(vals)
        names.append(name)
        used.append(name)
    return np.column_stack(cols), names, used


@dataclass(frozen=True)
class AruobaResult:
    intercept_k: float          # alpha — the first-print bias (feeds section 5A)
    r2: float                   # forecastable share of revision variance
    coef_names: list[str]
    coeffs: np.ndarray
    n: int


def aruoba_regression(revision_k: np.ndarray, X: np.ndarray, names: list[str]) -> AruobaResult:
    """Fit revision = alpha + gamma'.X. Drops rows with any non-finite entry."""
    revision_k = np.asarray(revision_k, dtype=float)
    mask = np.isfinite(revision_k) & np.isfinite(X).all(axis=1)
    Xm, ym = X[mask], revision_k[mask]
    res = ols(Xm, ym)
    return AruobaResult(intercept_k=float(res.coeffs[0]), r2=res.r2,
                        coef_names=list(names), coeffs=res.coeffs, n=res.n)


@dataclass(frozen=True)
class MZResult:
    alpha: float
    beta: float
    joint_stat: float    # Wald chi^2 for (alpha=0, beta=1)
    joint_p: float
    r2: float
    n: int


def mincer_zarnowitz(actual: np.ndarray, forecast: np.ndarray) -> MZResult:
    """Efficiency regression actual = alpha + beta*forecast; test alpha=0, beta=1."""
    from scipy.stats import chi2

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(forecast)
    a, f = actual[mask], forecast[mask]
    X = np.column_stack([np.ones(a.size), f])
    res = ols(X, a)
    theta = res.coeffs                       # [alpha, beta]
    theta0 = np.array([0.0, 1.0])
    diff = theta - theta0
    wald = float(diff @ np.linalg.inv(res.cov) @ diff)
    p = float(chi2.sf(wald, df=2))
    return MZResult(alpha=float(theta[0]), beta=float(theta[1]),
                    joint_stat=wald, joint_p=p, r2=res.r2, n=res.n)


def qcew_settled_changes(store_path=None) -> pl.DataFrame:
    """Latest-vintage QCEW national total-private over-the-month change (thousands).

    The settled QCEW level is the fair external target for QCEW-anchored competitors.
    Selects max(vintage_date) per ref month, level-differences, and returns the result
    in thousands (employment is already stored in thousands; no unit conversion needed).

    Series used: industry_code='05' (total private, NSA). The store does NOT contain
    industry_code='00' (total nonfarm) — total private is the only available QCEW proxy.

    THREE KNOWN LIMITATIONS (DONE_WITH_CONCERNS):
    1. NOT seasonally adjusted (NSA): The store holds only NSA QCEW; the model nowcast
       and CES first-print are seasonally adjusted. Comparing SA errors to this NSA truth
       mixes seasonality and overstates apparent errors in summer/winter months.
    2. Private-only (industry_code='05'): Excludes government workers — not directly
       comparable to CES total nonfarm (industry_code='00'). Total-nonfarm QCEW
       (industry_code='00') is absent from this store.
    3. Q1 hole (April gap): QCEW ref_dates cover months 4–12 only (Jan–Mar are absent
       from the store). The April shift(1) entry is a 4-month gap (Dec→Apr), not a
       monthly change. May–Dec differences are genuine month-over-month changes.
       Callers scoring metrics must exclude April to avoid corrupting ME/MAE/RMSE.
    These limitations are logged here so downstream callers can filter or caveat as needed.
    """
    from nfp_ingest.vintage_store import read_vintage_store
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    store_path = store_path or VINTAGE_STORE_PATH
    lf = read_vintage_store(
        store_path, source="qcew", seasonally_adjusted=False,
        geographic_type="national", geographic_code="00",
        industry_type="total", industry_code="05",
    )
    df = (
        lf.collect()
        # Keep only the aggregate (null size_class = all size classes combined).
        .filter(pl.col("size_class_code").is_null())
        .with_columns(pl.col("ref_date").dt.truncate("1mo"))  # store ref_date is day-12; align to month-start
        .sort(["ref_date", "vintage_date"])
        .group_by("ref_date").agg(pl.col("employment").last().alias("level"))
        .sort("ref_date")
        .with_columns(((pl.col("level") - pl.col("level").shift(1))).alias("qcew_settled_change_k"))
        .select(["ref_date", "qcew_settled_change_k"])
        .drop_nulls("qcew_settled_change_k")
    )
    return df
