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


def _third_print_changes(store_path=None, industry_code: str = "05") -> pl.DataFrame:
    """First-to-THIRD PRIVATE CES SA over-the-month change (thousands), gap-safe.

    The model nowcasts PRIVATE NFP (``industry_code='05'``); this is the third-print
    private level-change that, differenced against the private first print, defines
    the Aruoba revision LHS. ``industry_code='05'`` (total private) by default, SA.

    Per reference month the THIRD monthly print level ``L2(M)`` is the latest
    NON-benchmark (``benchmark_revision == 0``) vintage — equivalently the max
    ``revision`` among the monthly prints (``revision == 2`` when present; rev 0/1
    survive gracefully at the frontier). Benchmark vintages are excluded so annual-
    benchmark wedges never enter the revision. The ``-1.0`` "no print" shutdown
    sentinel (``employment <= 0``) is dropped. ref_date is truncated to month-start.

    The change is computed ONLY between ADJACENT months:
    ``change(M) = L2(M) - L2(M-1)`` where the prior row's ref_date is EXACTLY one
    month before ``M``. The store omits some months, so a naive ``.shift(1)`` would
    diff across month gaps and produce a garbage intercept — the adjacency guard (an
    explicit one-month-prior self-join, NOT shift) is mandatory. Non-adjacent months
    get a null change and are dropped.
    """
    from nfp_ingest.vintage_store import read_vintage_store
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    store_path = store_path or VINTAGE_STORE_PATH
    lf = read_vintage_store(
        store_path, source="ces", seasonally_adjusted=True,
        geographic_type="national", geographic_code="00",
        industry_type="total", industry_code=industry_code,
    )
    # Third-print level per ref month: latest bmr0 monthly vintage, sentinel dropped.
    levels = (
        lf.collect()
        .filter((pl.col("benchmark_revision") == 0) & (pl.col("employment") > 0))
        .with_columns(pl.col("ref_date").dt.truncate("1mo").alias("ref_date"))
        .sort(["ref_date", "revision", "vintage_date"])
        .group_by("ref_date").agg(pl.col("employment").last().alias("level"))
        .sort("ref_date")
        .with_columns(prev_month=pl.col("ref_date").dt.offset_by("-1mo"))
    )
    # Adjacency guard: join each month to its EXACT one-month-prior level. A self
    # join on the prior-month key (not shift) leaves the change null wherever the
    # store skips a month, so gaps never masquerade as monthly changes.
    prev = levels.select(
        pl.col("ref_date").alias("prev_month"),
        pl.col("level").alias("prev_level"),
    )
    return (
        levels.join(prev, on="prev_month", how="left")
        .with_columns((pl.col("level") - pl.col("prev_level")).alias("later_change_k"))
        .select(["ref_date", "later_change_k"])
        .drop_nulls("later_change_k")
    )


def _join_revision(fp: pl.DataFrame, later: pl.DataFrame) -> pl.DataFrame:
    return (
        fp.join(later, on="ref_date", how="inner")
        .with_columns((pl.col("later_change_k") - pl.col("first_print_change_k")).alias("revision_k"))
        .sort("ref_date")
    )


def build_revision_table(store_path=None, industry_code: str = "05") -> pl.DataFrame:
    """[ref_date, first_print_change_k, later_change_k, revision_k] over the store.

    PRIVATE '05' by default (Track A): the model nowcasts private NFP, so both the
    first-print and the third-print over-the-month changes are the CES total-private
    SA series. ``first_print_change_k`` is ``first_print_changes(industry_code='05')``;
    ``later_change_k`` is the gap-safe first-to-third private change; ``revision_k =
    later_change_k - first_print_change_k`` for matched ADJACENT months only.
    Public return columns are STABLE across the retarget so callers (run_a5_backtest,
    run_tier1_diagnostics) keep working.
    """
    from nfp_ingest.first_print import first_print_changes

    fp = first_print_changes(
        industry_code=industry_code,
        **({"store_path": store_path} if store_path else {}),
    )
    fp = fp.select(["ref_date", "first_print_change_k"])
    later = _third_print_changes(store_path, industry_code=industry_code)
    return _join_revision(fp, later)


def pooled_first_print_bias(rev_tbl: pl.DataFrame, *, method: str = "median") -> float:
    """Pooled private first-print bias δ (k-jobs) — the §5A post-hoc offset.

    The central location of ``revision_k = later_change_k - first_print_change_k``
    over the store. δ < 0 means the first print prints *above* the settled third
    print, so a model predicting the third-print value reads low against the first
    print; §5A subtracts δ from the nowcast to push it toward the first print.

    ``method="median"`` (default) is robust to the benchmark/COVID outlier months
    (e.g. the real +871k 2022-11 revision) that contaminate the mean; ``method=
    "mean"`` matches the unconditional Aruoba intercept. Null/non-finite revisions
    are dropped. This is a pooled constant — a month-type-specific δ is the obvious
    refinement (the harness already classifies month types).
    """
    vals = np.asarray(rev_tbl["revision_k"].to_list(), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no finite revision_k values to pool")
    if method == "median":
        return float(np.median(vals))
    if method == "mean":
        return float(np.mean(vals))
    raise ValueError(f"unknown method {method!r}; expected 'median' or 'mean'")


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


@dataclass(frozen=True)
class GateConfig:
    normal_r2_floor: float = 0.10           # below this, monthly revisions ~ noise
    turning_point_excess: float = 0.15      # tp R2 must exceed normal by this to fund BD


def gate_decision(r2_by_month_type: dict[str, float], cfg: GateConfig) -> dict:
    """Translate Aruoba R^2-by-month-type into Tier 2/3 funding decisions."""
    normal = r2_by_month_type.get("normal", 0.0)
    tp = r2_by_month_type.get("turning_point", 0.0)
    bench = r2_by_month_type.get("benchmark_window", 0.0)
    fund_rebuild = normal >= cfg.normal_r2_floor
    fund_bd = (tp - normal) >= cfg.turning_point_excess
    rationale = (
        f"normal R²={normal:.3f} ({'>=' if fund_rebuild else '<'} {cfg.normal_r2_floor}); "
        f"turning_point R²={tp:.3f}, benchmark R²={bench:.3f}; "
        f"BD funded={fund_bd} (turning_point excess {tp - normal:+.3f} "
        f"vs {cfg.turning_point_excess})."
    )
    return {
        "fund_first_release_rebuild": bool(fund_rebuild),
        "fund_tier3_bd": bool(fund_bd),
        "rationale": rationale,
    }


def qcew_settled_changes(store_path=None) -> pl.DataFrame:
    """Latest-vintage QCEW national total-private over-the-month change (thousands).

    The settled QCEW level is the fair external target for QCEW-anchored competitors.
    Selects max(vintage_date) per ref month, level-differences, and returns the result
    in thousands (employment is already stored in thousands; no unit conversion needed).

    Series used: industry_code='05' (total private, NSA). This is the model's INTENDED
    target, not a fallback: the model nowcasts PRIVATE NFP (it is QCEW-anchored, and
    this store's QCEW is total-private), so the private QCEW-settled value is the
    PRIMARY administrative truth the private nowcast is held to. The '00' (total
    nonfarm) QCEW would belong to a future, unbuilt total-NFP extension (Track B), not
    to this private track.

    TWO RESIDUAL CAVEATS (data-shape, not a target mismatch):
    1. NOT seasonally adjusted (NSA): The store holds only NSA QCEW; the model nowcast
       and CES first-print are seasonally adjusted. Comparing SA errors to this NSA truth
       mixes seasonality and overstates apparent errors in summer/winter months.
    2. Q1 hole (April gap): QCEW ref_dates cover months 4–12 only (Jan–Mar are absent
       from the store). The April shift(1) entry is a 4-month gap (Dec→Apr), not a
       monthly change. May–Dec differences are genuine month-over-month changes.
       Callers scoring metrics must exclude April to avoid corrupting ME/MAE/RMSE.
    These caveats are logged here so downstream callers can filter or caveat as needed.
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
