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
