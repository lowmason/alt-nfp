"""Tier 1 diagnostics — Aruoba revision regression, Mincer-Zarnowitz, gate.

Evaluation-side only; imports no nfp-model code. OLS is hand-rolled on numpy
(no statsmodels in the workspace). See specs/model_improvements.md section 4 and
plans/13-tier01-scoreboard-and-diagnostics.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
