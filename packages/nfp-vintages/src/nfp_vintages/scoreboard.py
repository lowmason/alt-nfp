# packages/nfp-vintages/src/nfp_vintages/scoreboard.py
"""Tier 0 scoreboard helpers — regime decomposition, calibration, venue tag.

Evaluation-side only; imports no nfp-model code. See specs/model_improvements.md
section 3 and specs/plans/completed/13-tier01-scoreboard-and-diagnostics.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

MonthType = str  # "normal" | "large_revision" | "turning_point" | "benchmark_window"


@dataclass(frozen=True)
class MonthTypeConfig:
    """Operational definitions for the four month buckets (spec section 3)."""

    large_revision_pctl: float = 90.0          # |first->third| above this pctl
    claims_momentum_k: float = 40.0            # |3mo claims change| (thousands) for a turn
    benchmark_months: tuple[int, ...] = (2,)   # Feb-release annual-benchmark window


def classify_month_types(
    ref_months: list[date],
    revision_abs_k: np.ndarray,
    claims_momentum_k: np.ndarray,
    cfg: MonthTypeConfig,
) -> dict[date, MonthType]:
    """Map each ref month to one bucket.

    Precedence (rarest signal wins): large_revision > turning_point >
    benchmark_window > normal. ``revision_abs_k`` is |first-print - later-print|
    in thousands; ``claims_momentum_k`` is the absolute 3-month change in initial
    claims (thousands), aligned index-for-index with ``ref_months``.
    """
    revision_abs_k = np.asarray(revision_abs_k, dtype=float)
    claims_momentum_k = np.asarray(claims_momentum_k, dtype=float)
    finite = revision_abs_k[np.isfinite(revision_abs_k)]
    thresh = np.percentile(finite, cfg.large_revision_pctl) if finite.size else np.inf

    out: dict[date, MonthType] = {}
    for i, m in enumerate(ref_months):
        rev = revision_abs_k[i]
        mom = abs(claims_momentum_k[i]) if np.isfinite(claims_momentum_k[i]) else 0.0
        if np.isfinite(rev) and rev >= thresh:
            out[m] = "large_revision"
        elif mom >= cfg.claims_momentum_k:
            out[m] = "turning_point"
        elif m.month in cfg.benchmark_months:
            out[m] = "benchmark_window"
        else:
            out[m] = "normal"
    return out


def interval_coverage(draws: np.ndarray, actual: float, level: float) -> bool:
    """True iff ``actual`` lies in the central ``level`` predictive interval."""
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.isfinite(actual):
        return False
    lo = np.percentile(d, 100.0 * (1.0 - level) / 2.0)
    hi = np.percentile(d, 100.0 * (1.0 + level) / 2.0)
    return bool(lo <= actual <= hi)


def crps_sample(draws: np.ndarray, actual: float) -> float:
    """Sample-based CRPS (energy form): E|X-y| - 0.5 E|X-X'|.

    Lower is better. For a point-mass forecast this reduces to |forecast - y|.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0 or not np.isfinite(actual):
        return float("nan")
    term1 = np.abs(d - actual).mean()
    # 0.5 * mean_{i,j} |x_i - x_j| via the sorted-array O(n log n) identity.
    s = np.sort(d)
    i = np.arange(1, n + 1)
    # mean pairwise abs diff = (2 / n^2) * sum_i (2i - n - 1) * s_i
    mean_pairwise = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * s)
    return float(term1 - 0.5 * mean_pairwise)


def change_draws_k(
    pred_growth_draws: np.ndarray, *, prev_index: float, idx_to_level: float
) -> np.ndarray:
    """Predictive sample of the first-print change (thousands) from growth draws.

    Mirrors the batch nowcast arithmetic (batch.py reduction): a month-over-month
    change of growth g multiplies the prior index, change_k = prev*(exp(g)-1)*scale.
    Uses the mean prior index as a fixed scale (a first-order linearization that is
    exact to O(g^2); adequate for interval coverage / CRPS).
    """
    g = np.asarray(pred_growth_draws, dtype=float).reshape(-1)
    return prev_index * (np.exp(g) - 1.0) * idx_to_level


def venue_for(*, providers_present: bool) -> str:
    """Tag a scored row by information regime (spec section 3)."""
    return "full" if providers_present else "public-only"
