# packages/nfp-vintages/src/nfp_vintages/scoreboard.py
"""Tier 0 scoreboard helpers — regime decomposition, calibration, venue tag.

Evaluation-side only; imports no nfp-model code. See specs/model_improvements.md
section 3 and plans/13-tier01-scoreboard-and-diagnostics.md.
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
