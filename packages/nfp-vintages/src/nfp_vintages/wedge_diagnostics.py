"""Diagnostics for the government wedge (specs/completed/government_wedge.md §3.2/§7).

These NEVER enter the model likelihood; they validate the wedge decomposition
and calibrate intervention priors from public government data.
"""
from __future__ import annotations

import numpy as np


def decomposition_residual(wedge_change, gov90_change) -> dict:
    """r = wedge - published_90: the SA-additivity residual we target the wedge to escape."""
    w = np.asarray(wedge_change, float)
    g = np.asarray(gov90_change, float)
    r = w - g
    # Sample std (ddof=1) for consistency with calibrate_intervention_sd.
    return {"r_mean": float(r.mean()), "r_std": float(r.std(ddof=1)),
            "wedge_std": float(w.std(ddof=1)),
            "r_share": float(r.std(ddof=1) / (w.std(ddof=1) or 1.0))}


def calibrate_intervention_sd(observed_federal_change, baseline_sd: float) -> float:
    """An honest prior sd for a federal-shock magnitude: max(empirical spread, baseline)."""
    obs = np.asarray(observed_federal_change, float)
    return float(max(obs.std(ddof=1) if obs.size > 1 else 0.0, baseline_sd))
