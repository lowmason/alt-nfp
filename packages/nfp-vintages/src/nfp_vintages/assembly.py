"""Assemble a Total-NFP change posterior from the private nowcast + wedge.

Total = private + wedge is exact by construction (we forecast the wedge directly).
The private leg is growth/index space and must be converted to change-k first.
"""
from __future__ import annotations

import numpy as np

from nfp_vintages.scoreboard import change_draws_k


def assemble_total(
    private_growth_draws: np.ndarray,
    wedge_change_draws: np.ndarray,
    *,
    prev_index: float,
    idx_to_level: float,
    eta: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Element-wise Total first-print change draws (thousands).

    N = wedge draw count (wedge authoritative); the private change draws are
    resampled to N. ``eta`` enables the (default-off) residual coupling: adds
    ``eta * z`` to the wedge leg, z = standardized mean-zero private residual —
    point-invariant, widens intervals only.
    """
    priv = change_draws_k(private_growth_draws, prev_index=prev_index,
                          idx_to_level=idx_to_level)        # flattened
    wedge = np.asarray(wedge_change_draws, float).reshape(-1)
    n = wedge.shape[0]
    rng = np.random.default_rng(seed)
    priv_n = rng.choice(priv, size=n, replace=True) if priv.shape[0] != n else priv
    total = priv_n + wedge
    if eta:
        z = (priv_n - priv_n.mean()) / (priv_n.std() or 1.0)
        total = total + eta * z
    if np.isnan(total).any():
        raise ValueError("assemble_total produced NaN — check prev_index/idx_to_level anchor")
    return total
