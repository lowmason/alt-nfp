"""Nowcast extraction from a fitted posterior.

Ports the nowcast arithmetic of the reference backtest loop
(``nfp-model-hmc`` ``backtest.py``): transform the latent SA growth path
through the CES observation equation, take the posterior-mean growth path,
rebuild the index path from the panel's base index, and read off the
target month. The target month (the as-of date itself) is never inside the
censored calendar, so the last latent state is the nowcast proxy —
``c_idx`` defaults accordingly.

Pure numpy; ``base_index`` and ``idx_to_level`` are scalars the caller
extracts from the data layer's levels frame (``ces_sa_index[0]`` and the
index→thousands conversion at the index-100 base row).
"""

from __future__ import annotations

import numpy as np


def ces_sa_predictive(posterior: dict[str, np.ndarray]) -> np.ndarray:
    """CES-SA observation-equation transform of the latent path.

    ``alpha_ces + lambda_ces * g_total_sa`` per draw → (chains, draws, T).
    """
    alpha = posterior["alpha_ces"]
    lam = posterior["lambda_ces"]
    return alpha[:, :, None] + lam[:, :, None] * posterior["g_total_sa"]


def nowcast_summary(
    posterior: dict[str, np.ndarray],
    *,
    base_index: float,
    idx_to_level: float,
    c_idx: int | None = None,
) -> dict:
    """Point nowcast + draw-level distribution at the target index.

    Returns ``nowcast_growth`` (posterior-mean log growth at ``c_idx``),
    ``nowcast_change_k`` (month-over-month jobs added, thousands, from the
    posterior-mean index path), ``pred_mean`` (the (T,) mean growth path),
    and ``pred_draws`` (the (chains, draws) predictive draws at ``c_idx``).
    """
    g_pred = ces_sa_predictive(posterior)
    pred_mean = np.nanmean(g_pred, axis=(0, 1))  # (T,)
    if c_idx is None:
        c_idx = len(pred_mean) - 1

    index_path = base_index * np.exp(np.cumsum(pred_mean))  # index after each month
    nowcast_index = index_path[c_idx]
    prev_index = base_index if c_idx == 0 else index_path[c_idx - 1]

    return {
        "c_idx": int(c_idx),
        "nowcast_growth": float(pred_mean[c_idx]),
        "nowcast_index": float(nowcast_index),
        "nowcast_change_k": float((nowcast_index - prev_index) * idx_to_level),
        "pred_mean": pred_mean,
        "pred_draws": g_pred[:, :, c_idx],
    }
