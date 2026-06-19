"""Standalone Bayesian government-wedge model (specs/government_wedge.md).

Imports ONLY jax/numpyro/numpy (no nfp_* package). Models the wedge MoM CHANGE
directly in change-space (units: thousands of jobs):

    mu_t = drift + season[month_t] + X_intervention @ coef
    y_t ~ Normal(mu_t, sigma)            (masked over COVID + the Oct-2025 hole)
"""
from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

WEDGE_DETERMINISTIC_SITES = ("mu", "season")


def wedge_model(data: dict) -> None:
    T = int(data["T"])
    moy = jnp.asarray(data["month_of_year"]) - 1          # 0..11
    X = jnp.asarray(data["X_intervention"])               # (T, K)
    K = X.shape[1]

    drift = numpyro.sample("drift", dist.Normal(0.0, 50.0))

    # 11 free monthly effects, 12th pinned by sum-to-zero (deterministic).
    tau = numpyro.sample("tau_season", dist.HalfNormal(30.0))
    s_raw = numpyro.sample("season_raw", dist.Normal(0.0, 1.0).expand([11]))
    s11 = s_raw * tau                                      # non-centered
    season = numpyro.deterministic(
        "season", jnp.concatenate([s11, -s11.sum()[None]]))   # length 12, sums to 0

    if K > 0:
        pm = jnp.asarray(data["iv_prior_mean"])
        ps = jnp.asarray(data["iv_prior_sd"])
        coef = numpyro.sample("iv_coef", dist.Normal(pm, ps).to_event(1))
        iv = X @ coef
    else:
        iv = jnp.zeros(T)

    mu = numpyro.deterministic("mu", drift + season[moy] + iv)
    sigma = numpyro.sample("sigma", dist.HalfNormal(30.0))

    mask = jnp.asarray(data["mask"], dtype=bool)
    with numpyro.handlers.mask(mask=mask):                 # inline idiom (no nfp_* import)
        numpyro.sample("y_obs", dist.Normal(mu, sigma), obs=jnp.asarray(data["y"]))
