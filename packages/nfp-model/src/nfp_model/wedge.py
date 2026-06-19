"""Standalone Bayesian government-wedge model (specs/government_wedge.md).

Imports ONLY jax/numpyro/numpy (no nfp_* package). Models the wedge MoM CHANGE
directly in change-space (units: thousands of jobs):

    mu_t = drift + season[month_t] + X_intervention @ coef
    y_t ~ Normal(mu_t, sigma)            (masked over COVID + the Oct-2025 hole)
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from nfp_model.config import PRESETS
from nfp_model.sampling import FitResult

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


def fit_wedge(
    data: dict, *, settings: str | object = "default", seed: int = 0, progress: bool = False
) -> FitResult:
    """NUTS fit of the wedge model. Mirrors fit_model's packaging (FitResult)."""
    numpyro.enable_x64()
    if isinstance(settings, str):
        settings = PRESETS[settings]
    kernel = NUTS(
        wedge_model,
        target_accept_prob=settings.target_accept,
        max_tree_depth=settings.max_tree_depth,
        init_strategy=init_to_median(num_samples=15),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=settings.num_chains,
        chain_method=settings.chain_method,
        progress_bar=progress,
    )
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), data=data, extra_fields=("diverging",))
    wall = time.time() - t0
    post = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    flat = mcmc.get_samples()
    dets = Predictive(
        wedge_model, posterior_samples=flat, return_sites=list(WEDGE_DETERMINISTIC_SITES)
    )(jax.random.PRNGKey(0), data=data)
    nc, nd = settings.num_chains, settings.num_samples
    for k, v in dets.items():
        arr = np.asarray(v)
        post[k] = arr.reshape(nc, nd, *arr.shape[1:])
    div = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
    return FitResult(
        posterior=post, num_divergences=div, settings=settings, seed=seed, wall_seconds=wall
    )


def wedge_pred_draws(fit: FitResult, target_idx: int, *, seed: int = 0) -> np.ndarray:
    """Posterior predictive of the wedge first-print CHANGE at target_idx (length N).

    The predictive includes observation noise (mu + Normal(0, sigma)), mirroring the
    private nowcast's first-print predictive, so the two convolve like-for-like.
    """
    mu = fit.posterior["mu"][..., target_idx].reshape(-1)   # (N,)
    sigma = fit.posterior["sigma"].reshape(-1)               # (N,)
    eps = np.random.default_rng(seed).standard_normal(mu.shape[0])
    return mu + sigma * eps
