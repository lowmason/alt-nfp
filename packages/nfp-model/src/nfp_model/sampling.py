"""NUTS sampling and posterior packaging.

:func:`fit_model` runs NumPyro NUTS and returns a :class:`FitResult` whose
``posterior`` maps every sample site *and* deterministic site to a numpy
array of shape ``(chains, draws, ...)`` — the same layout as the reference
``idata.posterior[...].values``, so comparison code is symmetric.

Deterministic sites are not collected by NumPyro's MCMC; they are
reconstructed exactly from the posterior draws via ``Predictive``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from .config import PRESETS, ModelPriors, SamplerSettings
from .data import model_inputs
from .model import DETERMINISTIC_SITES, nfp_model


@dataclass
class FitResult:
    """Posterior draws plus run metadata."""

    posterior: dict[str, np.ndarray]
    num_divergences: int
    settings: SamplerSettings
    seed: int
    wall_seconds: float
    priors: ModelPriors = field(default_factory=ModelPriors)


def fit_model(
    data: dict,
    priors: ModelPriors | None = None,
    *,
    settings: SamplerSettings | str = "default",
    seed: int = 0,
    progress: bool = False,
) -> FitResult:
    """Fit the model to a ModelData dict (or ``from_snapshot`` output).

    Parameters
    ----------
    data
        Output of ``nfp_ingest.model_data.build_model_data`` or
        ``nfp_model.data.from_snapshot`` — reduced internally via
        ``model_inputs`` so frames never reach JAX tracing.
    settings
        A :class:`SamplerSettings` or a preset name (``"default"``,
        ``"light"``, ``"medium"``).
    """
    numpyro.enable_x64()  # parity is defined in float64
    if isinstance(settings, str):
        settings = PRESETS[settings]
    if priors is None:
        priors = ModelPriors()

    inputs = model_inputs(data)

    kernel = NUTS(
        nfp_model,
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
    mcmc.run(
        jax.random.PRNGKey(seed), data=inputs, priors=priors, extra_fields=("diverging",)
    )
    wall = time.time() - t0

    posterior = {
        k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()
    }

    # Deterministics, reconstructed from the (flat) draws; chain-major
    # flat order means a plain reshape restores (chains, draws, ...).
    flat = mcmc.get_samples()
    dets = Predictive(
        nfp_model, posterior_samples=flat, return_sites=list(DETERMINISTIC_SITES)
    )(jax.random.PRNGKey(0), data=inputs, priors=priors)
    n_chains, n_draws = settings.num_chains, settings.num_samples
    for k, v in dets.items():
        arr = np.asarray(v)
        posterior[k] = arr.reshape(n_chains, n_draws, *arr.shape[1:])

    divergences = int(
        np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum()
    )
    return FitResult(
        posterior=posterior,
        num_divergences=divergences,
        settings=settings,
        seed=seed,
        wall_seconds=wall,
        priors=priors,
    )
