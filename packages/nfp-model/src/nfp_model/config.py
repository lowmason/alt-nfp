"""Prior and sampler configuration, frozen from the PyMC reference.

Every default here is pinned to the frozen reference implementation
(``nfp-model-hmc``'s ``NowcastConfig`` defaults); A3 parity is defined
against these values. Change them only behind a new parity baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatentPriors:
    """AR(1) continuing-units growth (tau = stationary SD parametrization)."""

    log_tau_mu: float = math.log(0.013)
    log_tau_sd: float = 0.5
    phi_alpha: float = 18.0
    phi_beta: float = 2.0
    phi_cap: float = 0.99
    mu_g_mu: float = 0.001
    mu_g_sd: float = 0.005


@dataclass(frozen=True)
class CESPriors:
    """CES observation equation: shared bias/loading and per-vintage LogNormal sigmas."""

    log_sigma_mu: float = math.log(0.002)
    log_sigma_sd: float = 0.5
    alpha_sd: float = 0.005
    lambda_mu: float = 1.0
    lambda_sd: float = 0.1
    lambda_lower: float = 0.5


@dataclass(frozen=True)
class QCEWPriors:
    """Student-t anchor; the tight M2 prior prevents QCEW precision dominance."""

    nu: int = 5
    log_sigma_mid_mu: float = math.log(0.0005)
    log_sigma_mid_sd: float = 0.15
    log_sigma_boundary_mu: float = math.log(0.002)
    log_sigma_boundary_sd: float = 0.5


@dataclass(frozen=True)
class FourierPriors:
    """Fourier seasonal block: harmonic count and the annually-evolving GRW step SDs."""

    n_harmonics: int = 4
    log_sigma_mu: float = math.log(0.0003)
    log_sigma_sd: float = 0.5
    init_sd: float = 0.015


@dataclass(frozen=True)
class BirthDeathPriors:
    """Structural birth/death block priors: intercept phi_0, shock SD sigma_bd, and cyclical-covariate loading SD phi_3 (the v2 BD term is 3-term: bd_t = phi_0 + sigma_bd*xi + phi_3*X_cycle)."""

    phi0_mu: float = 0.001
    phi0_sd: float = 0.002
    log_sigma_mu: float = math.log(0.003)
    log_sigma_sd: float = 0.5
    phi3_sd: float = 0.3


@dataclass(frozen=True)
class ProviderPriors:
    """Per-provider measurement model: bias/loading, InverseGamma noise SD, and AR(1) persistence."""

    alpha_sd: float = 0.005
    lambda_mu: float = 1.0
    lambda_sd: float = 0.15
    sigma_concentration: float = 3.0
    sigma_rate: float = 0.004
    rho_alpha: float = 2.0
    rho_beta: float = 3.0


@dataclass(frozen=True)
class ModelPriors:
    """Bundle of every prior block plus the structural knobs."""

    latent: LatentPriors = field(default_factory=LatentPriors)
    ces: CESPriors = field(default_factory=CESPriors)
    qcew: QCEWPriors = field(default_factory=QCEWPriors)
    fourier: FourierPriors = field(default_factory=FourierPriors)
    birth_death: BirthDeathPriors = field(default_factory=BirthDeathPriors)
    provider: ProviderPriors = field(default_factory=ProviderPriors)
    n_eras: int = 2
    #: phi_3 covariate order — must stay stable for draw comparability
    indicator_names: tuple[str, ...] = ("claims", "jolts")


@dataclass(frozen=True)
class SamplerSettings:
    """NUTS settings; field names follow NumPyro, defaults follow the reference."""

    num_samples: int = 4000
    num_warmup: int = 3000
    num_chains: int = 4
    target_accept: float = 0.95
    max_tree_depth: int = 10
    chain_method: str = "sequential"


#: presets mirroring the reference SamplingConfig (default / light / medium)
PRESETS: dict[str, SamplerSettings] = {
    "default": SamplerSettings(),
    "light": SamplerSettings(num_samples=2000, num_warmup=2000, num_chains=2),
    "medium": SamplerSettings(),
}
