"""Defaults pinned to the frozen reference (nfp-model-hmc NowcastConfig)."""

import math

from nfp_model.config import PRESETS, ModelPriors


def test_prior_defaults_match_frozen_reference():
    p = ModelPriors()
    assert p.latent.log_tau_mu == math.log(0.013)
    assert p.latent.log_tau_sd == 0.5
    assert (p.latent.phi_alpha, p.latent.phi_beta, p.latent.phi_cap) == (18.0, 2.0, 0.99)
    assert (p.latent.mu_g_mu, p.latent.mu_g_sd) == (0.001, 0.005)
    assert p.ces.log_sigma_mu == math.log(0.002)
    assert (p.ces.alpha_sd, p.ces.lambda_mu, p.ces.lambda_sd, p.ces.lambda_lower) == (
        0.005, 1.0, 0.1, 0.5,
    )
    assert p.qcew.nu == 5
    assert p.qcew.log_sigma_mid_mu == math.log(0.0005)
    assert p.qcew.log_sigma_mid_sd == 0.15
    assert p.qcew.log_sigma_boundary_mu == math.log(0.002)
    assert p.qcew.log_sigma_boundary_sd == 0.5
    assert p.fourier.n_harmonics == 4
    assert p.fourier.log_sigma_mu == math.log(0.0003)
    assert p.fourier.init_sd == 0.015
    assert (p.birth_death.phi0_mu, p.birth_death.phi0_sd) == (0.001, 0.002)
    assert p.birth_death.log_sigma_mu == math.log(0.003)
    assert p.birth_death.phi3_sd == 0.3
    assert (p.provider.alpha_sd, p.provider.lambda_mu, p.provider.lambda_sd) == (
        0.005, 1.0, 0.15,
    )
    assert (p.provider.sigma_concentration, p.provider.sigma_rate) == (3.0, 0.004)
    assert (p.provider.rho_alpha, p.provider.rho_beta) == (2.0, 3.0)
    assert p.n_eras == 2
    assert p.indicator_names == ("claims", "jolts")


def test_sampler_presets_match_frozen_reference():
    d = PRESETS["default"]
    assert (d.num_samples, d.num_warmup, d.num_chains, d.target_accept) == (
        4000, 3000, 4, 0.95,
    )
    li = PRESETS["light"]
    assert (li.num_samples, li.num_warmup, li.num_chains, li.target_accept) == (
        2000, 2000, 2, 0.95,
    )
    m = PRESETS["medium"]
    assert (m.num_samples, m.num_warmup, m.num_chains) == (4000, 3000, 4)
    assert d.max_tree_depth == 10
