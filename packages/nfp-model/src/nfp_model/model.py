"""NumPyro state-space model for employment growth nowcasting.

Faithful translation of the frozen PyMC reference (``nfp-model-hmc``
``model.py``): identical priors and likelihoods, so the posterior is the
same distribution up to Monte Carlo error (the A3 parity gate). Components:

1. **Latent continuing-units growth** — non-centered AR(1) with era-specific
   means, parametrized by stationary SD ``tau`` (breaks the phi–sigma ridge).
2. **Fourier seasonal** — annually-evolving harmonic amplitudes; the
   reference's centered Gaussian random walk is reparametrized non-centered
   (identical prior law: init N(0, init_sd), increments N(0, sigma_k)).
3. **Structural birth/death** — ``bd_t = phi_0 + phi_3·X_cycle + sigma_bd·xi``,
   with all-zero covariates gated out (avoids unidentified parameters).
4. **QCEW likelihood** — Student-t anchor, two estimated LogNormal base
   sigmas (M2 vs boundary) times per-observation revision multipliers.
5. **CES likelihood** — best-available print per month, vintage-indexed
   sigmas, shared bias/loading.
6. **Provider likelihoods** — per-provider iid or AR(1) measurement error.

Site names match the reference so posterior comparisons are key-for-key.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .config import ModelPriors

#: deterministic sites recorded for downstream use (paths over the calendar)
DETERMINISTIC_SITES = (
    "g_cont", "seasonal", "fourier_coefs_det", "bd", "g_total_sa", "g_total_nsa",
)


def nfp_model(data: dict, priors: ModelPriors | None = None) -> None:
    """The QCEW-anchored model. *data* is a ``model_inputs``-style dict."""
    p = priors if priors is not None else ModelPriors()

    T = int(data["T"])
    month_of_year = np.asarray(data["month_of_year"])

    # =============================================================
    # QCEW observation noise: estimated base by tier × revision mult.
    # LogNormal avoids the funnel HalfNormal creates when sigma
    # collapses toward zero (extreme QCEW precision → bimodality).
    # =============================================================
    sigma_qcew_mid = numpyro.sample(
        "sigma_qcew_mid", dist.LogNormal(p.qcew.log_sigma_mid_mu, p.qcew.log_sigma_mid_sd)
    )
    sigma_qcew_boundary = numpyro.sample(
        "sigma_qcew_boundary",
        dist.LogNormal(p.qcew.log_sigma_boundary_mu, p.qcew.log_sigma_boundary_sd),
    )
    qcew_is_m2 = np.asarray(data["qcew_is_m2"], dtype=bool)
    base_sigma = jnp.where(qcew_is_m2, sigma_qcew_mid, sigma_qcew_boundary)
    qcew_sigma = base_sigma * jnp.asarray(np.asarray(data["qcew_noise_mult"], dtype=float))

    # =============================================================
    # Latent continuing-units growth: AR(1), tau = stationary SD
    # =============================================================
    tau = numpyro.sample("tau", dist.LogNormal(p.latent.log_tau_mu, p.latent.log_tau_sd))
    phi_raw = numpyro.sample("phi_raw", dist.Beta(p.latent.phi_alpha, p.latent.phi_beta))
    phi = jnp.minimum(phi_raw, p.latent.phi_cap)
    sigma_g = tau * jnp.sqrt(1 - phi**2)
    eps_g = numpyro.sample("eps_g", dist.Normal(0.0, 1.0).expand([T]))

    era_idx = data.get("era_idx")
    if era_idx is not None:
        mu_g_era = numpyro.sample(
            "mu_g_era", dist.Normal(p.latent.mu_g_mu, p.latent.mu_g_sd).expand([p.n_eras])
        )
        mu_g = mu_g_era[np.asarray(era_idx, dtype=int)]  # (T,)
    else:
        mu_g_scalar = numpyro.sample(
            "mu_g", dist.Normal(p.latent.mu_g_mu, p.latent.mu_g_sd)
        )
        mu_g = jnp.broadcast_to(mu_g_scalar, (T,))

    g0 = mu_g[0] + sigma_g * eps_g[0]

    def ar1_step(g_prev, xs):
        e_t, mu_t = xs
        g_t = mu_t + phi * (g_prev - mu_t) + sigma_g * e_t
        return g_t, g_t

    _, g_rest = jax.lax.scan(ar1_step, g0, (eps_g[1:], mu_g[1:]))
    g_cont = jnp.concatenate([g0[None], g_rest])
    numpyro.deterministic("g_cont", g_cont)

    # =============================================================
    # Fourier seasonal with annually-evolving amplitudes (GRW across
    # years, non-centered): rows 0..K-1 are A_k, rows K..2K-1 are B_k.
    # =============================================================
    K = p.fourier.n_harmonics
    n_years = int(data["n_years"])
    year_of_obs = np.asarray(data["year_of_obs"], dtype=int)

    # Innovation std per harmonic (decreasing with k in log-space)
    sigma_fourier_mu = p.fourier.log_sigma_mu - jnp.log(jnp.arange(1, K + 1))
    sigma_fourier = numpyro.sample(
        "sigma_fourier", dist.LogNormal(sigma_fourier_mu, p.fourier.log_sigma_sd)
    )
    sigma_vec = jnp.tile(sigma_fourier, 2)  # (2K,)

    fourier_z = numpyro.sample("fourier_z", dist.Normal(0.0, 1.0).expand([2 * K, n_years]))
    fourier_steps = jnp.concatenate(
        [p.fourier.init_sd * fourier_z[:, :1], sigma_vec[:, None] * fourier_z[:, 1:]],
        axis=1,
    )
    fourier_coefs = jnp.cumsum(fourier_steps, axis=1)  # (2K, n_years)

    k_vals = np.arange(1, K + 1)
    cos_basis = jnp.asarray(np.cos(2 * np.pi * k_vals * month_of_year[:, None] / 12))
    sin_basis = jnp.asarray(np.sin(2 * np.pi * k_vals * month_of_year[:, None] / 12))

    A_t = fourier_coefs[:K, year_of_obs].T  # (T, K)
    B_t = fourier_coefs[K:, year_of_obs].T  # (T, K)
    s_t = jnp.sum(A_t * cos_basis + B_t * sin_basis, axis=1)  # (T,)
    numpyro.deterministic("seasonal", s_t)
    numpyro.deterministic("fourier_coefs_det", fourier_coefs.T)  # (n_years, 2K)

    # =============================================================
    # Structural birth/death offset: bd_t = phi_0 + phi_3·X + sigma_bd·xi.
    # Covariates are centred upstream; unavailable months are zeroed so
    # bd collapses to phi_0 + sigma_bd·xi there.
    # =============================================================
    phi_0 = numpyro.sample("phi_0", dist.Normal(p.birth_death.phi0_mu, p.birth_death.phi0_sd))
    sigma_bd = numpyro.sample(
        "sigma_bd", dist.LogNormal(p.birth_death.log_sigma_mu, p.birth_death.log_sigma_sd)
    )
    xi_bd = numpyro.sample("xi_bd", dist.Normal(0.0, 1.0).expand([T]))
    bd_t = phi_0 + sigma_bd * xi_bd

    cyclical_arrays = []
    for ind_name in p.indicator_names:
        arr = data.get(f"{ind_name}_c")
        if arr is not None and np.any(np.asarray(arr) != 0.0):
            cyclical_arrays.append(np.asarray(arr, dtype=float))
    if cyclical_arrays:
        phi_3 = numpyro.sample(
            "phi_3", dist.Normal(0.0, p.birth_death.phi3_sd).expand([len(cyclical_arrays)])
        )
        for i, arr in enumerate(cyclical_arrays):
            bd_t = bd_t + phi_3[i] * jnp.asarray(arr)
    numpyro.deterministic("bd", bd_t)

    # =============================================================
    # Composite growth signals
    # =============================================================
    g_cont_nsa = g_cont + s_t
    g_total_sa = g_cont + bd_t
    g_total_nsa = g_cont + s_t + bd_t
    numpyro.deterministic("g_total_sa", g_total_sa)
    numpyro.deterministic("g_total_nsa", g_total_nsa)

    # =============================================================
    # QCEW likelihood — truth anchor
    # =============================================================
    qcew_obs = np.asarray(data["qcew_obs"], dtype=int)
    numpyro.sample(
        "obs_qcew",
        dist.StudentT(p.qcew.nu, g_total_nsa[qcew_obs], qcew_sigma),
        obs=jnp.asarray(np.asarray(data["g_qcew"], dtype=float)[qcew_obs]),
    )

    # =============================================================
    # CES likelihood — best-available print, vintage-indexed sigma
    # =============================================================
    alpha_ces = numpyro.sample("alpha_ces", dist.Normal(0.0, p.ces.alpha_sd))
    lambda_ces = numpyro.sample(
        "lambda_ces",
        dist.TruncatedNormal(p.ces.lambda_mu, p.ces.lambda_sd, low=p.ces.lambda_lower),
    )
    n_ces_v = int(data["n_ces_vintages"])
    sigma_ces_sa = numpyro.sample(
        "sigma_ces_sa",
        dist.LogNormal(p.ces.log_sigma_mu, p.ces.log_sigma_sd).expand([n_ces_v]),
    )
    sigma_ces_nsa = numpyro.sample(
        "sigma_ces_nsa",
        dist.LogNormal(p.ces.log_sigma_mu, p.ces.log_sigma_sd).expand([n_ces_v]),
    )

    ces_sa_obs = np.asarray(data["ces_sa_obs"], dtype=int)
    ces_nsa_obs = np.asarray(data["ces_nsa_obs"], dtype=int)
    if len(ces_sa_obs) > 0:
        vidx = np.asarray(data["ces_sa_vintage_idx"], dtype=int)
        numpyro.sample(
            "obs_ces_sa",
            dist.Normal(alpha_ces + lambda_ces * g_total_sa[ces_sa_obs], sigma_ces_sa[vidx]),
            obs=jnp.asarray(np.asarray(data["g_ces_sa"], dtype=float)[ces_sa_obs]),
        )
    if len(ces_nsa_obs) > 0:
        vidx = np.asarray(data["ces_nsa_vintage_idx"], dtype=int)
        numpyro.sample(
            "obs_ces_nsa",
            dist.Normal(alpha_ces + lambda_ces * g_total_nsa[ces_nsa_obs], sigma_ces_nsa[vidx]),
            obs=jnp.asarray(np.asarray(data["g_ces_nsa"], dtype=float)[ces_nsa_obs]),
        )

    # =============================================================
    # Provider likelihoods — config-driven, iid or AR(1) errors
    # =============================================================
    for pp in data["pp_data"]:
        name = str(pp["name"]).lower()
        error_model = pp.get("error_model", "iid")
        obs_idx = np.asarray(pp["pp_obs"], dtype=int)
        if len(obs_idx) == 0:
            continue  # censored backtest iterations can empty a provider
        y_np = np.asarray(pp["g_pp"], dtype=float)[obs_idx]
        y = jnp.asarray(y_np)

        alpha_p = numpyro.sample(f"alpha_{name}", dist.Normal(0.0, p.provider.alpha_sd))
        lam_p = numpyro.sample(
            f"lam_{name}", dist.Normal(p.provider.lambda_mu, p.provider.lambda_sd)
        )
        sigma_p = numpyro.sample(
            f"sigma_pp_{name}",
            dist.InverseGamma(p.provider.sigma_concentration, p.provider.sigma_rate),
        )
        mu_base = alpha_p + lam_p * g_cont_nsa[obs_idx]

        if error_model == "ar1":
            rho_p = numpyro.sample(
                f"rho_{name}", dist.Beta(p.provider.rho_alpha, p.provider.rho_beta)
            )
            mu_cond = jnp.concatenate(
                [mu_base[:1], mu_base[1:] + rho_p * (y[:-1] - mu_base[:-1])]
            )
            sigma_cond = jnp.concatenate(
                [
                    (sigma_p / jnp.sqrt(1.0 - rho_p**2))[None],
                    jnp.broadcast_to(sigma_p, (len(obs_idx) - 1,)),
                ]
            )
            numpyro.sample(f"obs_{name}", dist.Normal(mu_cond, sigma_cond), obs=y)
        elif error_model == "iid":
            numpyro.sample(f"obs_{name}", dist.Normal(mu_base, sigma_p), obs=y)
        else:
            # The reference silently skipped unknown error models — that is a
            # data-loss footgun, not a behavior worth parity.
            raise ValueError(f"unknown provider error_model: {error_model!r}")
