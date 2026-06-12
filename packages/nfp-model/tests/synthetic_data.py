"""Shared synthetic ModelData builders for nfp-model tests.

Small, fully synthetic dicts shaped exactly like
``nfp_ingest.model_data.build_model_data`` output (minus frames) — no store,
no network, no proprietary values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class FakeProviderConfig:
    """Attribute-style provider config (mimics nfp_lookups.ProviderConfig)."""

    name: str = "G"
    error_model: str = "iid"


def make_synthetic_data(
    T: int = 40,
    *,
    error_model: str = "iid",
    with_claims: bool = True,
    with_jolts: bool = True,
    with_ces: bool = True,
    provider_obs: bool = True,
    era: bool = True,
    config_as_dict: bool = False,
) -> dict:
    rng = np.random.default_rng(7)
    n_years = (T + 11) // 12

    g_ces_sa = rng.normal(0.0015, 0.002, T)
    g_ces_nsa = g_ces_sa + rng.normal(0.0, 0.003, T)
    g_ces_sa[-1] = np.nan
    g_ces_nsa[-1] = np.nan
    ces_obs = np.arange(0, T - 1) if with_ces else np.array([], dtype=int)
    vintage_idx = np.full(len(ces_obs), 2, dtype=int)
    if len(vintage_idx) >= 2:
        vintage_idx[-1] = 0
        vintage_idx[-2] = 1

    qcew_obs = np.arange(0, T - 8)
    g_qcew = rng.normal(0.0015, 0.004, T)
    g_qcew[T - 8:] = np.nan
    qcew_is_m2 = np.array([i % 3 == 1 for i in range(len(qcew_obs))])
    qcew_noise_mult = np.where(qcew_is_m2, 1.0, 2.5)

    g_pp = np.full(T, np.nan)
    pp_obs = np.arange(3, T - 2) if provider_obs else np.array([], dtype=int)
    g_pp[pp_obs] = rng.normal(0.0015, 0.005, len(pp_obs))

    claims_c = rng.normal(0.0, 0.5, T)
    claims_c[T - 2:] = 0.0
    jolts_c = rng.normal(0.0, 0.4, T)
    jolts_c[T - 3:] = 0.0

    config: object = {"name": "G", "error_model": error_model} if config_as_dict \
        else FakeProviderConfig(error_model=error_model)

    def _month_offset(i: int) -> date:
        y, m = divmod(i, 12)
        return date(2017 + y, m + 1, 12)

    return {
        "T": T,
        "n_years": n_years,
        "n_ces_vintages": 3,
        "n_providers": 1,
        "dates": [_month_offset(i) for i in range(T)],
        "ces_vintage_map": {0: 0, 1: 1, 2: 2},
        "month_of_year": np.array([i % 12 for i in range(T)]),
        "year_of_obs": np.array([i // 12 for i in range(T)]),
        "era_idx": np.array([0] * (T - 10) + [1] * 10) if era else None,
        "g_ces_sa": g_ces_sa,
        "ces_sa_obs": ces_obs,
        "ces_sa_vintage_idx": vintage_idx,
        "g_ces_nsa": g_ces_nsa,
        "ces_nsa_obs": ces_obs.copy(),
        "ces_nsa_vintage_idx": vintage_idx.copy(),
        "g_qcew": g_qcew,
        "qcew_obs": qcew_obs,
        "qcew_is_m2": qcew_is_m2,
        "qcew_noise_mult": qcew_noise_mult,
        "pp_data": [
            {
                "name": "G",
                "config": config,
                "g_pp": g_pp,
                "pp_obs": pp_obs,
                "emp_col": "g_employment",
                "births": None,
                "births_obs": None,
            }
        ],
        "birth_rate": np.full(T, np.nan),
        "bd_proxy": np.full(T, np.nan),
        "bd_qcew_lagged": np.full(T, np.nan),
        "claims_c": claims_c if with_claims else None,
        "jolts_c": jolts_c if with_jolts else None,
    }
