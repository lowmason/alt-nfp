import os

import numpy as np
import pytest
from dotenv import load_dotenv

pytestmark = [pytest.mark.slow, pytest.mark.real_store]


@pytest.mark.skipif(not os.environ.get("NFP_STORE_URI"), reason="needs real store")
def test_wedge_fit_converges_on_clean_window():
    load_dotenv(".env")
    from datetime import date

    from nfp_ingest.wedge_data import build_wedge_model_data
    from nfp_model.wedge import fit_wedge
    from numpyro.diagnostics import gelman_rubin

    data = build_wedge_model_data(as_of=None, target_month=date(2026, 1, 1))
    # The gate intentionally fits settings="default" (4 chains, so R-hat is
    # well-defined) — it certifies the model, not the production preset
    # (cmd_total runs PRESET="light" for speed).
    fit = fit_wedge(data, settings="default", seed=0)
    # Build gate (spec §10): convergence, not accuracy.
    assert fit.num_divergences == 0
    assert np.isfinite(fit.posterior["mu"]).all()
    # R-hat over the key SAMPLED sites (skip the deterministics mu/season).
    # fit.posterior arrays are (chains, draws[, ...]); gelman_rubin reduces over
    # the chain/draw axes, leaving any trailing event dims (e.g. season_raw → 11).
    rhat = max(
        float(np.max(gelman_rubin(fit.posterior[site])))
        for site in ("drift", "sigma", "tau_season", "season_raw")
    )
    assert rhat <= 1.01, f"R-hat {rhat:.4f} exceeds 1.01"
    # The heavier spec §10 gate parts — 80% interval coverage ∈ [60%, 95%],
    # ppc-mean RMSE ≤ 2×23k, per-calendar-month R-hat warn — are deferred to the
    # Bloomberg port.
