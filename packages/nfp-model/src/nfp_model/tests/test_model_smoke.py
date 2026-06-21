"""End-to-end sampling smoke on synthetic data (tiny chains, marked slow)."""

import numpy as np
import pytest
from nfp_model import SamplerSettings, fit_model, nowcast_summary
from synthetic_data import make_synthetic_data

TINY = SamplerSettings(num_samples=40, num_warmup=40, num_chains=2)


@pytest.fixture(scope="module")
def tiny_fit():
    return fit_model(make_synthetic_data(), settings=TINY, seed=1)


@pytest.mark.slow
class TestFitSmoke:
    def test_posterior_layout(self, tiny_fit):
        post = tiny_fit.posterior
        T = 40
        assert post["tau"].shape == (2, 40)
        assert post["mu_g_era"].shape == (2, 40, 2)
        assert post["g_cont"].shape == (2, 40, T)
        assert post["g_total_sa"].shape == (2, 40, T)
        assert post["fourier_coefs_det"].shape == (2, 40, 4, 8)
        assert tiny_fit.num_divergences >= 0
        for name, arr in post.items():
            assert np.all(np.isfinite(arr)), name

    def test_deterministics_consistent_with_draws(self, tiny_fit):
        """Predictive reshape must align chain-major with the sample sites."""
        post = tiny_fit.posterior
        data = make_synthetic_data()
        bd = post["phi_0"][:, :, None] + post["sigma_bd"][:, :, None] * post["xi_bd"]
        cyclical = [data["claims_c"], data["jolts_c"]]  # gating order: claims, jolts
        for i, arr in enumerate(cyclical):
            bd = bd + post["phi_3"][:, :, i, None] * arr[None, None, :]
        np.testing.assert_allclose(bd, post["bd"], rtol=1e-10, atol=1e-12)

    def test_nowcast_summary(self, tiny_fit):
        nc = nowcast_summary(
            tiny_fit.posterior, base_index=100.0, idx_to_level=1500.0
        )
        assert nc["c_idx"] == 39
        assert np.isfinite(nc["nowcast_growth"])
        assert np.isfinite(nc["nowcast_change_k"])
        assert nc["pred_mean"].shape == (40,)
        assert nc["pred_draws"].shape == (2, 40)
        # change_k must equal the level delta implied by the mean growth path
        path = 100.0 * np.exp(np.cumsum(nc["pred_mean"]))
        expected = (path[-1] - path[-2]) * 1500.0
        assert nc["nowcast_change_k"] == pytest.approx(expected)

    def test_seed_reproducibility(self, tiny_fit):
        again = fit_model(make_synthetic_data(), settings=TINY, seed=1)
        np.testing.assert_array_equal(again.posterior["tau"], tiny_fit.posterior["tau"])
