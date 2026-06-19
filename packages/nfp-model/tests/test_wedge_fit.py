import numpy as np
import pytest
from nfp_model.wedge import fit_wedge, wedge_pred_draws


@pytest.mark.slow
def test_fit_recovers_drift_and_predicts_target():
    rng = np.random.default_rng(0)
    T = 60
    moy = np.array([(i % 12) + 1 for i in range(T)])
    true_season = np.array([0, 5, 30, -10, -5, 20, 25, 10, -15, -20, 0, -40], float)
    true_season -= true_season.mean()
    y = 8.0 + true_season[moy - 1] + rng.normal(0, 20, T)
    mask = np.ones(T, bool)
    mask[-1] = False  # last = target, unobserved
    data = {"y": np.where(mask, y, 0.0), "month_of_year": moy, "T": T, "mask": mask,
            "X_intervention": np.zeros((T, 0)), "iv_prior_mean": np.zeros(0),
            "iv_prior_sd": np.zeros(0)}
    fit = fit_wedge(data, settings="light", seed=0)
    assert fit.num_divergences == 0
    drift_mean = fit.posterior["drift"].mean()
    assert 8.0 - 12 < drift_mean < 8.0 + 12             # recovered, loosely
    draws = wedge_pred_draws(fit, target_idx=T - 1, seed=1)
    assert draws.ndim == 1 and draws.shape[0] == (
        fit.posterior["drift"].shape[0] * fit.posterior["drift"].shape[1])
