import numpy as np
import jax
import numpyro
from numpyro.infer import Predictive
from nfp_model.wedge import wedge_model, WEDGE_DETERMINISTIC_SITES

numpyro.enable_x64()


def _data(T=24, K=1):
    rng = np.random.default_rng(0)
    return {
        "y": rng.normal(10, 24, T), "month_of_year": np.array([(i % 12) + 1 for i in range(T)]),
        "T": T, "mask": np.ones(T, bool),
        "X_intervention": np.zeros((T, K)), "iv_prior_mean": np.zeros(K),
        "iv_prior_sd": np.full(K, 20.0),
    }


def test_priors_run_and_sum_to_zero():
    d = _data()
    pred = Predictive(wedge_model, num_samples=8)(jax.random.PRNGKey(0), data=d)
    season = np.asarray(pred["season"])           # (8, 12)
    assert season.shape[1] == 12
    assert np.allclose(season.sum(axis=1), 0.0, atol=1e-8)   # sum-to-zero pin
    assert np.asarray(pred["mu"]).shape == (8, d["T"])


def test_units_are_thousands():
    d = _data()
    pred = Predictive(wedge_model, num_samples=200)(jax.random.PRNGKey(1), data=d)
    # drift ~ Normal(0,50): O(tens), NOT tens-of-thousands
    assert np.abs(np.asarray(pred["mu"])).mean() < 1000
