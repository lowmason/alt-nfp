import numpy as np
from nfp_vintages.wedge_diagnostics import calibrate_intervention_sd, decomposition_residual


def test_decomposition_residual_small_when_wedge_tracks_90():
    rng = np.random.default_rng(0)
    g90 = rng.normal(15, 20, 80)
    wedge = g90 + rng.normal(0, 3, 80)          # small SA-additivity residual r
    out = decomposition_residual(wedge, g90)
    assert out["r_std"] < out["wedge_std"]      # r is a small fraction of wedge variance
    assert abs(out["r_mean"]) < 5


def test_calibrate_intervention_sd_is_robust():
    fed = np.array([-45.0, -52.0, -48.0])       # observed federal moves around a RIF
    sd = calibrate_intervention_sd(fed, baseline_sd=25.0)
    assert sd > 0
