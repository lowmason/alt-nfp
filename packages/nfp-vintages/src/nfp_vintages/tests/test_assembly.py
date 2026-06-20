import numpy as np
from nfp_vintages.assembly import assemble_total


def test_alignment_and_pure_sum_eta0():
    priv_growth = np.zeros((2, 50))                 # growth 0 -> change 0
    wedge = np.full((3, 100), 7.0)                  # change-k draws
    total = assemble_total(priv_growth, wedge, prev_index=1000.0, idx_to_level=1.0, eta=0.0)
    assert total.shape == (300,)                    # N = wedge chains*draws
    assert np.allclose(total, 7.0)                  # 0 (private) + 7 (wedge)
    assert not np.isnan(total).any()


def test_coupling_is_point_invariant():
    rng = np.random.default_rng(0)
    priv_growth = rng.normal(0.001, 0.0005, (4, 100))
    wedge = rng.normal(10.0, 20.0, (4, 100))
    base = assemble_total(priv_growth, wedge, prev_index=1500.0, idx_to_level=1.0, eta=0.0)
    coup = assemble_total(priv_growth, wedge, prev_index=1500.0, idx_to_level=1.0, eta=0.5, seed=0)
    assert abs(base.mean() - coup.mean()) < 1.0     # mean-zero z -> point ~invariant
    assert coup.std() > base.std()                  # intervals widen
