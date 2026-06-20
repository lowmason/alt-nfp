"""End-to-end vmapped batch vs serial fits on synthetic data (slow).

Statistical agreement only — exactness is proven at the log-density level
in test_batch_unit.py; here we check the whole vmapped MCMC machinery
(adaptation per date, masks under NUTS, in-graph reduction) lands on the
same posterior the serial path finds.
"""

import numpy as np
import pytest
from nfp_model import (
    SamplerSettings,
    fit_model,
    fit_model_batch,
    model_inputs,
    pad_model_inputs,
)
from nfp_model.parity import _mcse, collect_parity_arrays, compare_scalar
from synthetic_data import make_synthetic_data

pytestmark = pytest.mark.slow

TINY = SamplerSettings(
    num_samples=300, num_warmup=300, num_chains=2,
    target_accept=0.95, max_tree_depth=10, chain_method="vectorized",
)
BASE_INDEX = 100.0
IDX_TO_LEVEL = 1500.0
SCALARS = ("draws__tau", "draws__phi_raw", "draws__alpha_ces", "draws__lambda_ces",
           "draws__sigma_pp_g")


@pytest.fixture(scope="module")
def batch_and_serial():
    datasets = [
        make_synthetic_data(40, era_break_at=25),
        make_synthetic_data(50, era_break_at=25),
    ]
    inputs = [model_inputs(d) for d in datasets]
    bi = pad_model_inputs(inputs)
    batch = fit_model_batch(
        bi, settings=TINY, seed=0, base_index=BASE_INDEX, idx_to_level=IDX_TO_LEVEL
    )
    serial = []
    for i, d in enumerate(datasets):
        fit = fit_model(d, settings=TINY, seed=100 + i)
        serial.append(
            collect_parity_arrays(
                fit,
                base_index=BASE_INDEX,
                idx_to_level=IDX_TO_LEVEL,
                c_idx=int(d["T"]) - 1,
            )
        )
    return bi, batch, serial


class TestBatchMatchesSerial:
    def test_scalar_means_agree(self, batch_and_serial):
        _, batch, serial = batch_and_serial
        for i in range(batch.n_dates):
            got, _ = batch.date_arrays(i)
            want, _ = serial[i]
            for key in SCALARS:
                g, w = got[key], want[key]
                pooled_sd = np.sqrt((g.std() ** 2 + w.std() ** 2) / 2)
                d = abs(float(g.mean()) - float(w.mean()))
                assert d <= 0.6 * pooled_sd, (
                    f"date {i} {key}: |Δmean|={d:.3e} vs 0.6·SD={0.6 * pooled_sd:.3e}"
                )

    def test_latent_path_agrees(self, batch_and_serial):
        _, batch, serial = batch_and_serial
        for i in range(batch.n_dates):
            got, _ = batch.date_arrays(i)
            want, _ = serial[i]
            frac = np.abs(got["path_mean__g_total_sa"] - want["path_mean__g_total_sa"])
            frac /= np.maximum(want["path_sd__g_total_sa"], 1e-12)
            assert frac.max() <= 0.6, f"date {i}: max|Δ|/SD={frac.max():.3f}"

    def test_nowcast_agrees(self, batch_and_serial):
        """Draw-level z-test + MCSE-derived change_k bound (the A3 criteria)."""
        _, batch, serial = batch_and_serial
        for i in range(batch.n_dates):
            got, got_meta = batch.date_arrays(i)
            want, want_meta = serial[i]
            rows = compare_scalar(
                "pred_draws", want["nowcast_pred_draws"], got["nowcast_pred_draws"]
            )
            assert all(r.passed for r in rows), f"date {i}: {[r.detail for r in rows]}"
            level = got_meta["nowcast_index"] * IDX_TO_LEVEL
            pooled_mcse = np.sqrt(
                _mcse(want["nowcast_pred_draws"]) ** 2
                + _mcse(got["nowcast_pred_draws"]) ** 2
            )
            bound = max(4.0 * level * pooled_mcse, 1.0)
            d = abs(got_meta["nowcast_change_k"] - want_meta["nowcast_change_k"])
            assert d <= bound, f"date {i}: |Δnowcast|={d:.2f}k, bound={bound:.2f}k"

    def test_no_divergences(self, batch_and_serial):
        _, batch, _ = batch_and_serial
        for i in range(batch.n_dates):
            _, meta = batch.date_arrays(i)
            assert meta["num_divergences"] == 0


class TestBatchMechanics:
    def test_shapes_and_meta(self, batch_and_serial):
        bi, batch, _ = batch_and_serial
        assert batch.n_dates == 2
        for i, T in enumerate(bi.T_real):
            arrays, meta = batch.date_arrays(i)
            assert meta["T"] == int(T)
            assert meta["c_idx"] == int(T) - 1
            assert meta["n_cyclical"] == 2
            assert arrays["path_mean__g_total_sa"].shape == (int(T),)
            assert arrays["nowcast_pred_mean"].shape == (int(T),)
            assert arrays["nowcast_pred_draws"].shape == (TINY.num_chains, TINY.num_samples)
            assert arrays["draws__tau"].shape == (TINY.num_chains, TINY.num_samples)

    def test_seed_reproducible(self, batch_and_serial):
        bi, batch, _ = batch_and_serial
        again = fit_model_batch(
            bi, settings=TINY, seed=0, base_index=BASE_INDEX, idx_to_level=IDX_TO_LEVEL
        )
        np.testing.assert_array_equal(batch.arrays["draws__tau"], again.arrays["draws__tau"])
        np.testing.assert_array_equal(
            batch.arrays["nowcast_change_k"], again.arrays["nowcast_change_k"]
        )
