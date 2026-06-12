"""Unit tests for nfp_model.batch: padding, masking, structure assertions.

The core proof is exact log-density equality: substitute one parameter draw
into both the padded+masked model and the unpadded model — every likelihood
site's log-prob must agree (padding contributes exactly zero) and the
deterministic paths must agree on the real calendar. MCMC agreement (the
smoke test) follows from this; this is the part that can be exact.
"""

import numpy as np
import pytest
from nfp_model import ModelPriors, model_inputs, nfp_model, pad_model_inputs
from nfp_model.batch import assemble_data
from numpyro import handlers
from synthetic_data import make_synthetic_data

PRIORS = ModelPriors()


def _inputs(T: int, **kw) -> dict:
    return model_inputs(make_synthetic_data(T, era_break_at=25, **kw))


def _crop(value: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Leading-slice a padded parameter down to the unpadded site shape."""
    if value.shape == tuple(target_shape):
        return value
    return value[tuple(slice(0, n) for n in target_shape)]


def _site_values(data: dict, substituted: dict) -> dict:
    """Observed-site log-probs (post-mask) + deterministic values."""
    with handlers.trace() as tr, handlers.seed(rng_seed=2), \
            handlers.substitute(data=substituted):
        nfp_model(data, PRIORS)
    out: dict = {}
    for name, site in tr.items():
        if site["type"] == "sample" and site.get("is_observed", False):
            out[name] = float(np.sum(np.asarray(site["fn"].log_prob(site["value"]))))
        elif site["type"] == "deterministic":
            out[f"det:{name}"] = np.asarray(site["value"])
    return out


class TestPaddedLogDensity:
    @pytest.mark.parametrize("error_model", ["iid", "ar1"])
    def test_padding_changes_nothing(self, error_model):
        inputs = [_inputs(40, error_model=error_model), _inputs(50, error_model=error_model)]
        bi = pad_model_inputs(inputs)
        date = 0  # the shorter date — actually padded
        padded = assemble_data(bi, {k: np.asarray(v[date]) for k, v in bi.batched.items()})

        # One parameter draw at padded shapes; crop to unpadded shapes.
        with handlers.seed(rng_seed=0):
            padded_tr = handlers.trace(lambda: nfp_model(padded, PRIORS)).get_trace()
        params_padded = {
            k: np.asarray(s["value"]) for k, s in padded_tr.items()
            if s["type"] == "sample" and not s.get("is_observed", False)
        }
        with handlers.seed(rng_seed=1):
            shape_tr = handlers.trace(lambda: nfp_model(inputs[date], PRIORS)).get_trace()
        params_unpadded = {
            k: _crop(params_padded[k], np.shape(s["value"]))
            for k, s in shape_tr.items()
            if s["type"] == "sample" and not s.get("is_observed", False)
        }

        got = _site_values(padded, params_padded)
        want = _site_values(inputs[date], params_unpadded)

        lik_sites = [k for k in want if not k.startswith("det:")]
        assert set(lik_sites) <= set(got)
        for site in lik_sites:
            assert got[site] == pytest.approx(want[site], rel=1e-12, abs=1e-10), site

        T = int(bi.T_real[date])
        for var in ("g_cont", "seasonal", "bd", "g_total_sa", "g_total_nsa"):
            np.testing.assert_allclose(
                got[f"det:{var}"][:T], want[f"det:{var}"], rtol=1e-12, atol=1e-14,
            )

    def test_longest_date_is_unpadded_identity(self):
        """The T_max member's padded slice differs only in obs-array padding."""
        inputs = [_inputs(40), _inputs(50)]
        bi = pad_model_inputs(inputs)
        sl = {k: np.asarray(v[1]) for k, v in bi.batched.items()}
        np.testing.assert_array_equal(sl["qcew_mask"], np.ones_like(sl["qcew_mask"]))
        np.testing.assert_array_equal(sl["g_ces_sa"], np.asarray(inputs[1]["g_ces_sa"]))


class TestPadStructure:
    def test_static_calendar_comes_from_longest(self):
        bi = pad_model_inputs([_inputs(40), _inputs(50)])
        assert bi.static["T"] == 50
        assert bi.static["n_years"] == 5
        assert len(bi.static["month_of_year"]) == 50
        assert len(bi.static["era_idx"]) == 50
        assert bi.static["cyclical_active"] == ("claims", "jolts")

    def test_masks_count_real_observations(self):
        inputs = [_inputs(40), _inputs(50)]
        bi = pad_model_inputs(inputs)
        for i, d in enumerate(inputs):
            assert bi.batched["qcew_mask"][i].sum() == len(d["qcew_obs"])
            assert bi.batched["ces_sa_mask"][i].sum() == len(d["ces_sa_obs"])
            assert bi.batched["pp__G__mask"][i].sum() == len(d["pp_data"][0]["pp_obs"])

    def test_c_idx_defaults_to_last_real_state(self):
        bi = pad_model_inputs([_inputs(40), _inputs(50)])
        np.testing.assert_array_equal(bi.c_idx, [39, 49])

    def test_explicit_c_idx_validated(self):
        inputs = [_inputs(40), _inputs(50)]
        with pytest.raises(ValueError, match="c_idx"):
            pad_model_inputs(inputs, c_idx=[40, 49])  # 40 >= T_real[0]
        bi = pad_model_inputs(inputs, c_idx=[12, 24])
        np.testing.assert_array_equal(bi.c_idx, [12, 24])

    def test_mixed_cyclical_gating_raises(self):
        with pytest.raises(ValueError, match="cyclical gating"):
            pad_model_inputs([_inputs(40), _inputs(50, with_claims=False)])

    def test_mixed_provider_structure_raises(self):
        with pytest.raises(ValueError, match="provider structure"):
            pad_model_inputs([_inputs(40, error_model="iid"), _inputs(50, error_model="ar1")])

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="no observations"):
            pad_model_inputs([_inputs(40), _inputs(50, provider_obs=False)])

    def test_mixed_era_presence_raises(self):
        with pytest.raises(ValueError, match="era_idx"):
            pad_model_inputs([_inputs(40), _inputs(50, era=False)])

    def test_calendar_prefix_violation_raises(self):
        a, b = _inputs(40), _inputs(50)
        a["month_of_year"] = a["month_of_year"].copy()
        a["month_of_year"][5] = (a["month_of_year"][5] + 1) % 12
        with pytest.raises(ValueError, match="month_of_year"):
            pad_model_inputs([a, b])

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pad_model_inputs([])

    def test_assemble_round_trips_provider_fields(self):
        bi = pad_model_inputs([_inputs(40), _inputs(50)])
        data = assemble_data(bi, {k: np.asarray(v[0]) for k, v in bi.batched.items()})
        (pp,) = data["pp_data"]
        assert pp["name"] == "G"
        assert pp["error_model"] == "iid"
        assert set(pp) == {"name", "error_model", "g_pp", "pp_obs", "mask"}
        assert "c_idx" not in data
