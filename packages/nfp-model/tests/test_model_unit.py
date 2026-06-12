"""Model-structure tests on synthetic data: sites, shapes, gating, branches."""

import jax
import numpy as np
import pytest
from nfp_model import model_inputs, nfp_model
from numpyro import handlers
from numpyro.infer.util import log_density
from synthetic_data import make_synthetic_data


def _trace(data_dict, priors=None):
    inputs = model_inputs(data_dict)
    model = handlers.seed(nfp_model, jax.random.PRNGKey(0))
    return handlers.trace(model).get_trace(data=inputs, priors=priors), inputs


class TestSites:
    def test_expected_sites_present(self):
        tr, _ = _trace(make_synthetic_data())
        expected = {
            "tau", "phi_raw", "mu_g_era", "eps_g",
            "sigma_fourier", "fourier_z",
            "phi_0", "sigma_bd", "xi_bd", "phi_3",
            "sigma_qcew_mid", "sigma_qcew_boundary",
            "alpha_ces", "lambda_ces", "sigma_ces_sa", "sigma_ces_nsa",
            "alpha_g", "lam_g", "sigma_pp_g",
            "obs_qcew", "obs_ces_sa", "obs_ces_nsa", "obs_g",
            "g_cont", "seasonal", "fourier_coefs_det", "bd",
            "g_total_sa", "g_total_nsa",
        }
        assert expected <= set(tr)

    def test_deterministic_shapes(self):
        data = make_synthetic_data(T=40)
        tr, _ = _trace(data)
        T, n_years = 40, data["n_years"]
        for site in ("g_cont", "seasonal", "bd", "g_total_sa", "g_total_nsa"):
            assert tr[site]["value"].shape == (T,), site
        assert tr["fourier_coefs_det"]["value"].shape == (n_years, 8)
        assert tr["mu_g_era"]["value"].shape == (2,)
        assert tr["sigma_fourier"]["value"].shape == (4,)
        assert tr["sigma_ces_sa"]["value"].shape == (3,)
        assert tr["phi_3"]["value"].shape == (2,)

    def test_observed_values_finite(self):
        tr, _ = _trace(make_synthetic_data())
        for site in ("obs_qcew", "obs_ces_sa", "obs_ces_nsa", "obs_g"):
            assert tr[site]["is_observed"]
            assert np.all(np.isfinite(np.asarray(tr[site]["value"]))), site

    def test_log_density_finite(self):
        data = make_synthetic_data()
        tr, inputs = _trace(data)
        params = {
            name: site["value"]
            for name, site in tr.items()
            if site["type"] == "sample" and not site["is_observed"]
        }
        ld, _ = log_density(nfp_model, (), {"data": inputs, "priors": None}, params)
        assert np.isfinite(float(ld))


class TestCyclicalGating:
    def test_all_zero_and_missing_indicators_drop_phi3(self):
        data = make_synthetic_data(with_jolts=False)
        data["claims_c"] = np.zeros(data["T"])
        tr, _ = _trace(data)
        assert "phi_3" not in tr

    def test_single_surviving_indicator(self):
        data = make_synthetic_data(with_jolts=False)
        tr, _ = _trace(data)
        assert tr["phi_3"]["value"].shape == (1,)

    def test_bd_responds_to_covariate(self):
        # With identical seeds, bd differs once a nonzero covariate enters.
        gated = make_synthetic_data(with_claims=False, with_jolts=False)
        full = make_synthetic_data()
        tr_g, _ = _trace(gated)
        tr_f, _ = _trace(full)
        assert "phi_3" not in tr_g
        assert not np.allclose(tr_g["bd"]["value"], tr_f["bd"]["value"])


class TestProviderBranches:
    def test_ar1_branch_samples_rho(self):
        tr, _ = _trace(make_synthetic_data(error_model="ar1"))
        assert "rho_g" in tr
        assert "obs_g" in tr

    def test_iid_branch_has_no_rho(self):
        tr, _ = _trace(make_synthetic_data(error_model="iid"))
        assert "rho_g" not in tr

    def test_provider_without_observations_is_skipped(self):
        tr, _ = _trace(make_synthetic_data(provider_obs=False))
        assert "alpha_g" not in tr
        assert "obs_g" not in tr

    def test_unknown_error_model_raises(self):
        with pytest.raises(ValueError, match="error_model"):
            _trace(make_synthetic_data(error_model="garch"))

    def test_dict_config_duck_typing(self):
        tr, _ = _trace(make_synthetic_data(error_model="ar1", config_as_dict=True))
        assert "rho_g" in tr


class TestStructuralBranches:
    def test_no_ces_observations_drops_ces_likelihood(self):
        tr, _ = _trace(make_synthetic_data(with_ces=False))
        assert "obs_ces_sa" not in tr
        assert "obs_ces_nsa" not in tr
        assert "obs_qcew" in tr

    def test_no_era_index_uses_scalar_mu_g(self):
        tr, _ = _trace(make_synthetic_data(era=False))
        assert "mu_g" in tr
        assert "mu_g_era" not in tr

    def test_composites_are_consistent(self):
        tr, _ = _trace(make_synthetic_data())
        g = np.asarray(tr["g_cont"]["value"])
        s = np.asarray(tr["seasonal"]["value"])
        bd = np.asarray(tr["bd"]["value"])
        np.testing.assert_allclose(tr["g_total_sa"]["value"], g + bd, rtol=1e-12)
        np.testing.assert_allclose(tr["g_total_nsa"]["value"], g + s + bd, rtol=1e-12)

    def test_float64_enabled_by_package_import(self):
        tr, _ = _trace(make_synthetic_data())
        assert np.asarray(tr["g_cont"]["value"]).dtype == np.float64


class TestBoundary:
    def test_no_data_package_imports(self):
        """The inference layer must not import any nfp_* data package."""
        import pathlib

        import nfp_model

        src = pathlib.Path(nfp_model.__file__).parent
        offenders = []
        for py in src.rglob("*.py"):
            for line in py.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("import nfp_", "from nfp_")) and not stripped.startswith(
                    ("import nfp_model", "from nfp_model")
                ):
                    offenders.append(f"{py.name}: {stripped}")
        assert not offenders, offenders
