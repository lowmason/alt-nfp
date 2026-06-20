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


def _is_exempt_nfp_module(name: str) -> bool:
    """Return True if *name* is nfp_model itself or one of its sub-modules."""
    return name == "nfp_model" or name.startswith("nfp_model.")


def _find_nfp_boundary_violations(source: str) -> list[str]:
    """AST-scan *source* and return descriptions of any forbidden nfp_* imports.

    Catches:
    - ``import nfp_foo`` / ``import numpy, nfp_foo`` (ast.Import, per alias)
    - ``from nfp_foo import x`` (ast.ImportFrom)
    - ``importlib.import_module("nfp_foo")`` / ``__import__("nfp_foo")``
      with a constant string first argument (ast.Call)

    nfp_model and nfp_model.* sub-modules are treated as allowed.
    """
    import ast

    tree = ast.parse(source)
    violations: list[str] = []

    for node in ast.walk(tree):
        # --- plain imports: import foo, bar ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("nfp_") and not _is_exempt_nfp_module(name):
                    violations.append(f"import {name}")

        # --- from-imports: from foo import ... ---
        elif isinstance(node, ast.ImportFrom):
            mod = node.module  # may be None for relative imports
            if mod is not None and mod.startswith("nfp_") and not _is_exempt_nfp_module(mod):
                violations.append(f"from {mod} import ...")

        # --- dynamic imports: importlib.import_module("nfp_...") / __import__("nfp_...") ---
        elif isinstance(node, ast.Call):
            func = node.func
            is_dynamic = (
                # importlib.import_module(...)
                (isinstance(func, ast.Attribute) and func.attr == "import_module")
                # bare import_module(...) or __import__(...)
                or (isinstance(func, ast.Name) and func.id in ("import_module", "__import__"))
            )
            if is_dynamic and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    mod = first.value
                    if mod.startswith("nfp_") and not _is_exempt_nfp_module(mod):
                        violations.append(f"dynamic import of {mod!r}")

    return violations


class TestBoundaryChecker:
    """Self-tests: verify _find_nfp_boundary_violations catches what it must."""

    def test_plain_import_flagged(self):
        assert _find_nfp_boundary_violations("import nfp_ingest")

    def test_from_import_flagged(self):
        assert _find_nfp_boundary_violations("from nfp_ingest import x")

    def test_multi_import_flagged(self):
        # Single ast.Import node with two aliases — only nfp_ingest is forbidden.
        violations = _find_nfp_boundary_violations("import numpy, nfp_ingest")
        assert any("nfp_ingest" in v for v in violations)

    def test_dynamic_import_module_flagged(self):
        assert _find_nfp_boundary_violations('import importlib\nimportlib.import_module("nfp_ingest")')

    def test_dunder_import_flagged(self):
        assert _find_nfp_boundary_violations('__import__("nfp_ingest")')

    def test_nfp_model_self_import_allowed(self):
        # nfp_model importing its own sub-modules must not be flagged.
        assert not _find_nfp_boundary_violations("import nfp_model\nfrom nfp_model.config import ModelPriors")

    def test_nfp_model_submodule_import_allowed(self):
        assert not _find_nfp_boundary_violations("from nfp_model.data import model_inputs")

    def test_clean_source_allowed(self):
        src = "import jax\nimport numpy as np\nfrom numpyro import handlers"
        assert not _find_nfp_boundary_violations(src)

    def test_relative_import_not_flagged(self):
        # Relative imports have module=None; must not raise.
        assert not _find_nfp_boundary_violations("from . import utils")

    def test_dynamic_import_with_variable_not_flagged(self):
        # Dynamic import of a variable is statically undetectable; must not crash.
        violations = _find_nfp_boundary_violations("importlib.import_module(mod_name)")
        assert not violations  # can't flag non-constant args


class TestBoundary:
    def test_no_data_package_imports(self):
        """The inference layer must not import any nfp_* data package."""
        import pathlib

        import nfp_model

        src = pathlib.Path(nfp_model.__file__).parent
        offenders = []
        for py in src.rglob("*.py"):
            # Tests now live under src/nfp_model/tests/ and import data packages
            # by design; the boundary governs shippable inference code only.
            if "tests" in py.relative_to(src).parts:
                continue
            source = py.read_text()
            for violation in _find_nfp_boundary_violations(source):
                offenders.append(f"{py.name}: {violation}")
        assert not offenders, offenders
