# nfp-model

JAX/NumPyro state-space model for NFP nowcasting — the inference layer.
ModelData arrays in, posterior out.

## Overview

Faithful translation of the frozen PyMC reference
(`~/Projects/alt_nfp/packages/nfp-model-hmc`), gated by A3 posterior parity
(`plans/5-a3_model_parity.md`). Provides:
- **Model** (`model.py`): `nfp_model(data, priors)` — non-centered AR(1)
  latent with era means, Fourier-GRW seasonal, structural birth/death with
  gated cyclical covariates (claims/jolts), QCEW Student-t anchor with
  tiered LogNormal sigmas, vintage-indexed CES, per-provider iid/AR(1)
  likelihoods. Site names match the reference for key-for-key comparison.
- **Config** (`config.py`): `ModelPriors` + `SamplerSettings`/`PRESETS`
  (default 4000/3000/4chains, light 2000/2000/2) — every default pinned to
  the frozen reference; change only behind a new parity baseline.
- **Data intake** (`data.py`): `model_inputs()` normalizes
  `build_model_data` dicts; `from_snapshot(arrays, meta)` rebuilds from
  snapshot artifacts (inverse of `nfp_ingest.snapshots.collect_snapshot`).
- **Sampling** (`sampling.py`): `fit_model(data, settings=…, seed=…)` →
  `FitResult` with posterior as `(chains, draws, …)` numpy arrays
  (deterministic sites reconstructed via `Predictive`).
- **Nowcast** (`nowcast.py`): CES observation-equation transform + the
  reference backtest's index/jobs-added arithmetic.
- **Parity** (`parity.py`): A3 gate criteria (MCSE z-tests, SD-ratio bands,
  path and nowcast comparison) shared by `scripts/run_a3_parity.py` and the
  spot-check test.

## Hard boundary

**`nfp_model` imports only jax, numpyro, numpy** — no `nfp_*` packages, no
polars, no store access, no plotting. Enforced by
`tests/test_model_unit.py::TestBoundary`. Provider configs are duck-typed
(dataclass or dict). Importing the package enables **JAX float64 globally**
(`numpyro.enable_x64()`): the parity contract is double precision.

## Tech Stack

- **Dependencies**: jax, numpyro, numpy — nothing else (tests may import
  data packages; `src/` must not)
- **Build**: hatchling; requires-python >= 3.12 (current jax)

## Key Commands

```bash
pytest packages/nfp-model -m "not slow"      # fast structure/intake tests
pytest packages/nfp-model -m slow            # MCMC smoke (~30 s)
NFP_A3_PARITY=1 pytest packages/nfp-model/tests/test_parity_golden.py  # minutes

# Full A3 parity (14 fits; restartable; see plans/5)
uv run python scripts/run_a3_parity.py fit data/golden_a3_staging data/a3
uv run python scripts/run_a3_parity.py compare data/golden_a3_staging data/a3
```

## Test Mapping

- `test_config.py` — defaults pinned to frozen-reference literals
- `test_data.py` — `model_inputs` normalization, snapshot round trip
  (schema v2 `error_model`, v1 fallback)
- `test_model_unit.py` — sites/shapes/log-density on synthetic data,
  cyclical gating, iid/ar1/empty-provider branches, the import boundary
- `test_model_smoke.py` — tiny-MCMC end-to-end (`slow`): layout, chain-major
  deterministic consistency, seed reproducibility, nowcast arithmetic
- `test_parity_golden.py` — single-fixture parity spot check vs
  `s3://alt-nfp/golden/a3` (opt-in: `NFP_A3_PARITY=1` + store env;
  manifest committed in `tests/golden/`)
- `synthetic_data.py` — shared synthetic ModelData builders (not a test)

## Key Patterns

- **Parametrization vs parity**: the reference's centered
  `GaussianRandomWalk` for Fourier coefficients is implemented non-centered
  (identical prior law; posterior-invariant; better geometry). The AR(1)
  latent and BD shocks were already non-centered in the reference and are
  ported as-is. `g0` uses the *innovation* SD (`sigma_g`), not the
  stationary SD — that matches the reference, don't "fix" it.
- **Covariate gating**: all-zero or missing cyclical arrays are dropped
  before `phi_3` is sampled (reference behavior; avoids unidentified
  parameters in censored backtests). Order is `ModelPriors.indicator_names`.
- **Unknown provider `error_model` raises** (the reference silently skipped
  the likelihood — deliberate deviation, can't affect parity).
- **The model never sees a vintage_date**: all censoring happens in
  `nfp_ingest` before arrays arrive (A2 boundary).
