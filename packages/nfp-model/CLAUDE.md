# nfp-model

JAX/NumPyro state-space model for NFP nowcasting — the inference layer.
ModelData arrays in, posterior out.

## Overview

Faithful translation of the frozen PyMC reference
(`~/Projects/alt_nfp/packages/nfp-model-hmc`), gated by A3 posterior parity
(`specs/plans/5-a3_model_parity.md`).

**Parity here is port-fidelity — the JAX model reproduces the reference's
posterior — not correctness: the reference is a WIP, and model correctness is
validated against external ground truth (see `specs/plans/0`). A3 parity is banked,
but a correctness-driven change to a pinned default is legitimate behind a new
baseline.** (The Track-B government-wedge model `wedge.py` is the exception to the
A3 frame: it is a *separate* model, not a translation of the reference, and is
A3-parity-free — see Provides below.)

Provides:
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
  spot-check test. Also the A4 serial-vs-batched equivalence instrument.
- **Batch** (`batch.py`, A4): `pad_model_inputs(list_of_inputs)` pads a
  date grid to common shapes (likelihood masks for padded slots; static
  calendar/structure asserted uniform and closed over);
  `fit_model_batch(...)` fits the whole grid in one `jit(vmap(MCMC.run))`
  program (vectorized inner chains) and reduces each date *in graph* to
  the A3 fixture schema (`BatchFitResult.date_arrays(i)`).
- **Government wedge** (`wedge.py`, Track B — a **separate, standalone** model):
  a NumPyro change-space STS for the government wedge
  `published_00 − published_05` — constant drift + shrunk monthly-seasonal block
  + an announcement-priored intervention layer + masked iid-Normal likelihood.
  Exposes `wedge_model`, `fit_wedge`, `wedge_pred_draws`,
  `WEDGE_DETERMINISTIC_SITES` (all re-exported from `__init__`). Honors the import
  boundary (jax/numpyro/numpy only) and **reimplements the mask idiom inline** —
  it does **not** import `model._maybe_mask`. It does **not** touch
  `model.py`/`nowcast.py` and needs **no A3 parity baseline** (a new model, not a
  parity-gated change). Its predictive draws are convolved with the private
  nowcast into a Total-NFP posterior by `nfp_vintages.assembly.assemble_total`
  (harness side; `nfp-model` stays assembly-free). Spec:
  `specs/completed/government_wedge.md`; plan: `specs/plans/completed/14`.

## Hard boundary

**`nfp_model` imports only jax, numpyro, numpy** — no `nfp_*` packages, no
polars, no store access, no plotting. Enforced by
`src/nfp_model/tests/test_model_unit.py::TestBoundary`. Provider configs are duck-typed
(dataclass or dict). Importing the package enables **JAX float64 globally**
(`numpyro.enable_x64()`): the parity contract is double precision.

## Tech Stack

- **Dependencies**: jax, numpyro, numpy — nothing else (tests may import
  data packages; `src/` must not)
- **Build**: hatchling; requires-python >= 3.12 (current jax)

## Key Commands

```bash
pytest packages/nfp-model -m "not slow"      # fast structure/intake tests
pytest packages/nfp-model -m slow            # MCMC smoke incl. vmapped batch (~4 min)
NFP_A3_PARITY=1 pytest packages/nfp-model/src/nfp_model/tests/test_parity_golden.py  # minutes

# Full A3 parity (14 fits; restartable; see specs/plans/5)
uv run python scripts/run_a3_parity.py fit data/golden_a3_staging data/a3
uv run python scripts/run_a3_parity.py compare data/golden_a3_staging data/a3

# A4 vmapped 24-month backtest (snapshot grid → serial baseline → batched → report)
uv run python scripts/run_a4_backtest.py snapshot data/backtests
uv run python scripts/run_a4_backtest.py serial   data/backtests   # restartable, ~1 h
uv run python scripts/run_a4_backtest.py batched  data/backtests   # minutes
uv run python scripts/run_a4_backtest.py compare  data/backtests   # report + exit code
```

## Test Mapping

- `test_config.py` — defaults pinned to frozen-reference literals
- `test_data.py` — `model_inputs` normalization, snapshot round trip
  (schema v2 `error_model`, v1 fallback)
- `test_model_unit.py` — sites/shapes/log-density on synthetic data,
  cyclical gating, iid/ar1/empty-provider branches, the import boundary
- `test_model_smoke.py` — tiny-MCMC end-to-end (`slow`): layout, chain-major
  deterministic consistency, seed reproducibility, nowcast arithmetic
- `test_batch_unit.py` — padding/masking exactness: substituted-draw
  log-density equality between padded+masked and unpadded models (iid and
  ar1), uniform-structure assertions, mask bookkeeping
- `test_batch_smoke.py` — vmapped batch vs serial fits (`slow`): scalar/
  path/nowcast agreement under the A3 criteria, batch seed reproducibility
- `test_wedge_model.py` — wedge model sites/shapes/log-density + change-space
  intervention-shape encodings on synthetic data (no store, no MCMC)
- `test_wedge_fit.py` — wedge `fit_wedge`/`wedge_pred_draws` smoke (`slow`)
- `test_parity_golden.py` — single-fixture parity spot check vs
  `s3://alt-nfp/golden/a3` (opt-in: `NFP_A3_PARITY=1` + store env;
  manifest committed in `src/nfp_model/tests/golden/`)
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
- **Batched mode = padding + masks, never new structure**: padded latent
  timesteps are prior-only N(0,1) dimensions (posterior-invariant, proven
  exactly in `test_batch_unit.py`); anything that changes a site's
  *dimension* (cyclical gating, provider set, era presence) must be uniform
  across a batch and is asserted in `pad_model_inputs`. Calendar keys
  (`T`, `n_years`, `month_of_year`, `year_of_obs`, `era_idx`,
  `n_ces_vintages`) stay concrete/static under `vmap`; obs
  values/indices/masks are traced.
