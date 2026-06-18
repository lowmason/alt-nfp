# Implementation Plan: A3 — `nfp-model` posterior parity

> **⚠️ Post-rebuild status (2026-06-18).** These A3 references are **frozen
> pre-rebuild** — PyMC posteriors on the OLD 2012+ vintage store. The store rebuild
> ([`plans/10`](10-store_rebuild.md) T8) moved the canonical store to the 2017+
> rebuilt schema, so the references **no longer match `build_model_data` against the
> canonical store** (different data window + the Oct-2025 shutdown). They were
> intentionally NOT re-baselined: the port validation below is historical and sound
> (the model code is unchanged). To re-run parity, point at the preserved old-store
> backup — `NFP_STORE_URI=s3://alt-nfp/store-prev-20260618`. A forward
> JAX-on-canonical **regression** baseline is deferred until model iteration resumes;
> the spot-check test (`test_parity_golden`) now skips when pointed at the canonical
> store and tells you to use the backup.

> **Status: ✅ COMPLETE (2026-06-12). A3 PARITY PASS: 14 fixtures,
> 476/476 criteria.** Suite at **421 passed / 2 intentional skips** (+33
> nfp-model tests incl. the opt-in parity spot check), ruff clean; without
> `.env` the model tests still run (synthetic data), only store-gated and
> parity tests skip.
>
> - **Fixtures**: 14 seeded reference fits (corrected indicators config,
>   nutpie; 2 default-preset at 2023-07-12 + 2026-01-12, 12 light-preset
>   covering the 2025-02 … 2026-01 window incl. all four shutdown-affected
>   frontiers) → `s3://alt-nfp/golden/a3/`; manifest committed at
>   `packages/nfp-model/tests/golden/a3_manifest.json`. No unbuildable
>   dates; reference divergences ≤ 4 per fit.
> - **Posterior parity**: every scalar/vector site (z mostly < 1.5,
>   |Δmean| ≤ 0.07·SD, SD ratios within ~5% at the default preset); all
>   five latent paths max|Δmean|/SD ≤ 0.03 (default) / ~0.1 (light);
>   nowcast draws matched and |Δnowcast| ≤ 32k jobs on every window date,
>   inside the MCSE-derived bounds. New side: **0 divergences in all 14
>   fits** (vs 0–4 reference) and 61 vs 87 min total sampling wall.
> - **One criterion recalibrated during the run**: the fixed SD-ratio band
>   [0.8, 1.25] flagged `sigma_fourier[3]` at 2025-04-12 (ratio 0.784).
>   Root cause is reference-side MC noise, not model mismatch: the
>   reference's *centered* GRW gives ESS ≈ 175–290 for these scale
>   parameters at the light preset (vs ≈ 6,000+ non-centered), the new
>   side's SD is stable across adjacent as-of months while the reference
>   value is an outlier against its own neighbors, and **re-seeding the
>   reference moved its own SD 13.5% (5.44e-5 → 4.79e-5, toward the new
>   side's 4.26e-5)**. The SD criterion now also accepts ratios within
>   4 MC-σ of the kurtosis-aware log-SD error, `Var(log ŝ) ≈ (κ−1)/(4·ESS)`
>   — tight where ESS is high, calibrated where it isn't. Mean parity was
>   never violated anywhere.
> - Full report: `data/a3/parity_report.md` (regenerate via
>   `run_a3_parity.py compare`; fixtures in S3 are the durable record).

Phase A3 of `plans/0-port_and_staged_plan.md`: port the model layer to
JAX/NumPyro as a new `nfp-model` package (renamed from the plan's original
`nfp-model-jax` working title) and prove posterior parity against the frozen
PyMC/nutpie reference at `~/Projects/alt_nfp/packages/nfp-model-hmc`.

**Gate (from the plan of record):** on identical snapshots, posterior parity
with the HMC reference within Monte Carlo error (key params: era `mu_g`,
`phi`, sigma hierarchy, `lambda_G`, `alpha_G`, BD path; criterion:
|mean difference| small relative to pooled posterior SD and MCSE), plus
matched nowcast distributions across a 12-month backtest window.

**Critical constraint from A2:** the reference baseline MUST be generated
with the corrected indicators config
(`NowcastConfig(paths=PathsConfig(data_dir="../../data"))`) or the reference
posterior silently lacks the φ₃ cyclical block (the latent indicators-path
regression — see `plans/4-a2_seams_snapshots.md`).

## Scope decisions

- **Direct NumPyro translation only** (step 1 of the plan's A3 sequencing).
  Same priors, same likelihoods, same non-centered parametrizations where the
  reference uses them; the Fourier GRW is reparametrized non-centered
  (identical prior law, different sampling geometry — posterior-invariant).
  Kalman marginalization (step 2) is deferred: the gate is posterior parity,
  which the direct translation satisfies; marginalization is a speed play
  that belongs with A4's `vmap` work if profiling demands it.
- **Faithful-to-code port.** The reference *code's* BD form is
  `bd_t = φ₀ + φ₃·X^cycle + σ_bd·ξ_t` — there is no `φ₁·X^birth` term in
  `model.py` despite the plan-doc equation (birth_rate/bd_proxy arrays are
  built but unused by the model). Parity is against the code.
- **Dependency policy:** `nfp_model` imports **only** `jax`, `numpyro`,
  `numpy`. No `nfp_*` imports, no polars, no plotting, no store access —
  ModelData dicts/snapshot arrays in, posterior out. Provider configs are
  duck-typed (dataclass from `build_model_data` or plain dict from snapshot
  meta). This is the strictest possible reading of the A2 boundary.
- Plotting/report modules of the reference (`plots.py`, `forecast_and_plot`,
  `diagnostics.py`, …) are **not** ported in A3 — only the inference core
  (`model.py`, `sampling.py`) and the nowcast arithmetic from `backtest.py`
  needed for the gate's second half.

## Model spec being ported (reference `model.py`, frozen defaults)

| Block | Spec |
|---|---|
| QCEW noise | `σ_mid ~ LogN(log 5e-4, 0.15)`, `σ_bnd ~ LogN(log 2e-3, 0.5)`; per-obs σ = base(is_m2) × noise_mult |
| Latent AR(1) | `τ ~ LogN(log 0.013, 0.5)`, `φ_raw ~ Beta(18,2)`, `φ = min(φ_raw, .99)`, `σ_g = τ√(1−φ²)`, non-centered ε; era means `μ_g_era ~ N(0.001, 0.005)²`; `g₀ = μ₀ + σ_g ε₀` |
| Fourier | K=4; `σ_k ~ LogN(log 3e-4 − log k, 0.5)`; GRW across years, init N(0, 0.015), shape (2K, n_years) |
| BD | `φ₀ ~ N(0.001, 0.002)`, `σ_bd ~ LogN(log 0.003, 0.5)`, ξ non-centered; `φ₃ ~ N(0, 0.3)` per surviving indicator (all-zero arrays gated out) |
| QCEW lik | StudentT(ν=5) on `g_total_nsa[qcew_obs]` |
| CES lik | `α_ces ~ N(0, 0.005)`, `λ_ces ~ TruncN(1.0, 0.1, low=0.5)`, `σ_ces_{sa,nsa} ~ LogN(log 0.002, 0.5)` per vintage idx |
| Providers | per provider: `α ~ N(0,0.005)`, `λ ~ N(1, 0.15)`, `σ ~ InvGamma(3, 0.004)`; mean `α + λ·g_cont_nsa`; iid or AR(1) (`ρ ~ Beta(2,3)`, stationary first-obs SD); G = iid |
| Sampler presets | default: 4000 draws / 3000 tune / 4 chains / ta 0.95; light: 2000/2000/2/0.95 |

## Workstreams

### 5.1 Reference fixtures (old repo, read-only, background job)

`scripts/generate_a3_reference.py`, run with `~/Projects/alt_nfp/.venv/bin/python`:

- Corrected config; `build_panel(as_of_ref=D)` → `panel_to_model_data` →
  `build_model` → `sample_model` (nutpie, seeded).
- **Dates:** default preset at 2023-07-12 (mid-sample control) and 2026-01-12
  (frontier); light preset at the 12-month window 2025-02-12 … 2026-01-12
  (mirrors the reference backtest's preset choice). Unbuildable window dates
  (shutdown artifacts) are recorded as skipped, not fatal.
- **Per date, saved to npz:** full draws for every scalar/small-vector param
  (τ, φ_raw, μ_g_era, φ₀, φ₃, σ_bd, σ_qcew_mid/bnd, σ_fourier, σ_ces_sa/nsa,
  α_ces, λ_ces, α_g, λ_g, σ_pp_g); mean/SD paths for bd, g_cont, g_total_sa,
  g_total_nsa, seasonal; the CES-SA posterior-predictive mean path
  (`α + λ·g_total_sa` averaged over draws) and its full draws at the nowcast
  index `c_idx` (last state, per the reference backtest's proxy convention);
  nowcast_growth / nowcast_change_k computed exactly as `backtest.py` does
  (base_index and idx_to_level from the uncensored panel).
- Fixtures → `s3://alt-nfp/golden/a3/` (public repo; posterior draws derive
  from provider data). Committed: `packages/nfp-model/tests/golden/a3_manifest.json`.

### 5.2 The `nfp-model` package

```
packages/nfp-model/
├── pyproject.toml          # deps: jax, numpyro, numpy — nothing else
├── CLAUDE.md
├── src/nfp_model/
│   ├── __init__.py
│   ├── config.py           # ModelPriors + SamplerSettings/PRESETS (frozen reference defaults)
│   ├── data.py             # normalize ModelData dicts / snapshot (arrays, meta) → model inputs
│   ├── model.py            # the NumPyro model (faithful translation)
│   ├── sampling.py         # fit(...) → FitResult (posterior as (chains, draws, …) arrays)
│   └── nowcast.py          # CES-SA predictive transform + nowcast index/change arithmetic
└── tests/
    ├── test_config.py      # defaults pinned to frozen reference literals
    ├── test_model_unit.py  # synthetic-data trace, shapes, gating, iid/ar1 branches
    ├── test_model_smoke.py # tiny-MCMC smoke (no store needed)
    └── test_parity_golden.py  # env-gated single-date parity vs s3 fixtures
```

Root workspace: nfp-model added to root deps/sources + coverage addopts;
`slow` marker registered. The full-grid parity run stays a script
(`scripts/run_a3_parity.py`) — 14 MCMC fits don't belong in pytest.

### 5.3 Parity comparison

`scripts/run_a3_parity.py` (new venv): for each reference fixture, build the
same data via `build_model_data(as_of=D)`, fit with the matching preset and
NumPyro NUTS, then compare:

- **Scalar/vector params** (draws on both sides): flag unless
  `|Δmean| ≤ 0.15 × pooled SD` **or** `z = |Δmean|/√(MCSE_old² + MCSE_new²) ≤ 4`
  (ESS via `numpyro.diagnostics`, both sides from raw draws); SD ratio in
  [0.8, 1.25] **or** within 4 MC-σ of the log-SD estimates, where
  `Var(log ŝ) ≈ (κ−1)/(4·ESS)` (kurtosis-aware). The MC-σ escape exists
  because the *reference's* centered-GRW scale parameters sample poorly
  (ESS ≈ 175–290 for `sigma_fourier` at the light preset vs ≈ 6,000+
  non-centered on the new side) — a fixed band is miscalibrated at that
  ESS; the new side's SD for the one observed excursion (2025-04-12
  `sigma_fourier[3]`) is stable across adjacent as-of months while the
  reference value is an outlier against its own neighbors.
- **Paths** (bd, g_cont, g_total_sa, g_total_nsa, seasonal):
  `max_t |Δmean_t| / SD_t ≤ 0.25`; SD-ratio band per t in [0.7, 1.4]
  (path SDs are noisier).
- **Nowcast** (gate second half): per window date, z-test on the
  `c_idx` predictive draws' means, SD ratio, and |Δnowcast_change_k|
  reported in thousands of jobs with an MCSE-derived bound.

Report → `data/a3/parity_report.md` (gitignored) + summary in plans; gate
annotation quotes the numbers.

### 5.4 Rename + docs

`nfp-model-jax` → `nfp-model` everywhere forward-looking: root CLAUDE.md
(workspace table + intro), plans/0 (architecture table, A3 heading), new
package CLAUDE.md, mypy comment in root pyproject. Gate annotations in
plans/0 and this doc's status block.

## Sequencing

5.1 generator script first and launched in the background (longest pole) →
5.2 package + fast tests while it runs → 5.3 parity run (new side, then
comparison) → 5.4 docs + gates.
