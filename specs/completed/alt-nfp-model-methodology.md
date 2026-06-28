# alt-nfp Model Methodology

**Document purpose:** Comprehensive technical reference for the NFP nowcasting model methodology — current implementation (Phases A–C3) and planned extensions (C1–E1). Written for review by an independent agent or researcher.

**Last updated:** 2026-06-27

> **Incorporates an independent methodological review** (see `alt-nfp-methods-review.md`). The review's highest-impact findings are folded into the relevant sections below as inline flags (marked **⚑ Review**) and consolidated, with the prescribed fixes and citations, in **§19 (Methodological Review — Corrections & Prioritized Roadmap)**. A **§20 References** bibliography backs the citations. The three load-bearing findings: (i) *if* the observation likelihood is a smoothed-mean plug-in (the doc is internally inconsistent on this, and the code has since been upgraded — confirm first), it is not the exact marginal likelihood — fix via Student-t scale-mixture augmentation (§19.1); (ii) the pooled CES bias/loading cannot represent a first-print-specific bias separate from the QCEW-anchored truth — de-pool per vintage in a multi-vintage news/noise block (§19.2); (iii) the evaluation should be re-anchored on the consensus survey median as the primary benchmark, scored against the first print (§16, §19.7).

------------------------------------------------------------------------

## 1. Problem Statement

The model produces a Bayesian nowcast of U.S. total private nonfarm payroll employment change (the "NFP number") in advance of the BLS CES release. It combines:

- **QCEW** — universe-count ground truth (quarterly, lagged \~5 months)
- **CES** — BLS monthly survey estimate (three vintages: first, second, third print)
- **PSP (Private Survey Providers)** — payroll-processor microdata covering partial establishment panels, with known release schedules

Total NFP = private nowcast + government-sector forecast (the latter handled by a separate `wedge.py` model).

**⚑ Review — target identification.** The scoring target is the **CES first print**, but the only census-grade anchor (QCEW) measures something close to the **benchmark/final truth** and arrives ~5 months late. These are different objects: the first print carries a documented systematic revision component (Aruoba 2008; Neumark–Wascher 1991), so a model disciplined toward the QCEW truth is pulled "truth-ward" relative to what BLS will actually print first. The estimation principle (Koenig–Dolmas–Piger 2003) is to match the measurement equation to the release being targeted. This motivates the de-pooled multi-vintage CES block in §19.2 and the evaluation re-design in §16/§19.7.

------------------------------------------------------------------------

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  nfp-ingest (data layer)                                │
│  build_model_data(as_of=D) → ModelData dict (arrays)    │
│  snapshots.collect_snapshot → (arrays, meta) artifact    │
└──────────────────────┬──────────────────────────────────┘
                       │ numpy arrays only
                       ▼
┌─────────────────────────────────────────────────────────┐
│  nfp-model (inference layer)                            │
│  model_inputs() → normalized dict                       │
│  nfp_model(data, priors) → NumPyro model                │
│  fit_model(data) → FitResult (posterior + diagnostics)  │
│  nowcast_summary(posterior) → point estimate + draws     │
└─────────────────────────────────────────────────────────┘
```

**Hard boundary:** `nfp_model` imports only `jax`, `numpyro`, `numpy`. No data packages, no polars. The model never sees a vintage date — all censoring happens upstream in `nfp_ingest` before arrays arrive.

**Precision contract:** JAX float64 globally (`numpyro.enable_x64()`). All Kalman filter operations, all sampling, all posterior arithmetic in double precision.

------------------------------------------------------------------------

## 3. Inference Strategy: Rao-Blackwellized NUTS

The model uses **Rao-Blackwellized Hamiltonian Monte Carlo**:

1.  **NUTS samples only static parameters** (\~15–23 scalars depending on feature gates).
2.  A **custom pure-JAX Kalman filter** (`kalman.py`) analytically marginalizes the T-dimensional latent state trajectory, producing the marginal log-likelihood $\log p(y_{1:T} | \theta)$ conditioned on the NUTS-proposed static parameters $\theta$.
3.  The marginal log-likelihood is injected into the NumPyro model via `numpyro.factor('kalman_ll', ll)`.
4.  **Post-NUTS**, Forward-Filter Backward-Sample (FFBS) reconstructs latent state trajectories for each posterior draw, yielding the full joint posterior $p(\theta, x_{1:T} | y_{1:T})$.

This eliminates \~370 latent dimensions from the NUTS target (compared to pre-A4 which sampled $\epsilon_g$, `fourier_z`, $\xi_\text{bd}$ directly), reducing NUTS dimensionality from \~385 to \~15–23.

### Why Rao-Blackwellization works here

The model is a **conditionally linear Gaussian state-space model** (CLGSSM): given the static parameters $\theta = (\phi, \tau, \mu_g, \sigma_\text{fourier}, \phi_0, \sigma_\text{bd}, \ldots)$, the latent state transitions and (Gaussian) observation equations are linear. The Kalman filter computes the exact marginal likelihood analytically. After B1/B2 introduced Student-t observations, all observations moved *outside* the filter — the filter runs a pure prior-propagation pass (no observations update the state), and observation likelihoods are computed separately from smoothed state means.

> **⚑ Review — this is the most serious *potential* issue; confirm against the current code first.** **If** the scheme is what §5/§18.3 describe — scoring observation densities against RTS **smoothed means treated as fixed plug-in points** — then it is *not* the model's marginal likelihood: it (a) double-uses the data (the smoother conditions on the observations, which are then scored against the smoothed mean), (b) discards the smoothing covariance $P_{t|T}$ and so **understates predictive uncertainty** (overconfident intervals — exactly what a nowcast must calibrate), and (c) is not the collapsed/Rao-Blackwellized likelihood, which integrates the state out *analytically* rather than replacing it with a point estimate (Doucet, de Freitas, Murphy & Russell 2000). **But the description is internally inconsistent:** the genuinely Rao-Blackwellized idea stated just above ("Kalman filter analytically marginalizes the state, likelihood injected via `numpyro.factor`") is *correct* and contradicts the smoothed-mean step — so the first task is to verify which description matches the current (substantially upgraded) implementation. If the marginalization is in fact analytic and "smoothed mean" refers only to a post-hoc reconstruction for reporting, the error does not apply. The discriminating empirical test is the simulated-coverage gate in §19.1. **Fix (if confirmed):** keep the model exactly conjugate under Student-t errors via the **Gaussian scale-mixture / auxiliary-variable representation** ($t_\nu = \mathcal{N}/\sqrt{\Gamma}$), conditional on which every observation is Gaussian, the Kalman prediction-error-decomposition likelihood is exact, and FFBS samples states correctly. See **§19.1**.

------------------------------------------------------------------------

## 4. State-Space Formulation

### 4.1 State Vector

$$x_t = \begin{pmatrix} g_{\text{cont},t} \\ A_{1,y(t)} \\ B_{1,y(t)} \\ A_{2,y(t)} \\ B_{2,y(t)} \\ \vdots \\ A_{K,y(t)} \\ B_{K,y(t)} \end{pmatrix} \in \mathbb{R}^d, \quad d = 1 + 2K = 9 \text{ (with } K=4 \text{)}$$

| Index | Component | Description |
|------------------|-------------------------|-----------------------------|
| 0 | $g_{\text{cont},t}$ | Continuing-units employment growth (AR(1)) |
| $2k-1$ | $A_{k,y(t)}$ | Fourier cosine amplitude, harmonic $k$, year $y(t)$ |
| $2k$ | $B_{k,y(t)}$ | Fourier sine amplitude, harmonic $k$, year $y(t)$ |

**Key design decision:** The birth/death shock $\xi_t$ is NOT a Kalman state. It is iid (no temporal dependence), so $\sigma_\text{bd}^2$ is folded into the observation noise covariance $R_t$ for CES observations, and the deterministic component $\phi_0 + \phi_3 \cdot X_t$ enters the observation offset $d_t$.

### 4.2 Transition Equation

$$x_t = F x_{t-1} + b_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)$$

**Transition matrix** $F$ (block-diagonal, time-invariant):

$$F = \text{diag}(\phi, \underbrace{1, 1, \ldots, 1}_{2K})$$

The growth component has AR(1) persistence $\phi$; Fourier amplitudes are identity (random walk within a year).

**Transition offset** $b_t$:

$$b_t = (\mu_{g,e(t)} (1 - \phi), 0, \ldots, 0)^\top$$

where $\mu_{g,e(t)}$ is the era-indexed AR(1) mean, implementing mean-reversion: $\mathbb{E}[g_t | g_{t-1}] = \mu_g + \phi(g_{t-1} - \mu_g) = \phi \cdot g_{t-1} + \mu_g(1-\phi)$.

**Process noise** $Q_t$ (time-varying):

$$Q_t = \text{diag}\bigl(\sigma_{g,t}^2, \underbrace{q_{1,t}, q_{1,t}, q_{2,t}, q_{2,t}, \ldots, q_{K,t}, q_{K,t}}_{2K}\bigr)$$

where: - $\sigma_{g,t}$ is the growth innovation SD (scalar when SV is off; time-varying path $\sigma_g \exp(0.5 h_t)$ when SV is on) - $q_{k,t} = \sigma_{\text{fourier},k}^2 \cdot \mathbb{1}[\text{year\_boundary}(t)]$ — Fourier GRW steps only at year boundaries; within a year, amplitudes are frozen ($q_{k,t} = 0$)

### 4.3 Initial State

$$x_0 \sim \mathcal{N}(m_0, P_0)$$

- $m_0 = (\mu_{g,0}, 0, \ldots, 0)^\top$
- $P_0 = \text{diag}\bigl(\sigma_g^2 / (1 - \phi^2), \sigma_\text{init}^2, \ldots, \sigma_\text{init}^2\bigr)$ with $\sigma_\text{init} = 0.015$

The growth initial variance is the AR(1) stationary variance.

------------------------------------------------------------------------

## 5. Observation Equations

After Phase B1/B2, **all observations are handled outside the Kalman filter** as separate `numpyro.factor` terms using smoothed state means. The filter runs a pure prior-propagation pass (no observations update the state).

> **⚑ Review — see §3 flag and §19.1.** *If* the "smoothed state means" wording here reflects the current implementation (rather than a post-hoc reconstruction for reporting), it is the inference shortcut the review identifies as a first-order error — confirm against the code via the simulated-coverage gate (§19.1). The scale-mixture fix returns these observation factors *inside* an exact, conditionally-Gaussian Kalman/FFBS pass without abandoning the Student-t tails. Either way, the equations in §5.2–§5.4 are correct as measurement *specifications*; only the *mechanism* for evaluating their likelihood is in question.

### 5.1 Derived Quantities from State

From the smoothed state mean $\hat{x}_t = (\hat{g}_{\text{cont},t}, \hat{A}_{1,t}, \hat{B}_{1,t}, \ldots)$:

$$\text{seasonal}_t = \sum_{k=1}^K \hat{A}_{k,t} \cos(2\pi k m_t / 12) + \hat{B}_{k,t} \sin(2\pi k m_t / 12)$$

$$\text{det\_bd}_t = \phi_0 + \sum_i \phi_{3,i} X_{i,t}$$

$$g_{\text{total,sa},t} = \hat{g}_{\text{cont},t} + \text{det\_bd}_t$$

$$g_{\text{total,nsa},t} = \hat{g}_{\text{cont},t} + \text{seasonal}_t + \text{det\_bd}_t$$

### 5.2 QCEW Likelihood (Student-t Anchor)

$$y_t^\text{QCEW} \sim t_\nu(g_{\text{total,nsa},t},\ \sigma_{\text{QCEW},t})$$

- $\nu = 5$ (fixed)
- $\sigma_{\text{QCEW},t}$ = tiered base sigma × per-observation revision multiplier:
  - M2 months: $\sigma_\text{qcew,mid} \sim \text{LogNormal}(\log 0.0005, 0.15)$
  - M1/M3 months: $\sigma_\text{qcew,boundary} \sim \text{LogNormal}(\log 0.002, 0.5)$
- \~4–8 observations per fit (quarterly QCEW lagged \~5 months)

### 5.3 CES Likelihood (Student-t, Phase B1)

**CES SA** (seasonally adjusted): $$y_t^\text{CES,SA} \sim t_{\nu_\text{ces}}(\alpha_\text{ces} + \lambda_\text{ces} \cdot g_{\text{total,sa},t},\ \sigma_{\text{ces,sa},v(t)})$$

**CES NSA** (not seasonally adjusted): $$y_t^\text{CES,NSA} \sim t_{\nu_\text{ces}}(\alpha_\text{ces} + \lambda_\text{ces} \cdot g_{\text{total,nsa},t},\ \sigma_{\text{ces,nsa},v(t)})$$

- $\nu_\text{ces} = \nu_\text{ces,raw} + 2$, where $\nu_\text{ces,raw} \sim \text{Exp}(0.1)$ (prior median $\nu \approx 9$)
- $\alpha_\text{ces} \sim \mathcal{N}(0, 0.005)$ — pooled CES bias
- $\lambda_\text{ces} \sim \text{TruncatedNormal}(1.0, 0.1, \text{low}=0.5)$ — near-unit loading
- $\sigma_{\text{ces},v(t)} \sim \text{LogNormal}(\log 0.002, 0.5)$ — vintage-indexed noise

> **⚑ Review — pooling across vintages is a misspecification for a first-print target.** A single $(\alpha_\text{ces}, \lambda_\text{ces})$ shared across first/second/third prints (vintage-indexing only the noise $\sigma$) assumes the *systematic* state→print relationship is identical across vintages, with only dispersion changing. The payroll-revision literature contradicts this (Aruoba 2008; Neumark–Wascher 1991; Mankiw–Shapiro 1986): first prints carry a predictable, biased component distinct from later vintages, so a pooled scalar cannot represent a first-print-specific bias separate from the final-print bias. Give each vintage its own $(\alpha_v, \lambda_v)$ — nesting the pooled model as a testable restriction — inside the multi-vintage news/noise block (§19.2). This is anticipated, in part, by Phase C1 (§13.1).

### 5.4 Provider Likelihoods (Student-t, Phase B2)

**Iid providers:** $$y_t^\text{prov} \sim t_{\nu_p}(\alpha_p + \lambda_p \cdot g_{\text{obs},t},\ \sigma_{p,t})$$

**AR(1) providers** (conditional likelihood): $$y_t^\text{prov} | y_{t-1} \sim t_{\nu_p}(\mu_{\text{cond},t},\ \sigma_{\text{cond},t})$$

where $\mu_{\text{cond},t} = \mu_{\text{base},t} + \rho_p (y_{t-1} - \mu_{\text{base},t-1})$ and $\sigma_{\text{cond},t} = \sigma_{p,t}$ (with the first observation using stationary variance $\sigma_p / \sqrt{1 - \rho_p^2}$).

**Provider observation target** depends on `include_bd`: - `include_bd=True` (PSP providers): $g_{\text{obs},t} = g_{\text{cont},t} + \text{seasonal}_t + \text{det\_bd}_t$ (QCEW-calibrated growth includes BD) - `include_bd=False` (legacy providers): $g_{\text{obs},t} = g_{\text{cont},t} + \text{seasonal}_t$

Per-provider parameters: - $\alpha_p \sim \mathcal{N}(0, 0.005)$ - $\lambda_p \sim \mathcal{N}(1.0, 0.15)$ - $\sigma_p \sim \text{InverseGamma}(3.0, 0.004)$ - $\nu_p = \nu_{p,\text{raw}} + 2$, where $\nu_{p,\text{raw}} \sim \text{Exp}(0.1)$ - $\rho_p \sim \text{Beta}(2, 3)$ (AR(1) providers only)

------------------------------------------------------------------------

## 6. Latent Growth Process

### 6.1 AR(1) with Era-Specific Means

$$g_t = \mu_{g,e(t)} + \phi(g_{t-1} - \mu_{g,e(t)}) + \sigma_{g,t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$$

- **Stationary-SD parametrization:** $\sigma_g = \tau \sqrt{1 - \phi^2}$ ensures $\tau$ directly controls the marginal SD of the stationary distribution
- $\tau \sim \text{LogNormal}(\log 0.013, 0.5)$ — monthly private employment growth is \~1.3% annualized SD
- $\phi_\text{raw} \sim \text{Beta}(18, 2)$ → $\phi = \min(\phi_\text{raw}, 0.99)$ — prior mean \~0.9, capped below unit root
- $\mu_{g,e} \sim \mathcal{N}(0.001, 0.005)$ per era — `n_eras=2` (pre-COVID, post-COVID)

### 6.2 Fourier Seasonal (Annually-Evolving GRW)

$K=4$ harmonics. Amplitudes evolve as Gaussian random walks with steps only at year boundaries:

$$A_{k,y} = A_{k,y-1} + \sigma_{\text{fourier},k} \eta_{A,k,y}, \quad B_{k,y} = B_{k,y-1} + \sigma_{\text{fourier},k} \eta_{B,k,y}$$

Within a year, amplitudes are frozen. The seasonal contribution at month $m_t$ in year $y(t)$:

$$s_t = \sum_{k=1}^K A_{k,y(t)} \cos\left(\frac{2\pi k m_t}{12}\right) + B_{k,y(t)} \sin\left(\frac{2\pi k m_t}{12}\right)$$

Prior on innovation SDs: $\sigma_{\text{fourier},k} \sim \text{LogNormal}(\log 0.0003 - \log k, 0.5)$ — harmonic-weighted (higher harmonics shrink faster).

> **⚑ Review — defensible-but-suboptimal, plus an identification hazard.** A $K=4$ Fourier seasonal whose amplitudes step *only at year boundaries* is a rigid special case of the stochastic trigonometric seasonal (Harvey 1989; Durbin & Koopman 2012) and cannot reproduce BLS's **concurrent X-13ARIMA-SEATS** factors (re-estimated every month), so the internal seasonal will diverge from the published SA series the model also fits. The sharper concern is **loading**: the model ingests both CES-SA and CES-NSA. The latent construction in §5.1 already routes the seasonal to the NSA composite only ($g_\text{total,sa}$ excludes $s_t$), which is the correct *transform-specific* split — but the **shared** $\lambda_\text{ces}$ across the SA and NSA equations re-couples them. Make the loading transform-specific (or drop one CES transform) so the internal seasonal is identified off NSA without being imposed on SA. See §19.3.

### 6.3 Structural Birth/Death

The BD component is **deterministic given parameters** (not a latent state):

$$\text{bd}_t = \phi_0 + \sum_i \phi_{3,i} X_{i,t}$$

- $\phi_0 \sim \mathcal{N}(0.001, 0.002)$ — small positive BD intercept
- $\sigma_\text{bd} \sim \text{LogNormal}(\log 0.003, 0.5)$ — BD shock SD (enters $R_t$, not the state)
- $\phi_3 \sim \mathcal{N}(0, 0.3)$ per indicator — cyclical covariates (default: claims, jolts)

> **⚑ Review — birth/death deserves its own stochastic state.** Folding net firm birth/death into a deterministic-given-parameters function of observed covariates (intercept + cyclical drift, with an iid shock) imposes that the BD contribution has no persistent dynamics of its own. But benchmark revisions are *dominated* by birth/death and continuing-units misses — the preliminary March-2025 benchmark was **−911,000 (−0.6%)**, the largest on record in absolute terms, later finalized at **−862,000 (−0.5%)** (BLS USDL-25-1352, Sept 2025). A deterministic BD term is therefore a likely source of first-print miss at turning points. Promote BD to a **time-varying latent state** (an AR(1)/random-walk BD level with the cyclical covariates as drift terms), which the latent-true-state literature naturally accommodates (Cajner et al. 2018, 2022). See §19.4.

**Covariate gating:** If all values of a cyclical array are zero (e.g., censored backtest where the indicator isn't yet available), that covariate is dropped and $\phi_3$ dimension shrinks accordingly.

------------------------------------------------------------------------

## 7. Optional Extensions (Feature-Gated)

### 7.1 Common Stochastic Volatility (B3, `enable_sv=True`)

Time-varying growth innovation variance via log-AR(1):

$$\log h_t = \mu_h + \rho_h (\log h_{t-1} - \mu_h) + \sigma_\omega \epsilon_{h,t}$$

Effective innovation SD: $\sigma_{g,t} = \sigma_g \cdot \exp(0.5 \cdot h_t)$ where $\sigma_g$ retains its meaning as the unconditional stationary SD.

**Non-centered parametrization** (mandatory — centered form creates funnel geometry):

```
eps_h ~ N(0, 1)^{T-1}
log_h[0] = mu_h
log_h[t] = mu_h + rho_h * (log_h[t-1] - mu_h) + sigma_omega * eps_h[t-1]
```

Priors: - $\mu_h \sim \mathcal{N}(0, 1)$ - $\rho_h \sim \text{Beta}(18, 2)$ — prior mean \~0.9 - $\sigma_\omega \sim \text{LogNormal}(\log 0.1, 0.5)$

Adds $T-1$ latent dimensions to NUTS (the SV innovations `eps_h`).

### 7.2 Per-Provider Stochastic Volatility (C3, `enable_provider_sv=True`)

Same log-AR(1) pattern as B3 but per provider:

$$\log h_{p,t} = \mu_{h,p} + \rho_{h,p}(\log h_{p,t-1} - \mu_{h,p}) + \sigma_{\omega,p} \epsilon_{h,p,t}$$

Effective provider noise: $\sigma_{p,t} = \sigma_p \cdot \exp(0.5 \cdot h_{p,t})$. Adds $T-1$ latent dimensions per provider to NUTS. Uses same `ProviderSVPriors` (same hyperparameters as `SVPriors`).

------------------------------------------------------------------------

## 8. Kalman Filter Implementation

### 8.1 Forward Filter (`kalman_filter`)

Standard predict-update recursion via `jax.lax.scan`:

**Predict:** $$m_t^- = F m_{t-1} + b_t, \quad P_t^- = F P_{t-1} F^\top + Q_t$$

**Innovation:** $$v_t = y_t - (H_t m_t^- + d_t), \quad S_t = H_t P_t^- H_t^\top + R_t^{\text{eff}}$$

**Update:** $$K_t = P_t^- H_t^\top S_t^{-1}, \quad m_t = m_t^- + K_t v_t, \quad P_t = (I - K_t H_t) P_t^-$$

**Marginal log-likelihood:** $$\log p(y_{1:T} | \theta) = \sum_{t=1}^T \log \mathcal{N}(v_t; 0, S_t)$$

**Missing observations** are handled by inflating the observation noise: `R_eff[j,j] += 1e6` for unobserved channels. This makes the Kalman gain for that channel effectively zero while keeping array shapes static for JIT.

**Numerical stability:** - Kalman gain via Cholesky solve (`jnp.linalg.solve`), not matrix inversion - Covariance symmetrization after each update: $P \leftarrow (P + P^\top)/2$ - Jitter addition before Cholesky: $P \leftarrow P + 10^{-8} I$

### 8.2 RTS Smoother (`kalman_smoother`)

Rauch-Tung-Striebel backward pass via `jax.lax.scan(reverse=True)`:

$$G_t = P_t F^\top (P_{t+1}^-)^{-1}$$ $$m_t^s = m_t + G_t (m_{t+1}^s - m_{t+1}^-)$$ $$P_t^s = P_t + G_t (P_{t+1}^s - P_{t+1}^-) G_t^\top$$

Produces smoothed state means and covariances used for all observation likelihoods (QCEW, CES, providers).

### 8.3 FFBS (`ffbs_sample`)

Forward-Filter Backward-Sample draws one joint state trajectory:

1.  Sample $x_T \sim \mathcal{N}(m_T, P_T)$
2.  For $t = T-1, \ldots, 0$:
    - $G_t = P_t F^\top (P_{t+1}^-)^{-1}$
    - $m_t^{\text{back}} = m_t + G_t (x_{t+1} - m_{t+1}^-)$
    - $P_t^{\text{back}} = P_t - G_t P_{t+1}^- G_t^\top$
    - Sample $x_t \sim \mathcal{N}(m_t^{\text{back}}, P_t^{\text{back}})$

Used post-NUTS to reconstruct path draws for downstream consumption. Vmapped over all posterior draws.

------------------------------------------------------------------------

## 9. Sampling Pipeline

### 9.1 NUTS Configuration

| Setting          | Default                          | Light | Medium |
|------------------|----------------------------------|-------|--------|
| `num_samples`    | 4000                             | 2000  | 3000   |
| `num_warmup`     | 3000                             | 2000  | 3000   |
| `num_chains`     | 4                                | 2     | 4      |
| `target_accept`  | 0.95                             | 0.95  | 0.95   |
| `max_tree_depth` | 10                               | 10    | 10     |
| `init_strategy`  | `init_to_median(num_samples=15)` | —     | —      |

### 9.2 Post-NUTS FFBS Reconstruction

After NUTS completes: 1. For each of `n_chains × n_draws` posterior draws, rebuild SSM params from static parameters 2. Run `kalman_filter` on the pre-built emissions matrix 3. Draw one latent state trajectory via `ffbs_sample` 4. Reshape to `(chains, draws, T, d)` 5. Extract path variables: `g_cont`, `seasonal`, `bd`, `g_total_sa`, `g_total_nsa`

The reconstruction passes `include_ces=False` and empty provider lists (mirroring the model's prior-only filter).

### 9.3 Diagnostics

| Diagnostic | Threshold | Interpretation |
|----------------------|---------------------|-----------------------------|
| Max R-hat | \< 1.05 | Convergence across chains |
| Min ESS (bulk) | \> 100 | Effective sample size |
| Min BFMI | \> 0.2 | Bayesian fraction of missing information |
| Divergences | few per fit | HMC trajectory violations |
| $\nu$ posterior median | \> 2.5 | Student-t df not in infinite-variance regime |

------------------------------------------------------------------------

## 10. Nowcast Extraction

### 10.1 CES-SA Predictive

$$\hat{y}_t^\text{CES,SA} = \alpha_\text{ces} + \lambda_\text{ces} \cdot g_{\text{total,sa},t}$$

Applied per posterior draw → `pred_draws` array of shape `(chains, draws, T)`.

### 10.2 Point Nowcast

The posterior-mean growth path is converted to an index path via cumulative exponentiation:

$$\text{index}_t = \text{base\_index} \times \exp\left(\sum_{s=1}^t \overline{\hat{y}}_s\right)$$

The nowcast target is `c_idx` (typically the last timestep), producing: - `nowcast_growth`: posterior-mean log growth at `c_idx` - `nowcast_change_k`: month-over-month jobs change in thousands = $(\text{index}_{c} - \text{index}_{c-1}) \times \text{idx\_to\_level}$ - `pred_draws`: full distributional object at `c_idx`

------------------------------------------------------------------------

## 11. Batched Fitting

`fit_model_batch` fits multiple as-of dates simultaneously via `jax.jit(jax.vmap(...))`:

1.  **Padding:** Per-date inputs are padded to common shapes (T_max, max obs counts)
2.  **Masking:** Padded observation slots contribute zero log-probability
3.  **Structural uniformity:** Cyclical gating, provider set, era handling must be uniform across the batch
4.  **Per-date reduction:** Each date's posterior is reduced in-graph to scalar draws + path mean/SD + nowcast predictive draws

After B1/B2, path reconstruction uses the Kalman **smoother** mean (not FFBS draws) in batch mode for performance — path SDs reflect parameter uncertainty only, not state uncertainty.

------------------------------------------------------------------------

## 12. PSP (Private Survey Provider) Integration

### 12.1 PSP Store Schema

PSP data arrives from a separate provider store (`s3://alt-nfp/psp-store/psp={1,2,3}/series.parquet`):

| Column               | Description                                         |
|----------------------|-----------------------------------------------------|
| `ref_date`           | Reference month (day=12, BLS pay-period convention) |
| `release_date`       | When this estimate became available                 |
| `revision`           | 0 (initial), 1 (second), 2 (third)                  |
| `benchmark_revision` | 0 (non-benchmarked) or 1 (QCEW-benchmarked)         |
| `growth`             | QCEW-calibrated link-relative growth (NSA)          |
| `matched_emp`        | Sample employment count                             |
| `matched_clts`       | Sample client count                                 |
| `industry_code`      | BLS CES codes (supersector/sector level)            |

> **⚑ Review — validate provider-unit geography/industry against a worksite benchmark.** Where the pseudo-establishment `industry_code`/geography is inferred by clustering on employees' **residence** locations rather than **worksite** addresses, a systematic residence-vs-worksite wedge (commuting, telework) will misassign geography toward bedroom communities and bias industry assignment, most severely for the dense metros that dominate employment. The official infrastructure (Census **LEHD**, MWR→QCEW) solves the multi-establishment problem on worksite reports, not residences (LEHD TP-2006-01). **Recommendation:** prefer worksite ZIP/address fields from the payroll records where present, and validate the residence-clustered units against an LEHD/QWI or QCEW worksite benchmark at the county×industry level (the natural extension of the Dunn et al. 2026 "novel test"). See §19.5.

### 12.2 Provider Observation Equation

PSP providers are configured with `include_bd=True`, meaning they observe the full QCEW-calibrated growth (including the birth/death component):

$$y_t^\text{PSP} = \alpha_p + \lambda_p \cdot (g_{\text{cont},t} + \text{seasonal}_t + \text{det\_bd}_t) + \epsilon_t$$

This differs from legacy providers (`include_bd=False`) which observe $g_\text{cont} + \text{seasonal}$ only.

### 12.3 Revision Lifecycle

| Tuple (rev, bench_rev) | Availability                | Model role             |
|------------------------|-----------------------------|------------------------|
| (0, 0)                 | CES release − 7 days        | Early signal (noisier) |
| (1, 0)                 | CES release − 2 days        | Refined signal         |
| (2, 0)                 | Next month release − 7 days | Final pre-benchmark    |
| (2, 1)                 | After annual QCEW benchmark | Ground-truth-adjacent  |

**As-of correctness:** Filter `release_date <= backtest_asof` for proper backtesting.

------------------------------------------------------------------------

## 13. Planned Extensions

### 13.1 Phase C1: Release-Specific BLS Bias States

**Goal:** Stop collapsing CES to best-available vintage. Observe all three releases with per-release latent bias states.

$$y_{t,r}^\text{BLS} = e_t^* + b_{t,r} + \epsilon_{t,r}, \quad \epsilon_{t,r} \sim t_{\nu_r}(0, \sigma_r^2)$$

$$b_{t,r} = \phi_r b_{t-1,r} + \Gamma_r z_t + \psi_r u_{t,r}$$

Three AR(1) bias states (first, second, third release), gated by `enable_release_bias`.

> **⚑ Review — frame C1 as a Jacobs–van Norden multi-vintage block, and promote it.** This is the structural fix for the pooled-CES misspecification (§5.3 flag) and the target mismatch (§1 flag): treat first print, second print, third print, and QCEW as **distinct noisy reads of the latent true state, each with its own bias and loading** $(\alpha_v, \lambda_v)$, with measurement errors that admit **news, noise, and spillover** dynamics (Jacobs & van Norden 2011; Kishor & Koenig 2012; Cunningham et al. 2012). Anchoring the latent state with QCEW while letting the first-print equation carry its own bias is what identifies the first-print nowcast *separately* from the benchmark truth and removes the truth-ward bias. Given the first print is the scoring target, this should be treated as a near-term correction rather than an optional extension — see §19.2 and the Stage-1 roadmap (§19.7).

### 13.2 Phase C1b: Provider Release-Specific Observations

Same bias-state framework applied to PSP providers. Per-provider, per-release parameters with hierarchical shrinkage:

$$y_{t,r,p}^\text{prov} = \alpha_p + \lambda_p e_t^* + b_{t,r}^{(p)} + \epsilon_{t,r,p}$$

The pay-period composition effect (weekly vs biweekly vs monthly) drives provider-specific initial→third revision magnitudes, motivating per-provider parameters.

### 13.3 Phase C2: Annual Benchmark Shock

$$e_t^* = g_{\text{total,sa},t} + B_{a(t)} \cdot w_{t,a(t)}$$

where $B_a \sim t_{\nu_B}(0, \sigma_B)$ is the benchmark shock for year $a$, and $w_{t,a}$ is the BLS wedge-back weight (linear taper from April through March).

### 13.4 Phase D1: Supersector Vectorization

Replace scalar $g_\text{cont}$ with $(S,)$ vector ($S \approx 10$–12 private supersectors):

$$g_{s,t} = \mu_{g,s} + \phi_s(g_{s,t-1} - \mu_{g,s}) + \sigma_{g,s} \epsilon_{g,s,t}$$

Soft aggregation constraint: $g_\text{tot,t} = \sum_s \omega_s g_{s,t} + \eta_\text{resid}$ where $\omega_s$ are employment share weights. Kalman state dimension grows from $d \approx 9$ to $d \approx S \cdot (1 + 2K) \approx 110$.

### 13.5 Phase D3: Dynamic Factor

Low-rank common factor capturing supersector co-movement:

$$g_t = \mu_g + \Phi g_{t-1} + \Lambda f_t + \eta_t$$

$$f_t = A f_{t-1} + \zeta_t$$

with $f_t \in \mathbb{R}^k$ ($k=2$), loadings $\Lambda \in \mathbb{R}^{S \times k}$ (lower-triangular for identification).

### 13.6 Phase E1: Matrix DFM Challenger

Separate model treating the supersector × provider panel as a matrix:

$$Y_t = A F_t B^\top + E_t$$

with Kronecker error structure $\Sigma_c \otimes \Sigma_r$. Implemented as a standalone model file (like `wedge.py`).

------------------------------------------------------------------------

## 14. Prior Specification Summary

### Static Parameters (NUTS-sampled)

| Parameter | Distribution | Hyperparameters | Phase |
|----------------|------------------|-----------------------|----------------|
| $\tau$ | LogNormal | $\mu=\log(0.013), \sigma=0.5$ | Current |
| $\phi_\text{raw}$ | Beta | $\alpha=18, \beta=2$ | Current |
| $\mu_{g,e}$ | Normal | $\mu=0.001, \sigma=0.005$ | Current |
| $\sigma_\text{fourier,k}$ | LogNormal | $\mu=\log(0.0003)-\log k, \sigma=0.5$ | Current |
| $\phi_0$ | Normal | $\mu=0.001, \sigma=0.002$ | Current |
| $\sigma_\text{bd}$ | LogNormal | $\mu=\log(0.003), \sigma=0.5$ | Current |
| $\phi_3$ | Normal | $\mu=0, \sigma=0.3$ | Current |
| $\alpha_\text{ces}$ | Normal | $\mu=0, \sigma=0.005$ | Current |
| $\lambda_\text{ces}$ | TruncatedNormal | $\mu=1.0, \sigma=0.1$, low=0.5 | Current |
| $\sigma_\text{ces}$ | LogNormal | $\mu=\log(0.002), \sigma=0.5$ | Current |
| $\nu_\text{ces}$ | $2 + \text{Exp}(0.1)$ | — | B1 |
| $\alpha_p$ | Normal | $\mu=0, \sigma=0.005$ | Current |
| $\lambda_p$ | Normal | $\mu=1.0, \sigma=0.15$ | Current |
| $\sigma_p$ | InverseGamma | $\alpha=3, \beta=0.004$ | Current |
| $\rho_p$ | Beta | $\alpha=2, \beta=3$ | Current |
| $\nu_p$ | $2 + \text{Exp}(0.1)$ | — | B2 |
| $\mu_h$ | Normal | $\mu=0, \sigma=1$ | B3 |
| $\rho_h$ | Beta | $\alpha=18, \beta=2$ | B3 |
| $\sigma_\omega$ | LogNormal | $\mu=\log(0.1), \sigma=0.5$ | B3 |
| $\sigma_\text{qcew,mid}$ | LogNormal | $\mu=\log(0.0005), \sigma=0.15$ | Current |
| $\sigma_\text{qcew,boundary}$ | LogNormal | $\mu=\log(0.002), \sigma=0.5$ | Current |

### QCEW Fixed Parameters

| Parameter | Value | Rationale |
|---------------------------|------------------|---------------------------|
| $\nu_\text{QCEW}$ | 5 | Moderate tails; QCEW is ground truth but has boundary noise |

------------------------------------------------------------------------

## 15. Computational Profile

| Metric | Pre-A4 (NUTS over all) | Post-A4 (Rao-Blackwellized) |
|----------------|-------------------------|--------------------------------|
| NUTS dimension | \~385 (15 static + 370 latent) | \~15–23 (static + optional SV) |
| Kalman cost per NUTS step | 0 | $O(T \cdot d^3) \approx 150$k flops |
| Expected walltime per chain | \~17s | \~2–5s |
| With SV (B3) | N/A | \~15–23 + $(T-1)$ dims for `eps_h` |

The Kalman filter cost at $d=9, T=150$ is trivial relative to NUTS overhead. At Phase D ($d \approx 110, T = 150$): $O(T \cdot d^3) \approx 10^8$ flops per step — still fast on CPU.

------------------------------------------------------------------------

## 16. Model Validation Framework

### 16.1 Phase-Boundary Baselines

Each phase requires a new parity baseline. Validation uses: - **MCSE z-tests:** $|(\bar\theta_\text{new} - \bar\theta_\text{ref})| / \sqrt{\text{MCSE}_\text{new}^2 + \text{MCSE}_\text{ref}^2} < 4.0$ - **SD ratio bands:** posterior SD ratio within $[0.80, 1.25]$ - **Path comparison:** $\max_t |\Delta\text{mean}_t| / \text{SD}_t < 0.25$

### 16.2 Backtest Design

Rolling-origin real-time backtest. For each evaluation month: 1. Fit using only data knowable as of the corresponding as-of date 2. Extract nowcast at target index 3. Score against the **realized CES first print** (the primary target), with the QCEW-settled value retained as a secondary diagnostic of where the model lands relative to the benchmark truth.

The first print is the object the model is judged on, so it is the object every headline metric and every forecast-comparison test (§16.3–§16.5) is computed against. Scoring primarily against the QCEW-settled truth would reward truth-ward bias (§1 flag) and is therefore relegated to a diagnostic.

### 16.3 Scoring Metrics

| Metric                 | Target                    |
|------------------------|---------------------------|
| MAE, RMSE, Median AE   | Point accuracy            |
| CRPS, WIS              | Distributional accuracy   |
| 50%/80%/95% coverage   | Interval calibration      |
| Log predictive density | Sharpness                 |
| Sign accuracy          | Direction                 |
| PIT histogram          | Calibration diagnostic    |
| Diebold-Mariano tests  | Pairwise model comparison |

**Report calibration, not just point error.** A Bayesian nowcast lives or dies on its intervals: PIT histograms and 50/80/95% coverage are first-class outputs alongside RMSE/MAE, not afterthoughts. This is also the metric most degraded by the smoothed-mean plug-in (§3 flag), so calibration is the canary for whether the §19.1 inference fix is working.

### 16.4 Benchmarks and Forecast-Comparison Tests

**⚑ Review — the consensus survey median is the primary benchmark.** Because consensus economists are explicitly forecasting the **BLS first print** (that is what markets trade), the consensus median is the correct apples-to-apples competitor for a first-print target. Beating consensus — not beating the settled truth — is the meaningful bar.

- **Primary competitor:** the Bloomberg/consensus survey median, scored against the *first announced* NFP value. The academic evidence is that the consensus median is biased but hard to beat for NFP — it has historically printed modestly *below* the first-print actual (on the order of ~10k jobs in the Patel–Murphy 2017 sample) and beats model-based approaches (Döpke, Bürgi et al. 2021). These magnitudes should be **re-estimated on this project's own real-time vintage sample** before being cited in the A5 evaluation.
- **Provider data is an input, not a competitor:** the model's own provider/PSP layer (§12) is a model *input*, not a benchmark series; its contribution is isolated by the no-provider ablation in §16.6, not by a head-to-head.
- **Internal baselines:** a revision-aware AR and/or a Kishor–Koenig revision VAR, which the full state-space model nests, so the heavy machinery (Rao-Blackwellized NUTS / FFBS / SV) can be shown to earn its keep.

Test stack:

| Test | Use | When |
|------|-----|------|
| Diebold–Mariano (1995) / West (1996) | unconditional equal predictive accuracy | non-nested pairwise (vs consensus) |
| **Giacomini–White (2006)** | **conditional** predictive ability — the right frame for a model re-estimated each month under a fixed rolling window | **primary** comparison vs consensus and for the ablation |
| Clark–West (2007) | MSPE adjustment for **nested** comparisons (standard DM is undersized) | model vs the nested revision-aware AR / VAR baselines |
| Mincer–Zarnowitz (1969) | forecast-efficiency/bias regression of the realized first print on the nowcast (test intercept 0, slope 1) | efficiency/bias diagnostic |

### 16.5 The Accuracy Ceiling and Floor

A first-print nowcast cannot beat the **unforecastable ("news") share** of the gap between the first print and the truth, and — more fundamentally — the first print is itself a noisy survey draw with **irreducible sampling error**. BLS reports the 90% confidence interval for the monthly change in total nonfarm employment is on the order of **±122,000** (Employment Situation Technical Note). No nowcast of the first print can have RMSE below this floor.

The realistic target is therefore *"beat consensus by X,"* not *"approach the truth."* Establish the forecastable share explicitly:

- **News-vs-noise decomposition** (Mankiw–Shapiro 1986; Aruoba 2008) on this project's own real-time first-print revisions, to quantify how much of the first-print surprise is predictable at all. A rigorous total-nonfarm first-print news/noise verdict appears to be an open contribution this project is well-placed to make.
- Report the model's MAE/RMSE against both the **±122k sampling floor** and the **consensus** number, so a "win" is unambiguous: statistically beating consensus on Giacomini–White, with calibrated intervals — otherwise the model is a monitoring tool, not a forecast improvement.

### 16.6 Isolating Provider-Data Value-Add

Run the **identical** state-space model with and without the PSP provider layer and report the Giacomini–White conditional test of the difference (mirroring the Dunn et al. 2026 ~11% error-reduction framing). Until this ablation is run, the justification for the provider layer — and for the Rao-Blackwellized NUTS / FFBS / SV apparatus over a simple revision-aware baseline — is unproven. If the provider layer does not produce a significant conditional-predictive-ability gain, simplify.

------------------------------------------------------------------------

## 17. Government Wedge Model (Separate)

The government wedge model (`wedge.py`) is a standalone change-space STS:

$$\mu_t = \text{drift} + \text{season}[m_t] + X_\text{intervention} \cdot \text{coef}$$

$$y_t \sim \mathcal{N}(\mu_t, \sigma)$$

where $y_t$ is the government wedge month-over-month change (published total − published private, in thousands). Sum-to-zero seasonal constraint (11 free effects). Intervention dummies for structural breaks. Masked over COVID.

Total NFP forecast = private nowcast + government wedge prediction.

------------------------------------------------------------------------

## 18. Key Invariants and Design Decisions

1.  **BD shock is NOT a Kalman state** — iid, so folded into $R_t$ / $d_t$
2.  **Fourier GRW steps only at year boundaries** — within-year amplitudes frozen
3.  **All observations outside the filter (post-B1/B2)** — enables Student-t without breaking Gaussian Kalman assumption. **⚑ Review (§19.1):** if this means observations are scored against smoothed-mean plug-ins, it is the review's highest-priority correction; the Student-t **scale-mixture augmentation** keeps observations *inside* an exact conditionally-Gaussian filter, preserving Student-t tails without the plug-in — in which case this invariant should be retired in favor of the scale-mixture formulation. Confirm against the current code (simulated-coverage gate, §19.1) before acting.
4.  **QCEW handled separately** — Student-t ($\nu=5$), too few observations to justify Kalman integration
5.  **AR(1) provider errors stay NUTS-side** — non-Gaussian-marginalizable
6.  **Non-centered SV is mandatory** — centered form creates severe funnel geometry
7.  **FFBS post-processing is two vmapped passes** — filter, then backward sample
8.  **PSP `include_bd=True`** — QCEW-calibrated growth includes birth/death; differs from legacy providers
9.  **Unknown `error_model` raises** — deliberate deviation from reference (which silently skipped)
10. **Config defaults are frozen** — changes require new parity baseline

------------------------------------------------------------------------

## 19. Methodological Review — Corrections & Prioritized Roadmap

This section consolidates the independent methodological review (`alt-nfp-methods-review.md`). The inline **⚑ Review** flags above point here. Items are ordered by the review's assessed impact on first-print accuracy and calibration; §19.7 sequences them into a staged plan with explicit "benchmark-to-change-the-plan" off-ramps so a fix can be downgraded if the evidence does not support it.

### 19.1 Inference correction — Student-t scale-mixture augmentation (highest priority)

**Conditionality — confirm before acting.** This correction applies *only if* the current implementation actually scores observations against smoothed-mean plug-ins, as §5/§18.3 describe. The methodology doc is internally inconsistent on this point: §3 also describes the *correct* analytic marginalization ("Kalman filter analytically marginalizes the state, likelihood injected via `numpyro.factor`"). Because the deployed code has been substantially upgraded since this doc's description was first written, the **first step is to confirm which scheme is live** — the simulated-coverage gate below is the discriminating test. If the marginalization is already analytic (and "smoothed mean" is only a post-hoc reconstruction for reporting), there is no error and this subsection is moot. The remainder describes the fix *if* the plug-in is confirmed.

**Problem (if the plug-in is live).** Running a prior-propagation-only filter, taking RTS smoothed state *means*, and scoring Student-t observation densities against those fixed means (§3, §5, §18.3) is **not** the model's marginal likelihood. It (a) double-uses the data — the smoother conditions on the observations, which are then scored against the smoothed mean; (b) discards the smoothing covariance $P_{t|T}$, **understating predictive uncertainty** and producing overconfident intervals; and (c) is not the collapsed/Rao-Blackwellized likelihood, which integrates the state out *analytically* rather than replacing it by a point estimate. It is neither the exact non-Gaussian likelihood nor a named, error-analyzed approximation (Laplace/EP/variational).

**Fix.** The only non-Gaussianity is the Student-t observation density, so keep the model **exactly conjugate** via the Gaussian scale-mixture (auxiliary-variable) representation. For each Student-t observation $y_t \sim t_\nu(\mu_t, \sigma_t^2)$ introduce a per-observation latent scale $\lambda_t$:

$$\lambda_t \sim \text{InverseGamma}\!\left(\tfrac{\nu}{2}, \tfrac{\nu}{2}\right), \qquad y_t \mid x_t, \lambda_t \sim \mathcal{N}\!\left(\mu_t,\ \lambda_t\,\sigma_t^2\right),$$

and marginalizing $\lambda_t$ recovers exactly $t_\nu(\mu_t, \sigma_t^2)$ (Geweke 1993). **Conditional on $\{\lambda_t\}$, every observation equation is Gaussian** with inflated noise $R_t^{\text{eff}} = \lambda_t \sigma_t^2$, so:

- the Kalman filter's **prediction-error decomposition gives the exact conditional marginal likelihood** $\log p(y_{1:T}\mid\theta,\{\lambda_t\})$ — observations now update the state via the Kalman gain, as they must;
- **FFBS samples the latent trajectory exactly** (Carter & Kohn 1994; Frühwirth-Schnatter 1994);
- the only added unknowns are the scales, whose full conditional is **conjugate**:
  $$\lambda_t \mid y_t, x_t, \theta \sim \text{InverseGamma}\!\left(\tfrac{\nu+1}{2},\ \tfrac{\nu + r_t^2/\sigma_t^2}{2}\right), \quad r_t = y_t - \mu_t,$$
  so they are Gibbs-updated (or sampled by NUTS), never plugged in.

This is precisely the device Kim, Shephard & Chib (1998) use for the non-Gaussian SV observation density, and the general partial-non-Gaussian data-augmentation approach of Frühwirth-Schnatter & Wagner; it delivers the genuine analytic Rao-Blackwellization (Doucet, de Freitas, Murphy & Russell 2000) the model already intends. The cost is one conjugate scale per Student-t observation — trivial relative to the Kalman pass — and the heavy tails (robustness to QCEW boundary noise and CES outliers) are preserved. If augmentation is to be avoided, the defensible fallback is a **named** single-pass Laplace/EP/variational robust-Student-t Kalman filter with analyzed error, not the ad hoc plug-in.

**Verification gate.** On simulated data, posterior coverage of the latent path and of the first-print predictive must be nominal (50/80/95%). The smoothed-mean scheme should fail this; if it does not, the fix can be downgraded.

### 19.2 De-pooled multi-vintage CES measurement (Jacobs–van Norden)

**Problem.** The pooled $(\alpha_\text{ces}, \lambda_\text{ces})$ across first/second/third prints (§5.3) assumes the systematic state→print map is identical across vintages, with only dispersion changing. The payroll-revision literature contradicts this: first prints are biased and revisions are partly predictable (Aruoba 2008; Neumark–Wascher 1991; Mankiw–Shapiro 1986). Combined with a QCEW anchor that disciplines the latent state toward the benchmark truth, a single pooled scalar pulls the frontier-month nowcast **truth-ward**, away from the first print it is scored on.

**Fix.** Replace the pooled CES equation with a **multi-vintage news/noise measurement block** (Jacobs & van Norden 2011; Kishor & Koenig 2012; Cunningham et al. 2012). Treat first print, second print, third print, and QCEW as distinct noisy reads of a latent true state $e_t^*$, each with its own bias and loading:

$$y_{t,r} = \alpha_r + \lambda_r\, e_t^* + b_{t,r} + \varepsilon_{t,r}, \qquad b_{t,r} = \phi_r b_{t-1,r} + \Gamma_r z_t + \psi_r u_{t,r},$$

with QCEW anchoring $e_t^*$ (near-unit loading) and the **first-print equation carrying its own bias**, so the first-print nowcast is identified separately from the benchmark truth. This is Phase C1 (§13.1) reframed and **promoted to a near-term correction** (the first print is the target). The pooled model is the nested restriction $\alpha_r\equiv\alpha,\ \lambda_r\equiv\lambda$ — testable by posterior overlap / likelihood ratio. **Benchmark-to-change-the-plan:** if vintage-specific $(\alpha_v,\lambda_v)$ are statistically indistinguishable on this sample, keep the pooled form.

### 19.3 Transform-specific seasonal loading

**Problem.** The model ingests both CES-SA and CES-NSA. The latent construction (§5.1) already routes the seasonal to the NSA composite only — correct — but the **shared** $\lambda_\text{ces}$ across the SA and NSA equations re-couples them, and a $K=4$ within-year-frozen Fourier seasonal cannot match BLS's concurrent X-13ARIMA-SEATS factors (Harvey 1989; Durbin & Koopman 2012).

**Fix.** Make the loading transform-specific (seasonal loads only on NSA series; SA loads on the seasonally-adjusted latent), or drop one CES transform to remove the redundancy. Lower priority than §19.1–§19.2: defensible-but-suboptimal, not an error.

### 19.4 Stochastic birth/death state

**Problem.** Birth/death is the dominant driver of benchmark revisions (preliminary March-2025 benchmark **−911,000**, finalized **−862,000**; BLS USDL-25-1352), yet it is modeled as a deterministic-given-parameters function of cyclical covariates plus an iid shock (§6.3) — no persistent dynamics of its own, a likely source of turning-point first-print miss.

**Fix.** Promote BD to a **time-varying latent state** — an AR(1)/random-walk BD level carried in the Kalman state, with the cyclical covariates (claims, JOLTS) as drift terms — which the latent-true-state approach naturally accommodates (Cajner et al. 2018, 2022). **Benchmark:** improvement in turning-point first-print errors (2020; the benchmark-heavy 2024–25 episodes).

### 19.5 Worksite validation of provider units

Where pseudo-establishment geography/industry is inferred from employees' **residences** rather than **worksites**, a commuting/telework wedge biases geography toward bedroom communities and misassigns industry, worst for the dense metros that dominate employment. The official infrastructure (Census LEHD, MWR→QCEW) allocates on worksite reports (LEHD TP-2006-01). **Fix:** prefer worksite address/ZIP fields from the payroll records where present, and validate residence-clustered units against an LEHD/QWI or QCEW worksite benchmark at the county×industry level (extending the Dunn et al. 2026 test).

### 19.6 SOTA positioning

Relative to the nowcasting frontier — mixed-frequency dynamic factor models (Giannone, Reichlin & Small 2008; Bańbura, Giannone & Reichlin 2011; the New York Fed Staff Nowcast, Bok et al. 2017; Almuzara et al. 2023), mixed-frequency VARs (Mariano & Murasawa 2003), and MIDAS (Ghysels et al.) — this model occupies a legitimate, underexplored niche: a **single-target, revision-aware** state-space model with a bespoke provider-data layer. That niche is defensible, but it earns its complexity only once the inference error (§19.1) is fixed and the provider layer's value is isolated by ablation (§16.6) against a simple revision-aware AR / Kishor–Koenig VAR baseline.

### 19.7 Prioritized staged roadmap

| Stage | Action | Benchmark to change the plan |
|---|---|---|
| **1 — likelihood & target** (before any head-to-head) | (a) Replace the smoothed-mean plug-in with Student-t scale-mixture augmentation + exact Kalman/FFBS (§19.1); verify nominal simulated-data coverage. (b) De-pool the CES block into vintage-specific $(\alpha_v,\lambda_v)$ inside a Jacobs–van Norden multi-vintage block anchored by QCEW (§19.2). | Coverage already nominal under the current scheme (it should not be) ⇒ downgrade (a); vintage-specific $\alpha/\lambda$ statistically indistinguishable ⇒ downgrade (b). |
| **2 — seasonality & birth/death** | Transform-specific seasonal loadings (NSA-only; §19.3); promote net birth/death to its own stochastic state with cyclical covariates as drift (§19.4). | No improvement in turning-point first-print errors (2020; 2024–25 benchmark episodes). |
| **3 — evaluation re-design** | Make the **consensus survey median the primary benchmark**; Giacomini–White conditional predictive ability, Clark–West for nested baselines, Mincer–Zarnowitz efficiency; report PIT/coverage; establish the forecastable-share ceiling vs the ±122k sampling floor (§16.4–§16.5). | No statistically significant Giacomini–White win over consensus ⇒ the model is a monitoring tool, not a forecast improvement. |
| **4 — microdata layer** | Validate residence-clustered units against LEHD/QCEW worksite geography (§19.5); run the provider-data ablation (§16.6). | Provider layer shows no significant conditional-predictive-ability gain ⇒ simplify. |

------------------------------------------------------------------------

## 20. References

### Real-time data, revisions, and target identification

- Aruoba, S.B. (2008). "Data Revisions Are Not Well Behaved." *Journal of Money, Credit and Banking* 40(2–3):319–340.
- Aruoba, S.B., Diebold, F.X. & Scotti, C. (2009). "Real-Time Measurement of Business Conditions." *Journal of Business & Economic Statistics* 27(4):417–427.
- Clements, M.P. (2019). "Do forecasters target first or later releases of national accounts data?" *International Journal of Forecasting* 35(4):1240–1249.
- Cunningham, A., Eklund, J., Jeffery, C., Kapetanios, G. & Labhard, V. (2012). "A State Space Approach to Extracting the Signal From Uncertain Data." *Journal of Business & Economic Statistics* 30(2):173–180. (Bank of England WP 336.)
- Jacobs, J.P.A.M. & van Norden, S. (2011). "Modeling data revisions: Measurement error and dynamics of 'true' values." *Journal of Econometrics* 161(2):101–109.
- Kishor, N.K. & Koenig, E.F. (2012). "VAR Estimation and Forecasting When Data Are Subject to Revision." *Journal of Business & Economic Statistics* 30(2):181–190.
- Koenig, E.F., Dolmas, S. & Piger, J. (2003). "The Use and Abuse of Real-Time Data in Economic Forecasting." *Review of Economics and Statistics* 85(3):618–628.
- Mankiw, N.G. & Shapiro, M.D. (1986). "News or Noise: An Analysis of GNP Revisions." *Survey of Current Business*.
- Neumark, D. & Wascher, W. (1991). Payroll-employment revisions. *Journal of Business & Economic Statistics* 9(2).
- Stark, T. (2011). Payroll-employment revision studies. Federal Reserve Bank of Philadelphia.
- Guisinger, A. & Smith, A.L. (2019). On JOLTS revisions.

### State-space inference, Student-t augmentation, and FFBS

- Carter, C.K. & Kohn, R. (1994). "On Gibbs Sampling for State Space Models." *Biometrika* 81(3):541–553.
- Doucet, A., de Freitas, N., Murphy, K. & Russell, S. (2000). "Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks." *UAI*:176–183.
- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press.
- Frühwirth-Schnatter, S. (1994). "Data Augmentation and Dynamic Linear Models." *Journal of Time Series Analysis* 15(2):183–202.
- Frühwirth-Schnatter, S. & Wagner, H. Data augmentation for partial non-Gaussian state-space models.
- Geweke, J. (1993). "Bayesian Treatment of the Independent Student-t Linear Model." *Journal of Applied Econometrics* 8(S1):S19–S40.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Kim, S., Shephard, N. & Chib, S. (1998). "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models." *Review of Economic Studies* 65(3):361–393.

### Forecast evaluation and comparison

- Clark, T.E. & West, K.D. (2007). "Approximately normal tests for equal predictive accuracy in nested models." *Journal of Econometrics* 138(1):291–311.
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics* 13(3):253–263.
- Döpke, J., Bürgi, C., et al. (2021). Consensus vs. model forecasts for NFP. *Journal of Economic Behavior & Organization*.
- Giacomini, R. & White, H. (2006). "Tests of Conditional Predictive Ability." *Econometrica* 74(6):1545–1578.
- Mincer, J. & Zarnowitz, V. (1969). "The Evaluation of Economic Forecasts." In *Economic Forecasts and Expectations*, NBER.
- Patel, A. & Murphy, R. (2017). Consensus vs. first-print NFP (sample magnitudes).
- West, K.D. (1996). "Asymptotic Inference About Predictive Ability." *Econometrica* 64(5):1067–1084.

### Nowcasting frontier and payroll-provider microdata

- Almuzara, M., Baker, R., O'Keeffe, H. & Sbordone, A. (2023). New York Fed Staff Nowcast methodology.
- Bańbura, M., Giannone, D. & Reichlin, L. (2011). "Nowcasting." In *The Oxford Handbook of Economic Forecasting* (eds. M.P. Clements & D.F. Hendry), Oxford University Press.
- Bok, B., Caratelli, D., Giannone, D., Sbordone, A. & Tambalotti, A. (2017). "Macroeconomic Nowcasting and Forecasting with Big Data." FRBNY Staff Report.
- Cajner, T., Crane, L.D., Decker, R.A., Hamins-Puertolas, A., Kurz, C. & Radler, T. (2018). "Using Payroll Processor Microdata to Measure Aggregate Labor Market Activity." FEDS 2018-005.
- Cajner, T., Crane, L.D., Decker, R.A., Hamins-Puertolas, A. & Kurz, C. (2022). "Improving the Accuracy of Economic Measurement with Multiple Data Sources: The Case of Payroll Employment Data." In *Big Data for Twenty-First-Century Economic Statistics*, NBER (WP 26033).
- Dingel, J.I. & Neiman, B. (2020). "How Many Jobs Can Be Done at Home?" *Journal of Public Economics* 189:104235.
- Dunn, A., English, A., Hood, K., Mason, C. & Quistorff, B. (2026). Validation test for payroll-provider microdata.
- Ghysels, E., Santa-Clara, P. & Valkanov, R. MIDAS regressions.
- Giannone, D., Reichlin, L. & Small, D. (2008). "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics* 55(4):665–676.
- Mariano, R.S. & Murasawa, Y. (2003). "A new coincident index of business cycles based on monthly and quarterly series." *Journal of Applied Econometrics* 18(4):427–443.

### Official / data sources

- BLS, *Employment Situation* Technical Note — ±122,000 90% confidence interval for the monthly change in total nonfarm employment.
- BLS, *QCEW Handbook of Methods* — ~5-month publication lag; >95% coverage of U.S. jobs.
- BLS, USDL-25-1352 (Sept 9 2025) — preliminary March-2025 CES benchmark −911,000 (−0.6%); finalized −862,000 (−0.5%).
- BLS, CES seasonal adjustment — X-13ARIMA-SEATS on a concurrent basis.
- CRS Report IF12827 — UI covers ~97% of nonfarm payroll employment.
- U.S. Census Bureau, LEHD Technical Paper TP-2006-01.
