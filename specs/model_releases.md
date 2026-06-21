# Staged Implementation Plan: Bayesian NFP Nowcasting System

> **Revision note.** This document has been aligned to the model actually implemented in
> `packages/nfp-model` (the JAX/NumPyro national state-space model). The originally separate
> **Release 1** (national measurement error) and **Release 2** (national birth/death) are now
> described together as a single implemented national model in **Part I**, because the code is one
> unified model — there is no built intermediate without birth/death, QCEW, or seasonality.
> **Releases 3–5** (cell-level estimation, nested hierarchy, QCEW forecasting + time-varying bias)
> are **not implemented**; they are described as planned work in **Part II** and correspond to the
> port plan's Phase B (`specs/plans/0-port_and_staged_plan.md`).
>
> One deliberate exception to "match the code": the **provider priors are written as hierarchical**
> (the multi-provider pooling design), even though the current code realizes the single-provider
> *collapse* of that hierarchy with fixed weakly-informative priors. Hierarchical pooling remains the
> plan; see Part I §7 for the implemented-vs-target mapping.

## Overview

This document describes a Bayesian state-space NFP nowcasting system. The implemented system (Part I)
produces a national NFP nowcast under strict as-of censoring, fusing CES survey prints, lagged QCEW
administrative anchors, payroll-provider continuing-units microdata, and cyclical indicators. Planned
extensions (Part II) add geographic/industry decomposition and real-time QCEW conditioning while
maintaining backward compatibility with the national model.

### Design Principles

1.  **Forecast accuracy before narrative.** Birth/death correction (the fundamental limitation of
    continuing-units data) is prioritized over geographic/industry decomposition. *Realized:* the
    national model carries a structural birth/death component; decomposition is deferred to Part II.

2.  **Multi-provider by construction.** The measurement-error framework treats each provider as an
    independent noisy observation of latent truth, so additional providers slot in without
    retrofitting. *Realized:* per-provider likelihoods with config-driven loading and an iid/AR(1)
    error option. The hierarchical pooling over providers is the planned form (Part I §7); with a
    single provider it collapses to weakly-informative per-provider priors.

3.  **Coherent hierarchical forecasts (planned).** Once cell-level estimation begins (Release 3),
    dual national/cell estimation with MinT reconciliation will ensure cell contributions sum exactly
    to the national forecast.

### Implementation status

| Capability | Status |
|---|---|
| National latent state — non-centered AR(1) with era-specific means | **Implemented** |
| Fourier seasonal (annually-evolving amplitudes) + SA/NSA decomposition | **Implemented** |
| Structural birth/death with cyclical covariates (claims, JOLTS) | **Implemented** |
| QCEW Student-t anchor (tiered base σ, per-obs revision multipliers) | **Implemented** |
| CES likelihood — best-available print, vintage-indexed σ, shared bias/loading | **Implemented** |
| Multi-provider measurement (iid/AR(1) errors) | **Implemented** (fixed priors; hierarchical pooling planned) |
| Cell-level estimation + dual national/cell + MinT (Release 3) | Planned — Phase B |
| Nested geo/industry hierarchy (Release 4) | Planned — Phase B |
| Forecasted QCEW + time-varying provider bias (Release 5) | Planned — Phase B |

### Notation and implementation mapping

The model works in **monthly log-growth** units throughout. Latent paths and their composites map to
the deterministic sites in `nfp_model.model`:

| Symbol | Meaning | Code site |
|---|---|---|
| $g^{cont}_t$ | latent continuing-units (intensive-margin) growth, seasonally adjusted | `g_cont` |
| $s_t$ | seasonal component | `seasonal` |
| $BD_t$ | net birth/death contribution | `bd` |
| $g^{cont,nsa}_t = g^{cont}_t + s_t$ | continuing-units growth, not seasonally adjusted | `g_cont_nsa` |
| $g^{tot,sa}_t = g^{cont}_t + BD_t$ | total employment growth, seasonally adjusted | `g_total_sa` |
| $g^{tot,nsa}_t = g^{cont}_t + s_t + BD_t$ | total employment growth, not seasonally adjusted | `g_total_nsa` |

Every prior value below is pinned to `nfp_model.config` (the A3 parity contract). Change a prior only
behind a new parity baseline.

---

# PART I — IMPLEMENTED: National State-Space Model

## Objective

Produce a national NFP nowcast by jointly estimating a latent continuing-units growth path, an
annually-evolving seasonal, and a structural birth/death contribution, then reading those through the
QCEW, CES, and payroll-provider observation equations. The target is the official **seasonally
adjusted CES headline**, modeled as a biased, scaled, revision-dependent observation of latent total
growth.

## System Description

Continuing-units growth is the economic signal common to payroll providers and (after birth/death
adjustment) to CES and QCEW. Providers observe the not-seasonally-adjusted continuing-units signal
$g^{cont,nsa}_t$; CES and QCEW observe total growth (continuing units plus birth/death). QCEW is a
near-census, NSA, lagged anchor that lets the model learn bias, loading, and noise parameters and
project them forward to the censored frontier where the nowcast lives.

### Data Inputs

| Source | Frequency | Lag | Description |
|---|---|---|---|
| Payroll Provider *p* National Index | Monthly | ~3 weeks | Continuing-units (intensive-margin) growth from provider microdata on a rotating, internally frozen measurement panel aligned to the CES reference week (stabilized clients only; administrative exits removed without counting as job loss). **Not seasonally adjusted.** |
| CES headline (SA) | Monthly | ~3 weeks (rev-0) | BLS CES seasonally adjusted total employment growth; revisions rev-0/1/2. |
| CES headline (NSA) | Monthly | ~3 weeks (rev-0) | BLS CES not-seasonally-adjusted total employment growth; revisions rev-0/1/2. |
| QCEW National | Quarterly | 5–6 months | Near-census total employment, **not seasonally adjusted**; revision selected per quarter. |
| Cyclical indicators | Monthly | claims 1 mo, JOLTS 2 mo | Initial claims (ICNSA) and JOLTS job openings (JTSJOL), centered upstream. |

The provider series is computed on a rotating, internally frozen panel so that provider
onboarding/offboarding cannot masquerade as economic job flows. Within the frozen window,
month-to-month growth uses matched client observations only; clients that administratively exit are
removed from the panel and do not contribute negative change — the measurement target is the
intensive-margin employment change of continuing units. (Construction details are in the companion
data-methods document; the model consumes the finished continuing-units series.)

### Complete Model Specification

**Latent continuing-units growth — non-centered AR(1) with era-specific means.**
The latent path is a *stationary* AR(1) parametrized by its stationary standard deviation $\tau$
(this breaks the $\phi$–$\sigma$ ridge), with a regime break separating the pre- and post-COVID means
(persistence and marginal SD are shared across eras; only the mean differs):

$$\tau \sim \text{LogNormal}(\log 0.013,\ 0.5)$$

$$\phi_{raw} \sim \text{Beta}(18,\ 2),\qquad \phi = \min(\phi_{raw},\ 0.99),\qquad \sigma_g = \tau\sqrt{1-\phi^{2}}$$

$$\mu_{g,e} \sim N(0.001,\ 0.005^{2}),\quad e \in \{0,1\}\ \ (\text{era break at 2020-01})$$

$$\varepsilon_{g,t} \sim N(0,1),\qquad g^{cont}_{0} = \mu_{g,0} + \sigma_g\,\varepsilon_{g,0}$$

$$g^{cont}_{t} = \mu_{g,e(t)} + \phi\big(g^{cont}_{t-1} - \mu_{g,e(t)}\big) + \sigma_g\,\varepsilon_{g,t}$$

The 2020–2021 window is excluded from evaluation; the estimation sample begins ~2012.

**Fourier seasonal — annually-evolving amplitudes (non-centered Gaussian random walk across years).**
With $K=4$ harmonics, the cosine/sine amplitudes evolve year to year; innovation scale decreases with
harmonic order:

$$\sigma^{(k)}_{F} \sim \text{LogNormal}\big(\log 0.0003 - \log k,\ 0.5\big),\quad k = 1,\dots,K$$

$$z \sim N(0,1)^{(2K)\times n_{yr}},\qquad
\text{step}_{\cdot,1} = 0.015\cdot z_{\cdot,1},\quad
\text{step}_{\cdot,y>1} = \sigma^{(k)}_{F}\cdot z_{\cdot,y}$$

$$[A_{k,y};\,B_{k,y}] = \operatorname{cumsum}_{y}(\text{step}),\qquad
s_t = \sum_{k=1}^{K} A_{k,\,yr(t)}\cos\!\Big(\tfrac{2\pi k\,m_t}{12}\Big) + B_{k,\,yr(t)}\sin\!\Big(\tfrac{2\pi k\,m_t}{12}\Big)$$

where $m_t$ is month-of-year and $yr(t)$ the calendar year index. The first half of the coefficient
rows are the $A_k$ (cosine), the second half the $B_k$ (sine).

**Structural birth/death.** A constant plus cyclical covariates plus a structural innovation;
covariates are centered upstream and gated out when a column is identically zero (which keeps
backtest iterations identified):

$$\phi_0 \sim N(0.001,\ 0.002^{2}),\qquad \sigma_{BD} \sim \text{LogNormal}(\log 0.003,\ 0.5),\qquad \xi_t \sim N(0,1)$$

$$\phi_{3,i} \sim N(0,\ 0.3^{2})\ \text{ per active covariate } i \in \{\text{claims, JOLTS}\}$$

$$BD_t = \phi_0 + \sigma_{BD}\,\xi_t + \sum_{i \in \text{active}} \phi_{3,i}\,X^{(i)}_t$$

There is **no lagged-QCEW-birth/death proxy term** and **no separate birth-rate covariate**: both were
removed empirically (posteriors indistinguishable from zero). The surviving cyclical block is
$[\text{claims, JOLTS}]$ only.

**Composite growth signals.**

$$g^{cont,nsa}_t = g^{cont}_t + s_t,\qquad g^{tot,sa}_t = g^{cont}_t + BD_t,\qquad g^{tot,nsa}_t = g^{cont}_t + s_t + BD_t$$

**Data Likelihood — QCEW (near-census NSA total, lagged, truth anchor).**
A Student-t anchor with two estimated base scales (a tight M2 tier and a wider boundary tier for
M1/M3 publications) times a per-observation revision multiplier from the publication schedule
(post-COVID boundary months carry an additional era multiplier). The tight M2 prior prevents QCEW
precision from dominating identification; LogNormal (not HalfNormal) scale priors avoid funnel
geometry:

$$\sigma^{mid}_{Q} \sim \text{LogNormal}(\log 0.0005,\ 0.15),\qquad \sigma^{bnd}_{Q} \sim \text{LogNormal}(\log 0.002,\ 0.5)$$

$$\sigma_{Q,\,obs} = \big[\,\mathbb{1}\{M2\}\,\sigma^{mid}_{Q} + \mathbb{1}\{\text{boundary}\}\,\sigma^{bnd}_{Q}\,\big]\cdot r_{obs}$$

$$y^{QCEW}_{obs} \sim \text{Student-}t\big(\nu = 5,\ g^{tot,nsa}_{obs},\ \sigma_{Q,\,obs}\big)$$

QCEW arrives with a 5–6 month lag; the per-quarter maximum revision selected at a given horizon is
$\{Q1{:}4,\ Q2{:}3,\ Q3{:}2,\ Q4{:}1\}$. Because QCEW is lagged, its role is to pin bias/loading/noise
that then propagate forward to the real-time frontier.

**Data Likelihood — CES (best-available print, vintage-indexed σ, shared bias/loading).**
CES is modeled as a *biased, scaled* observation of latent total growth with **revision-indexed**
noise, separately for the SA and NSA series. One observation per month per series is used, at the
highest available revision (CES vintages are correlated at $\rho > 0.99$, so using all of them
overcounts information):

$$\alpha_{ces} \sim N(0,\ 0.005^{2}),\qquad \lambda_{ces} \sim \text{TruncatedNormal}(1,\ 0.1;\ \text{low}=0.5)$$

$$\sigma^{sa}_{ces,v} \sim \text{LogNormal}(\log 0.002,\ 0.5),\qquad \sigma^{nsa}_{ces,v} \sim \text{LogNormal}(\log 0.002,\ 0.5)\quad \text{per observed vintage } v$$

$$y^{ces,sa}_{obs} \sim N\big(\alpha_{ces} + \lambda_{ces}\,g^{tot,sa}_{obs},\ \sigma^{sa}_{ces,\,v(obs)}\big)$$

$$y^{ces,nsa}_{obs} \sim N\big(\alpha_{ces} + \lambda_{ces}\,g^{tot,nsa}_{obs},\ \sigma^{nsa}_{ces,\,v(obs)}\big)$$

The vintage index $v(obs)$ is the revision number (0/1/2) selected for that month, remapped to a
contiguous range so there are no ghost-parameter scales for vintages with zero observations.

**Data Likelihood — Payroll Provider *p* (continuing units, NSA).**
Each provider observes the not-seasonally-adjusted continuing-units signal with its own bias and
loading, and either iid or AR(1) measurement error:

$$y^{p}_{t} = \alpha_p + \lambda_p\,g^{cont,nsa}_t + \varepsilon_{p,t}$$

$$\text{iid: } \varepsilon_{p,t} \sim N(0,\ \sigma_p^{2}),\qquad
\text{AR(1): } \varepsilon_{p,t} = \rho_p\,\varepsilon_{p,t-1} + u_{p,t}\ \ (\text{stationary init})$$

See §7 for the provider priors (hierarchical target vs. implemented collapse).

### 7. Provider priors — hierarchical pooling (planned) and its implemented collapse

The provider parameters are designed as a **hierarchical pool**, so that with multiple providers the
model learns provider-specific bias/loading/noise while shrinking sparse providers toward shared
means. This is the planned form:

*Bias:*

$$\alpha_p \sim N(\mu_\alpha,\ \tau_\alpha^{2}),\qquad \mu_\alpha \sim N(0,\ 0.005^{2}),\qquad \tau_\alpha \sim \text{Half-}N(0,\ 0.005)$$

*Signal loading:*

$$\lambda_p \sim N(\mu_\lambda,\ \tau_\lambda^{2}),\qquad \mu_\lambda \sim N(1,\ 0.15^{2}),\qquad \tau_\lambda \sim \text{Half-}N(0,\ 0.1)$$

*Observation noise (log scale):*

$$\log \sigma_p \sim N(\mu_\sigma,\ \tau_\sigma^{2}),\qquad \mu_\sigma \sim N(\log 0.002,\ 0.5^{2}),\qquad \tau_\sigma \sim \text{Half-}N(0,\ 0.3)$$

*AR(1) persistence (when the AR(1) error model is selected):*

$$\rho_p \sim \text{Beta}(2,\ 3)$$

> **Implemented vs. target.** The current code (`nfp_model.config.ProviderPriors`) realizes the
> **single-provider collapse** of this hierarchy with fixed priors:
> $\alpha_p \sim N(0,\ 0.005)$, $\lambda_p \sim N(1,\ 0.15)$, $\sigma_p \sim \text{InverseGamma}(3,\ 0.004)$
> (median $\approx 0.002$), and $\rho_p \sim \text{Beta}(2,3)$ for the AR(1) option. The hyperprior
> locations and scales above are chosen so the pool reduces to these fixed priors when $P = 1$;
> activating the hierarchy ($P \ge 2$) is the open task that this section retains as the plan. The
> InverseGamma noise prior is the current collapse of the LogNormal hierarchical noise above.

### Forecast Production

1.  **Censored estimation.** Run NUTS on all data knowable as of date $D$ (two-layer as-of censoring
    upstream; the model never sees a vintage date). The latent $g^{cont}$, seasonal $s_t$, and $BD_t$
    paths are inferred jointly.
2.  **Nowcast transform.** Map the SA total path through the CES-SA observation equation,
    $\alpha_{ces} + \lambda_{ces}\,g^{tot,sa}_t$, take the posterior-mean growth path, rebuild the
    index path from the panel's base index, and read off the target month (the as-of month itself,
    which sits at the censored frontier, so the last latent state is the nowcast proxy).
3.  **Reporting.** Convert the month-over-month index change to jobs added (thousands).

### Output Specification

The packaged nowcast summary (`nfp_model.nowcast.nowcast_summary`) returns:

| Output | Description |
|---|---|
| `nowcast_growth` | Point estimate — posterior-mean log growth at the target month |
| `nowcast_index` | Implied index level at the target month |
| `nowcast_change_k` | Month-over-month jobs added (thousands) from the posterior-mean index path |
| `pred_mean` | The $(T,)$ posterior-mean predictive growth path |
| `pred_draws` | The $(\text{chains}\times\text{draws})$ predictive draws at the target month — the source for std and 80%/95% credible intervals |

Additional quantities are available directly as posterior sites: the decomposition paths
(`g_cont`, `seasonal`, `bd`, `g_total_sa`, `g_total_nsa`), the CES bias/loading
($\alpha_{ces}, \lambda_{ces}$) and per-vintage noise scales, the QCEW tier scales, and the
per-provider bias/loading/noise ($\alpha_p, \lambda_p, \sigma_p$, and $\rho_p$ under AR(1)) — the
latter being the provider signal-quality and bias diagnostics.

### Validation Metrics

-   Vintage-aware backtest RMSE against CES first/second/final prints (the canonical evaluation),
    scored at each information regime (e.g. T−12/9/6/3/1); LOO-CV is treated as a data-quality audit,
    not model evaluation.
-   Coverage of 80%/95% credible intervals.
-   Comparison to naive baselines (random walk on CES) as a sanity floor, and — once added (Phase A5)
    — to ADP and consensus-median competitors at each regime.
-   Birth/death estimates vs. realized QCEW birth/death (with lag).
-   Per-provider signal-quality and bias rankings.

### Limitations (of the national model)

-   National-level only — no geographic or industry decomposition (Part II).
-   Birth/death does not capture industry heterogeneity.
-   Provider bias is time-invariant (time-varying bias is Release 5).
-   Provider pooling is the single-provider collapse until the hierarchy is activated (§7).

---

# PART II — PLANNED EXTENSIONS (not implemented)

> The following releases are **design sketches, not built code.** They map to the port plan's Phase B
> (`specs/plans/0-port_and_staged_plan.md`): B1 supersector narrative, B2 forecasted QCEW + time-varying
> provider bias, B3 MinT reconciliation + production hardening. Do not treat the equations below as a
> description of `nfp-model`. Several Phase-B strategic questions (target = first print vs.
> benchmark-informed truth; consumer of the output; whether the banked model beats consensus/ADP)
> are to be resolved before this work begins.
>
> Note: the existing 44-cell (11 supersector × 4 region) QCEW-weighted **compositing** lives in the
> **data** layer (`nfp_ingest/compositing.py`) and collapses cell-level provider data into a single
> national series *before* it reaches the model. It is a representativeness correction, **not** the
> cell-level Bayesian model described below.

## Release 3 (planned): Cell-Level Estimation with Dual Framework and MinT

**Objective.** Geographic × industry (cell) decomposition with coherent national alignment, retaining
birth/death and multi-provider integration.

**Key additions.**

-   Cell-level latent states $\mu^{cont}_{j,t}$ (geo unit × supersector) alongside an **independent
    national state** $\mu^{nat}_t$ informed by CES (dual estimation).
-   Provider × cell parameters with exchangeable hierarchical shrinkage:
    $\alpha_{p,j} = \alpha^{prov}_p + \alpha^{cell}_j + \alpha^{resid}_{p,j}$, and likewise for the
    loading $\beta_{p,j}$ and a log-additive noise decomposition.
-   Cell-level birth/death with exchangeable industry effects.
-   A national-vs-sum-of-cells discrepancy state $\delta_t$ (random walk).
-   Cell-level QCEW observations.

**MinT reconciliation.** With base forecasts $\hat y_t = [\hat\mu_{1,t},\dots,\hat\mu_{J,t},\hat\mu^{nat}_t]'$,
reconciled forecasts $\tilde y_t = S(S'W^{-1}S)^{-1}S'W^{-1}\hat y_t$ with $W$ from historical forecast
errors, enforcing $\sum_j w_j \tilde\mu_{j,t} + \delta_t = \tilde\mu^{nat}_t$.

## Release 4 (planned): Nested Hierarchical Structure

**Objective.** Exploit the nesting of geography (region → division → state) and industry
(domain → supersector) for finer granularity without overfitting.

**Key additions.** Replace exchangeable cell priors with nested random effects; bias and loading
decompose into geographic, industry, provider, provider×industry, and residual components; birth/death
intensity follows the nested industry hierarchy. Example bias decomposition:

$$\alpha_{p,j} = \alpha^{prov}_p + \alpha^{geo}_{g(j)} + \alpha^{ind}_{s(j)} + \alpha^{prov\times ind}_{p,s(j)} + \alpha^{resid}_{p,j}$$

with $\alpha^{geo} = \alpha^{region} + \alpha^{div|region} + \alpha^{state|div}$ and
$\alpha^{ind} = \alpha^{domain} + \alpha^{supersector|domain}$, each level given its own Half-Normal
variance-component prior.

## Release 5 (planned): QCEW Conditioning and Time-Varying Bias

**Objective.** Let provider bias evolve over time with QCEW error-correction, and forecast QCEW to the
present to tighten conditioning during the 5–6 month publication lag.

**Key additions.**

-   Time-varying bias with RW1 dynamics and QCEW error-correction:
    $\alpha_{p,j,t} = \alpha_{p,j,t-1} + \omega_{p,j,t} - \kappa_p\,d_{p,j,t-L}$, where $d$ is the
    discrepancy vs. QCEW at lag $L$ and $\kappa_p \sim \text{Beta}(3,3)$ is a provider-specific
    error-correction speed.
-   A QCEW forecast model (location quotients + leading indicators) producing
    $\hat y^{QCEW}_{j,t}$, which then enters as an explicitly-noisier observation for real-time
    conditioning.
-   The model **conditions on** QCEW (a high-precision observation) rather than benchmarking
    (constraining sums) — consistent with the national model's treatment in Part I.

---

# Appendix A: Implemented vs. Planned — Capability Summary

| Release | Scope | Multi-provider | Birth/Death | Seasonal | Hierarchy | Dual + MinT | QCEW | Time-varying bias | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1–2 | National | Yes (pooling planned) | Yes (structural) | **Yes (Fourier)** | — | No | Lagged (Student-t, tiered) | No | **Implemented** |
| 3 | Geo × Supersector | Yes | Yes (cell) | Yes | Exchangeable | **Yes** | Lagged (cell) | No | Planned |
| 4 | State × Supersector | Yes | Yes (nested) | Yes | **Nested** | Yes | Lagged (cell) | No | Planned |
| 5 | State × Supersector | Yes | Yes (nested) | Yes | Nested | Yes | **Forecasted** | **Yes** | Planned |

Two features of the implemented model that the original staged plan omitted: an **annually-evolving
Fourier seasonal** with an explicit SA/NSA decomposition, and a **CES observation equation with bias,
loading, and vintage-indexed noise** (CES is not treated as a direct readout of latent truth).

# Appendix B: Multi-Provider Design Rationale

1.  **No retrofitting.** Adding providers later requires restructuring; building the measurement-error
    framework for it from the start is cleaner.
2.  **Graceful degradation.** With one provider the hierarchical priors collapse to weakly-informative
    per-provider priors (the current implemented form); the framework works without overcomplicating.
3.  **Immediate diagnostics.** Even with one provider, the model exposes provider-specific signal
    quality ($\lambda_p$) and bias ($\alpha_p$).
4.  **Provider comparison.** With multiple providers, the hierarchy automatically learns relative
    strengths; at the cell level (Release 3+) the
    $\alpha_{p,j} = \alpha^{prov}_p + \alpha^{cell}_j + \alpha^{prov\times ind}_{p,s(j)} + \alpha^{resid}_{p,j}$
    decomposition separates provider main effects, cell difficulty, provider×industry specialization,
    and residual.

For a given target, signals from multiple providers combine via precision weighting: providers with
higher loading and lower noise receive more weight.

# Appendix C: Implementation Notes

## Non-centered parameterization

The implemented model is non-centered throughout — the latent AR(1) (`eps_g` $\sim N(0,1)$ scaled by
$\sigma_g$ around the era mean), the Fourier GRW (`fourier_z` $\sim N(0,1)$ scaled by the per-harmonic
innovation), and the birth/death innovation (`xi_bd` $\sim N(0,1)$ scaled by $\sigma_{BD}$). The same
non-centering is essential for sparse cells and providers in Part II:

```
# Non-centered (recommended)
α̃_{p,j} ~ N(0, 1)
α_{p,j} = μ_α + τ_α × α̃_{p,j}
```

The latent parametrization uses the **stationary SD** $\tau$ with $\sigma_g = \tau\sqrt{1-\phi^2}$
specifically to break the $\phi$–$\sigma$ ridge that a direct $(\phi, \sigma)$ parametrization
induces.

## Computational scaling

| Release | Approx. parameters | Typical runtime |
|---|---|---|
| 1–2 (implemented) | latent path $T$ + seasonal $2K\times n_{yr}$ + BD path $T$ + $\mathcal{O}(10)$ scalars + per-provider $\mathcal{O}(3)\times P$ | Minutes (CPU); the batched vmap path fits an as-of grid in one program (GPU is the speed lever) |
| 3 | $\sim 50 + 10P + 5J + P\times J$ | 30–60 min |
| 4 | $\sim 100 + 10P + \text{hierarchy} + P\times J$ | 1–2 hours |
| 5 | Release 4 $+\ T\times P\times J$ (bias trajectories) | 3–6 hours |

$P$ = number of providers, $J$ = number of cells, $T$ = time periods, $K$ = Fourier harmonics ($=4$),
$n_{yr}$ = number of years in the sample.