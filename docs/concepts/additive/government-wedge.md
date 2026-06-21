# The Government Wedge

The government wedge is the second addend of the [additive nowcast
framework](overview.md): the piece that turns a private nowcast into a Total-NFP
nowcast. It is a small, standalone Bayesian model (`nfp_model.wedge.wedge_model`)
that forecasts the `00 − 05` difference directly, and its posterior draws are
convolved with the private nowcast's draws to assemble the Total. This page
folds the government-wedge design record and documents the model as it ships.

## Why a wedge, not government `'90'`

The product competes against the **Total** NFP consensus (`industry_code` `'00'`),
but the state-space model can only justify a **private** nowcast (`'05'`). The
accounting identity closes the gap:

$$
\underbrace{\text{Total}\,('00')}_{\text{what consensus measures}}
  = \underbrace{\text{Private}\,('05')}_{\text{state-space model}}
  + \underbrace{\text{Government}\,('90')}_{?}
$$

The naive move would be to build a model for published government (`'90'`). That
has a fatal flaw: BLS seasonally adjusts `'00'`, `'05'`, and `'90'` **independently**,
so in seasonally adjusted space they do not add up exactly —

$$
\mathrm{SA}('00') \neq \mathrm{SA}('05') + \mathrm{SA}('90'),
\qquad r = '00' - '05' - '90' \neq 0.
$$

Forecasting published `'90'` would bake that residual $r$ into the Total as an
irreducible error floor. Instead the model forecasts the **wedge** defined as the
exact difference:

$$
\text{wedge}_t := \text{published}\,'00'_t - \text{published}\,'05'_t,
\qquad
\text{Total} = \text{private nowcast} + \text{wedge forecast}.
$$

Because both legs come from the *same first-print release*, adding the wedge back
reproduces the published Total **by construction** — no seasonal-adjustment leakage,
no hierarchical reconciliation step. The wedge is *economically* government (it
carries education seasonality and federal shocks) but is *defined* as `00 − 05`, so
coherence with the published Total is free.

## What the wedge is worth, and where its edge lives

Measured over 2017–2026 (COVID excluded, $n \approx 82$), the first-print wedge
month-over-month change has a standard deviation of about **24k**, and a plain
calendar-month seasonal mean already forecasts it to roughly 23k RMSE. Propagated
into the Total alongside a ~45k-RMSE private nowcast, a good wedge model adds only
about **5k** of Total RMSE. Two consequences shape the entire design:

- **The wedge is secondary.** The consensus contest is won or lost on the *private*
  nowcast. The wedge model is deliberately minimal; complexity must be justified by
  measured Total-error reduction.
- **The edge lives in the tails.** Consensus forecasters also eat government
  uncertainty. Normal months are near-naive-forecastable for everyone, so the only
  edge over the street's implicit government forecast is modeling **known shocks**
  (RIFs, the decennial census, shutdowns) better than a seasonal mean can.

This is why the model is, by design, a coherent-posterior generalization of the
calendar-month mean, with its whole complexity budget spent on
**announcement-priored interventions**.

## The model — a change-space structural time series

`wedge_model` is a NumPyro program that models the wedge **month-over-month change**
$y_t$ directly in change-space (units: thousands of jobs). It imports only
`jax`/`numpyro`/`numpy` and reimplements the likelihood-mask idiom inline, honoring
the same import boundary as the private model. The change mean is

$$
\mu_t = \text{drift} + \text{season}\bigl[m(t)\bigr]
  + \sum_k X^{\text{iv}}_{t,k}\,c_k,
$$

and the likelihood is a masked iid Normal:

$$
y_t \sim \mathcal{N}(\mu_t,\ \sigma),\qquad
\text{drift}\sim\mathcal{N}(0,\ 50),\qquad
\sigma\sim\mathrm{HalfNormal}(30).
$$

The likelihood is applied **under a boolean mask** over the COVID window (2020–21)
and the October-2025 no-print hole — masked, never deleted, so drift and seasonality
stay on a contiguous calendar axis.

!!! note "A single constant drift — no random walk"
    The change mean has a single *constant* `drift`, not a random walk or AR term.
    A random walk on the change is an I(2) process on the wedge *level*, and that is
    exactly why a random-walk baseline lost to the seasonal mean in the diagnostic.
    Persistence is guilty until proven; an AR(1) residual is deferred.

### Seasonality — a sum-to-zero monthly block

Seasonality is **dummy effects, not Fourier** — the dominant signal is sharp
state/local **education** seasonality (the Aug/Sep/Jun cliffs) that a harmonic basis
would smear. Eleven free monthly effects are sampled non-centered, with the twelfth
pinned by a sum-to-zero constraint (which kills the drift/seasonal ridge):

$$
\text{season}[1{:}11] = \tau_{\text{season}}\cdot z,\quad
z\sim\mathcal{N}(0,1)^{11},\qquad
\text{season}[12] = -\sum_{m=1}^{11}\text{season}[m],
$$

$$
\tau_{\text{season}} \sim \mathrm{HalfNormal}(30).
$$

A federal/state-local seasonal split is **rejected on identifiability grounds**: the
likelihood observes only the *summed* wedge, so two seasonal blocks enter only
through their sum and their difference is never identified from wedge-only data.

### Interventions — the announcement-priored edge

The intervention layer is the only complexity sink and the model's entire edge. Each
known shock contributes a column $X^{\text{iv}}_{\cdot,k}$ whose **shape is fixed by
configuration** while only its **magnitude $c_k$ is sampled**, under an informative,
announcement-derived prior:

$$
c_k \sim \mathcal{N}\!\bigl(\text{magnitude}_k,\ \text{magnitude\_sd}_k\bigr).
$$

The prior is never flat: a brand-new shock at the frontier can only be *imposed* from
an announcement, not inferred from the one or two months of frontier data it would
otherwise chase. The X-13 shock vocabulary maps into change-space via
`nfp_lookups.government.intervention_column`:

| Event | Level-space shape | Change-space encoding |
|---|---|---|
| Permanent RIF (no back-pay) | level shift | `pulse` — a single change step (level steps and stays) |
| Phased RIF | ramp | `box` — $1/k$ over $k$ months |
| Census hire | temporary change | `tc` — a positive pulse with geometric givebacks |
| Back-paid shutdown | additive outlier | nets ≈ 0 — handled by the mask, no term |

Over the 2017–2026 ex-COVID window this is roughly **one active mechanism**: the
2025 federal RIF as a negative pulse, with a dormant census slot that activates in
2030. The interventions live in a frozen `KNOWN_INTERVENTIONS` table carrying an
**`announcement_date` axis**, so a backtest can censor to what was knowable at each
release-eve through `get_known_interventions_as_of(as_of)` — the lookahead guard is a
real date comparison, not a vacuous one.

!!! warning "Placeholder intervention priors"
    The shipping `KNOWN_INTERVENTIONS` table carries **placeholder** 2025-RIF
    magnitudes. The real announcement-derived numbers (permanent-separation count,
    honest SD, effective date, source URL) are a human input the maintainer supplies
    before any accuracy claim. The placeholder keeps the build unblocked.

## Government components — diagnostics, not regressors

BLS publishes government components `91` (federal), `92` (state), `93` (local), which
sum to `90`. These are **acquired but are not likelihood regressors**
(`GOVERNMENT_INDICATORS` in `nfp_lookups.government`). Two reasons:

- **Lagged government is a weak regressor.** The only government information available
  at release-eve is lagged (components publish *with* the total), and the wedge change
  is near-white around its seasonal mean — a lagged-`90` term spends degrees of freedom
  against a ~5k budget for little payoff.
- **Their real value is shock attribution.** The 2025 RIF and the 2030 census live
  entirely in federal (`91`). Seeing `91` separately lets the intervention priors be
  *sized and validated* against the clean federal series rather than against a wedge
  where state/local education noise drowns them — a diagnostic and prior-calibration
  use, not a likelihood input.

A component-structured *likelihood* would re-introduce the SA-additivity residual $r$
the wedge was designed to escape, so it is deferred to the Bloomberg port.

## Assembly into the Total

The wedge posterior predictive draws — including observation noise, mirroring the
private nowcast's first-print predictive so the two convolve like-for-like — are
combined with the private nowcast by an **independent, draw-wise convolution**:

$$
\text{Total change}^{(d)} = \text{private change}^{(d)} + \text{wedge change}^{(d)}.
$$

Adding-up is exact because the wedge is forecast directly (no reconciliation needed).
The one seam that bites is units: the persisted private nowcast draws are in
growth/index space and must be converted to change-in-thousands *before* the natively
change-space wedge draws are added. That conversion is an explicit, tested
`assemble_total` helper on the harness side (`nfp_vintages.assembly`) — `nfp_model`
itself stays assembly-free.

Independence is the one live modeling bet, and a defensible one: the private nowcast
is a business-cycle object and the wedge an education/policy object, with a measured
outcome correlation of only about +0.19. A residual-coupling knob that would widen the
Total intervals under positive error-correlation is specified but ships **off**,
deferred to the port where out-of-sample error correlation can be re-measured.

!!! note "No A3 parity baseline"
    The wedge is a *separate* model, not a translation of the frozen PyMC reference,
    so it carries no A3 parity baseline. Its build gate is convergence and a sane
    posterior (R-hat, divergences, posterior-predictive on the clean window) — not
    accuracy, which is a port-side verdict.
