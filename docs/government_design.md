# Government wedge forecast — design decision record

**Status:** **Documentation / decision record — NOT the implementation spec.** Records *how
we landed on the design* and the design as approved (2026-06-19), as background and
rationale. The **normative implementation spec is a separate artifact** (project `specs/`
convention); this file exists to explain the *why*, not to be implemented from. The model
is built behind parity baselines per `specs/model_improvements.md` §8.

**Post-approval resolutions** (override the panel's earlier "resolved" stamps where noted):
the full government **store axis is NOT realized** (the heavy Option 3 was considered and
dropped); government `90/91/92/93` SA are acquired via a **lightweight published-SA
artifact** and used for **diagnostics + intervention-prior calibration only — not as
likelihood regressors**; the **single-block wedge** model is kept (a component-structured
*likelihood* is deferred to the port). See §8.4 / §9 for the reasoning.

**Date:** 2026-06-19
**Owner:** maintainer
**Companion specs:** `specs/model_improvements.md` (Track B is deferred there; this is
its scoping), `specs/bloomberg_consensus.md` (the Total consensus contract),
`specs/a5_real_competitors.md` (the eval harness this extends).

---

## 1. The problem

The existing nowcast models **private** NFP (CES `industry_code='05'`, SA, MoM net
change, thousands) and emits a full Bayesian posterior. The product we want to sell —
and the thing the **Bloomberg consensus** measures — is **Total** NFP (`'00'`). So the
nowcast and its yardstick are not the same object: scoring a private nowcast against the
total-nonfarm consensus is a target mismatch. `specs/model_improvements.md` already
named this gap and deferred its fix as **"Track B"**:

> Track B (deferred): Total NFP = **private nowcast + government forecast** → compared
> to the Total-NFP consensus. … The first Track-B task is to **scope it**.

This document is that scoping.

## 2. The accounting identity (verified)

The hierarchy is encoded in `nfp_lookups.industry` (`DOMAIN_DEFINITIONS`,
`get_supersector_components`):

```
00  Total nonfarm   (includes_govt = True)
05  Total private   (includes_govt = False)
90  Government       = 91 Federal + 92 State + 93 Local
00 = 05 + 90
```

So the user's premise holds exactly: **Private (`05`) + Government (`90`) = Total
(`00`)**. To turn a private nowcast into a Total nowcast we need the government piece.

## 3. The pivotal reframe — forecast the *wedge*, not published `90`

The naive plan ("build a model for supersector `90`") has a subtle, fatal flaw, surfaced
in advisor review and confirmed by BLS methodology research (§6, F-items): **BLS
seasonally adjusts `00`, `05`, and `90` independently**, so in SA space

```
SA(00) ≠ SA(05) + SA(90)        (a small but nonzero residual r = 00 − 05 − 90)
```

If we forecast published `90`, that residual `r` becomes an **irreducible error floor**
in Total that no amount of forecasting skill can remove. Instead we forecast the
**wedge**:

```
g  :=  published_00  −  published_05
Total  =  private_nowcast  +  wedge_forecast        ← reproduces published 00 BY CONSTRUCTION
```

The wedge is government *economically* (it carries education seasonality and federal
shocks), but it is defined as the exact `00 − 05` difference, so **coherence with the
published Total is free** — no seasonal-additivity leakage, and (see F10/F11) no
hierarchical reconciliation step is needed.

**User decision (2026-06-19):** target = the `00 − 05` wedge. Published `90` and components
`91/92/93` were initially intended as model *signals*; after working through the reasoning
(§8.4, §9) this was refined: they are **not** likelihood regressors (lagged government is
low-persistence) but **are** used for **diagnostics + intervention-prior calibration**
(their value is federal-shock attribution). The target stays leak-free; the components
sharpen the shock layer without entering the likelihood.

## 4. Empirical grounding — the wedge diagnostic

Both `00` and `05` first prints are already in the real-time store, so the wedge target
needs **no new acquisition**. We measured it directly (`first_print_changes('00')` −
`first_print_changes('05')`; throwaway script `scripts/_wedge_diag.py`):

**First-print wedge MoM change (COVID-excluded, n ≈ 82, 2017–2026):** std ≈ **24k**.

| Naive wedge forecast | MAE | RMSE |
|---|---|---|
| predict-zero | 24.1k | 31.4k |
| random-walk | 22.5k | 30.2k |
| calendar-month mean (seasonal) | 18.5k | 23.0k |

**Error budget.** Propagated into Total (private nowcast ≈ 45k RMSE, assuming
independence), a ~23k-RMSE wedge model lifts Total RMSE to ≈ **50.5k** — it adds ≈ **5k**.
The published consensus ceiling is ≈ 48k MAE / 60–65k RMSE. Two consequences drive
every later design choice:

1. **The wedge is secondary.** The consensus contest is won or lost mostly on the
   *private* nowcast (Track A). The wedge model should be **deliberately minimal** —
   complexity must be justified by measured Total-error reduction, not anticipated
   federal weirdness.
2. **The edge lives in the tails.** Consensus forecasters *also* eat government
   uncertainty (they predict Total). Normal months are near-naive-forecastable for
   everyone, so our only edge over the street's *implicit* government forecast is
   modeling **known shocks** (Census, shutdowns, RIFs) better than a seasonal mean.

**Caveats from the diagnostic.** (a) The full-sample wedge std is 132.9k, dominated by
COVID (April-2020 wedge = −980k) — COVID stays excluded from headline metrics. (b) A
"settled-level" wedge came out *noisier* (std 41k) than the first-print wedge; that is an
artifact of mixing annual-benchmark cohorts across ref-months, and is exactly why the
**real-time first-print** wedge — internally consistent and the thing we score against —
is the right target.

## 5. Decisions locked with the user

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Forecast target | `00 − 05` wedge | leak-free coherence with published Total (§3) |
| D2 | Government `90/91/92/93` | **diagnostics + intervention-prior calibration only**, *not* likelihood regressors | lagged govt is low-persistence (weak as a regressor, §8.4); their real value is federal-shock attribution, captured via calibration |
| D3 | Design scope | wedge model **+** Total assembly **+** scoring scaffold | close the loop end-to-end; the contest is the point |
| D4 | Method | **Bayesian** (NumPyro), posterior predictive | coherent convolution with the private posterior → honest Total posterior |
| D5 | Process | **research-first** | learn how this is actually forecast before committing structure |
| D6 | Consensus wiring | reuse pluggable `load_consensus()` (→ `—` until the Bloomberg file lands) | build-here / validate-on-port |
| D7 | Data acquisition | **lightweight published-SA artifact** (FRED/`CYCLICAL_INDICATORS` pattern); **NOT** the full government store axis | store-axis (Option 3) considered and dropped — buys real-time vintage reconstruction we don't use; gold-plating (§8.6) |
| D8 | Model structure | **single-block wedge** (not component-structured) | meets the ~5k budget cleanly; a component *likelihood* re-introduces the `90 ≠ wedge` residual `r` reconciliation (§8.4) |
| D9 | Component-structured likelihood | **deferred to the Bloomberg port**, only if federal-shock attribution earns its keep | thin in-window shock support locally (only 2025 RIF); verdict belongs to the port |

## 6. What the literature says (deep-research synthesis)

A fan-out deep-research pass (5 angles → 21 primary sources → 101 claims → **25 verified
3-0**, 0 refuted) returned findings that map almost one-to-one onto the model design.
All are primary BLS / Census / peer-reviewed sources.

- **F1 — BLS builds government SA bottom-up.** Federal/state/local are each SA'd
  separately then summed, because their seasonal patterns *diverge* and a direct
  aggregate adjustment would mask them. → a single shared seasonal block for the whole
  wedge is likely mis-specified; federal vs state/local seasonality differ.
- **F2 — Named government calendar regressors.** BLS layers deterministic dummies on
  X-13: a poll-worker dummy in **local** government (election-year November) and a
  **December USPS** dummy. → encode these named calendar effects explicitly.
- **F3 — Shocks are deterministic, not regimes.** BLS removes known federal events
  (Census, strikes) as *prior adjustments* (remove-before-SA, add-back), explicitly so
  the abnormal change is **not** learned as a recurring seasonal pattern. **Decisive: use
  deterministic interventions, not Markov regime-switching.**
- **F4 — The intervention menu.** X-13 / regARIMA gives the usable shock-shape
  vocabulary: **AO** (single-month spike), **LS** (permanent step), **TC** (spike with
  exponential decay), **ramp** (linear transition). Mapping: Census = TC; permanent
  RIF / hiring-freeze = LS or ramp; short back-paid shutdown = AO ≈ null.
- **F5 — Level shifts persist in the published SA series.** Interventions are removed
  before seasonal-factor estimation but **remain** in the published SA value — so a real
  RIF level shift *does* appear in the wedge we nowcast; the model must represent
  persistent level shifts in the target.
- **F6 — The frontier constraint (first-order for a release-eve nowcast).** A brand-new
  shock at the end of a series can *only* be an AO; TC/LS need later data. At release-eve
  the modeler must **impose the shock's shape from external priors (announcements)**, not
  infer it from frontier data.
- **F7 — Census is a known, published, huge intervention.** BLS publishes a dedicated
  NSA census-worker table (level + OTM change). Normal-cycle peak ≈ **+400k OTM**
  (May-2010 +411k); 2020 was re-timed to August (+238k) by COVID — so the deterministic
  profile's *timing* needs a real-time feed. Next decennial = 2030.
- **F8 — Back-paid shutdowns ≈ null in CES.** Furloughed workers paid for the pay period
  including the 12th count as **employed**; Jan-2019 federal was +1k. Permanent RIFs (no
  back pay) *are* real level shifts. (This project's store already carries a "no-print"
  hole at Oct-2025 from that shutdown's delayed release.)
- **F9 — The target is concurrent & revision-prone.** CES SA is concurrent; first-print
  SA is the correct competition target — consistent with the project framing.
- **F10 — Component-summation is the theoretically weaker branch…** Hendry & Hubrich
  (2011): adding disaggregate info *into* an aggregate model beats both pure
  component-summation and a univariate aggregate. Our separate-private + separate-wedge
  architecture is the weaker branch *in theory* — **but** modularity + estimation
  uncertainty defend it, and forecasting the wedge directly makes the adding-up **exact**
  (coherence free; no MinT needed).
- **F11 — Coherent uncertainty combination.** MinT reconciliation (closed-form GLS) for
  separate components, **or** — for a small Bayesian model — impose the adding-up
  constraint in a *joint* posterior. Since we forecast the wedge directly, the only open
  question is private/wedge **error correlation**: independent convolution (simple) vs a
  joint model (captures correlation).

**Research gaps (honest limits).** No published quantitative benchmark for
government-forecast accuracy exists (our naive numbers in §4 are the de-facto bar); no
proven monthly *leading indicator* for government hiring beyond the census schedule
(state/local **education seasonality** is the dominant signal; JOLTS-government-openings
and state-budget data are untested hypotheses — do not over-invest); and the **2025
federal RIFs** and **Oct-2025 shutdown** real-time CES treatment are undocumented —
handle by the back-pay analogy (permanent RIF = level shift, back-paid furlough = null).

**Primary sources (verified):**
BLS CES seasonal-adjustment technical notes (`bls.gov/web/empsit/cesseasadjtn.htm`); CES
State & Area SA (`bls.gov/sae/seasonal-adjustment/`); BLS 2022 large-revisions blog;
BLS CPI intervention-analysis (IASA); Census X-13ARIMA-SEATS reference manual; BLS CES
decennial-census-workers table; BLS 2019-shutdown Employment-Situation Q&A; Hendry &
Hubrich (2011, *JBES* 29(2):216–227); Wickramasuriya/Hyndman MinT
(`robjhyndman.com/papers/MinT.pdf`); Scott & Varian "Predicting the Present with BSTS".

## 7. From findings to a model — the design panel

With the target, scope, method, and research fixed, the remaining choices are model
*structure* (seasonality representation, intervention layer, signal use, assembly
coherence, acquisition). Rather than hand-pick one, we ran a **design judge-panel**:
four independent Bayesian wedge-model architectures under distinct lenses
(*minimalist/YAGNI*, *BLS-faithful/component-structured*, *integration/joint-posterior*,
*shock-first/robustness*), each scored by three adversarial judges on six dimensions
(budget-fit, Bayesian coherence, real-time honesty, data cost, convergence risk,
codebase fit), then synthesized.

**Outcome:** 4 designs proposed, 3 survived (≥2/3 keep). Ranked by mean judge score
(of 30):

| Design | Avg | Keeps |
|---|---|---|
| **Wedge-as-Bayesian-Seasonal-Mean** (change-space STS, minimalist) | **29.7** | 3/3 |
| Residual-coupled joint posterior (private⟂̸wedge mean-zero link) | 27.0 | 3/3 |
| Intervention-layered robust local-seasonal (fed/state-local split) | 22.7 | 3/3 |

The winner became the **spine**; the synthesis grafted the runners-up's best ideas as
*default-off* options and rejected their flawed centerpieces (the always-on coupling; the
non-identified seasonal split).

## 8. Recommended design

**Wedge-as-Bayesian-Seasonal-Mean with an announcement-priored intervention layer** —
a change-space structural time-series with a single seasonal block, assembled to Total by
independent convolution, with a residual-coupling escalation knob specified but off.

It is, deliberately, a **coherent-posterior generalization of the calendar-month-mean**
(the ~23k-RMSE floor it must not underperform). The entire complexity budget is spent on
the one place the ~5k error budget permits an edge: **deterministic, announcement-priored
interventions on known-shock months** — exactly what a seasonal mean cannot represent and
what the street's implicit government forecast eats blindly.

### 8.1 Model structure — change-space STS
A standalone NumPyro program (`nfp_model/wedge.py` or a sibling), importing only
jax/numpyro/numpy (honors the import boundary). It models the **first-print wedge MoM
change** `y_t` directly in **change-space**:

```
mu_t   = drift + season[month(t)] + Σ_k intervention_k(t)
y_t   ~ Normal(mu_t, sigma)        # StudentT(ν≈6) coded-ready but OFF (YAGNI)
```

- `drift ~ Normal(0, 30k)` — a single **constant** level for the change. **No RW/AR on
  the change-mean**: that is I(2) on the wedge *level* and is precisely why the
  random-walk baseline lost (30.2k vs the seasonal mean's 23.0k). Persistence is
  guilty-until-proven; an AR(1) residual is deferred.
- Non-centered throughout (house style). Likelihood under a **boolean mask** (the
  `handlers.mask` idiom) over COVID 2020–21 and the Oct-2025 no-print hole — *mask, never
  delete rows*, to keep drift/seasonal on a contiguous calendar axis.
- Because the target is built from `00` & `05` **first prints**, there is no
  vintage-indexed observation equation and no pooled-vs-first-print bias — the wedge
  **sidesteps the private model's first-print problem entirely**.

### 8.2 Seasonality — single shrunk monthly block
11 free monthly effects `season[1..11]` with the 12th pinned by **sum-to-zero** (kills
the drift/seasonal ridge), under hierarchical shrinkage `season[m] ~ Normal(0, τ)`,
`τ ~ HalfNormal(~15k)`. **Dummy effects, not Fourier** — the dominant signal is sharp
state/local **education** seasonality (Aug/Sep/Jun cliffs) a harmonic basis would smear.
The named F2 calendar effects (Nov poll-worker, Dec USPS) are absorbed into the relevant
monthly effect, kept stable by partial pooling.

**A federal/state-local seasonal split is rejected on identifiability grounds, not just
budget:** the likelihood observes only the *summed* wedge (`00−05`), so the two blocks
enter only via their sum — their difference is prior-dominated and **never identified**
from wedge-only data. F1 says the *true* seasonal is a sum of two shapes; it does not
license recovering them from the sum alone.

### 8.3 Interventions — the edge (and the only complexity sink)
Deterministic prior adjustments (F3: **not** regime-switching), in change-space. Shape is
**fixed by config; only magnitude is sampled**, under an **informative,
announcement-derived prior** (never flat — F6: a brand-new end-of-series shock can only be
*imposed*, not inferred from the 1–2 frontier months it would otherwise chase). The X-13
vocabulary maps into change-space (F4/F5 — note AO/LS/TC/ramp are *level*-space shapes):

| Event | Level shape | Change-space encoding |
|---|---|---|
| Permanent RIF (no back-pay) | level shift | single negative **pulse** (cumsum steps down & stays — F5 free) |
| Phased RIF | ramp | short negative **box** (−m/k over k months) |
| Census hire | temporary change | positive pulse + smaller decaying-reversal pulses |
| Back-paid shutdown | additive outlier | +spike then −giveback, nets ≈ 0 |

Concretely over the 2017–2026 ex-COVID window this is **~1 active mechanism**: (a) the
**2025 federal RIF** — the only live in-sample shock — as a negative pulse/box,
magnitude `~Normal(announced_headcount, tight_sd)`; (b) a **dormant Census slot** (built,
unfit; activates 2030 from the F7 NSA table); (c) the **Oct-2025 back-paid shutdown** →
AO ≈ null (F8), handled by the mask, no term. Operationalized as a frozen
`KNOWN_INTERVENTIONS` table (mirroring `lookups/benchmark_revisions.py`) plus a
`predict_wedge_change(..., interventions=[...])` operator interface for release-eve.
**Critical safety:** an as-of-censored backtest fixture + a lookahead-guard test must
assert the per-as-of table carries only *announcement-knowable* values — else the headline
edge is fake.

> **Spec-review correction:** the panel framed this table as "mirroring
> `benchmark_revisions`," but that is a flat `dict` with **no as-of axis**, which would make
> the lookahead guard vacuous. The normative spec (`specs/completed/government_wedge.md` §4.3) instead
> gives `KNOWN_INTERVENTIONS` an **`announcement_date` axis** + a
> `get_known_interventions_as_of(as_of)` helper, so the guard is a real *date* comparison.

### 8.4 Signals — components for diagnostics & calibration, not the likelihood
`90/91/92/93` SA are **not** likelihood regressors, but they **are** acquired and used —
the distinction matters and was worked out with the user (post-approval).

**Why not regressors.** The only government info available at release-eve is *lagged*
government (the components publish *with* the total, in the same release), and the
diagnostic shows the wedge change is near-white around its seasonal mean (random-walk
30.2k > seasonal-mean 23.0k; predict-zero ≈ random-walk). So a lagged-`90` regressor is a
weak persistence term, the seasonal block already carries the dominant (education) signal,
and adding regressors spends degrees of freedom against a ~5k budget for little payoff.
No proven external leading indicator exists either (JOLTS-government-openings, state
budgets are untested — do not over-invest).

**Why acquire them anyway — federal-shock attribution.** The components' real value is not
persistence but **identifiability and attribution**: the 2025 RIF and 2030 Census live
entirely in **federal (`91`)**, and seeing `91` separately lets us *size and validate* the
intervention priors against the federal series cleanly (e.g., check how a past RIF landed
in observed `91` to set an honest prior SD), instead of against a wedge where state/local
education noise drowns them. This is **diagnostic + prior-calibration** use, honoring F6
(at release-eve a *new* shock is still imposed from announcements, never inferred from
frontier `91`).

**Why not a component-structured *likelihood* (D8).** Modeling `91+92+93 → 90` and then
reconciling to the wedge re-introduces the SA-additivity residual `r = wedge − 90` we
targeted the wedge to escape, plus extra NUTS surface — for a payoff concentrated in shock
months that are barely testable in-window (only the 2025 RIF). So the component likelihood
is **deferred to the port** (D9); the single-block wedge stays the v1 model. The only feed
the *model* itself consumes is the F7 Census table (dormant slot, intervention
timing/magnitude).

### 8.5 Assembly — independent convolution (default)
`Total_change_draws = private_change_draws + wedge_change_draws`, summed elementwise per
draw. Adding-up is exact because we forecast the wedge directly (no MinT). **The seam that
bites (verified in code):** the persisted private `nowcast_pred_draws` are in
**growth/index** space and must pass through `scoreboard.change_draws_k(prev_index,
idx_to_level)` to become change-k *before* adding the natively-change-k wedge draws; a bad
anchor reintroduces the `base_index` NaN class the project already hit. Make this an
explicit, **tested** `assemble_total` helper in the harness, not inside `nfp-model`.

Independence is the one live bet — defensible: private = business-cycle object, wedge =
education/policy object; measured *outcome*-ρ is only +0.19 (the *error*-ρ is almost
certainly lower), and at ρ=0.3 Total RMSE moves ~3k inside a 48–65k ceiling. A
**residual-coupling knob** (add `η·z_t`, the standardized *mean-zero* private residual, to
the wedge mean) is specified but **default OFF**: it leaves the Total *point* forecast
invariant and only widens intervals — pure interval honesty, deferred to the port where
out-of-sample error-ρ can be re-measured.

**Scoring scaffold (scope item 3):** extend `scripts/run_a5_backtest.py` +
`nfp_vintages.scoreboard` to score assembled Total change against (i) the Total `00` first
print and (ii) the consensus, reusing the existing `change_draws_k` / `crps_sample` /
`interval_coverage` machinery. Consensus stays pluggable/None-tolerant via the existing
`load_consensus()` / `Consensus` class (verified T-1-only).

### 8.6 Data acquisition — hybrid (resolved D7)
**Target** from the store (no new acquisition): `first_print_changes('00') −
first_print_changes('05')`, both legs from the *same* first-print release vintage.
**Signals** (`90/91/92/93` SA + the F7 Census table) as a lightweight published-SA artifact
under `data/` following the existing `CYCLICAL_INDICATORS` / `indicators.py` FRED pattern
(`fred_id` + `pub_lag`, read via `read_indicator`). Used for **diagnostics + intervention-
prior calibration** (§8.4), *not* as likelihood regressors; the only feed the model itself
consumes is the dormant Census slot. **Reject** extending the reserved
`ownership='government'` store axis (the user's Option 3, then dropped): it buys full
real-time vintage reconstruction we don't use (signals are lagged, barely consumed,
tiny-revision) — store-rebuild-scale gold-plating. That axis becomes the right home only if
a future phase promotes `90–93` to *modeled* real-time likelihood inputs (D9, port).

### 8.7 Phasing — build here / validate on port
**v1 (build locally):** the Design-1 core (constant drift + 11 shrunk monthly effects +
masked iid-Normal likelihood); one live intervention slot (2025 RIF) + a dormant census
slot; the `KNOWN_INTERVENTIONS` table + operator interface **with** an as-of-censored
fixture and lookahead guard; hybrid acquisition; independent convolution with the explicit
`change_draws_k` conversion; the Total-vs-consensus scaffold (None-tolerant, exercised by a
synthetic fixture). **Build gate = convergence + sane posterior** (R-hat, divergences,
posterior-predictive on the clean window), **not** accuracy. Because `wedge.py` is a
*separate* model it needs **no new A3 parity baseline** on the private model.
**Defer to port:** the accuracy/keep-drop verdict vs consensus; the residual-coupling
knob; StudentT-by-default; any seasonal split (needs component observations). **Defer to
2030:** census activation.

### 8.8 Alternatives (kept on the shelf)
1. **Always-on residual coupling** (Design 2) — turn the §8.5 knob on. *Prefer when* the
   port shows assembled-Total intervals miscalibrated under independence **and**
   out-of-sample error-ρ is materially positive. *Tradeoff:* all three judges flagged it
   as over-promoted — if error-ρ≈0 it earns ~0k and couples geometry for nothing; correct
   to ship off, wrong as a centerpiece.
2. **Two-block federal/state-local seasonal** (Design 3) — *prefer only if* component-level
   `91/92/93` observations are acquired and entered as separate observed series (then the
   split becomes identified). *Tradeoff:* with wedge-only observations it adds ~12
   prior-dominated dimensions and worsens NUTS geometry for zero forecasting gain.

## 9. Decisions resolved at approval, and the one input still needed

**Resolved (2026-06-19, with the user):**
- **Role of `90/91/92/93` (was open fork #1).** Not likelihood regressors / not a
  drift-anchor — **diagnostics + intervention-prior calibration** (D2/D8, §8.4). The
  reasoning was the deciding exchange: lagged government is low-persistence (weak as a
  regressor), but observing the components is valuable for **federal-shock attribution**;
  the panel had under-weighted that because its winner assumed wedge-only data.
- **Acquisition (was open fork D7).** Lightweight published-SA artifact, **not** the
  government store axis (Option 3 dropped). See §8.6.
- **Model structure (D8/D9).** Single-block wedge for v1; component-structured likelihood
  deferred to the port.

**Still needed from the human (an input, not an architecture choice):**

1. **The 2025 federal RIF intervention priors** (shape / magnitude / timing) — the design's
   headline edge. These are **announcement-derived human inputs** that cannot be pinned
   from data without violating F6. *Recommendation lean:* permanent **level-shift** (no
   back-pay ⇒ real per F8), magnitude centered on the announced permanent-separation count
   with an honest sd, timing from the effective-date announcement censored to each
   release-eve. You supply the actual numbers into `KNOWN_INTERVENTIONS`.

### Key risks (carry into implementation)
- **Assembly-seam units mismatch** (highest priority) — private draws are growth/index;
  must convert via `change_draws_k` before summing, or hit the `base_index` NaN class
  again. *Mitigation:* explicit tested `assemble_total`.
- **Intervention-table lookahead** — the edge is fake if the as-of table ever encodes
  realized post-hoc shock sizes. *Mitigation:* as-of-censored fixture + guard test.
- **Independence assumption** — mildly overconfident on Total intervals if error-ρ>0.
  *Mitigation:* ship independent; port-side coverage triggers the coupling knob.
- **Seasonal-vs-intervention confound at n≈82** — a shock in month *m* trades off against
  month *m*'s seasonal effect. *Mitigation:* shrinkage + informative announcement prior.
- **`τ_season` funnel** — ~7 obs/month vs large Apr/May variance. *Mitigation:*
  non-centered, HalfNormal `τ`, sum-to-zero; watch R-hat.
- **Edge is only partially validatable locally** — the *baseline* is (public data), but
  the *intervention edge* has thin in-window support (2020 census COVID-excluded, 2018-19
  shutdown null, 2030 future, leaving only the 2025 RIF months); the keep/drop verdict
  genuinely needs the port.

## 10. How this record was produced (provenance)

- **Code/identity verification** — `nfp_lookups.industry`, `nfp_ingest.first_print`,
  `nfp_ingest.vintage_store`; specs `model_improvements.md`, `bloomberg_consensus.md`,
  `store_rebuild.md` §11.
- **Advisor review** — surfaced the wedge-vs-`90` SA-additivity reframe and the
  error-budget framing before any design was committed.
- **Empirical diagnostic** — `scripts/_wedge_diag.py` (throwaway) over the real store.
- **Deep research** — fan-out web research workflow (run id `wf_80672e35-ba1`); 25
  verified claims.
- **Design panel** — adversarial judge-panel workflow (run id `wf_2b280bbd-fa2`).
