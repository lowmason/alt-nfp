# Model improvements — private first-print targeting, diagnostics, and turning-point edge (design)

Status: **design, revised 2026-06-19; amended 2026-06-22 (§12 — consensus on both tracks)**. The **model-side counterpart** to `specs/completed/a5_real_competitors.md` (evaluation-side firewall — no `nfp-model` changes). Motivated by `specs/model_research.md` (consolidated literature review) and the **validate-first pivot** in `specs/plans/0-port_and_staged_plan.md` (parity is a port-fidelity floor, **not** correctness). This spec reopens Phase A's "parity-is-done" freeze — behind new baselines — to fix two things A5's prong-2 surfaced: the model is not modeling the **object** it should (it nowcasts **private** NFP, not total), nor the **vintage** it is scored on (first print).

> **Amendment 2026-06-22 — consensus is available on BOTH tracks (supersedes TL;DR #3 / §3).**
> The Bloomberg consensus file is now on MinIO (`s3://alt-nfp/competitors/consensus.parquet`) and
> carries **both** Total (`'00'`, ownership total) **and private** (`'05'`, ownership private) survey
> medians/means — **173 months each, 2012-01 → 2026-05**, on the model's exact object (total private,
> SA, first-print headline change); the adapter already keys on `industry_code` (`consensus.py`). So the
> premise that "consensus is a Total object, meaningless against the private nowcast" is **false**. The
> private nowcast gains its first real competitor; the **government wedge gains its first competitor ever**
> (`Total − Private` = an *implied-government* consensus); and a model–consensus **combination** is now on
> the table behind a falsifiable gate. The two-track A/B framing is **re-priced, not removed** — the wedge
> stays (load-bearing for the strong-Total headline). Full design, gate, and scope in **§12**.

## TL;DR

1.  **The model nowcasts PRIVATE NFP — target the private first print.** The model's signal is inherently private: it is anchored to QCEW (private in this store) and its richest inputs are **private payroll providers**. It cannot see government employment. So its object is **CES total private** (`industry_code='05'`), and it must be scored against the **private** first print and **private** QCEW truth. The run-path currently defaults to `'00'` (total nonfarm) — a **latent mismatch** being corrected here, *not* the intended target. (The `'00'` total data exists for the **Track B** "private nowcast + government forecast = total" assembly — whose government forecast, the `00 − 05` wedge, is now **built**; §2 below.)
2.  **Second, even on private it targets the wrong vintage.** The latent passes through a single, vintage-*pooled* CES observation equation, so the nowcast predicts a *revised/third-print*-ish value — then it is scored against the **first** print. The headline model fix (§5) makes it a genuine **first-print** predictor.
3.  **Two evaluation tracks.** **Track A (now):** the private nowcast vs **naive floors only**, scored on the private first print + the private QCEW-settled truth (the **primary** truth comparison), decomposed by regime, judged on **calibration**. **Track B (built locally 2026-06-19; validate-on-port):** Total NFP = **private nowcast + government forecast** → compared to the Total-NFP **consensus** + Total first print. Consensus is a **Total** object — it has no meaning against the private nowcast alone. **[Superseded 2026-06-22 — §12: the consensus file also carries the private `'05'` series, so the private nowcast now has its own consensus competitor and the two-track framing is re-priced, not removed.]** The government forecast (the **wedge** `00 − 05`), the Total assembly, and the consensus scoring are now **designed + built + tested** (`specs/completed/government_wedge.md`, `specs/plans/completed/14-government_wedge.md`); only the **accuracy/keep-drop verdict vs consensus** is deferred to the port — exactly Track A's build-here/validate-on-port posture (TL;DR #4).
4.  **Build here, validate on the port.** This compute (Opus 4.8, no limits) is for *building*; the local eval is a providerless **skeleton** that can confirm code **runs and converges** but cannot **validate accuracy**. So build the full stack here (Tiers 1–3, each behind its parity baseline), confirm correctness + convergence locally, and defer the **accuracy verdict, tuning, and keep/drop** to the Bloomberg port. Tier 1 diagnostics are **instrumentation** (the Aruoba intercept feeds §5A) — **not** a build-blocker.
5.  **Out of scope / deferred:** a benchmark / Early-Benchmark revision layer; a full first-release-vintage store rebuild (unless §4 justifies); supersector logic (Phase B). **ADP is out entirely** — not a competitor, not a regressor. (The **government forecast** — formerly listed here as Track B's undesigned critical path — is now **built**: the `00 − 05` wedge, `specs/completed/government_wedge.md` / `specs/plans/completed/14`. What remains deferred to the port is its **accuracy verdict vs consensus**, the port-only refinements, and two input gates — §2.) Model changes land behind **new parity baselines** (§8).

## 1. Motivation — the model must target the private first print

**The object is private.** The model's latent `g_total_sa` is pinned to QCEW by the Student-t anchor (`model.py:187-193`), and its differentiating inputs are **private payroll-provider microdata**. Both are *private*: this store's QCEW national series is total-private (`industry_code='05'`; see `store-industry-layout`), and payroll providers by construction cannot observe government employment. So the model nowcasts **private** NFP. Scoring it against **total nonfarm** (which folds in government) is a target mismatch; comparing it to the **total-nonfarm consensus** is doubly mismatched. The fair truth for the model is the **private** first print and the **private** QCEW-settled value. The run-path's `industry_code='00'` default (`first_print_changes`, `panel_to_model_data`, `nfp_vintages.a5`) is the mismatch this spec corrects — every CES/QCEW series the model trains on, predicts, and is scored against moves to `'05'`.

**The vintage is also wrong.** Even on the private object, the nowcast applies a **vintage-pooled** CES observation equation `alpha_ces + lambda_ces·g_total_sa` whose bias/loading are shared across all vintages (`model.py:198-202`; only `sigma_ces_sa` is vintage-indexed); the nowcast applies those pooled parameters directly (`nowcast.py:21-28`). Because the as-of training diagonal is dominated by third-print rows, the pooled `alpha_ces` encodes a *revised-print* bias — so the nowcast predicts a generic/revised print and, scored against the first print, carries a systematic offset. §5 is the fix.

**Two edges — of the private nowcast.** Against a *public-information* private predictor, this model's value proposition is two edges: (1) **private payroll-provider microdata** — richer and earlier than any public print — which should lift *normal-month* private-nowcast accuracy; and (2) **explicit turning-point birth/death modeling**, which should win on *direction at cyclical turns* (where Klein 2022 shows public forecasters under-react). Both are **untested locally**: the rebuilt public store has no provider data (Bloomberg-only), so every local run scores a providerless *skeleton* that understates the model. These edges describe the **private nowcast's quality**; whether the *Total* product beats *consensus* is a **Track B** question (it depends on the private nowcast *and* the government forecast) — Track B is now **built** (§2), with the accuracy verdict vs consensus taken on the port.

## 2. Scope & non-goals

**In scope (Track A — the private nowcast).** (1) **Evaluation-side** — a regime-decomposed, calibration-aware scoreboard (§3) and revision/efficiency diagnostics (§4) extending the A5 harness, all on the **private** target, with **naive floors** as the only competitors. (2) **Model-side, behind new parity baselines** — a first-print observation layer (§5) and a turning-point-aware birth/death extension (§6), making the model a good *private first-print* predictor. The §4 diagnostics **gate** which model layers get built.

**Track B — the Total NFP product (built locally 2026-06-19; accuracy validated on the port).** The product that competes with consensus is **Total NFP = private nowcast + government forecast**. The **government forecast is now built** — the `00 − 05` **wedge**, a thin Bayesian change-space STS scored vs the Total `00` first print and vs consensus, designed in `specs/completed/government_wedge.md` and implemented per `specs/plans/completed/14-government_wedge.md` (`nfp_model/wedge.py`, `nfp_vintages/assembly.py:assemble_total`, `competitors/consensus.py:load_consensus`, `score_total`, the `cmd_total` backtest — committed + unit-tested). Track B has thus joined Track A's **build-here / validate-on-port** posture: the machinery runs and converges locally, but the **accuracy/keep-drop verdict vs consensus** is deferred to the Bloomberg port (the intervention *edge* is only thinly testable locally; the consensus *file* is Bloomberg-only). **Two open input gates remain** (not engineering): (1) the **2025 federal RIF intervention magnitudes** — a maintainer human-input gate; v1 ships placeholder priors (`government_wedge.md` §8); (2) the **Bloomberg consensus file** — absent locally, so the consensus column renders `—` until it lands (`specs/bloomberg_consensus.md`).

**Evaluation venue — a first-class constraint.** The model has two information regimes: **full** (Bloomberg compute, payroll-provider microdata present) and **public-only** (local, rebuilt public store, providers absent). The provider edge lives in the *full* regime; a beat-a-public-baseline claim from providers is valid only there. Locally we can develop, unit-test, and validate the one provider-independent improvement — the **first-print fairness fix** (§5A) — and the entire evaluation harness. The provider-ablation and the full capability readout are **forward-looking** to the Bloomberg run.

**Non-goals.**

-   **The government forecast is built** (the `00 − 05` wedge — `specs/completed/government_wedge.md` / `specs/plans/completed/14`), so Track B is no longer blocked on it; only its accuracy verdict vs consensus is deferred to the port (see above).
-   **No ADP** — removed entirely as competitor and regressor. The private nowcast is judged against the private first print + private QCEW truth + naive floors; the Total contest (Track B) is against consensus, not ADP.
-   **No benchmark / Early-Benchmark revision layer** — a different product (annual-benchmark accuracy).
-   **No full first-release-vintage store rebuild** — unless §4's Aruoba R² shows materially predictable monthly revisions. Absent that, the as-of diagonal stands (KDP hygiene only, §7).
-   **No removal of the QCEW truth anchor** — the latent stays QCEW-anchored (private QCEW); §5 *adds* a first-print observation layer on top.
-   **No supersector logic** (Phase B / B1) — national private total only.

**Relationship to existing design & parity governance.** This is the **model-side counterpart** to the evaluation-firewall `a5_real_competitors.md`: it opens `nfp-model` to change under the validate-first pivot (`specs/plans/0`; parity ≠ correctness). Every model-side change lands **behind a new A3-style parity baseline** (§8). NB: the private retarget itself (feeding the model `'05'` instead of `'00'` arrays) is a **data-layer** change — the model has only ever been validated on `'00'`, so a `'05'` fit must be confirmed to converge (no divergences, sane posterior) before any `'05'` eval is trusted (§11).

## 3. Tier 0 — Private scoreboard correctness *(evaluation-side; extends the A5 harness, no model code)*

Score the **private nowcast** against the **private** first print, by `competitor × horizon-regime × month-type`. **Competitors: naive floors only** (random-walk, trailing-mean). **No consensus** (a Total object → Track B); **no ADP**. **[Superseded 2026-06-22 — §12: the private consensus (`'05'`) is now available and IS a competitor on this private board; see §12.1/§12.3. ADP stays out.]** The **private QCEW-settled scoreboard** (§Task-5 in the plan) is the **primary truth** comparison — the model is QCEW-anchored, so the QCEW-settled private value is the closest administrative truth it can be held to. Tier 0 adds:

1.  **Month-type decomposition.** Split every metric by **normal / large-revision / turning-point / benchmark-window**, alongside the T−7/T−1 horizon split. Definitions: *large-revision* = |first→third **private** revision| above a fixed percentile of the historical distribution; *turning-point* = a cyclical-state flag (claims-momentum / direction-change); *benchmark-window* = the Feb-release months most affected by annual-benchmark + seasonal-factor updates; *normal* = the complement. Pooled MAE hides where edge lives — the **provider edge in normal months, the BD edge at turns**.
2.  **Calibration metrics.** Interval coverage (80/90% hit-rate) and CRPS beside the point metrics. In normal months — and in every providerless local run — the Bayesian model's value is *honest uncertainty*, not a point-MAE edge (ties to §10).
3.  **Venue tag.** Every scored row records whether providers were in the information set (**full** vs **public-only**, §2).

COVID (2020–21) stays excluded from headline metrics; shutdown-frontier months (e.g., 2026-01) are flagged, not silently pooled. Home: `scripts/run_a5_backtest.py` scoring + `a5_report.md`; reads the model reduction and `first_print_changes(industry_code='05')`; touches no pinned path.

## 4. Tier 1 — Diagnostics (instrumentation, not a gate) *(evaluation-side; on the store; no model change)*

**Instrument the forecastable share; let the port deliver the verdict.** Three diagnostics, all on the **private** series. Locally they confirm the store behaves like published US-payroll evidence (small α, low normal-month R²) and the Aruoba intercept feeds §5A; on the **port**, their numbers inform tuning and keep/drop. They do **not** block building Tiers 2–3 here (TL;DR #4).

1.  **Aruoba revision regression** — `(later_private_vintage − private_first_print) = α + γ'·X + u`, the LHS being the **first-to-third private** revision (`industry_code='05'`, excluding annual benchmark), with `X = {claims, jolts, biz_apps, nfci, lagged revisions, cyclical-state}` (all **public** indicators — kept; **no ADP**), run pooled and by month-type. The **intercept α is the private first-print bias** (consumed by §5's offset); the **R² is the forecastable share.** *On the port this informs tuning:* normal-month R² near zero suggests the as-of diagonal is adequate (a first-release-vintage rebuild, §7, would add little); R² concentrated in turning-point / benchmark regressors confirms where §6's BD layer earns its keep. Locally it is a correctness check, not a build decision.
2.  **Mincer–Zarnowitz efficiency regression** — `actual = α + β·forecast`, testing α=0, β=1, on the **model's private nowcast** (a self-check for left-on-the-table information). *Consensus MZ moves to Track B* — consensus forecasts the Total number, so an MZ on consensus belongs with the Total assembly, not the private track.
3.  **Provider-ablation** *(full regime; Bloomberg-only, forward-looking)* — private-nowcast-with-providers vs without on identical dates, by month-type. The **direct test of the §1 provider hypothesis**; runs only where providers exist.

Home: a diagnostics module (`nfp_vintages.diagnostics`) reading the store + `first_print_changes('05')`; unit-tested against the literature priors (small α, low normal-month R²) as a check that our store behaves like published US-payroll evidence. Tier 1 is **built alongside** Tiers 2–3 (it needs no model change); its numbers inform tuning **on the port** — they do not decide what gets built **here**.

## 5. Tier 2 — A first-print observation equation *(the headline model fix; staged)*

The fix for the §1 vintage mismatch (on the private target). Today the nowcast predicts `alpha_ces + lambda_ces·g_total_sa` with pooled, vintage-shared parameters (`nowcast.py:21-28`, `model.py:198-202`); dominated by third-print training rows, that predicts a *revised* private print. The revision is ≈ a mean shift, i.e. an **α effect** — the lever is the intercept, not the loading.

**5A — Post-hoc first-print offset** *(cheap first cut; no parity break; locally testable).* **BUILT 2026-06-20 (harness-side).** Subtract the measured private first-print bias `δ` from the nowcast at score time. *Implementation refinement vs the original sketch* (recorded so the doc tracks the code): the scored nowcast comes from the JAX batch reducer (`batch.py:_reduce_one`), **not** `nowcast.py`, and the Aruoba revision LHS is natively in **k-jobs** — so δ is applied **post-hoc in jobs space** in the harness (`run_a5_backtest.py:cmd_score`) as a new **`model_5a`** competitor row alongside `model`: point `model − δ`, with the predictive draws shifted by δ so coverage/CRPS move too. For a constant offset this equals the original "growth space before the index arithmetic" to O(δ²)≈1e-8, and it touches **zero `nfp-model` code → no A3 baseline break** (strictly cleaner than editing `nowcast.py`). δ is the **pooled** bias — the robust **median** of `revision_k` (≈ −8k locally; `pooled_first_print_bias` in `diagnostics.py`), median (not the OLS Aruoba intercept) to resist the real benchmark/COVID outlier months (+871k 2022-11, −714k 2024-11). **Provider-independent** → the one improvement validatable on the local public-only skeleton; the local ME-shift is a **sign-and-wiring check** (pooled in-sample δ), not an accuracy verdict. Limitations / refinements deferred to the port: a **month-type-specific** δ (the harness already classifies month types) and an **as-of-estimated** δ.

**5B — Vintage-indexed observation equation** *(principled; new baseline).* Index `alpha_ces` (and optionally `lambda_ces`) by CES vintage in `model.py` — today only `sigma_ces_sa` is vintage-indexed (`model.py:204-211`) — and have the nowcast select the **first-print vintage's** parameters. This lets the model *learn* the private first-print bias and its uncertainty in the likelihood rather than import a point estimate. It **breaks A3 parity → new baseline** (§8). *Identifiability caveat:* the as-of diagonal shows one print per month, so first-print rows concentrate at the frontier; cleanly identifying a first-print α may need a first-release-vintage CES target history (§7) — §4's Aruoba structure tells us whether the signal is there.

**Staging.** Ship 5A first — it is the fairness fix, costs no baseline, and is locally validatable. Escalate to 5B only if 5A's offset proves insufficient *or* §4 shows vintage structure worth learning.

## 6. Tier 3 — Turning-point-aware birth/death *(the edge layer; gated, dirty-month-validated)*

§1's second edge. Today BD is **linear and symmetric** — `bd_t = phi_0 + Σ phi_3[i]·X[i] + sigma_bd·xi` (`model.py:157-172`), default covariates just `("claims","jolts")` (`config.py:88`) — and the latent has only a coarse calendar-fixed `n_eras = 2` mean. The documented failure is **asymmetric**: the net birth/death model cannot see business deaths in real time, so it overstates jobs *entering* downturns — exactly what a symmetric linear `phi_3` cannot capture. (This is *inherently a private-sector* phenomenon — firm births/deaths are private — which is part of why the model's object is private.)

Three enrichments, **cheapest-first**:

1.  **Wire `biz_apps` into the BD covariate set** — the business-applications birth proxy is configured in the data layer but absent from `indicator_names`. Lowest effort.
2.  **Asymmetric / hinge claims loading** — let claims load more strongly when rising / above a threshold (state-dependent `phi_3`), capturing the death surge entering downturns.
3.  **Cyclical-state regime on the BD intercept** — a third "downturn" regime or a Markov-switching `phi_0`, distinct from the calendar eras.

**Gated** on §4's turning-point R²; each enrichment lands **behind a new baseline** (§8); build cheapest-first, stop when turning-point error stops improving. **Validated on dirty months** (2008-09, 2020, the 2024-25 large-revision episodes), **not** the clean window. Heaviest lift, most contingent, only fully testable on the Bloomberg full regime — **last in sequence**.

## 7. Vintage discipline (KDP hygiene)

Koenig–Dolmas–Piger (2003) Strategy 1 — first-release on the LHS target, real-time vintage on the RHS — is *already* approximated by the as-of diagonal. The hygiene task is narrow: ensure benchmark re-anchoring does not contaminate the historical **private** first-print training *levels*. A *full* first-release-vintage CES reconstruction is gated on §4's R² (and is the same data that resolves §5B's identifiability). Necessary maintenance — not the primary lever.

## 8. Parity governance

Tier 0/1 and §5A touch no `model.py` → no baseline. The **private retarget** changes only the *data* fed to the model (`'05'` arrays), not `model.py` — but because the model was validated on `'00'`, treat the first `'05'` fit as a **fresh validation** (confirm convergence + sane posterior; §11), and **pin a `'05'` reference fit** as the baseline for subsequent `'05'` model changes. Every model-side change (§5B, §6.1–6.3) lands **behind its own new A3-style baseline**: regenerate the golden, pin the new posterior, and **record the divergence from the frozen reference as intentional**. Never relax the existing baseline; the reference stays the port target, not an oracle (`specs/plans/0`). Each cutover follows the existing A3 machinery (`nfp_model.parity`, `scripts/run_a3_parity.py`, golden fixtures under `s3://alt-nfp/golden/a3`). Record divergences here and in `specs/plans/0`'s gate log.

## 9. Sequencing

**Track A — build here, in order; each step confirmed to *run + converge*, not validated for accuracy:**
1.  **Retarget to private `'05'`** — thread `industry_code` through the snapshot/model target, `first_print`, the a5 index, and the Tier 1 LHS; confirm a `'05'` fit converges (§8, §11). *(Done 2026-06-19 except the convergence fit — the next session's first step.)*
2.  **Tier 0 + Tier 1** (private scoreboard + diagnostics instrumentation) — no model change.
3.  **§5A offset** — the fairness fix (provider-independent), confirmed to shift ME sanely.
4.  **§5B** (vintage-indexed α) and **§6** (turning-point BD enrichments) — **built now**, each behind its own new parity baseline (§8), confirmed to sample cleanly. **Not** gated on the local Tier 1 numbers (build-here/validate-on-port, TL;DR #4).
5.  **Defer to the Bloomberg port:** the accuracy verdict, layer tuning, keep/drop, provider-ablation, and the private-nowcast capability test — the full regime is the only place these are *valid*.

**Track B (government forecast built 2026-06-19; accuracy on the port):** the **government forecast** is designed + built (the `00 − 05` wedge — `specs/completed/government_wedge.md` / `specs/plans/completed/14`); **Total = private nowcast + government forecast** is assembled (`assemble_total`) and scored against the **Total-NFP consensus** + Total first print (`score_total` / `cmd_total`). This is the only valid consensus contest, and it now runs locally. Remaining: feed the **2025 RIF intervention magnitudes** (maintainer input) + the **Bloomberg consensus file**, then take the **accuracy verdict vs consensus** on the port.

## 10. Reality check

The published consensus ceiling (~48k MAE / 60–65k RMSE; sampling SE ~67.5k) is a **Total-NFP** figure — it binds **Track B**, not the private nowcast directly. The private nowcast is judged on **private** truth (first print + QCEW-settled) and **calibration**, not against consensus. A providerless local clean-window readout landing at ~parity with the naive/public baselines is **expected, not a failure** — it is the skeleton, not the experiment; the real edges (providers in normal months, BD at turns) are forward-looking to the full regime. Jan-2026 (dirty, shutdown-frontier, providerless) is triple-disqualified as a capability signal. Judge clean-window normal months on **calibration**.

## 11. Open items & risks

-   **Government forecast — RESOLVED 2026-06-19 (built).** Track B's critical path is closed: the `00 − 05` **wedge** (slow-moving, seasonal, a small standalone change-space STS, exactly as anticipated) is designed (`specs/completed/government_wedge.md`), planned (`specs/plans/completed/14`), and committed + unit-tested. The remaining Track-B open items are the **2025 RIF intervention magnitudes** (a maintainer human-input gate, not engineering — `government_wedge.md` §8), the **Bloomberg consensus file** (Bloomberg-only; column `—` locally), and the **accuracy verdict vs consensus** (port). The wedge's own residual risks (assembly-seam units, intervention lookahead, independence on Total intervals) are tracked in `government_wedge.md` §12.
-   **`'05'` is untested in the model.** A1–A3 goldens and A4 were all on `'00'`. Before trusting any `'05'` eval: confirm `first_print_changes('05')` yields a sane private series (✓ verified 2026-06-19: 108 months, 2017→2026) **and** that one `'05'` model fit converges (divergences, R-hat, posterior sanity). Pin a `'05'` reference fit (§8).
-   **Aruoba private LHS definition.** The first-to-third *private* revision must exclude annual benchmark wedges and avoid month gaps (a naive "exclude benchmark rows + shift" corrupts the series). Get the cohort matching right before trusting the intercept (§5A input) or the gate.
-   **§5B identifiability.** Cleanly identifying a first-print α may force the first-release-vintage CES rebuild (§7); §4's Aruoba structure decides. If weak, 5A's offset is the durable answer.
-   **`biz_apps` wiring.** Verify it threads through `panel_adapter` / `model_inputs` into the model path before §6.1 — data-layer config (`CYCLICAL_INDICATORS`), not a model default (`indicator_names`).
-   **Tuning against a skeleton.** The provider edge cannot be tested locally; defer edge-layer tuning to the full regime.
-   **Dirty-month validation set.** COVID (2020–21) and shutdown exclusions overlap the very turning-point months §6 needs; curate the Tier-3 validation set deliberately.

## 12. Consensus on both tracks — benchmarks + the combination gate *(2026-06-22 amendment)*

The premise behind TL;DR #3 and §3 — "consensus is a Total object, meaningless against the private
nowcast" — is **false as of 2026-06-22**. `s3://alt-nfp/competitors/consensus.parquet` carries both the
Total (`'00'`) and private (`'05'`, `ownership='private'`) survey median/mean, **173 months each,
2012-01 → 2026-05**, on the model's exact object (total private, SA, first-print headline change in
k-jobs). `consensus.py` already keys on `industry_code`, so consuming the private series needs **no new
adapter**. This unblocks three things the original two-track design could not express. **Deliverable
context (maintainer, 2026-06-22): the product is *both* Total and private, with strong emphasis on
Total** — so the headline is the additive assembly, and its quality is gated by its weakest link.

### 12.1 The fully-benchmarked additive stack

Each layer of `Total = private nowcast + government wedge` now has a professional competitor:

```
  private nowcast    → vs  Private consensus ('05')     (NEW; replaces "naive floors only" on §3's board)
+ government wedge    → vs  Total − Private             (NEW; the wedge's first competitor ever)
─────────────────
= Total (headline)   → vs  Total consensus ('00')       (existing Track B; now DECOMPOSABLE)
```

The payoff is the decomposition: a Total-headline miss splits into a **private-side** miss and a
**government-side** miss, so you can see *where* the headline is lost — which the single Total contest
(§9 Track B) could not show.

### 12.2 The implied-government benchmark — the underrated prize

Under strong-Total emphasis the headline is gated by the **government wedge**, the one place the model
has no signal and whose priors are placeholder 2025-RIF magnitudes (`government_wedge.md` §8, a *blocking
maintainer input-gate*). `Total − Private` consensus is the street's monthly government-employment
expectation, and it does two things the wedge never had: (a) gives the wedge an **external accuracy
check** for the first time (its verdict was deferred entirely — §11), and (b) is a candidate
**prior/anchor that may obviate the RIF input-gate**. Validate-first (consistent with the gate below):
score the wedge against implied-govt first; fold implied-govt into the wedge's priors only if it beats
the RIF-priored wedge. **Implication: strong-Total should redirect real attention onto the wedge** — the
part of the system this data finally makes improvable instead of a leap of faith.

### 12.3 The combination gate — the go/no-go for touching the model

**Decision (maintainer, 2026-06-22):** the private consensus is a **benchmark first; a model-combination
partner only if diagnostics justify** — and "justify" is a **regime- and horizon-decomposed optimal
combination**, not a vibe:

- For each **month-type × horizon** cell (month-types per §3.1: normal / large-revision / turning-point /
  benchmark-window; horizons T−7/T−1), fit the forecast-encompassing regression
  `first_print_private = a + b·model + c·private_consensus` and read the implied optimal combination
  weight `w*` (Granger–Ramanathan / Fair–Shiller combination).
- The gate **fires for a cell** when the encompassing test rejects "consensus encompasses the model"
  *and* the (model, consensus) blend beats **both** standalone on MAE/RMSE/CRPS.
- **Expected shape if the §1 two-edge thesis holds:** `w*(model)` high in the **turning-point** cells
  (the BD edge; Klein 2022 street under-reaction), low in **normal** months (consensus is the near-
  efficient ceiling). If it does *not* come out that way, that is the honest signal the edge isn't there.

### 12.4 Consensus is T−1-only → the combination is a T−1 phenomenon

The street median locks release-eve: `consensus.py` withholds the value until
`as_of ≥ release_date − _LOCK_LAG` (1 day). So **at T−7 there is no consensus** — the model stands alone
(vs naive floors), the cleanest expression of its *earliness* edge. The combination cells are therefore
**T−1 × {month-type}**; the **T−7-vs-T−1 contrast measures what the model's early call is worth versus
waiting to blend with consensus.**

### 12.5 Phase 2 — the combination layer (gated, post-hoc, a port decision)

If the gate fires, add a **`model_combo`** competitor row: a month-type×horizon-weighted blend of
`model_5a` (or `model`) and `consensus`, weights from §12.3, applied **post-hoc in jobs-space** in
`run_a5_backtest.py` — **the §5A pattern exactly** (blend the point *and* the predictive draws so
coverage/CRPS move; **zero `nfp-model` code → no A3 baseline break**; reversible). In-model consensus
conditioning is the later, principled escalation (a 5A→5B analog) — built only if post-hoc proves
insufficient. Because the gate cannot truly fire on the providerless local **skeleton** (TL;DR #4),
**Phase 2 — and whether the gate fires at all — is a Bloomberg-port decision.**

### 12.6 Scope — built here vs deferred to the port

**Build now, off the firewall, locally correctness-testable** (extends the A5 harness; touches no
`nfp-model` and no pinned path):

1. Wire the three consensus series as **series-keyed competitors** — private `'05'` on the private board,
   implied-govt `Total − Private` on a new wedge board, Total `'00'` on the Total board.
2. The **Total-error decomposition** (private-side vs government-side attribution).
3. The **gate computation** — the encompassing regression + optimal combination weights, by
   month-type × horizon (T−1 cells).

**Do NOT build locally:** the `model_combo` combination layer (§12.5) — it is gated and the gate cannot
fire on the skeleton; and any in-model consensus conditioning. The gate firing, the weights, the wedge
prior change (§12.2), and Phase 2 are the **port's** decisions (build-here/validate-on-port).

**Wiring:** set `NFP_CONSENSUS_PATH=s3://alt-nfp/competitors/consensus.parquet` — today only
`NFP_STORE_URI` is set in `.env`, so the harness renders consensus as `—` until this is added.

### 12.7 Updates to earlier sections

This amendment **supersedes** TL;DR #3 and §3 (the private nowcast now has a consensus competitor) and
**extends** §4.2 (the consensus Mincer–Zarnowitz can now run on the **private** track too, not only Track
B) and §10 (the private nowcast now has its **own** consensus ceiling — no longer only a Total figure).
The §9 sequencing gains: Track A step 2 wires the private/implied-govt consensus competitors + the gate
computation; Track B step gains the implied-govt wedge benchmark. The two-track A/B framing **stands**,
re-priced.

**Strategic framing (the product story).** Honest expectation: the model **matches** consensus in normal
months and **beats** it at turning points (the first print is near-efficient month to month — Klein 2022;
the defensible separation is at turns). That *is* the product proposition — *indistinguishable from the
street in the months nobody remembers, visibly better in the ~20% that move markets* — so the scoreboard
must **surface the turning-point cell**, not bury it in a blended MAE.

## Appendix — source map

Which research finding motivates which section (full detail in `specs/model_research.md`):

| Section | Motivating findings |
|------------------------------------|------------------------------------|
| §1, §4 | Aruoba 2008 (revisions not well-behaved); Guisinger–Smith 2019 (CES revisions = news, small relative variance); BLS revision statistics (51k mean abs first→third) |
| §1, §3, §10 | Klein 2022 (consensus biased/inefficient, turning-point under-reaction) — a **Total-NFP** comparator (Track B); BLS sampling SE ~67.5k (Total ceiling) |
| §1, §6 | Birth/death is inherently private (firm births/deaths); turning-point literature (Phillips–Nordlund; quarterly B/D since 2011); Cleveland Fed 2026 (benchmark serial correlation) |
| §5, §7 | Koenig–Dolmas–Piger 2003 (first-release LHS for a first-print target; vintage-aligned RHS) |
