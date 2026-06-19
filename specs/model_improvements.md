# Model improvements — private first-print targeting, diagnostics, and turning-point edge (design)

Status: **design, revised 2026-06-19**. The **model-side counterpart** to `specs/a5_real_competitors.md` (evaluation-side firewall — no `nfp-model` changes). Motivated by `specs/model_research.md` (consolidated literature review) and the **validate-first pivot** in `plans/0-port_and_staged_plan.md` (parity is a port-fidelity floor, **not** correctness). This spec reopens Phase A's "parity-is-done" freeze — behind new baselines — to fix two things A5's prong-2 surfaced: the model is not modeling the **object** it should (it nowcasts **private** NFP, not total), nor the **vintage** it is scored on (first print).

## TL;DR

1.  **The model nowcasts PRIVATE NFP — target the private first print.** The model's signal is inherently private: it is anchored to QCEW (private in this store) and its richest inputs are **private payroll providers**. It cannot see government employment. So its object is **CES total private** (`industry_code='05'`), and it must be scored against the **private** first print and **private** QCEW truth. The run-path currently defaults to `'00'` (total nonfarm) — a **latent mismatch** being corrected here, *not* the intended target. (The `'00'` total data exists only for a future, unbuilt "private nowcast + government forecast = total" extension — Track B below.)
2.  **Second, even on private it targets the wrong vintage.** The latent passes through a single, vintage-*pooled* CES observation equation, so the nowcast predicts a *revised/third-print*-ish value — then it is scored against the **first** print. The headline model fix (§5) makes it a genuine **first-print** predictor.
3.  **Two evaluation tracks.** **Track A (now):** the private nowcast vs **naive floors only**, scored on the private first print + the private QCEW-settled truth (the **primary** truth comparison), decomposed by regime, judged on **calibration**. **Track B (deferred):** Total NFP = **private nowcast + government forecast** → compared to the Total-NFP **consensus** + Total first print. Consensus is a **Total** object — it has no meaning against the private nowcast alone.
4.  **Build here, validate on the port.** This compute (Opus 4.8, no limits) is for *building*; the local eval is a providerless **skeleton** that can confirm code **runs and converges** but cannot **validate accuracy**. So build the full stack here (Tiers 1–3, each behind its parity baseline), confirm correctness + convergence locally, and defer the **accuracy verdict, tuning, and keep/drop** to the Bloomberg port. Tier 1 diagnostics are **instrumentation** (the Aruoba intercept feeds §5A) — **not** a build-blocker.
5.  **Out of scope / deferred:** the **government forecast** (Track B's critical path — captured, not designed or built here); a benchmark / Early-Benchmark revision layer; a full first-release-vintage store rebuild (unless §4 justifies); supersector logic (Phase B). **ADP is out entirely** — not a competitor, not a regressor. Model changes land behind **new parity baselines** (§8).

## 1. Motivation — the model must target the private first print

**The object is private.** The model's latent `g_total_sa` is pinned to QCEW by the Student-t anchor (`model.py:187-193`), and its differentiating inputs are **private payroll-provider microdata**. Both are *private*: this store's QCEW national series is total-private (`industry_code='05'`; see `store-industry-layout`), and payroll providers by construction cannot observe government employment. So the model nowcasts **private** NFP. Scoring it against **total nonfarm** (which folds in government) is a target mismatch; comparing it to the **total-nonfarm consensus** is doubly mismatched. The fair truth for the model is the **private** first print and the **private** QCEW-settled value. The run-path's `industry_code='00'` default (`first_print_changes`, `panel_to_model_data`, `nfp_vintages.a5`) is the mismatch this spec corrects — every CES/QCEW series the model trains on, predicts, and is scored against moves to `'05'`.

**The vintage is also wrong.** Even on the private object, the nowcast applies a **vintage-pooled** CES observation equation `alpha_ces + lambda_ces·g_total_sa` whose bias/loading are shared across all vintages (`model.py:198-202`; only `sigma_ces_sa` is vintage-indexed); the nowcast applies those pooled parameters directly (`nowcast.py:21-28`). Because the as-of training diagonal is dominated by third-print rows, the pooled `alpha_ces` encodes a *revised-print* bias — so the nowcast predicts a generic/revised print and, scored against the first print, carries a systematic offset. §5 is the fix.

**Two edges — of the private nowcast.** Against a *public-information* private predictor, this model's value proposition is two edges: (1) **private payroll-provider microdata** — richer and earlier than any public print — which should lift *normal-month* private-nowcast accuracy; and (2) **explicit turning-point birth/death modeling**, which should win on *direction at cyclical turns* (where Klein 2022 shows public forecasters under-react). Both are **untested locally**: the rebuilt public store has no provider data (Bloomberg-only), so every local run scores a providerless *skeleton* that understates the model. These edges describe the **private nowcast's quality**; whether the *Total* product beats *consensus* is a **Track B** question (it depends on the private nowcast *and* the government forecast), deferred.

## 2. Scope & non-goals

**In scope (Track A — the private nowcast).** (1) **Evaluation-side** — a regime-decomposed, calibration-aware scoreboard (§3) and revision/efficiency diagnostics (§4) extending the A5 harness, all on the **private** target, with **naive floors** as the only competitors. (2) **Model-side, behind new parity baselines** — a first-print observation layer (§5) and a turning-point-aware birth/death extension (§6), making the model a good *private first-print* predictor. The §4 diagnostics **gate** which model layers get built.

**Track B — the Total NFP product (deferred; spec-only, NOT built here).** The product that competes with consensus is **Total NFP = private nowcast + government forecast**. The **government forecast is a new, undesigned component** and the **critical path** to any consensus comparison: government employment is slow-moving and seasonal, a tractable separate model, but it does not exist yet. Until it does, there is **no valid consensus comparison** — consensus is a Total object. This spec *captures* Track B as the gateway to the consensus contest; it does not design or build it.

**Evaluation venue — a first-class constraint.** The model has two information regimes: **full** (Bloomberg compute, payroll-provider microdata present) and **public-only** (local, rebuilt public store, providers absent). The provider edge lives in the *full* regime; a beat-a-public-baseline claim from providers is valid only there. Locally we can develop, unit-test, and validate the one provider-independent improvement — the **first-print fairness fix** (§5A) — and the entire evaluation harness. The provider-ablation and the full capability readout are **forward-looking** to the Bloomberg run.

**Non-goals.**

-   **No government forecast** in this round — Track B's critical path; deferred (see above).
-   **No ADP** — removed entirely as competitor and regressor. The private nowcast is judged against the private first print + private QCEW truth + naive floors; the Total contest (Track B) is against consensus, not ADP.
-   **No benchmark / Early-Benchmark revision layer** — a different product (annual-benchmark accuracy).
-   **No full first-release-vintage store rebuild** — unless §4's Aruoba R² shows materially predictable monthly revisions. Absent that, the as-of diagonal stands (KDP hygiene only, §7).
-   **No removal of the QCEW truth anchor** — the latent stays QCEW-anchored (private QCEW); §5 *adds* a first-print observation layer on top.
-   **No supersector logic** (Phase B / B1) — national private total only.

**Relationship to existing design & parity governance.** This is the **model-side counterpart** to the evaluation-firewall `a5_real_competitors.md`: it opens `nfp-model` to change under the validate-first pivot (`plans/0`; parity ≠ correctness). Every model-side change lands **behind a new A3-style parity baseline** (§8). NB: the private retarget itself (feeding the model `'05'` instead of `'00'` arrays) is a **data-layer** change — the model has only ever been validated on `'00'`, so a `'05'` fit must be confirmed to converge (no divergences, sane posterior) before any `'05'` eval is trusted (§11).

## 3. Tier 0 — Private scoreboard correctness *(evaluation-side; extends the A5 harness, no model code)*

Score the **private nowcast** against the **private** first print, by `competitor × horizon-regime × month-type`. **Competitors: naive floors only** (random-walk, trailing-mean). **No consensus** (a Total object → Track B); **no ADP**. The **private QCEW-settled scoreboard** (§Task-5 in the plan) is the **primary truth** comparison — the model is QCEW-anchored, so the QCEW-settled private value is the closest administrative truth it can be held to. Tier 0 adds:

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

**5A — Post-hoc first-print offset** *(cheap first cut; no parity break; locally testable).* Subtract the measured private first-print bias at nowcast time: `δ` = the §4 Aruoba intercept (pooled or month-type-specific), applied in growth space before the index arithmetic. Touches only `nowcast.py` + the harness — **no `model.py` change, no baseline break.** It is **provider-independent**, so it is the *one* model improvement validatable locally now, and it should measurably shrink the clean-window **ME** even without providers. Limitation: a constant/month-type correction, not a likelihood-learned vintage effect.

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

Tier 0/1 and §5A touch no `model.py` → no baseline. The **private retarget** changes only the *data* fed to the model (`'05'` arrays), not `model.py` — but because the model was validated on `'00'`, treat the first `'05'` fit as a **fresh validation** (confirm convergence + sane posterior; §11), and **pin a `'05'` reference fit** as the baseline for subsequent `'05'` model changes. Every model-side change (§5B, §6.1–6.3) lands **behind its own new A3-style baseline**: regenerate the golden, pin the new posterior, and **record the divergence from the frozen reference as intentional**. Never relax the existing baseline; the reference stays the port target, not an oracle (`plans/0`). Each cutover follows the existing A3 machinery (`nfp_model.parity`, `scripts/run_a3_parity.py`, golden fixtures under `s3://alt-nfp/golden/a3`). Record divergences here and in `plans/0`'s gate log.

## 9. Sequencing

**Track A — build here, in order; each step confirmed to *run + converge*, not validated for accuracy:**
1.  **Retarget to private `'05'`** — thread `industry_code` through the snapshot/model target, `first_print`, the a5 index, and the Tier 1 LHS; confirm a `'05'` fit converges (§8, §11). *(Done 2026-06-19 except the convergence fit — the next session's first step.)*
2.  **Tier 0 + Tier 1** (private scoreboard + diagnostics instrumentation) — no model change.
3.  **§5A offset** — the fairness fix (provider-independent), confirmed to shift ME sanely.
4.  **§5B** (vintage-indexed α) and **§6** (turning-point BD enrichments) — **built now**, each behind its own new parity baseline (§8), confirmed to sample cleanly. **Not** gated on the local Tier 1 numbers (build-here/validate-on-port, TL;DR #4).
5.  **Defer to the Bloomberg port:** the accuracy verdict, layer tuning, keep/drop, provider-ablation, and the private-nowcast capability test — the full regime is the only place these are *valid*.

**Track B (deferred, after Track A is solid):** design the **government forecast** → assemble **Total = private nowcast + government forecast** → score the Total against the **Total-NFP consensus** + Total first print. This is the only valid consensus contest.

## 10. Reality check

The published consensus ceiling (~48k MAE / 60–65k RMSE; sampling SE ~67.5k) is a **Total-NFP** figure — it binds **Track B**, not the private nowcast directly. The private nowcast is judged on **private** truth (first print + QCEW-settled) and **calibration**, not against consensus. A providerless local clean-window readout landing at ~parity with the naive/public baselines is **expected, not a failure** — it is the skeleton, not the experiment; the real edges (providers in normal months, BD at turns) are forward-looking to the full regime. Jan-2026 (dirty, shutdown-frontier, providerless) is triple-disqualified as a capability signal. Judge clean-window normal months on **calibration**.

## 11. Open items & risks

-   **Government forecast is undesigned** — Track B's critical path and the gateway to *any* consensus comparison. Captured here; not built. The first Track-B task is to scope it (government employment: slow-moving, seasonal, administratively well-measured — likely a small standalone model).
-   **`'05'` is untested in the model.** A1–A3 goldens and A4 were all on `'00'`. Before trusting any `'05'` eval: confirm `first_print_changes('05')` yields a sane private series (✓ verified 2026-06-19: 108 months, 2017→2026) **and** that one `'05'` model fit converges (divergences, R-hat, posterior sanity). Pin a `'05'` reference fit (§8).
-   **Aruoba private LHS definition.** The first-to-third *private* revision must exclude annual benchmark wedges and avoid month gaps (a naive "exclude benchmark rows + shift" corrupts the series). Get the cohort matching right before trusting the intercept (§5A input) or the gate.
-   **§5B identifiability.** Cleanly identifying a first-print α may force the first-release-vintage CES rebuild (§7); §4's Aruoba structure decides. If weak, 5A's offset is the durable answer.
-   **`biz_apps` wiring.** Verify it threads through `panel_adapter` / `model_inputs` into the model path before §6.1 — data-layer config (`CYCLICAL_INDICATORS`), not a model default (`indicator_names`).
-   **Tuning against a skeleton.** The provider edge cannot be tested locally; defer edge-layer tuning to the full regime.
-   **Dirty-month validation set.** COVID (2020–21) and shutdown exclusions overlap the very turning-point months §6 needs; curate the Tier-3 validation set deliberately.

## Appendix — source map

Which research finding motivates which section (full detail in `specs/model_research.md`):

| Section | Motivating findings |
|------------------------------------|------------------------------------|
| §1, §4 | Aruoba 2008 (revisions not well-behaved); Guisinger–Smith 2019 (CES revisions = news, small relative variance); BLS revision statistics (51k mean abs first→third) |
| §1, §3, §10 | Klein 2022 (consensus biased/inefficient, turning-point under-reaction) — a **Total-NFP** comparator (Track B); BLS sampling SE ~67.5k (Total ceiling) |
| §1, §6 | Birth/death is inherently private (firm births/deaths); turning-point literature (Phillips–Nordlund; quarterly B/D since 2011); Cleveland Fed 2026 (benchmark serial correlation) |
| §5, §7 | Koenig–Dolmas–Piger 2003 (first-release LHS for a first-print target; vintage-aligned RHS) |
