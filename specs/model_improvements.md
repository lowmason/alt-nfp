# Model improvements — first-print targeting, diagnostics, and turning-point edge (design)

Status: **design, draft (2026-06-19)**. The **model-side counterpart** to `specs/a5_real_competitors.md` (which is an evaluation-side firewall — no `nfp-model` changes). Motivated by `specs/model_research.md` (consolidated literature review — forecastability, competitor targets, vintage design) and the **validate-first pivot** in `plans/0-port_and_staged_plan.md` (parity is a port-fidelity floor, **not** correctness). This spec is where Phase A's "parity-is-done" freeze is deliberately reopened — behind new baselines — to address what A5's prong-2 surfaced: the model is not yet modeling the target it is scored on.

## TL;DR

1.  **The model targets the wrong object.** Its latent is pinned to QCEW (the benchmark universe) and passed through a single, vintage-*pooled* CES observation equation, so the nowcast predicts a *revised/third-print*-ish value — then it is scored against the **first print**. That is the same target mismatch the reports say unfairly handicaps ADP. The headline fix converts the truth-nowcaster into a genuine **first-print predictor**.
2.  **Two edges over consensus, both forward-looking.** Consensus lacks (a) **private payroll-provider microdata** (a normal-month level edge) and (b) **explicit turning-point birth/death modeling** (a direction-at-turns edge). The published "low ceiling" binds *public-information* models; these two edges are the value proposition — but both are **untested locally** because provider data is **Bloomberg-only**. Every local backtest scores a providerless *skeleton* that understates the model.
3.  **Measure before building.** Tier 1 diagnostics (Aruoba revision regression, Mincer–Zarnowitz efficiency, provider-ablation) run first and **gate** which model layers get built. The Aruoba intercept *is* the input to the §5 fix.
4.  **Model changes land behind new parity baselines** (post-pivot governance, §8) — never by relaxing the existing one. The reference stays the port target, not an oracle.
5.  **Out of scope:** a benchmark / Early-Benchmark revision layer (different product), a full first-release-vintage store rebuild (unless §4 justifies), supersector logic (Phase B). The public-info ceiling is low; a providerless clean-window result at \~parity is **expected, not a failure** — judge normal months on **calibration**, not point-MAE.

## 1. Motivation — the model targets the wrong object

**Literature** (condensed from `specs/model_research.md`). CES monthly first prints are largely *news*, not *noise* (Guisinger–Smith 2019): small, slightly biased (\~+9k mean first→third; \~51k mean absolute), low forecastable share in normal months. Sampling error dominates — BLS pegs the monthly SE at \~67.5k (±122k 90% CI) — so the realistic ceiling over the first print is **low in normal regimes**. The forecastable structure concentrates in exactly two places: the **annual QCEW benchmark** (serially correlated; Cleveland Fed 2026) and **turning points** (the net birth/death model cannot see business deaths in real time, so it overstates jobs entering downturns). The fair, apples-to-apples bar is the **consensus median** (\~48k MAE / 60–65k RMSE vs the first print; itself \~11k-low and inefficient — Klein 2022), **not ADP**, which post-2022 targets the QCEW universe and is structurally mismatched to first-print scoring.

**The code-grounded reframe.** Our model sits on ADP's side of that line. The latent `g_total_sa` is pinned to QCEW (the benchmark universe) by the Student-t anchor (`model.py:187-193`), then passed through a **single, vintage-pooled** CES observation equation `alpha_ces + lambda_ces·g_total_sa` whose bias/loading are shared across all vintages (`model.py:198-202`; only `sigma_ces_sa` is vintage-indexed). The nowcast applies those pooled parameters directly (`nowcast.py:21-28`). Because the as-of training diagonal is dominated by third-print observations, the pooled `alpha_ces` encodes a *revised-print* bias — so the nowcast predicts a generic/revised print and, scored against the first print, carries a systematic offset. This is the same target mismatch the reports flag for ADP; the A5 spec's premise that the model "already emits nowcasts" scored fairly is therefore incomplete.

**Thesis.** The published "low ceiling" binds models limited to *public* information. This model's value proposition is two edges consensus structurally lacks: (1) **private payroll-provider microdata** — richer and earlier than the public ADP print the street sees — which should lift *normal-month* accuracy above consensus; and (2) **explicit turning-point birth/death modeling**, which should beat consensus on *direction at cyclical turns* (where Klein 2022 shows the street systematically under-reacts). Both are currently **untested locally**: the rebuilt public store has no provider data (Bloomberg-only), so every local A5 run scores a providerless *skeleton* that understates the model. The program therefore has two jobs — make the model a *fair* first-print predictor (testable locally now) and stand up the evaluation so the provider + turning-point edges can be **demonstrated or falsified** on the Bloomberg run (forward-looking). Matching consensus is the *floor*; beating it — on providers in normal months, on BD at turns — is the goal.

## 2. Scope & non-goals

**In scope.** (1) **Evaluation-side** — revision/efficiency diagnostics (§4) and a regime-decomposed scoreboard (§3) extending the A5 harness, plus a provider-ablation measurement. (2) **Model-side, behind new parity baselines** — a first-print observation layer (§5) and a turning-point-aware birth/death extension (§6). The §4 diagnostics **gate** which model layers get built — no speculative structure.

**Evaluation venue — a first-class constraint.** The model has two information regimes: **full** (Bloomberg compute, payroll-provider microdata present) and **public-only** (local, rebuilt public store, providers absent). The provider edge and the turning-point edge live in the *full* regime; a *beat-consensus* claim is valid only there. Locally we can develop, unit-test, and validate the one provider-independent improvement — the **first-print fairness fix** (§5), whose benefit (removing a systematic per-month bias) is present regardless of inputs. The clean-window capability readout and any provider-ablation are **forward-looking** to the Bloomberg run. The spec is written *toward* that deploy target; everything it specifies is buildable and testable on the public store.

**Non-goals.**

-   **No benchmark / Early-Benchmark revision layer** — it targets a different product (annual-benchmark accuracy); the product here is the first-print nowcast. Deferred; revisit only if the roadmap adds a benchmark deliverable.
-   **No full first-release-vintage store rebuild** — unless the §4 Aruoba R² shows materially predictable monthly revisions. Absent that, the as-of diagonal stands (KDP hygiene only, §7).
-   **No removal of the QCEW truth anchor** — the latent stays QCEW-anchored; §5 *adds* a first-print observation layer on top, it does not rip out the anchor that makes the model coherent across CES/QCEW.
-   **No supersector logic** (Phase B / B1) — national only, as in A5.

**Relationship to existing design & parity governance.** This is the **model-side counterpart** to the evaluation-firewall `a5_real_competitors.md`: it deliberately opens `nfp-model` to change under the validate-first pivot (`plans/0`; parity ≠ correctness). Every model-side change lands **behind a new A3-style parity baseline** (§8) — never by relaxing the existing one. The post-hoc offset (§5A) touches no `model.py` and breaks no baseline; the vintage-indexed observation equation (§5B) does, and gets its own.

## 3. Tier 0 — Scoreboard correctness *(evaluation-side; extends the A5 harness, no model code)*

The A5 scoreboard already scores `competitor × horizon-regime` against the first print (ME/MAE/RMSE), with **consensus as the intended primary bar** (currently stubbed — `load_consensus()` returns `None` until the Bloomberg file lands, `run_a5_backtest.py:169`) and **ADP off the first-print board** (`a5_real_competitors.md` §4, target-mismatch). Tier 0 keeps those and adds the three things the reports require:

1.  **Month-type decomposition.** Split every metric by **normal / large-revision / turning-point / benchmark-window**, alongside the existing T−7/T−1 horizon split. Definitions: *large-revision* = \|first→third revision\| above a fixed percentile of the historical distribution (computable now from `first_print_changes()` + later vintages); *turning-point* = a cyclical-state flag (claims-momentum / direction-change; recession-dating optional); *benchmark-window* = the Feb-release months most affected by annual-benchmark + seasonal-factor updates; *normal* = the complement. Pooled MAE hides exactly where edge lives — the **provider edge in normal months, the BD edge at turns** — so the decomposition is what makes the value proposition *measurable*.
2.  **Calibration metrics.** Add interval coverage (80/90% hit-rate) and CRPS beside the point metrics. In normal months — and in every providerless local run — the Bayesian model's value is *honest uncertainty*, not a point-MAE edge; scoring only ME/MAE/RMSE would declare "no edge" precisely where the real deliverable is calibration (ties to §10).
3.  **Venue tag.** Every scored row records whether providers were in the information set (**full** vs **public-only**, §2), so a providerless local result is never misread as a full-regime capability readout.

COVID (2020–21) stays excluded from headline metrics; shutdown-frontier months (e.g., 2026-01) are flagged, not silently pooled. Home: `scripts/run_a5_backtest.py` scoring + `a5_report.md`; reads the model reduction and `first_print_changes()`; touches no pinned path.

## 4. Tier 1 — Diagnostics, the gate *(evaluation-side; on the store; no model change)*

Both reports converge on one instruction: **measure the forecastable share before building any model layer.** Tier 1 runs three diagnostics whose outputs *gate* Tiers 2–3 and *feed* §5.

1.  **Aruoba revision regression** — `(later_vintage − first_print) = α + γ'·X +    u`, with `X = {claims, jolts, biz_apps, nfci, lagged revisions,    cyclical-state}`, run pooled and by month-type. The **intercept α is the first-print bias** (consumed directly by §5's offset); the **R² is the forecastable share.** *Decision rule:* normal-month R² below \~0.1 → the as-of diagonal is adequate, do **not** fund a first-release-vintage rebuild (§7); R² concentrated in turning-point / benchmark regressors → fund §6 (and confirm §5's vintage treatment). Feasible now from `first_print_changes()` + store later vintages + the as-of-censored cyclical arrays.
2.  **Mincer–Zarnowitz efficiency regression** — `actual = α + β·forecast`, testing α=0, β=1, on **both** consensus and the model's nowcast. Rejection means information left on the table; the *shape* of consensus's inefficiency (Klein's turning-point under-reaction) is exactly what the model aims to exploit, and the same test on the model is a self-check.
3.  **Provider-ablation** *(full regime; Bloomberg-only, forward-looking)* — score model-with-providers vs model-without on identical dates, by month-type. The **direct test of the §1 provider hypothesis**: it isolates the provider contribution to point accuracy *and* calibration, and only runs where providers exist.

The A5 "smart baseline" (bridge regression on vintage-censored claims/JOLTS, `a5_real_competitors.md` §4) is the **public-info ceiling** and shares this machinery: the gap between model-with-providers and the bridge baseline *is* the provider edge. Home: a diagnostics module (`nfp_vintages.evaluation` or a `scripts/` diagnostic) reading the store + `first_print_changes()` + competitor outputs; unit-tested against the literature priors (small α, low normal-month R²) as a check that our store behaves like published US-payroll evidence. **Tier 1 runs first** — it needs no model change, and its numbers decide what (if anything) Tiers 2–3 build.

## 5. Tier 2 — A first-print observation equation *(the headline model fix; staged)*

The fix for the §1 reframe. Today the nowcast predicts `alpha_ces + lambda_ces·g_total_sa` with pooled, vintage-shared parameters (`nowcast.py:21-28`, `model.py:198-202`); dominated by third-print training rows, that predicts a *revised* print. The revision is ≈ a mean shift (first→third ≈ +9k), i.e. an **α effect** — so the lever is the intercept, not the loading.

**5A — Post-hoc first-print offset** *(cheap first cut; no parity break; locally testable).* Subtract the measured first-print bias at nowcast time: `δ` = the §4 Aruoba intercept (pooled or month-type-specific), applied in growth space before the index arithmetic. Touches only `nowcast.py` + the harness — **no `model.py` change, no baseline break.** It is **provider-independent**, so it is the *one* model improvement validatable locally now, and it should measurably shrink the clean-window **ME** even without providers. Limitation: a constant/month-type correction, not a likelihood-learned vintage effect.

**5B — Vintage-indexed observation equation** *(principled; new baseline).* Index `alpha_ces` (and optionally `lambda_ces`) by CES vintage in `model.py` — today only `sigma_ces_sa` is vintage-indexed (`model.py:204-211`) — and have the nowcast select the **first-print vintage's** parameters. This lets the model *learn* the first-print bias and its uncertainty in the likelihood rather than import a point estimate. It **breaks A3 parity → new baseline** (§8). *Identifiability caveat:* the as-of diagonal shows one print per month, so first-print rows concentrate at the frontier; cleanly identifying a first-print α may need a first-release-vintage CES target history (§7) — §4's Aruoba structure tells us whether the signal is there.

**Staging.** Ship 5A first — it is the fairness fix for prong-2, costs no baseline, and is locally validatable. Escalate to 5B only if 5A's offset proves insufficient *or* §4 shows vintage structure worth learning. This front-loads the win and minimizes parity disruption.

## 6. Tier 3 — Turning-point-aware birth/death *(the edge layer; gated, dirty-month-validated)*

§1's second edge. Today BD is **linear and symmetric** — `bd_t = phi_0 + Σ phi_3[i]·X[i] + sigma_bd·xi` (`model.py:157-172`), default covariates just `("claims","jolts")` (`config.py:88`) — and the latent has only a coarse calendar-fixed `n_eras = 2` mean, no endogenous cyclical-state detection. The documented failure is **asymmetric**: the net birth/death model cannot see business deaths in real time, so it overstates jobs *entering* downturns — exactly what a symmetric linear `phi_3` cannot capture.

Three enrichments, **cheapest-first**:

1.  **Wire `biz_apps` into the BD covariate set** — the business-applications birth proxy is configured in the data layer but absent from `indicator_names`. Lowest effort, directly motivated by the births literature. (It helps the *birth* side; the death-at-turns asymmetry needs 2–3.)
2.  **Asymmetric / hinge claims loading** — let claims load more strongly when rising / above a threshold (state-dependent `phi_3`), capturing the death surge entering downturns.
3.  **Cyclical-state regime on the BD intercept** — a third "downturn" regime or a Markov-switching `phi_0`, distinct from the calendar eras, raising BD-death risk when the cyclical state flips.

**Gated** on §4's turning-point R² (don't build absent the signal); each enrichment lands **behind a new baseline** (§8); build cheapest-first, stop when turning-point error stops improving. **Validated on dirty months** (2008-09, 2020, the 2024-25 large-revision episodes), **not** the clean window — which excludes turns by construction and cannot test this layer. This is the heaviest lift and most contingent: it is where the *direction-at-turns* edge over consensus lives, but it is gated, baseline-breaking, and only fully testable on the Bloomberg full regime — so it is **last in sequence**.

## 7. Vintage discipline (KDP hygiene)

Koenig–Dolmas–Piger (2003) Strategy 1 — first-release on the LHS target, real-time vintage on the RHS — is *already* approximated by the as-of diagonal, which uses first prints for recent months where it matters. The hygiene task is narrow: ensure benchmark re-anchoring does not contaminate the historical first-print training *levels* (older months sit at third prints + re-anchored levels — verify no future information leaks into the first-print target). A *full* first-release-vintage CES reconstruction is gated on §4's R² (and is the same data that resolves §5B's identifiability). Necessary maintenance — not the primary lever. The diagonal stays for the monthly model; the rebuild is funded only if revisions prove materially predictable.

## 8. Parity governance

Tier 0/1 and §5A touch no `model.py` → no baseline. Every model-side change (§5B, §6.1–6.3) lands **behind its own new A3-style baseline**: regenerate the golden, pin the new posterior, and **record the divergence from the frozen reference as intentional** (a correctness change, not a regression). Never relax the existing baseline; the reference stays the port target, not an oracle (`plans/0`, parity ≠ correctness). Each baseline cutover follows the existing A3 machinery (`nfp_model.parity`, `scripts/run_a3_parity.py`, golden fixtures under `s3://alt-nfp/golden/a3`). A change that improves first-print accuracy *and* diverges from the reference is the expected, sanctioned outcome under validate-first — it must be recorded here and in `plans/0`'s gate log, not silently absorbed.

## 9. Sequencing

1.  **Tier 0 + Tier 1** (scoreboard + diagnostics) — no model change; Tier 1 gates the rest.
2.  **§5A offset** — the locally-testable fairness fix (provider-independent).
3.  **Prong-2** (clean window, scored vs *live* consensus, by month-type, judged on calibration) — providerless local = a *skeleton* readout.
4.  **Gate decision** → **§5B** and/or **§6** if §4 justifies; each behind a new baseline (§8).
5.  **Full-regime (Bloomberg) readout** with providers + provider-ablation — the actual beat-consensus test.

## 10. Reality check

The public-info ceiling is low (sampling SE \~67.5k; consensus \~48k MAE). A providerless local clean-window prong-2 landing at \~parity is **expected, not a failure** — it is the skeleton, not the experiment. The real edges (providers in normal months, BD at turns) are forward-looking to the full regime. Jan-2026 (−309k vs +130k first print) is dirty, shutdown-frontier, *and* providerless — triple-disqualified as a capability signal, and not evidence against the model in the clean regime. Judge clean-window normal months on **calibration** (coverage / CRPS), not point-MAE; a Bayesian model's value in normal months is honest uncertainty, not beating a \~48k-MAE consensus on the mean.

## 11. Open items & risks

-   **Consensus data is staged.** The primary bar renders "—" until the Bloomberg file lands (`specs/bloomberg_consensus.md`); Tier 0 metrics against consensus are blocked on it. Track as a follow-up, not a code blocker.
-   **§5B identifiability.** Cleanly identifying a first-print α may force the first-release-vintage CES rebuild (§7); §4's Aruoba structure decides. If the signal is weak, 5A's offset is the durable answer.
-   **`biz_apps` wiring.** Verify it threads through `panel_adapter` / `model_inputs` into the model path before §6.1 — it is data-layer config (`CYCLICAL_INDICATORS`), not a model default (`indicator_names`).
-   **Tuning against a skeleton.** The two headline edges cannot be tested locally; guard against over/under-fitting the providerless skeleton by deferring edge-layer tuning to the full regime.
-   **Bridge-baseline build status.** The A5 smart baseline (`a5` §4) is the public-info ceiling Tier 1 leans on; confirm it is built (it is the A5 scope cut-line and may be deferred there).
-   **Dirty-month validation set.** COVID (2020–21) and shutdown exclusions overlap the very turning-point months §6 needs; curate the Tier-3 validation set deliberately so the layer is testable without reintroducing excluded regimes into headline metrics.

## Appendix — source map

Which research finding motivates which section (full detail in `specs/model_research.md`):

| Section | Motivating findings |
|------------------------------------|------------------------------------|
| §1, §4 | Aruoba 2008 (revisions not well-behaved); Guisinger–Smith 2019 (CES revisions = news, small relative variance); BLS revision statistics (51k mean abs first→third) |
| §1, §3, §10 | Klein 2022 (consensus biased/inefficient, turning-point under-reaction); BLS sampling SE \~67.5k (ceiling) |
| §1, §2, §4 | ADP target-mismatch (Stanford Digital Economy Lab / ADP methodology — "nowcast of the QCEW," not a forecast of the BLS print); Fed Board ADP-microdata study (modest marginal provider value: \~61k→58k RMSE) |
| §5, §7 | Koenig–Dolmas–Piger 2003 (first-release LHS for a first-print target; vintage-aligned RHS) |
| §6 | Cleveland Fed 2026 (benchmark serial correlation); birth/death turning-point literature (Phillips–Nordlund; quarterly B/D since 2011) |