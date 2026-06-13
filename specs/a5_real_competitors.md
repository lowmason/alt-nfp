# A5 — Real competitors in the harness (design)

Status: **design, in review (2026-06-13)**. Implements the A5 gate in
`plans/0-port_and_staged_plan.md` ("Real competitors in the harness") and
resolves the evaluation-convention question left open by
`specs/ces_growth_convention.md` and the A4 dual-track finding
(`plans/6-a4_vmap_backtests.md`) — by choosing **Option A (the first
print)** as the sole A5 target (plans/0 SQ1, answered 2026-06-13). Companion
spec for the consensus data source: `specs/bloomberg_consensus.md`.

## TL;DR

1. A5 is **evaluation-side only**. It adds *targets* (what we score against)
   and *competitors* (what we compare the model to) plus a scoreboard
   report. It touches **no** `nfp-model` code, **no** `transform_to_panel`,
   **no** `build_model_data`, and **no** A1/A2/A3 golden-mastered path. The
   model already emits nowcasts through the A4 batched harness; A5 wraps
   scoring around them. The parity firewall stays intact.
2. **Target: the first print** (plans/0 strategic question 1, **answered
   2026-06-13**). Every competitor is scored against the within-release
   headline change BLS announces — Option A, an additive extractor over store
   levels. This is exactly what consensus aims at, so the comparison is
   apples-to-apples. Benchmark-informed / "revised-truth" targeting is
   **deferred** to a separate later model (this model tweaked to target
   benchmark revisions, or a new one). `final_view` revised truth may appear
   as an *unscored* reference column (to show revision magnitude beside the
   first-print scores) but is **not** an A5 target and gates nothing.
3. **Two near-release regimes.** The model is scored at **T−7** and **T−1**
   (BLS-release(M) − 7 and − 1 days), both new grids. These sit in the window
   where the payroll-provider inputs are near-complete — provider series
   quality rises as more of month M's payrolls are processed, so day-12 of M
   (the A4 grid) starves the model of its best input. Consensus is a **T−1**
   competitor (the street median locks ~release-eve); the smart baseline and
   naive floors run at both. T−7 is the model's read a week early; T−1 is the
   fair head-to-head against the locked street.
4. **Competitor set.** model · consensus (pluggable, Bloomberg-sourced —
   staged, T−1) · smart baseline (bridge regression on **vintage-censored**
   claims/JOLTS) · naive floors (random-walk, trailing-12-month mean). **ADP
   is intentionally dropped** — see §4.

## 1. Scope and non-goals

**In scope.** New evaluation-side library code (targets + competitor
adapters), two new near-release snapshot grids (T−7, T−1), and an
`a5_report.md` scoreboard.

**Explicit non-goals (the firewall).**

- No change to `nfp-model`. The model's nowcast is an input to A5, not a
  subject of it.
- No change to `transform_to_panel`, `build_model_data`, the vintage store,
  or any golden-mastered selection/growth path. The first-print target is an
  **additive** read over store levels (per `ces_growth_convention.md` §5
  Option A) — it adds a new derived series, it does not alter the `growth`
  column A1/A2 pin.
- No new model features (Phase A's parity-is-done rule still holds; competitor
  comparison is measurement, not modeling).
- The **store is append-only and read-only** for A5 (root CLAUDE.md hard
  rule). A5 reads levels; it never writes.

## 2. Target — the first print (Option A)

A5 scores against **one** actual: the within-release headline change BLS
announces. plans/0 strategic question 1 is **answered (2026-06-13): the
target is the first print.** Benchmark-informed ("revised-truth") targeting
is a separate, later model's concern (this model tweaked to target benchmark
revisions, or a new one) and is out of A5 scope.

### 2a. First print (Option A) — the scored target

The within-release headline change BLS announces on release day:
`headline(p, R) = log L(p at R) − log L(p−1 at R)`, both levels from the same
release. Built by a new extractor reading store **levels** under the
structural pairing rule from `ces_growth_convention.md` §5:

- partner of `(p, rev r, bmr 0)` is `(p−1, rev r+1, bmr 0)` for r ∈ {0, 1};
- at benchmark months, fall back to `p−1`'s latest `(rev 2, bmr ≥ 1)` row;
- **never** join on `vintage_date` equality (historical stamps misalign);
- for r = 2 outside benchmark season headline == cohort by construction.

Coverage is 271/273 rev-0 months (the two misses — 2003-05 history edge,
2026-01 Dec-rev-1 shadow with a bmr-1 fallback — are handled per the spec).
Validated against published headlines (+73k for 2025-07, +143k for 2025-01,
+130k for 2026-01).

**Home:** new `nfp_ingest/first_print.py` (alongside `model_data.py`, the
spec's recommended home). Returns per-period
`(first_print_growth, first_print_change_k, vintage_date)`. Unit-tested
against the published headlines above. Zero contact with pinned paths.

### 2b. Revised truth — unscored reference only (deferred)

The diff of one internally coherent latest-published level path
(`nfp_vintages.views.final_view` semantics) is basis-consistent by
construction and remains trivially available. It MAY be rendered as an
*unscored* reference column so the magnitude of subsequent revision is
visible beside the first-print scores — but it is **not** an A5 target, it
gates nothing, and the model is not scored on it. Full revised-truth
evaluation belongs to the future benchmark-targeting model (plans/0 SQ1,
resolved). If shown, it cleanly replaces the A4 report's rough
"best-available" track.

**Home (only if the reference column is wanted):** `nfp_vintages.evaluation`
gains a small `revised_truth_change_k(panel)` helper on `final_view`. Defer
if not needed.

## 3. Information regimes — the harness shape

The model is scored at **two near-release horizons**: **T−7**
(BLS-release(M) − 7 days) and **T−1** (BLS-release(M) − 1 day, release-eve).
Both sit in the window where the model's inputs are at their best, and the
one-week gap isolates the value of the final week of information before the
print.

**Why not day-12 of M (the A4 grid)?** The model's signal is dominated by the
payroll-provider series, whose quality is a function of *how many of month
M's payrolls have been processed* when the model runs. At day-12 of M the
reference month is barely underway and the provider series for M is thin; by
T−7 / T−1 (early M+1) M's pay periods are essentially complete and the as-of
censoring automatically surfaces the higher-quality, later provider vintages.
Scoring at the near-release horizons therefore measures the model on the
information set it is actually designed to use — not an artificially starved
one.

| Regime | As-of date | Competitors present | Measures |
|---|---|---|---|
| **T−7** | BLS-release(M) − 7 days | model, smart baseline, naive | the model's read **one week before** the print, on near-complete provider data (consensus has not locked yet) |
| **T−1** | BLS-release(M) − 1 day (release-eve) | model, consensus, smart baseline, naive | the **fair head-to-head**: the model on the full pre-release information set, against the locked street median |

The T−7 → T−1 gap quantifies how much the final week of provider payments and
claims is worth; the T−1 regime is where the consensus comparison lives (the
Bloomberg survey median is a release-eve snapshot, so it aligns to T−1 — see
`specs/bloomberg_consensus.md`; a week earlier the street has not yet formed a
reliable median, so consensus renders `—` at T−7).

**Grid construction.** Two new snapshot grids analogous to the A4 `snapshot`
command, with `as_of = _ces_publication_date(M) − {7, 1} days` (release date
from `release_dates/vintage_dates.py` `_ces_publication_date` /
`_first_friday`; use the scraped/actual release date where available, the
computed first-Friday rule otherwise). Censoring at either horizon still
excludes M's own print (its `vintage_date` is release day R > T−1, so
`vintage_date <= as_of` drops it), leaving M the unobserved nowcast target;
M−1's print and all of M's high-frequency data (claims, JOLTS at its lag,
provider data at its lag) are included. Both grids are **new compute** (the A4
day-12 grid is not reused for A5 scoring) but reuse `fit_model_batch`
**unchanged** — only the as-of dates differ; ~2 × 24 fits, tractable on the
batched harness. Buildability failures surface cheaply via the A1
negative-master pattern (record, don't crash).

## 4. Competitors

All competitors are reduced to a single comparable quantity per reference
month: **MoM change in total nonfarm payrolls, thousands** (`change_k`), the
same unit the model nowcast and the first-print target use.

- **Model.** Nowcast `change_k` from the batched fits at each regime's as-of
  (produced by the T−7 / T−1 grids; A5 reads the reduction).
- **Consensus.** Pluggable adapter `load_consensus(path=None) -> DataFrame |
  None`: reads a configured file in the contract schema if present, else
  returns `None` and the scoreboard renders the column as `—`. **Staged** —
  structure ships now; the Bloomberg-sourced data drops in later per
  `specs/bloomberg_consensus.md`. A **T−1-only** competitor: the final survey
  median locks ~release-eve, so it aligns to the T−1 regime and is absent at
  T−7.
- **ADP — dropped (decision 2026-06-13).** Removed from the competitor set.
  Post-Aug-2022 (the Stanford Digital Economy Lab redesign) ADP publicly
  states it is "an independent measure of private-sector employment… not
  intended to forecast the Bureau of Labor Statistics monthly jobs report,"
  so for the recent half of the window it is not a fair first-print
  competitor; pre-2022 history would cover only part of the sample, and as a
  private-only measure it never aligned cleanly to total nonfarm anyway. The
  literal plans/0 gate names ADP, but its *intent* (real competitors beyond
  naive floors) is met by consensus plus the smart bridge baseline. This
  amends the gate; the amendment is recorded in plans/0.
- **Smart baseline.** A bridge regression predicting M's growth from
  **vintage-censored** claims/JOLTS — the *same* as-of-censored cyclical
  arrays the model consumes (via `build_model_data(as_of=D)` / the snapshot),
  **not** revised series. Using revised regressors would be lookahead and
  make it an unfair (cheating) competitor; the vintage discipline is what
  makes this the honest *ceiling*. It is the one piece beyond the literal
  gate and is the scope cut-line: if effort must be trimmed, the bridge
  baseline is the first thing to defer (the gate is still met by
  model/consensus/naive).
- **Naive floors.** random-walk (last published print repeats) and
  trailing-12-month mean. Sanity floors, never gates.

## 5. Report — the scoreboard

A new `scripts/run_a5_backtest.py` (sibling to `run_a4_backtest.py`, reusing
its `snapshot`/`batched` machinery for the T−7 and T−1 grids) emits
`data/backtests/a5_report.md` + `a5_results.parquet`:

- a **scoreboard** of `competitor × regime` scored against the **first
  print**, with ME / MAE / RMSE (and, where dispersion exists, hit-rate /
  direction accuracy);
- per-date detail (each reference month: every competitor's `change_k`, the
  first-print actual, errors; plus the *unscored* revised-truth reference
  column if enabled);
- COVID (2020–2021) excluded from headline metrics (decided-questions rule).

The A4 report (`a4_report.md`) stays as-is — it is a parity-gate artifact
(batched-vs-serial). A5 is the evaluation layer above it; it does not modify
A4 outputs.

**Gate satisfied when** the report scores the model against consensus, the
smart baseline, and the naive floors at each regime (ADP dropped per §4).
Consensus may render `—` until the Bloomberg data is wired (staged) and is a
T−1-only competitor; the column, join, and scoring path exist and are
exercised by a fixture, so dropping in the file completes the gate with no
code change.

**Forward compatibility (B1).** The output consumer is decided (plans/0 SQ2,
2026-06-13): an accurate first-print nowcast for Bloomberg publication, plus
a supersector "why" narrative — so **B1 (the supersector narrative layer)
leads Phase B**. A5 itself stays national, but the scoreboard and competitor
protocol are keyed on a **series identifier** (national now; supersectors
later) so B1 can score supersector nowcasts through the same harness without
a rebuild. This is a structural choice (one key column), **not** added scope
— no supersector logic ships in A5.

## 6. Module layout

```
nfp_ingest/first_print.py            # Option A first-print extractor (store levels)
nfp_vintages/evaluation.py           # + revised_truth_change_k(final_view), competitor protocol
nfp_vintages/competitors/
    __init__.py                      # Competitor protocol + registry
    consensus.py                     # load_consensus() pluggable adapter (see bloomberg_consensus.md)
    bridge.py                        # vintage-censored claims/JOLTS bridge regression
    naive.py                         # random-walk + trailing-mean floors
scripts/run_a5_backtest.py           # T−7 / T−1 grids + scoreboard; reuses A4 batched harness
```

Tests: `nfp_ingest/tests/test_first_print.py` (extractor vs published
headlines), `nfp_vintages/tests/test_competitors.py` (each adapter's
units/alignment + the consensus fixture), `nfp_vintages/tests/test_a5_*`
(scoreboard assembly, regime alignment).

## 7. Sequencing

Retire risk first, then build outward from the cheapest pieces:

1. **First-print extractor** (`first_print.py`) — pure store-levels read,
   testable immediately against the published headlines. No grid needed.
2. **Revised-truth reference** (optional, *unscored* — `evaluation.py` on
   `final_view`) — small; build only if the reference column is wanted,
   else skip.
3. **Near-release grids (T−7, T−1)** — the new compute; reuse
   `fit_model_batch`. Build both snapshot grids, fit them batched, surface
   unbuildable dates.
4. **Competitor adapters** in increasing cost order: naive → consensus
   (with staged fixture) → bridge baseline.
5. **Scoreboard** (`run_a5_backtest.py`) wiring competitors × regimes scored
   against the first print; emit `a5_report.md`.
6. Gate annotation in plans/0; spec → archive on completion; memory updated.

## 8. Open items and risks

- **Consensus data is staged.** The gate is structurally met now; the
  consensus *column* fills when the Bloomberg file lands
  (`specs/bloomberg_consensus.md`). Track as a follow-up, not a blocker.
- **Near-release buildability (T−7, T−1)** — frontier/shutdown months may be
  unbuildable at either horizon; same negative-master handling, recorded not
  fatal.
- **Bridge-baseline fairness** hinges on vintage censoring of its regressors;
  if that proves fiddly, defer the baseline (it is beyond the literal gate).
- Carry forward the `ces_growth_convention.md` §3 model-input observations
  (triple benchmark wedge at February as-ofs; permanent November rev-2
  outliers) into the Phase-B model-evidence agenda — not an A5 task.

## Appendix: gate restatement (plans/0)

> **A5 — Real competitors in the harness.** Add ADP prints (FRED; mind the
> Aug-2022 methodology break) and consensus survey median (Bloomberg/Econoday
> history — sourcing is a real acquisition task). Naive baselines stay as
> sanity floors, not gates. **Gate:** every backtest report scores model vs.
> ADP vs. consensus vs. naive, at each information regime.

This design meets it with: consensus (staged, Bloomberg) + a smart bridge
baseline + naive floors, scored against the **first print** across the
**T−7 and T−1** regimes. **ADP is intentionally dropped** (§4): post-2022 it
no longer forecasts the BLS print, so it is not a fair first-print
competitor; the gate's intent (real competitors beyond naive floors) is met
by consensus + the smart baseline.
