# A5 — Real competitors in the harness (design)

Status: **design, approved 2026-06-13**. Implements the A5 gate in
`plans/0-port_and_staged_plan.md` ("Real competitors in the harness") and
resolves the evaluation-convention question left open by
`specs/ces_growth_convention.md` (Options A + C) and the A4 dual-track
finding (`plans/6-a4_vmap_backtests.md`). Companion spec for the consensus
data source: `specs/bloomberg_consensus.md`.

## TL;DR

1. A5 is **evaluation-side only**. It adds *targets* (what we score against)
   and *competitors* (what we compare the model to) plus a scoreboard
   report. It touches **no** `nfp-model` code, **no** `transform_to_panel`,
   **no** `build_model_data`, and **no** A1/A2/A3 golden-mastered path. The
   model already emits nowcasts through the A4 batched harness; A5 wraps
   scoring around them. The parity firewall stays intact.
2. **Dual-track targets.** Every competitor is scored against both the
   **first print** (Option A — the within-release headline change, built by
   an additive extractor over store levels) and **revised truth** (Option C
   — the diff of a single coherent `final_view` level path). First print is
   the primary scoreboard (it is what consensus/ADP aim at); revised truth is
   secondary (it is what the model is structurally built to win, and answers
   plans/0 strategic question 1).
3. **Two information regimes.** The model is scored at **early** (day-12 of
   the reference month — the existing A4 grid) and at **release-eve**
   (BLS-release(M) − 1 day — a new grid). Consensus, ADP, and the smart
   baseline only exist at release-eve; the model and naive baselines run at
   both. The early regime measures the model's *early-warning* edge (it
   produces a number ~3 weeks before any competitor exists); release-eve is
   the *fair head-to-head* on a common information set.
4. **Competitor set.** model · consensus (pluggable, Bloomberg-sourced —
   staged) · ADP (FRED, regime-split at the Aug-2022 methodology break) ·
   smart baseline (bridge regression on **vintage-censored** claims/JOLTS) ·
   naive floors (random-walk, trailing-12-month mean).

## 1. Scope and non-goals

**In scope.** New evaluation-side library code (targets + competitor
adapters), a new release-eve snapshot grid, and an `a5_report.md` scoreboard.

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

## 2. Targets — dual-track

A5 scores against two "actuals", because the competitors aim at different
things and the project has two distinct questions (plans/0 SQ1).

### 2a. First print (Option A) — the primary scoreboard

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

### 2b. Revised truth (Option C) — the secondary scoreboard

The diff of one internally coherent latest-published level path —
`nfp_vintages.views.final_view` semantics — so the target is basis-consistent
by construction (no cohort splices in the actual). This replaces the A4
report's rough "best-available" track and its `†` exclusions with a
well-defined target. The trade-off (documented, accepted): the target moves
when benchmarks land.

**Home:** `nfp_vintages.evaluation` (already exists; gains a
`revised_truth_change_k(panel)` helper built on `final_view`).

## 3. Information regimes — the harness shape

The A4 grid runs the model at **day-12 of reference month M** and nowcasts
M's print (which BLS publishes ~the first Friday of M+1). Consensus and ADP
for M do not exist until ~release-eve (first Thursday of M+1), ~3 weeks
later, on a much richer information set. Scoring the day-12 model against them
would penalize an information *deficit*, not skill. So A5 defines two regimes:

| Regime | As-of date | Competitors present | Measures |
|---|---|---|---|
| **early** | day-12 of M (existing A4 grid) | model, naive | early-warning edge: a number ~3 weeks before anyone else has one |
| **release-eve** | BLS-release(M) − 1 day (new grid) | model, consensus, ADP, smart baseline, naive | fair head-to-head on a common information set |

**Release-eve grid construction.** A new snapshot grid analogous to the A4
`snapshot` command, but with `as_of = _ces_publication_date(M) − 1 day`
(BLS release date computable from `release_dates/vintage_dates.py`
`_ces_publication_date` / `_first_friday`; use the scraped/actual release
date where available, the computed first-Friday rule otherwise). Censoring at
release-eve still excludes M's own print (its `vintage_date` is the next day,
so `vintage_date <= as_of` drops it), leaving M the unobserved nowcast target;
M−1's print and all of M's high-frequency data (claims, JOLTS at its lag,
provider data at its lag) are included — the same information set the street
had. Buildability failures surface cheaply via the A1 negative-master
pattern (record, don't crash). The release-eve fits reuse
`fit_model_batch` **unchanged** — only the as-of dates differ.

**Both grids feed the same scoreboard;** the early regime is near-free since
its snapshots/fits already exist from A4.

## 4. Competitors

All competitors are reduced to a single comparable quantity per reference
month: **MoM change in total nonfarm payrolls, thousands** (`change_k`), the
same unit the model nowcast and both targets use.

- **Model.** Nowcast `change_k` from the batched fits at each regime's as-of
  (already produced; A5 just reads the reduction).
- **Consensus.** Pluggable adapter `load_consensus(path=None) -> DataFrame |
  None`: reads a configured file in the contract schema if present, else
  returns `None` and the scoreboard renders the column as `—`. **Staged** —
  structure ships now; the Bloomberg-sourced data drops in later per
  `specs/bloomberg_consensus.md`. Naturally a release-eve competitor (the
  final survey median locks ~release-eve), so it aligns to that regime.
- **ADP.** `nfp_download.fred.fetch_fred_series("ADPMNUSNERSA")` (total
  private payroll *level*, SA → MoM change). **Regime-split at Aug-2022**
  (the Stanford Digital Economy Lab redesign): pre-break ADP was an explicit
  BLS-print predictor and is scored as a genuine competitor; post-break ADP
  publicly disclaims forecasting BLS and is labeled an *independent measure*
  (a reference series / a finding about its decoupling, not a fair forecast).
  The split is a boolean flag + a report footnote. *Implementation note:
  confirm FRED `ADPMNUSNERSA` pre-2022 coverage and whether its history is a
  new-methodology backcast; if pre-break ADP needs the discontinued
  ADP-Moody's series, source it separately or restrict ADP scoring to the
  post-break era and document the gap.* **Comparison basis (decided):** ADP
  is a *private* payroll measure while the model/consensus/targets are *total
  nonfarm*. Default to scoring ADP against the total-nonfarm targets directly,
  with the omitted government-employment MoM change carried as a documented,
  usually-small bias and flagged in the months where it is not small (Census
  2020, shutdowns). The cleaner private-vs-private basis (score ADP against a
  CES *total-private* first print) is a noted upgrade for when industry
  detail is wired in (a Phase-B-adjacent capability); it is not required for
  the gate.
- **Smart baseline.** A bridge regression predicting M's growth from
  **vintage-censored** claims/JOLTS — the *same* as-of-censored cyclical
  arrays the model consumes (via `build_model_data(as_of=D)` / the snapshot),
  **not** revised series. Using revised regressors would be lookahead and
  make it an unfair (cheating) competitor; the vintage discipline is what
  makes this the honest *ceiling*. This is the one piece beyond the literal
  gate and is the scope cut-line: if effort must be trimmed, the bridge
  baseline is the first thing to defer (the gate is still met by
  model/ADP/consensus/naive).
- **Naive floors.** random-walk (last published print repeats) and
  trailing-12-month mean. Sanity floors, never gates.

## 5. Report — the scoreboard

A new `scripts/run_a5_backtest.py` (sibling to `run_a4_backtest.py`, reusing
its `snapshot`/`batched` machinery for the release-eve grid) emits
`data/backtests/a5_report.md` + `a5_results.parquet`:

- a **scoreboard** of `competitor × target × regime` with ME / MAE / RMSE
  (and, where dispersion exists, hit-rate / direction accuracy);
- per-date detail (each reference month: every competitor's `change_k`, both
  actuals, errors);
- the ADP pre/post-2022 split called out explicitly;
- COVID (2020–2021) excluded from headline metrics (decided-questions rule).

The A4 report (`a4_report.md`) stays as-is — it is a parity-gate artifact
(batched-vs-serial). A5 is the evaluation layer above it; it does not modify
A4 outputs.

**Gate satisfied when** the report scores model vs ADP vs consensus vs naive
at each regime. Consensus may render `—` until the Bloomberg data is wired
(staged), but the column, join, and scoring path exist and are exercised by a
fixture; dropping in the file completes the gate with no code change.

## 6. Module layout

```
nfp_ingest/first_print.py            # Option A first-print extractor (store levels)
nfp_vintages/evaluation.py           # + revised_truth_change_k(final_view), competitor protocol
nfp_vintages/competitors/
    __init__.py                      # Competitor protocol + registry
    adp.py                           # FRED ADPMNUSNERSA → change_k, Aug-2022 regime split
    consensus.py                     # load_consensus() pluggable adapter (see bloomberg_consensus.md)
    bridge.py                        # vintage-censored claims/JOLTS bridge regression
    naive.py                         # random-walk + trailing-mean floors
scripts/run_a5_backtest.py           # release-eve grid + scoreboard; reuses A4 batched harness
```

Tests: `nfp_ingest/tests/test_first_print.py` (extractor vs published
headlines), `nfp_vintages/tests/test_competitors.py` (each adapter's
units/alignment + the consensus fixture), `nfp_vintages/tests/test_a5_*`
(scoreboard assembly, regime alignment).

## 7. Sequencing

Retire risk first, then build outward from the cheapest pieces:

1. **First-print extractor** (`first_print.py`) — pure store-levels read,
   testable immediately against the published headlines. No grid needed.
2. **Revised-truth target** (`evaluation.py` on `final_view`) — small.
3. **Release-eve grid** — the only new compute; reuses `fit_model_batch`.
   Build the snapshot grid, fit it batched, surface unbuildable dates.
4. **Competitor adapters** in increasing cost order: naive → ADP → consensus
   (with staged fixture) → bridge baseline.
5. **Scoreboard** (`run_a5_backtest.py`) wiring competitors × targets ×
   regimes; emit `a5_report.md`.
6. Gate annotation in plans/0; spec → archive on completion; memory updated.

## 8. Open items and risks

- **Consensus data is staged.** The gate is structurally met now; the
  consensus *column* fills when the Bloomberg file lands
  (`specs/bloomberg_consensus.md`). Track as a follow-up, not a blocker.
- **ADP pre-2022 history** (see §4 note) — verify FRED coverage; worst case
  restrict to post-break and document.
- **Release-eve buildability** — the frontier/shutdown months that were
  unbuildable at day-12 may still be unbuildable; same negative-master
  handling, recorded not fatal.
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

This design meets it with: ADP (regime-split) + consensus (staged, Bloomberg)
+ naive floors + a smart bridge baseline, scored on dual targets across the
early and release-eve regimes.
