# NFP first-print nowcast — forecastability, competitor targets, and vintage design (research synthesis)

Status: **reference, synthesized 2026-06-19**. Consolidates two independent literature
reviews (formerly `model_research_1.md`, `model_research_2.md`) into one record. This is
the **evidence base** behind `specs/model_improvements.md` (the design) and the
validate-first pivot in `plans/0-port_and_staged_plan.md`; it is descriptive, not a design
or implementation doc. The single question it answers: *for a nowcast scored against the
BLS first-print headline change in seasonally adjusted total nonfarm payrolls, what is the
realistic accuracy ceiling, who is the fair competitor, and how should the training
vintages be built?*

## TL;DR

1.  **The first print is close to an efficient ("news") estimate of its own monthly
    revisions, so the forecastable share of monthly revision variance is small.** The
    realistic ceiling in normal months is set by the **consensus median**, not ADP.
    Predictability concentrates in two non-monthly places: the **annual QCEW benchmark**
    (serially correlated, partly forecastable) and **business-cycle turning points** (net
    birth/death bias).
2.  **ADP and the consensus target different objects.** Post-2022 ADP is an independent
    *nowcast of the QCEW* (truth/benchmark target); the consensus median genuinely predicts
    the first print. Scoring both against the first print is fair to the consensus and
    structurally handicaps ADP — so "beating ADP" can be a target-mismatch artifact, not
    skill.
3.  **Three numbers that read like "the ceiling" are three different objects** — keep them
    apart (see §4). Consensus *forecast* error (~48k MAE / 60–65k RMSE), BLS *sampling* SE
    of the monthly change (~67.5k, irreducible survey noise), and an in-sample
    *revision-prediction* RMSE for final-private CES (61k→58k with provider data) are not
    interchangeable.
4.  **For a first-print target, Koenig–Dolmas–Piger prescribes first-release data on the
    LHS and real-time-vintage data on the RHS.** Because monthly CES revisions are largely
    news, the current "as-of diagonal" is approximately adequate for the monthly model;
    vintage discipline matters most for benchmark-affected months and turning points — where
    a measurement / bias-correction layer, not just a vintage change, is the higher-value fix.

## 1. The measurement object being benchmarked

The CES first print is a sample-based estimate, released quickly and then revised twice in
the next two monthly Employment Situation releases, followed by an **annual benchmark
revision** that re-anchors the series to the QCEW universe (unemployment-insurance
near-census counts). The Cleveland Fed's 2026 benchmark study (Pinheiro & Quinlan),
relying on BLS documentation, frames three distinct layers of change: **monthly revisions**,
**annual benchmark revisions**, and **seasonal-factor revisions** — they are separate
phenomena with separate forecastability, and conflating them is the most common error.

The benchmark anchor is the QCEW, and its **lag is the operational constraint**: the QCEW
is not public until roughly **five to six months after quarter-end** (ADP methodology note;
the Philadelphia Fed's Early Benchmark page puts it at "just over five months"). Annual
benchmarking is released **with January data in February**; over the prior decade the
absolute benchmark revision averaged **0.2%** of total nonfarm employment (BLS).

The **net birth/death model** is central to the cyclical story. In current-CES estimation it
imputes jobs from firm births and deaths that the sample cannot see; the annual benchmark
later corrects it. The Philadelphia Fed's state-employment analysis explains why this bites
at turns: near a cyclical turning point the historically stable birth-to-death relationship
breaks down, generating large one-sided revisions relative to the original sample estimate.

## 2. Q1 — Forecastability of first-print revisions (news vs. noise)

**The framework.** Mankiw & Shapiro (1986, "News or Noise? An Analysis of GNP Revisions")
established the distinction: if revisions are **news**, the early estimate is an efficient
forecast of the later value (the revision is uncorrelated with the first release, correlated
with the final) and little can be done to improve it; if **noise**, the early estimate
carries measurement error (the revision is correlated with the first release) and is
forecastable from information available at release. Mankiw–Shapiro found GNP revisions
largely news.

**Aruoba's challenge.** Aruoba (2008, "Data Revisions Are Not Well Behaved," *Journal of
Money, Credit and Banking* 40(2–3):319–340) shows revisions to many major US macro series
fail two "well-behaved" criteria: (P1) revisions are mean-zero / unbiased; (P2) the standard
deviation of revisions is small (less than half) relative to the variance of the final value.
Many series violate both and are predictable from the initial information set — implying the
initial announcements "are not rational forecasts."

**Where payroll employment sits.** Payroll employment is a comparatively *better-behaved*
case. Guisinger & Smith (2019), applying Aruoba's criteria to CES employment, find CES
revisions positively correlated with the final value and able to "always be classified as
news," satisfying the small-relative-variance criterion (P2) — though the mean revision is
statistically different from zero, so a small bias remains (P1 fails). That is far cleaner
than GNP/GDP or JOLTS (which they cannot classify and find forecastable). But payroll
revisions are **not** fully efficient noise: the Atlanta Fed finds revisions contain
information that anticipates future revisions, and the Cleveland Fed (2026) finds past
*benchmark* revisions retain statistically significant predictive content even recently (chi-square significant at the 1% level over 2010–2025, extending Haltom–Mitchell–Tallman 2005 and Phillips–Nordlund 2012).
Payroll thus sits closer to the "not well-behaved" side than strict efficiency allows — *once
benchmark episodes are included*. The older payroll-specific literature leans "small but
real": Neumark & Wascher (1991), Stark, Phillips & Nordlund (2012, cyclical bias), Owyang &
Vermann (2014, systematic bias), Gregory & Zhu (2008, VAR on the revision process).

**Magnitudes (BLS's own statistics).**

-   *Monthly (post-2003 probability-sample era), seasonally adjusted over-the-month change:*
    mean revision **+7k** (second − first), **+1k** (third − second), **+8k** (third − first);
    mean **absolute** revisions **33k / 34k / 51k**. The 51k third-minus-first mean absolute is
    the headline "monthly wedge" (61k over 1979–2003; the coarser CRS In Focus IF13084 puts the mean *signed* first→third at **+9k**, while over the longer 1979–present window ADP Research, citing BLS, reports a **−11k** mean — the sign is sample-dependent). Collection rates in 2024 averaged 60.4%
    (first closing) → 89% → 90.9%, which is why most revision occurs first→second.
-   *Bias estimate:* Stark (Philadelphia Fed, "Revisions to Nonfarm Payroll Employment: 1964
    to 2011") finds initial estimates biased **down ≈18k** on average (first-to-second), with
    no bias across expansions as a whole but a significant positive bias in the most recent one.
-   *Annual benchmark:* 2002–2023 the absolute benchmark revision (prior-March level) averaged
    **255k**; largest **−902k** (March 2009, −0.7%), smallest **−7k** (March 2021). Most level
    revisions sit within **±0.5%**, and the Cleveland Fed (2026) reports a mean benchmark revision to the monthly *change* of **−4,512** with a cumulative (first-to-latest) revision of **+6,844**. The preliminary March-2025 benchmark was
    **−911k** (−0.6%) total nonfarm and **−880k** (−0.7%) total private; the final (Feb 2026) revised SA total nonfarm down
    **−898k** (≈−0.54%, rounded −0.6%) — marginally outside the ±0.5% band but, per the Cleveland Fed, **not** a
    structural break.

**Regime dependence — where predictability actually lives.** The forecastable share of
*monthly* error is modest in normal times but rises sharply in specific regimes. The Federal
Reserve Board paper using ADP payroll microdata flags the **August revision problem
(2011–2014)** and shows real-time ADP data helped anticipate those large positive revisions.
The San Francisco Fed's 2025–2026 analyses show revisions were exceptionally large in
**2020–2021** and elevated around recession onset as early as **1990**. The pattern is
consistent across sources: revision size and revision forecastability are **state-dependent**
and concentrate at turning points and benchmark windows.

**Deliverable (Q1).** CES first-print monthly revisions are slightly biased (down over the
full sample, with cyclical sign flips), largely news, and only modestly forecastable in
normal regimes — the ceiling over the first print is low month-to-month. Explicitly modeling
the revision process is *not* warranted for the average month, but *is* warranted (i) for the
annual benchmark (serially correlated, QCEW-anchored) and (ii) at turning points (birth/death
bias). A **regime-aware bias-correction layer** is the higher-value investment than a generic
revision model.

## 3. Q2 — What ADP and the consensus actually target

**ADP (post-August-2022 redesign).** The redesigned National Employment Report (first
released 2022-08-31; ADP Research with the Stanford Digital Economy Lab) is explicitly "an
independent measure of the US labor market, rather than a forecast of the BLS monthly jobs
number." ADP's FAQ confirms the break from the prior model, which "sought to forecast changes
in the Current Employment Statistics survey." Current methodology reweights ADP's matched
payroll sample to QCEW industry × state × establishment-size cells and benchmarks to a QCEW
base period; ADP describes **both** its and BLS's estimates as "a timely estimate or nowcast
of the QCEW," with ADP re-benchmarking to the Q1 QCEW each year (preliminary in September,
final in February with the January release). **ADP is a QCEW/truth-targeter, not a first-print
predictor.**

ADP–BLS first-print divergence is large at monthly frequency. Since the 2022 redesign the two
series' monthly changes are weakly related: Chinn (Econbrowser, 2022) finds adjusted R²
≈ **0.13** ex-2020 (≈ −0.03 over 2021M01–2022M05). Vivid single months: October 2024 ADP
**+233k** vs BLS **−28k** private. (Pre-2022 ADP was modeled to track the BLS print and had a
mechanical R² ≈ 0.98, much of it from 2020.)

**Consensus / survey median (Bloomberg, Reuters, Econoday, WSJ).** Professional forecasters explicitly
predict the headline number BLS will announce, making the consensus median the genuine
first-print predictor and the apples-to-apples bar. Two independent accuracy estimates, from
different samples — **report both, do not average**:

-   **UNC "Pros vs. Joes" (Patel & Murphy, 2017)** — Bloomberg consensus, ~100 economists/mo,
    2014–2017, scored explicitly vs. the *initial* release: mean absolute error **≈48k**, RMSE
    **≈60k**, a small **~11k downward** bias, tight dispersion (SD ≈31k; "90% within 100k of each
    other," ≈3.3 SD — evidence of herding). Consensus beat crowdsourced Estimize by ~5k/mo.
-   **Klein (2022, *J. of Economic Behavior & Organization*, "Agree to disagree?")** —
    Bloomberg qualified-economist survey, 2008–2020. Pre-COVID out-of-sample RMSE **≈65k**
    relative to the **first** publication. Klein rejects equal predictive ability across
    economists, finds **significantly biased** forecasts, and shows the best 25% of regular
    participants beat the raw consensus — consensus is useful but **not fully efficient**, while
    still beating model-based and deep-learning benchmarks (and improving further when the best
    economists are combined).

The inefficiency is sharpest in stress: Klein shows forecasters **under-predict job losses in
turmoil and under-predict the recovery afterward** (acute in COVID). That residual,
state-dependent inefficiency is exactly the gap a disciplined model can target.

**The asymmetry to make explicit.** If ADP targets QCEW/benchmark truth and the consensus
targets the first print, then scoring both against the first print is *fair to the consensus,
biased against ADP*. ADP will look worse on first-print loss even if it is the better QCEW
nowcast, so "beating ADP on first-print MAE" can be target mismatch, not skill.
**Recommendation:** report the **consensus as the primary first-print bar**; show ADP with an
explicit truth-target caveat and, ideally, a **second scoreboard** scoring the model and ADP
against the benchmarked/QCEW value — the only fair test for ADP.

**No public national first-print Fed nowcast.** There is no widely published dedicated
*national* first-print NFP nowcast analogous to GDPNow. The St. Louis Fed Economic News Index
nowcasts real GDP, not NFP. Regional-Fed **Early Benchmark** products (Philadelphia, Dallas,
Chicago) target the QCEW/benchmark (truth), not the first print. Practitioner first-print
nowcasts (Chinn/Hamilton's ADP- and claims-based Econbrowser regressions; sell-side desk
models) exist but lack a long, audited public track record. The consensus median is thus the
de facto first-print benchmark; ADP and the Early Benchmarks are truth-targeters.

**Target map.**

| Competitor | What it targets | First-print vs. truth | Documented accuracy |
|---|---|---|---|
| Consensus median (Bloomberg/Reuters) | The headline BLS first print | First-print predictor (apples-to-apples) | MAE ≈48k / RMSE ≈60k (UNC 2014–17); RMSE ≈65k pre-COVID (Klein 2008–20); ~11k-low; biased/inefficient |
| ADP NER (post-2022) | QCEW/benchmark "truth" (nowcast of QCEW) | Truth-targeter (mismatched to first print) | Weak month-to-month link to BLS first print (R²≈0.13 ex-2020) |
| Philadelphia/Dallas Fed Early Benchmark | Annual QCEW benchmark revision (truth) | Truth-targeter; direction/breadth of the national revision | Reduces benchmark-revision size ~11–14% (state/district); not a national point estimate |

## 4. The accuracy ceiling — three distinct objects

A recurring error is to treat several "~60k jobs" figures as one ceiling. They are three
different measurements on three different objects; the design must keep them on separate axes.

| Object | Figure | Source | What it measures |
|---|---|---|---|
| **Consensus forecast error** (vs first print) | MAE ≈48k / RMSE ≈60k; RMSE ≈65k pre-COVID | UNC 2014–17 (Patel & Murphy); Klein 2008–20 | How well the *fair competitor* predicts the first print — the bar to beat |
| **BLS sampling SE** of the monthly change | ≈67,500 (90% CI ≈ ±122k) | Fed Board ADP-microdata paper; BLS Employment Situation Technical Note | *Irreducible survey noise* in the first print — no pre-release model can eliminate it |
| **Revision-prediction RMSE** (final *private* CES) | ≈61k public controls → ≈58k with real-time ADP | Fed Board ADP-microdata paper | In-sample difficulty of predicting the *final private* number once the first print is known; provider data's marginal value is real but **small** |

Three implications. (a) The **sampling SE (~67.5k)** is a floor on first-print uncertainty,
not a forecast target — part of the gap between the first print and later vintages is noise no
model should expect to remove. (b) The **revision-prediction RMSE (61k→58k)** says that, in
normal months, public information beyond the first print buys little, and even rich provider
microdata buys only a few thousand jobs of RMSE on the *ex-post correction* problem — the
clearest upside is **pre-release forecasting of the first print**, not heroic ex-post
correction of it. (c) The **consensus error (~48k/~60–65k)** is the one number that is
actually a competitor bar; a first-print nowcast that cannot materially beat consensus on the
object consensus is trying to hit has not demonstrated its core value.

## 5. Q3 — Real-time-vintage estimation for a first-print target

**Koenig–Dolmas–Piger (2003, "The Use and Abuse of 'Real-Time' Data in Economic
Forecasting," *Review of Economics and Statistics* 85(3):618–628).** KDP distinguish three
strategies:

-   *Strategy 1 (preferred):* first-release data on the LHS, **real-time-vintage** data on the
    RHS (each in-sample date uses RHS variables as they appeared at that date).
-   *Strategy 2:* latest-available vintage on the LHS, real-time-vintage on the RHS.
-   *Strategy 3 (common but "to be avoided"):* latest/current vintage on both sides.

Their core recommendation: use as many vintages as there are dates, and **use first-available
official estimates for the LHS even if you ultimately care about final-revised values.** Under
the efficiency assumption (the initial release's revisions are unpredictable at release), OLS
on a first-release LHS is unbiased *and* has strictly lower error variance than regressing on
revised data, because any revision adds information uncorrelated with the real-time RHS —
"extraneous noise" to the estimator. KDP show Strategy 1 yields GDP-growth forecasts
competitive with the Blue Chip consensus and degraded performance under Strategy 3, and note
it is practically easier to assemble many short vintages than long reconstructed single-vintage
series. This principle recurs in later practice: the Philadelphia Fed's SPF benchmark models
are estimated with the vintage panelists actually had; Croushore (2011, *Journal of Economic
Literature* 49(1):72–100) and Croushore & Stark (2001 *J. of Econometrics*; 2003 *REStat*)
show data vintage materially affects forecast evaluation and that the safe way to evaluate is
with real-time data. Moving down the **main diagonal** of a real-time dataset gives the
initial release for each date — and in principle any vintage can serve as the actual value
(first release, annual revision, or latest), so the choice should match the forecasting target
(Croushore 2011). Production nowcasters increasingly model the vintage/revision structure
rather than ignore it: the dynamic-factor tradition of Giannone–Reichlin–Small, and
specifically the Brave et al. state-space CES benchmark model that embeds the revision process
plus a labor-indicator dynamic factor — direct prior art for a Bayesian state-space NFP method
like this project's. Clark & McCracken supply the companion real-time forecast-evaluation
toolkit (tests for equal predictive accuracy in nested models).

**Where the current "as-of diagonal" sits.** Training on recent months at early prints and
older months at original third prints — with benchmark information entering only through QCEW
— is a hybrid: **closer to Strategy 1 than Strategy 3** on the target side (it uses early
prints for the LHS where it matters), but not a *pure* first-release-vintage sample, because
older observations sit at third prints rather than their own first prints, and benchmark
re-anchoring can contaminate historical first-print levels. The efficiency loss versus a pure
first-release sample is governed directly by Q1: **if monthly revisions are news with low
forecastable content (the payroll case), the loss is small; if they carry predictable
structure (benchmark months, turning points), the diagonal injects revision information the
real-time forecaster would not have had, biasing in-sample fit and overstating apparent
accuracy.**

**Deliverable (Q3).** The *target side* should already use first-release values (the current
diagonal largely does — keep it). A **full** first-release-vintage reconstruction is worth the
data-layer cost only to the extent Q1 shows monthly revisions carry predictable content; for
US payrolls that content is concentrated in benchmark-affected months and turning points. The
proportionate design: keep the diagonal for the monthly model, but (a) ensure benchmark
re-anchoring does not contaminate historical first-print training levels, and (b) add a
vintage-aware / bias-correction treatment specifically for benchmark months and cyclically
extreme periods. *This does not mean rebuilding every predictor at intra-month timestamps on
day one — only that the target and the most important release-driven predictors should sit as
close as possible to the true decision-time information set.*

## 6. Synthesis

**(a) The realistic ceiling and the right bar.** The fair, apples-to-apples first-print
benchmark is the **consensus median** (~48k MAE / 60–65k RMSE vs the first print, ~11k-low,
documented inefficiency). The first print is itself largely an efficient (news) estimate of
its own later sample-based revisions, with sampling error (≈±122k 90% CI / ~67.5k SE)
dwarfing the monthly revision wedge (51k mean absolute first→third — most of it news, not forecastable). The realistic ceiling for a
monthly model is therefore *roughly match-to-slightly-beat the consensus by exploiting its
known inefficiencies* (under-reaction to claims/ADP at turning points) — not large,
systematic outperformance. ADP should be shown but flagged as a QCEW-truth target.

**(b) Can a Bayesian state-space model targeting the first print reach the ceiling?**
Structurally yes for the **normal regime**: a well-specified state-space / dynamic-factor
model ingesting real-time indicators (initial/continuing claims, JOLTS, ADP, financial
conditions, cyclical state) and targeting the first print can capture the consensus-level
signal and harvest its residual inefficiencies. To do *better* than consensus where it matters
— turning points and benchmark months — the model needs (i) a measurement / bias-correction
layer for the birth/death-driven first-print bias and (ii) explicit incorporation of
QCEW/Early-Benchmark information for benchmark-affected months. A pure first-print model
without these layers will replicate the consensus's turning-point errors. **The vintage change
(Q3) is necessary hygiene; the bias-correction layer is the source of edge.**

**(c) What to instrument in the harness.** Decompose forecast error by regime:

-   **Normal months** — error vs consensus, ADP scored separately against truth; expect
    near-parity with consensus.
-   **Large-revision months / turning points** — condition on cyclical state (recession/recovery
    flags, claims momentum); test whether the model's and the consensus's first-print errors are
    predictable from claims/ADP/JOLTS. Define large-revision months mechanically from the
    historical first-to-second/first-to-third distribution. **This is where edge appears or
    disappears**, and where misses tend to be one-sided in slowdowns and rebounds.
-   **Benchmark months** — isolate the February-release benchmark window; score against both the
    first print and the eventual benchmarked value; track the Early Benchmark's directional
    signal.

## 7. In-house diagnostics (run on the project vintage store)

The literature gives the *prior*; the project's own vintage store must supply the
NFP-specific *posterior*. Two regressions are the decisive next-step diagnostics:

-   **Mincer–Zarnowitz efficiency regression.** Regress the realized outturn on a constant and
    the forecast, `actual_t = α + β·forecast_t + ε_t`; test the joint null α=0, β=1. Rejection
    (β≠1 or α≠0) means forecast errors are systematically related to information at the forecast
    origin — the forecast is inefficient. Apply to **both** the consensus and the model's
    first-print forecast.
-   **Aruoba-style revision regression.** Regress the revision (later vintage − first print) on
    the information set knowable at first-print time,
    `(y^later_t − y^first_t) = α + γ'·X_t + u_t`, reading the **intercept α as the bias** and the
    **R² as the forecastable share** of revision variance. Use
    `X_t = {claims, JOLTS, ADP, financial conditions, lagged revisions, cyclical state}`. Priors
    from the literature: small/modest α (a few thousand to ~18k, sign cyclically dependent) and
    low R² in normal months, rising once benchmark/turning-point regressors enter. *The intercept
    α is the direct input to the first-print fairness fix in `model_improvements.md` §5; the R²
    decides whether a full first-release-vintage rebuild (§7 there) is worth the cost.*

## 8. Recommendations — staged, with escalation thresholds

The staging below is the literature's natural order; it maps directly onto the **Tier**
structure of `specs/model_improvements.md` (one program, two documents). The one deliberate
divergence: the research's "ingest Early-Benchmark/QCEW signal for benchmark months" is held
**out of scope** in the design as a different product (annual-benchmark accuracy vs the
first-print nowcast) — recorded here as evidence, deferred there as roadmap.

| Research stage | Design Tier (`model_improvements.md`) | Escalation threshold |
|---|---|---|
| **Stage 1 — set the scoreboard correctly.** Consensus = primary first-print bar; second QCEW-scored scoreboard for ADP; regime-decomposed error from day one. | **Tier 0** (§3) | If the model cannot match consensus MAE (~48k) in normal months, fix the monthly model before adding layers. |
| **Stage 2 — run the Q1 diagnostics.** Mincer–Zarnowitz (consensus + model) and the Aruoba revision regression (α = bias, R² = forecastable share). | **Tier 1** (§4) — *the gate* | If revision-regression R² is low (<~0.1) in normal months and concentrated in benchmark/turning-point regressors, do **not** invest in a full first-release-vintage reconstruction — the diagonal is adequate. |
| **Stage 3 — build the layers that create edge.** (a) first-print observation / bias layer (the δ offset = Stage-2 Aruoba intercept) → Tier 2 §5; (b) birth/death-aware bias-correction keyed to cyclical state → Tier 3 §6; (c) [deferred in design] Early-Benchmark/QCEW signal for benchmark months. Vintage discipline = hygiene, not the primary lever. | **Tier 2 / Tier 3** (§5, §6) | Pursue the full first-release-vintage rebuild only if Stage-2 R² shows materially predictable monthly revisions. |
| **Stage 4 — validate at turning points.** Stress-test on 2008–09, 2020, and the 2024–25 large-revision episodes. | **Tier 3 validation** (§6) | If the model's turning-point error is statistically indistinguishable from the consensus's, the bias-correction layer is not yet earning its keep. |

## 9. Caveats & limitations

-   **No single published forecastable-variance R² for US payroll monthly revisions exists.**
    The "low monthly, higher at benchmarks/turning points" conclusion is synthesized from
    multiple sources (Guisinger–Smith; Cleveland Fed; the Fed Board ADP-microdata paper; BLS),
    not one headline statistic. The in-house Aruoba regression (§7) is the right way to pin the
    project-specific number.
-   **The consensus figures span two samples** — UNC 2014–17 (~48k MAE / 60k RMSE) and Klein
    2008–20 (~65k RMSE pre-COVID). Both predate or only partly cover COVID-era volatility; a
    long-window, peer-reviewed standard deviation of the NFP *surprise* (first print − consensus)
    was not located. Treat as an approximate ceiling, not a precise constant.
-   **The ~67.5k sampling SE and the 61k→58k revision-prediction RMSE are private-payroll /
    final-CES objects** from the Fed Board paper — not first-print forecast accuracy. Do not cite
    them as the consensus bar (see §4).
-   **The R²≈0.13 ADP–BLS link figure is from Econbrowser** (credible, not peer-reviewed); the
    ADP-minus-BLS monthly gap statistics floated elsewhere (one computation: mean absolute
    difference ≈68k/month, SD ≈99k over 2010-onward) come from
    non-peer-reviewed sources. Directionally consistent (ADP is a poor month-to-month first-print
    proxy) but re-estimate in-house before citing as authoritative.
-   **Early-Benchmark accuracy is documented at the state/district level** (≈11–14% revision-size
    reduction, Brave, Gascon, Kluender & Walstrum 2021, *International Journal of Forecasting*
    37(3):1261–1275); the Philadelphia Fed explicitly does **not** publish a national point-estimate
    RMSE and warns the sum-of-states is not a national measure — use it for direction/breadth, not a
    national number.
-   **Benchmark methodology and birth/death handling have changed over time** (quarterly birth/death
    updates since 2011; pandemic-era tweaks; 2024–25 partial-year adjustments), so historical
    revision statistics mix regimes. Structural-break tests (Cleveland Fed 2026) find no break, but
    caution is warranted when pooling pre- and post-2020 data.
-   **Public evidence is stronger for private-payroll revisions, state benchmark revisions, and
    professional-survey behavior than for a single clean national estimate of the forecastable share
    of total-nonfarm first-print revision variance.** That is a reason to interpret A5 as the
    benchmark that tells you whether a later measurement layer is warranted — not a reason to delay it.

## Appendix — primary sources

-   **News-vs-noise framework:** Mankiw & Shapiro (1986); Aruoba (2008, *JMCB* 40(2–3):319–340).
-   **Payroll revisions:** Guisinger & Smith (2019); Neumark & Wascher (1991); Stark (Philadelphia
    Fed, 1964–2011); Phillips & Nordlund (2012); Owyang & Vermann (2014); Gregory & Zhu (2008);
    Atlanta Fed; San Francisco Fed (2025–2026); Federal Reserve Board ADP-microdata paper; CRS In
    Focus IF13084; BLS Employment Situation Technical Note.
-   **Benchmark revisions:** Cleveland Fed (Pinheiro & Quinlan, 2026); Haltom–Mitchell–Tallman
    (2005); Phillips–Nordlund (2012); Brave, Gascon, Kluender & Walstrum (2021, *IJF*
    37(3):1261–1275); Walstrum (2015, Chicago Fed); CRS In Focus IF12827; Philadelphia/Dallas Fed
    Early Benchmark.
-   **Competitor targets:** Stanford Digital Economy Lab / ADP NER redesign note (2022) + ADP
    methodology; Chinn (Econbrowser, 2022); Patel & Murphy (2017, UNC "Pros vs. Joes"); Klein (2022,
    *J. of Economic Behavior & Organization*).
-   **Real-time vintages:** Koenig–Dolmas–Piger (2003, *REStat* 85(3):618–628); Croushore (2011,
    *JEL* 49(1):72–100); Croushore & Stark (2001, *J. of Econometrics*; 2003, *REStat*); Clark &
    McCracken; Philadelphia Fed SPF documentation.
