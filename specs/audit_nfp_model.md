# Package Review — `nfp-model`

**Scope this turn (read first).** Retrieval is still unavailable, but this is the
package I have in the most depth from the system pass, so the review is
near-complete:

- **`model.py`** — good depth (latent AR(1) + Fourier seasonal, era means, the
  QCEW / CES / provider likelihoods, the gated cyclical block, site naming, the
  `phi` clip).
- **`config.py`** — good (`ModelPriors` nested frozen dataclasses, `PRESETS`).
- **`data.py`** — good (`from_snapshot`, `model_inputs`, the v1→v2 fallback).
- **`batch.py`** — good depth (padding/masking, the static/batched split, the
  vmap fit, the closure binding).
- **`parity.py`** — good (`collect_parity_arrays`, `compare_reduced`, the MCSE
  z-test and the kurtosis-aware SD escape hatch).
- Tests: `test_model_unit.py` (incl. `TestBoundary`), `test_data.py`, and the
  batch/parity tests in part.

**Deferred (thin/partial coverage):** `sampling.py` (I have the `fit_model` /
`FitResult` shape but not the full NUTS config and seed handling) and
**`nowcast.py`** (the predictive / level-reconstruction arithmetic — the
highest-value remaining target, since a wrong cumulation or base level yields
wrong headline jobs-added numbers). Plus the full `CLAUDE.md` and some test
bodies.

**Calibration up front:** this is the strongest, most disciplined package in the
workspace. I'm not going to manufacture findings. The model is principled, the
batching is *provably* correct (log-density equality, not just MCMC agreement),
the parity machinery is statistically literate, and the architectural boundary is
enforced by a test. Almost nothing here is a bug. The findings are: one genuine
architectural gap (reproducibility covers the data but not the fit config), a
couple of boundary-hygiene items, and a set of **intentional-for-parity choices
that should be tracked so "parity-faithful" doesn't calcify into "permanently
un-fixable."** I'll say clearly where the code is simply excellent.

Findings new to this review are **[NEW]**; carried ones are **[SYS]**. Where I'm
reconstructing exact prior parameters / NUTS settings / index conventions from
memory, I say so — those are parity-verified against the reference even where I
can't independently check them.

---

## Package role

The inference library: a pure NumPyro model plus the machinery to feed it
(`data`), fit it (`sampling`, `batch`), reduce it (`parity`), and project it
(`nowcast`). Its defining constraint is **fidelity to a frozen PyMC reference** —
"parity, not novelty, defines done." Its one architectural rule (imports only
`jax`/`numpyro`/`numpy`, never an `nfp_*` package) is the seam that makes "the
model never sees a `vintage_date`" a structural fact. Importing the package
enables JAX float64 globally.

---

## `model.py`

The crown jewel, and it earns it. Non-centered reparametrizations for geometry,
funnel-avoiding `LogNormal` scale priors, a robust (Student-t) QCEW likelihood, a
contiguous-vintage-indexed CES noise vector, per-provider AR(1)-or-IID error
models switched by config, and a data-gated cyclical block — all with site names
chosen to match the reference key-for-key (a smart choice that makes parity a
dict comparison). The findings are modeling *notes*, not defects.

**[NEW] MOD-3 (Medium as a modeling note; low practical severity) — `phi =
jnp.minimum(phi_raw, p.latent.phi_cap)` clips persistence inside the model.** The
prior is `phi_raw ~ Beta(...)` (I recall ~`Beta(18, 2)`, mean ≈ 0.9) and the clip
caps it (≈0.99). Two consequences: a non-differentiable kink at the cap (zero
gradient above it, a flat direction for NUTS), and a posterior **atom** at exactly
the cap (all mass above collapses there) — a modeling artifact rather than a
belief. This is **parity-faithful** (the reference clips identically) and
**practically harmless** for this application (employment-growth `phi` sits well
below the cap, so it almost never binds in the posterior). But clipping inside a
model is a known anti-pattern; the principled form is a prior with bounded support
(a scaled Beta, or a soft cap). **Action:** leave it for parity, but put it on the
post-parity backlog so it's revisited when the model is allowed to diverge from
the reference.

**[SYS, H-4a, model side] MOD-7 (Low–Medium) — the cyclical block is silently
data-dependent.** `active = data.get('cyclical_active')` builds the `phi_3` sites
only for covariates with nonzero arrays; if the indicators were absent upstream
(the ingest H-4a footgun), `active` is empty and `phi_3` is **never sampled**, so
the *model structure itself* changes with the input data and the posterior carries
no flag that the cyclical block was dropped. Mitigation that already exists: the
active set is recorded in the **snapshot meta**, so a consumer working from the
snapshot can detect it — but a consumer working from the posterior alone cannot.
**Action:** consider emitting a `deterministic` site (or copying `cyclical_active`
into the fit result) so "this fit had no cyclical block" is visible downstream
without the snapshot.

**[SYS, H-3, model side] MOD-8 (Low) — the model never reads the BD covariates.**
`birth_rate` / `bd_proxy` / `bd_qcew_lagged` are reconstructed by `from_snapshot`
and ignored by the model. The model-side half of the fix is to stop
reconstructing them in `data.py` once ingest stops producing them.

**What's simply excellent here.** The non-centered AR(1) and Fourier-GRW
reparametrizations; the `LogNormal` scale priors; the Student-t QCEW likelihood;
the contiguous CES-vintage sigma indexing; the per-provider error-model switch.
The `if len(data['ces_sa_obs']) > 0:` Python branch is correct in **both** the
unbatched path (real length) and the vmap path (padded length is concrete under
vmap, masked slots contribute zero) — not a tracer-in-conditional bug. The
seasonal `month_of_year` indexing and the `jnp.take(g_total_nsa, qcew_obs)`
gather are parity-verified; I can't independently check the month convention or
`take`'s OOB behaviour, but the padded-index slots are mask-protected and the
log-density equality test would catch an off-by-one.

---

## `config.py`

`ModelPriors` as nested **frozen** dataclasses with every default pinned to the
reference, plus `PRESETS` for sampler budgets. Clean and appropriate. One
architectural finding emerges here but is really about the whole package:

**[NEW] MOD-1 (Medium — the one genuine, fix-now gap) — the reproducibility
boundary covers the data but not the fit.** The architecture's central selling
point is the content-hashed snapshot as the reproducibility anchor — but
`content_hash` covers only the **data arrays + data meta**. A posterior is a
function of `(data, priors, preset, seed)`, and three of those four are *not*
part of the snapshot identity. So two fits with different priors (or a different
seed) on the same data produce the **same snapshot hash** and **different
posteriors**, and the artifact that's supposed to pin reproducibility doesn't
pin the fit. The parity scripts presumably fix priors/preset/seed by convention,
but that lives in the harness, not in any content-addressed artifact. **Action:**
record (and ideally hash) the fit config alongside the data — e.g. a
`fit_manifest` carrying the `ModelPriors`, the preset, and the PRNG seed next to
the snapshot hash — so a posterior is fully reproducible from artifacts, not from
remembering which script invoked it. This is *not* a parity concern, so it should
be fixed now rather than deferred; it shores up the architecture's headline
claim. (Hedge: confirm against `sampling.py` whether the seed is already
threaded/recorded — if it is, this reduces to "surface priors+preset+seed in one
manifest.")

---

## `data.py`

The intake. `from_snapshot` (strip `__`-keyed globals, reconstruct provider
arrays from `{name}__g_pp`, default `error_model='iid'` for v1 snapshots) and
`model_inputs` (normalize to the model's data dict). The v1→v2 fallback is a
genuine strength — schema evolution handled with a tested backward path.

**[NEW] MOD-2 (Low–Medium) — `model_inputs` trusts the snapshot's internal
consistency.** `content_hash` verifies the snapshot wasn't corrupted *in transit*,
but not that it's internally *consistent* — a snapshot produced by a buggy
`build_model_data` (e.g. `qcew_obs` longer than `g_qcew`, or an obs index ≥ T)
would hash fine and then fail deep inside the model with an opaque shape error.
This is the model-side instance of the recurring "fail loud at the boundary"
theme. **Action:** a few cheap asserts in `model_inputs` (obs-value/obs-index
lengths match; indices in `[0, T)`; provider arrays aligned) localize the failure
to the boundary. **Also [SYS]:** `from_snapshot` depends on provider names being
`__`-free (the same invariant `batch.py` asserts and `collect_snapshot` doesn't);
mirror the assertion at the mint site.

---

## `batch.py`

The most impressive engineering in the codebase, and it holds up to scrutiny. The
padding-to-max-T with per-date masks is *proven* posterior-invariant by a
log-density equality test (padded latents are extra prior-only `N(0,1)` draws
touching no likelihood; padded obs slots masked to exactly zero), and the modest
CPU speedup was reported honestly. The closure-in-loop binding (`_n=name`,
`_j=j`) correctly dodges late-binding.

**[NEW] MOD-6 (Low) — the batch assumes a common time axis, and padding waste
grows with backtest span.** The static/batched split puts `T`, `month_of_year`,
and `cyclical_active` in the *static* dict (not vmapped), which is only correct
if every date in the batch shares one epoch and calendar — i.e. all dates sit on
a common time axis and each as-of date *masks* its future, rather than each
starting at its own month 0. That's exactly the right framing for a backtest, and
the equality test confirms the real region matches the unbatched model — but it's
a real constraint (the batch can't mix series with different epochs/calendars),
worth a docstring line. The performance corollary: the earliest date in a
wide-span backtest is padded up to the latest date's `T`, so it wastes
`1 − T_early/T_late` of its compute on prior-only padding. On GPU (lock-step)
this is free; on CPU a very wide span can erode the vmap win (the plans measured
≈1.4× and flagged this). **Action (situational):** if CPU backtests over wide
spans become routine, bucket dates by similar `T` and pad within buckets.

---

## `parity.py`

Statistically literate and honestly documented. `collect_parity_arrays` gathers
scalars, paths, and per-provider params (by prefix); `compare_reduced` uses
**MCSE-scaled z-tests** plus ESS-derived bounds and a **kurtosis-aware SD-ratio
escape hatch** for the reference's poorly-mixing centered-GRW scales. This is the
right toolkit. Two observations about the *gate's power*, not its correctness:

**[NEW] MOD-4 (Low — gate-power note) — the parity gate is weakest exactly where
a port bug is most likely.** Both mechanisms loosen the comparison for
poorly-mixing parameters: an MCSE-scaled z-test is automatically lenient when
MCSE is large (low ESS ⇒ wide tolerance), and the SD escape hatch explicitly
widens the band for the badly-mixing `sigma_fourier`-family. Statistically this
is correct — you can't demand tight agreement on a poorly-estimated quantity —
but it means the gate has the least power on the parameters most prone to harbor
a translation error. The escape hatch is well-reasoned (the reference's
centered-GRW geometry, not the v2's), but it *is* a discretionary loosening.
**Action:** corroborate the loosely-gated parameters with a **posterior-predictive
check** (the predictive is well-identified even when the marginal mixes badly), so
the gate doesn't rest solely on the widened marginal bands.

**[SYS, L-11] trivial:** the `'lam_ces'` exclusion in `collect_parity_arrays` is
dead (the CES loading site is `lambda_ces`, collected via `SCALAR_VARS`); the
provider sites are `lam_<name>` while CES is `lambda_ces` — cosmetic naming
inconsistency. Delete the dead entry.

---

## The architectural boundary test, and the import side effect

**[SYS, M-7] MOD-5a — the boundary test is line-based, not AST-based.**
`test_model_unit.py::TestBoundary` is the *only* enforcement of the
"no `nfp_*` imports" rule (the workspace installs all packages together, so
pyproject deps don't enforce it at runtime), which makes its line-based scan
worth hardening. It catches `import nfp_` / `from nfp_` at line-start but misses
dynamic imports (`importlib.import_module('nfp_ingest')`) and `import numpy,
nfp_ingest`-style lines. **Action:** walk the AST for `Import`/`ImportFrom` and
flag `import_module(` string args.

**[NEW] MOD-5b (Low) — importing `nfp_model` flips JAX to float64 process-wide.**
This is *necessary* (float32 would lose precision in the log-growth cumulation and
the likelihood) and is documented, but it's a global side effect of a plain
`import`: any other JAX code in the same process silently gets float64 too. Fine
for this pipeline (nfp_model is the main JAX user), but the kind of thing that
surprises — keep it prominently flagged.

---

## Test coverage

The best-tested package. The boundary, the model's site/shape structure, the
AR1-vs-IID provider branch, the from_snapshot round trip + v1→v2 fallback, and —
crucially — the **batch log-density equality** all run on synthetic data in CI.
The gaps are the familiar ones: the **golden-master parity** (the package's actual
definition of done) is S3/credential-gated and so invisible to CI (the system
review's H-2), and `nowcast.py`'s arithmetic is not something I can see being
exercised this turn. So CI verifies the model's *structure* and the batch's
*correctness* but not parity and not the headline-number projection.

---

## Prioritized findings (package-scoped)

| ID | Sev | Location | One-liner |
|----|-----|----------|-----------|
| **MOD-1** | Medium | `config`/architecture | Reproducibility anchor (content hash) covers the data, not `(priors, preset, seed)`; same hash can map to different posteriors. **[NEW]** |
| **MOD-3** | Medium (note) | `model.phi` | Clips persistence inside the model (kink + posterior atom); parity-faithful, rarely binds; post-parity backlog. **[NEW]** |
| **MOD-2** | Low–Med | `data.model_inputs` | No internal-consistency check on the snapshot; a valid-hash-but-malformed snapshot fails deep in the model. **[NEW]** |
| **MOD-7** | Low–Med | `model` cyclical gate | Cyclical block silently data-dependent; active set is in snapshot meta but not the posterior. **[SYS]** |
| **MOD-4** | Low (note) | `parity.compare_reduced` | Gate is weakest (MCSE-lenient + SD escape hatch) exactly where a port bug is likeliest; add predictive corroboration. **[NEW]** |
| **MOD-5a** | Low | `TestBoundary` | The sole boundary guardrail is line-based, not AST-based. **[SYS]** |
| Low | Low | `model`/`data`/`parity` | x64 import side effect (MOD-5b); BD arrays reconstructed-but-unread (MOD-8); batch common-axis assumption + wide-span padding waste (MOD-6); dead `lam_ces` (L-11). **[NEW/SYS]** |

---

## Synthesis

**What's good — and it's a lot.** `nfp-model` is the most carefully engineered
package in the workspace. The model is principled in every choice I can see; the
batching is *proven* correct rather than merely tested; the parity machinery is
statistically literate and honest about its own escape hatches; the architectural
boundary is a structural fact, not a convention; and the schema-evolution path is
handled with a tested fallback. The CI-visible tests genuinely verify the model's
structure and the batch's correctness. Where this package is excellent, it is
excellent, and most of the surface I can see has no defect.

**The pattern under the findings.** This package's quality *is* its constraint: it
is a faithful port of an imperfect reference, and the same discipline that makes
parity work — freezing every choice — also **preserves the reference's flaws**
(the `phi` clip, the badly-mixing centered-GRW geometry) and even **loosens the
verification gate** where the reference is worst (MCSE-lenient z-tests + the SD
escape hatch). None of that is wrong for the parity phase. The risk is purely
temporal: "parity-faithful" is a reason to *defer*, and deferred items calcify if
nobody tracks them. The plans anticipate a post-parity novelty phase; the
discipline to capture now is a **tracked backlog** of the inherited choices so
they get revisited when the model is allowed to diverge.

The one finding that is *not* parity-shaped — and therefore should be fixed now,
not deferred — is **MOD-1**: the content-hashed snapshot, the architecture's
headline reproducibility mechanism, identifies the data but not the fit. That's a
real gap in the central claim, and it's cheap to close.

**Top changes for `nfp-model`, in order.**

1. **Close the reproducibility gap (MOD-1):** record/hash `(priors, preset,
   seed)` in a fit manifest beside the data snapshot, so a posterior is fully
   reproducible from artifacts. Fix-now; not parity-gated.
2. **Validate snapshot internal consistency at the boundary (MOD-2)** — the
   model-side fail-loud — and mirror the `__`-free provider-name assertion.
3. **Make the boundary test AST-based (MOD-5a)** — it's the only guardrail.
4. **Start a tracked post-parity backlog** for the inherited choices (MOD-3 `phi`
   clip, MOD-4 gate power / centered-GRW geometry, MOD-7 silent cyclical gate) so
   they're revisited when novelty begins, and add predictive corroboration for the
   loosely-gated parameters.
5. **Tidy:** stop reconstructing the dead BD arrays in `data.py` (MOD-8), flag the
   x64 import side effect prominently (MOD-5b), delete the dead `lam_ces` entry.

**Bottom line.** `nfp-model` is the package to hold up as the standard for the
rest of the codebase: principled model, proven batching, literate parity,
enforced boundary, tested schema evolution. Its weaknesses are a genuine but
cheap-to-fix reproducibility gap and a set of intentional, parity-driven
deferrals whose only real danger is going untracked. The remaining
correctness-sensitive unknown is `nowcast.py`'s projection arithmetic, which I
couldn't see this turn.

---

## Deferred to next pass (retrieval permitting)

- **`nowcast.py`** — the highest-value remaining target in this package: the
  predictive draw, the growth-path → level reconstruction, and the headline
  jobs-added computation. A wrong cumulation, a wrong base level, or an SA/NSA
  mix-up here produces wrong public-facing numbers with no upstream guard.
- **`sampling.py`** — the NUTS configuration (`target_accept`, `max_tree_depth`),
  the seed threading and `FitResult` shape (directly relevant to MOD-1), and
  divergence handling.
- The full `CLAUDE.md` and the remaining test bodies (`test_batch.py`,
  `test_parity_golden.py`).

Re-run project-knowledge retrieval and I'll complete these — `nowcast.py` first,
since it's the one place in this otherwise-excellent package where an unreviewed
arithmetic bug would reach the headline number.