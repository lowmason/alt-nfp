# alt-nfp (v2) — Whole-Codebase Review

**Scope.** System-level review of the five-package `uv` workspace. This is the
architectural / cross-cutting pass; the per-package deep dive is deferred to a
follow-up. Findings here favour the *seams between packages*, the *load-bearing
invariants*, and the *control/data flow*, because that is where systemic issues
live.

**Method & honest caveat.** I reviewed the source by retrieval over the project
knowledge base, not a full checkout. Coverage is broad and centred on the
load-bearing code (the censoring selection, the snapshot/hash boundary, the
NumPyro model, the vmap batching, the parity machinery, the CLI, the download
layer, the data→model transform) plus all configuration and the `plans/` design
record. I did **not** read every line of every file. In particular I inferred
`.github/workflows/ci.yml` and the root `conftest.py` from the CLAUDE.md /
`plans/` descriptions rather than the YAML/py itself, and I did not yet read
`sampling.py`, `nowcast.py`, `compositing.py`, `aggregate.py`, `views.py`,
`evaluation.py`, `indicators.py`, the `processing/` modules, or the
ingest-side `ces_national.py` / `qcew.py` at the source level. Where a claim
rests on inference rather than the code in front of me, I say so. Those files
are the right target for the per-package pass.

---

## Phase 1 — Orientation (architecture as it actually is)

The provided sketch is accurate; this section records what I verified and the
few places reality has drifted from the sketch.

### Components and the dependency graph

A virtual workspace root (`alt-nfp`, not installable) over `packages/*`, with
`[tool.uv.sources]` pinning each member to `workspace = true`. The four data
packages form a strict linear chain; the model package sits apart.

```
nfp-lookups → nfp-download → nfp-ingest → nfp-vintages
                                 ⇣ (arrays/snapshots only — NO import)
                             nfp-model   (jax / numpyro / numpy only)
```

- **nfp-lookups** — foundation. `paths` (the single source of all filesystem
  layout; `_find_base_dir`, `_store_location`, `storage_options_for`,
  `is_remote`), `schemas`, `series_ids`, `revision_schedules`,
  `provider_config`, `benchmark_revisions`. Imports no other `nfp_*`.
- **nfp-download** — fetching only. `client.py` (httpx + a Chrome-impersonating
  curl_cffi session for `www.bls.gov`), `fred.py`, `bls/` (`_http`, `bulk`,
  `ces_*`, `qcew`), `release_dates/` (async scraper + HTML parser).
- **nfp-ingest** — the data engine. `vintage_store` (Hive parquet I/O + the
  two-layer as-of censoring), `panel`, `model_data` (`build_model_data(as_of=D)`
  — the "what was knowable on D" entry point), `snapshots` (the content-hashed
  `.npz` artifact), plus compositing/indicators/releases/tagger/aggregate.
- **nfp-vintages** — pipeline + `alt-nfp` CLI. `__main__` (Typer), `build_store`,
  `processing/`, `views`, `evaluation`.
- **nfp-model** — inference. `model` (the NumPyro model), `config`
  (`ModelPriors`/`PRESETS`, every default pinned to the frozen reference),
  `data` (`model_inputs`, `from_snapshot`), `sampling`, `batch` (vmap),
  `nowcast`, `parity`. Imports only jax/numpyro/numpy; importing it enables
  global float64.

### The central design choice (verified, and it holds)

The boundary between *knowability* and *inference* is **a serialized artifact,
not a function call**: a content-hashed `.npz` (arrays + embedded JSON meta).
This is load-bearing in the best sense — it is simultaneously the censoring
contract, the GPU/vmap-batching enabler, and the parity anchor. The
import-direction rule (`nfp-model` imports no `nfp_*`) is enforced by a test,
`tests/test_model_unit.py::TestBoundary`, which makes "the model never sees a
`vintage_date`" a structural fact rather than a convention.

### Control / data flow

`alt-nfp` orchestrates acquisition + store maintenance + snapshot baking as
idempotent, rarely-run steps. The model is a pure library invoked by `scripts/`
harnesses or any holder of a snapshot. Because the seam is a hash-pinned file,
the GPU/backtest loop never touches the network and failures localise cleanly.

### Drift from the sketch (minor, worth correcting)

1. **Bare `alt-nfp` runs one step further than documented.** The sketch and the
   vintages `CLAUDE.md` say bare `alt-nfp` runs `download → download-indicators →
   process → current`. The actual callback (`__main__.py`) also calls
   `build(None)`, so it runs through **`build`**. Given finding **C-1** below,
   that the headline command writes the store is exactly the thing that makes
   the missing guard dangerous.
2. **The BD model in the sketch is larger than the model in the code.** The
   sketch (and `technical_methods…docx`) describe birth/death with
   birth-rate and lagged-QCEW covariates; the actual `model.py` uses only the
   gated cyclical block (`claims`, `jolts`). `plans/5` is explicit that the
   reference *code* never wired `φ₁·X^birth`, and `plans/0` records that the
   `φ₂·BD^QCEW` proxy was pruned as indistinguishable from zero. See **H-3**.

Build/test/tooling, as configured: `uv sync` (+ `dev` group); `pytest` with
`testpaths = packages`, markers `network`/`slow`; `ruff` (E,W,F,I,B,C4,UP, line
100, `E501` deferred to black); `mypy` soft. CI (per CLAUDE.md/`plans/1`, not
read directly): checkout → setup-uv (cached) → `uv sync` → `ruff check .` →
`pytest -m "not network" --no-cov` on ubuntu-latest / 3.12.

---

## Phase 2 — Assessment by dimension

### Correctness & logic

The hardest correctness problem in the system — real-time vintage censoring —
is handled with unusual care. `transform_to_panel(as_of_ref=D)` applies
`vintage_date ≤ D` + `ref_date < D`, then rank-based selection
(`_select_ces_at_horizon`, `_select_qcew_at_horizon`) reconstructs the
triangular diagonal, with explicit frontier fallbacks and a fail-fast guard
(`_validate_censored_selection` rejects duplicate ref_dates, calendar gaps,
null/zero employment, null growth before anything reaches the sampler). The
growth-before-selection ordering (log-diff within revision cohort, *then* pick
one row per `(series, ref_date)` by recency rank) is subtle but internally
consistent and well-tested (`test_consecutive_ref_dates`, the quarter-rule and
fallback cases). I did not find a correctness bug in this layer; it is a
strength.

The snapshot identity is correct: `content_hash` hashes sorted
`(name, dtype, shape, raw bytes)` of every array plus canonical meta JSON, and
explicitly **not** the npz bytes (zip headers embed timestamps). `save`/`load`
are symmetric (meta's `content_hash` is popped before hashing on both sides),
and `load_snapshot` re-verifies and raises on mismatch. The documented
endianness caveat is acceptable for a single-arch research repo.

The model is a principled translation (non-centered AR(1) and Fourier-GRW for
geometry; LogNormal sigmas to avoid funnels; tiered QCEW noise; era means;
covariate gating). The vmap batching is *provably* posterior-invariant (padded
latents are prior-only `N(0,1)` touching no likelihood; padded likelihood slots
masked to exactly zero), and that invariance is tested by log-density equality,
not merely MCMC agreement. The closure-in-loop binding in `batch.py`
(`_n=name`, `_j=j`) correctly avoids the late-binding trap.

The real correctness *risks* are not in the math — they are at the data-quality
boundaries, where the system tends to produce a **valid-but-wrong** artifact
instead of failing loud (findings **C-1**, **H-4**).

### Architecture & design

This is the codebase's strongest dimension. Concerns are physically separated;
the artifact boundary is enforced by a test; paths are centralised in
`nfp_lookups.paths` and threaded as one object that is `Path` locally and
`UPath` on S3, with `storage_options_for` handling the Polars side. The
asymmetry that *does* exist is telling: `append_to_vintage_store` is a careful,
append-only, anti-join-deduped writer, while `build_store` — sitting in a
different package — bulldozes partitions. The store has *two* writers with
*opposite* safety postures (**C-1**, **M-6**).

### Readability & maintainability

Clean, consistently documented, user conventions mostly honoured (single
quotes, method-style Polars expressions in the newer modules). The `plans/`
gate logs + per-package `CLAUDE.md` + `specs/`/`archive/` form unusually good
connective tissue: bugs are recorded *with their fixes*, decisions carry
"do not reopen without evidence," gates have status annotations with concrete
numbers. The maintainability drag is *vestigial scope from the port*: built-but-
unused arrays (**H-3**), dead config (**L-10**), a dead exclusion name
(**L-11**), and sketch/code drift (**L-9**).

### Testing

Test *quality* is high where it runs: the censoring selection, the snapshot
hash, the model's site/shape/log-density structure, and the padding-equality
proof all assert meaningful behaviour on synthetic data, and the negative
masters (unbuildable horizons must raise) are a nice touch. The systemic
problem is *what CI can reach* (**H-2**): in local mode with no credentials,
every store-dependent test and every golden-master parity check self-skips, so
the project's actual definition of "done" — parity against the frozen reference
— is never exercised by automation.

### Performance

Appropriate for the workload. `build_panel(as_of_ref=D)` collects the lazy
frame to do the rank selection — a necessary materialisation, and amortised by
the snapshot (build once per `as_of`, pin the hash, reuse). The vmap speedup
was *measured honestly* (≈1.4× on CPU; the lock-step tree-doubling waste is
free on GPU, which is the deferred lever) rather than asserted. No hot-path
concern rises to "fix this now."

### Security

Low surface (a research/inference repo, not a service). Secrets come from a
gitignored `.env` via `conftest`/CLI; the BLS registration key is appended only
to `bls.gov` URLs. No injection surface of note. The one thing to keep honest:
`data/` is proprietary and the repo is public, so the gitignore discipline and
the "tests self-skip without the store" pattern are doing real work — keep them.

### Dependencies & tooling

Reproducible (committed `uv.lock`, pinned dev tooling). Two consistency issues:
the `requires-python` floors disagree with the actual 3.12 contract (**M-8**),
and `numpy` is bounded (`<2.4`) in two packages but unbounded in `nfp-model`,
an inconsistency the single lockfile currently papers over.

---

## Phase 3 — Findings by severity

Severity reflects blast radius × likelihood, given that this is a single-
maintainer research system (not a multi-tenant service).

### CRITICAL / top of HIGH

#### C-1 — `build_store` can silently and irreversibly destroy the canonical store
**Location:** `packages/nfp-vintages/src/nfp_vintages/build_store.py`
(partition wipe loop + `out_path = store_path or VINTAGE_STORE_PATH`);
`packages/nfp-vintages/src/nfp_vintages/__main__.py::build` calls
`build_store(releases_path=...)` with **no** `store_path` override.

**What's wrong.** `build_store` writes to the canonical store by default and
clears each partition before rewriting:

```python
# build_store.py
out_path = store_path or VINTAGE_STORE_PATH        # default = canonical store
...
if partition_dir.exists():
    for f in partition_dir.glob('*.parquet'):
        f.unlink()                                  # wipe, then rewrite
```

The root `CLAUDE.md` states the hazard in its own words:

> **Never rebuild the canonical store in place.** `s3://alt-nfp/store` contains
> live-captured release-day vintage rows … that exist in no raw input — a
> from-scratch `alt-nfp build` to that URI would silently destroy them.

The `plans/0` A0 gate confirms those rows are *irreproducible* (≈64
national-headline CES rows present in no raw input, live-captured Mar 2025–Jan
2026). So the danger is acknowledged — but the only mitigation is a comment.
`alt-nfp build` is the most natural command, the default `NFP_STORE_URI` is the
canonical store, and bare `alt-nfp` now runs `build` (Phase 1, drift #1).

**Why it matters.** One run of the headline pipeline command against the
documented default configuration silently deletes the only copy of months of
irreplaceable data. "The author currently remembers the rule" is not a
safeguard. This is the single most dangerous operation in the repo.

**Fix (any one; the third is cleanest).**
1. Refuse the canonical URI without an explicit opt-in:
   ```python
   if is_remote(out_path) and str(out_path).rstrip('/').endswith('/store') and not allow_canonical:
       raise RuntimeError(
           'refusing to rebuild the canonical store in place; '
           'write to a scratch prefix (…/store-rebuild) or pass allow_canonical=True'
       )
   ```
2. Write to a staging prefix and require an explicit promotion step.
3. **Best:** make `build_store` append + dedup instead of wipe + rewrite —
   reuse the semantics `append_to_vintage_store` already implements correctly,
   so the two store writers stop disagreeing about safety.

### HIGH

#### H-2 — The parity gates (the project's definition of "done") are not enforced in CI
**Location:** `.github/workflows/ci.yml` (per docs: `pytest -m "not network"
--no-cov`, local mode); the env-gating in `test_golden_masters.py` /
`test_parity_golden.py` (`pytestmark = skipif(not _golden_available())`,
`NFP_A3_PARITY=1` opt-in).

**What's wrong.** CI runs without `NFP_STORE_URI` and without S3 credentials, so
*every* store-dependent test and *every* golden-master/parity check self-skips.
The A3/A4 parity is run only by hand (`scripts/run_a3_parity.py`,
`run_a4_backtest.py`) and by env-gated tests. A change that breaks parity — a
prior tweak, a censoring-rank edit, a noise-multiplier change — passes CI as
long as the synthetic unit tests still pass.

**Why it matters.** "Parity, not novelty, defines done in Phase A" is the
stated contract, yet CI structurally cannot see it. The safety net for the core
invariant is human discipline plus manual script runs. (Understandable — CI has
no data/credentials and 14 MCMC fits don't belong in pytest — but the gap should
be explicit and partially closed.)

**Fix.** Commit a small, credential-free synthetic fixture and add a fast
end-to-end check that runs `build_model_data`-shaped inputs (or a frozen tiny
ModelData) → `fit_model(light)` → `collect_parity_arrays` and asserts against a
committed golden reduction at a generous tolerance. Optionally add a *scheduled*
CI job with read-only S3 creds for the real golden masters. At minimum, document
in CI output that parity is not covered.

#### H-3 — Dead covariate arrays flow through the entire data → snapshot → model pipeline
**Location:** `packages/nfp-ingest/src/nfp_ingest/model_data.py` (computes
`birth_rate`, `bd_proxy`, `bd_qcew_lagged` and returns them);
`snapshots.py::GLOBAL_ARRAY_KEYS` (serialises all three into every `.npz`);
`nfp_model/data.py::from_snapshot` (rebuilds them); `model.py` (**never reads
them** — confirmed; `plans/5` acknowledges it).

**What's wrong.** Three arrays are computed, serialised into every snapshot, and
reconstructed on intake, but the model consumes none of them. They correspond to
terms that were pruned (`φ₂·BD^QCEW` "indistinguishable from zero," `plans/0`)
or never implemented (`φ₁·X^birth`, `plans/5`). `bd_proxy = g_qcew - g_pp_avg`
and the lag loop run on every build for no consumer.

**Why it matters.** Wasted computation and larger snapshots are minor; the real
cost is a *misleading contract*. The snapshot schema implies these covariates
matter to inference. A future maintainer wiring `phi_3` to `bd_qcew_lagged`
would reasonably assume the array is meaningful and censored correctly. Parity-
faithfulness ("the reference built them too") is a reason they exist, not a
reason to keep shipping them.

**Fix.** Drop them from `build_model_data`'s return and from
`GLOBAL_ARRAY_KEYS`, or move them to a clearly-labelled diagnostics sidecar that
never enters the model path. Bump `SCHEMA_VERSION` and keep a v2-read fallback
(the codebase already does v1→v2 fallbacks, so this is in-pattern).

#### H-4 — Silent degradation at data-quality boundaries (valid-but-wrong, not error)
**Location (a):** `model_data.py` cyclical load + censor. **Location (b):**
`packages/nfp-download/src/nfp_download/release_dates/scraper.py::parse_index_page`.

**What's wrong.**
(a) If the cyclical indicator parquets are missing or mispathed, the loaded
arrays are all-zero, the model's gating drops them, and `phi_3` is **silently
never sampled**. This is not hypothetical — `plans/5` records exactly this
footgun ("the reference posterior silently lacks the φ₃ cyclical block" under the
wrong `indicators` path). The censoring sentinel makes it worse: censored months
are set to `0.0`, the same value as a legitimately-zero centred indicator, so
"censored" and "present-but-zero" are indistinguishable.
(b) The calendar scraper parses BLS HTML heuristically (find `h4` year → next
`ul` → `li` → anchor matching an href regex). A page-structure drift makes
`parse_index_page` return an empty/partial list with **no error**, feeding wrong
vintage dates into the censoring layer. The *fetch* path degrades gracefully
(catches the 403/`FetchError`, falls back to cached pages — good), but the
*parse* path has no "found at least N entries" sanity check.

**Why it matters.** For a system whose entire value is correctness-under-
censoring, the default at every data boundary should be fail-loud. The censoring
*selection* already does this beautifully (`_validate_censored_selection`); that
discipline simply needs to extend outward to the inputs that feed it.

**Fix.** Assert non-empty cyclical arrays when `as_of` post-dates the lag
horizon (or at least `warnings.warn` when a configured covariate loads all-zero);
assert expected cardinality / monotonic coverage on calendar parse; consider a
distinct sentinel (NaN, as births already use) for censored cyclicals so it
can't collide with a real zero.

### MEDIUM

#### M-5 — `get_with_retry` retries on HTTP status only, not transport exceptions
**Location:** `packages/nfp-download/src/nfp_download/client.py`.

```python
for attempt in range(max_retries):
    r = client.get(url, timeout=timeout, params=params)   # raises on timeout/reset → no retry
    if r.status_code == 429 or r.status_code >= 500:
        time.sleep(min(2**attempt, 120)); continue
    r.raise_for_status(); return r
```

A `ConnectTimeout` / `RequestException` / reset propagates immediately, despite
this being the retry helper. Transient transport blips are exactly what retry
exists to absorb. Tolerable for a rarely-run pipeline, but easy to fix: wrap
`client.get` in `try/except` over the httpx and curl_cffi transport exception
types and back off on those too (cap the attempts the same way).

#### M-6 — `append_to_vintage_store` and `compact_partition` disagree on the vintage_date tie-break
**Location:** `packages/nfp-ingest/src/nfp_ingest/vintage_store.py`.
Both use a uniqueness key that **excludes** `vintage_date`. `append` keeps the
*existing* row (first-seen vintage wins — correct for a vintage store);
`compact_partition` keeps `max(vintage_date)` (last wins). If two rows with
identical `(ref_date, geo, industry, revision, benchmark_revision)` but
different `vintage_date` ever coexist across fragments, the two operations
resolve them *oppositely*. In practice append prevents the second write so they
shouldn't coexist — but the latent inconsistency is real and undocumented.
**Fix:** make both keep the earliest `vintage_date` (the first time the print
was observed) and document the rule on the uniqueness key.

#### M-7 — The import-boundary test is line-based, not AST-based
**Location:** `packages/nfp-model/tests/test_model_unit.py::TestBoundary`.
It scans stripped lines for `import nfp_` / `from nfp_` (exempting
`nfp_model`). This catches the realistic case, but misses dynamic imports
(`importlib.import_module('nfp_ingest')`) and `import numpy, nfp_ingest`-style
lines. Because the workspace installs all packages together, the pyproject deps
don't enforce the boundary at runtime — this test is the *only* guardrail, which
makes its line-based nature a (minor) weakness. **Fix:** walk the AST for
`Import`/`ImportFrom` nodes and also flag `import_module(` string arguments.

#### M-8 — `requires-python` floors contradict the actual 3.12 contract
**Location:** root + `nfp-model` declare `>=3.12`; `nfp-lookups`/`-download`/
`-ingest`/`-vintages` declare `>=3.10`; several `CLAUDE.md` say "requires >=
3.10"; the lockfile is `requires-python >=3.12`; the prose says "3.12
throughout." Net effect today is benign (resolves to 3.12), but the per-package
floors are untested fiction — installing `nfp-ingest` standalone on 3.11 would
"work" by declaration but is never exercised. Also `numpy>=1.24.0,<2.4` in
lookups/ingest vs unbounded in nfp-model. **Fix:** set every package to
`>=3.12` to match the real contract and align the numpy bounds.

### LOW / nits

- **L-9** Sketch/doc drift: bare `alt-nfp` runs through `build`, not `current`
  (callback calls `build(None)`); the vintages `CLAUDE.md` command list also
  omits `build` from the bare run.
- **L-10** Coverage is configured elaborately in `[tool.pytest.ini_options]
  .addopts` (five `--cov` trees + html), but every documented invocation (CI,
  local fast/full) passes `--no-cov`. Coverage is never actually produced —
  effectively dead config.
- **L-11** `parity.py::collect_parity_arrays` excludes a dead site name
  `"lam_ces"` (the model's CES loading is `lambda_ces`, collected via
  `SCALAR_VARS`); harmless residue. Relatedly, provider sites use `lam_<name>`
  while CES uses `lambda_ces` — cosmetic naming inconsistency.
- **L-12** Root `pyproject` `keywords` advertises `dynamax`, which is only a
  deferred option, not a dependency.
- **L-13** `BLSHttpClient.__init__` opens an `httpx.Client` with no visible
  `close()`/context-manager surface — minor connection-pool hygiene (cannot
  fully confirm without reading the rest of `_http.py`).

---

## Phase 4 — Synthesis

### Genuine strengths (not padding)

- **The artifact boundary is excellent and load-bearing in the right way.** A
  content-hashed `.npz` between knowability and inference is one decision that
  simultaneously *is* the censoring contract, the vmap/GPU enabler, and the
  parity anchor. The hash design (array bytes + canonical meta, never zip bytes)
  and the corruption-detection-on-load are correct.
- **The two-layer as-of censoring is the hard part of the problem, done with
  care.** Rank rules, quarter-dependent QCEW logic, frontier fallbacks, and a
  fail-fast validator that runs before the sampler. Exceptionally well tested,
  including negative masters. I did not find a bug here.
- **The model and its batching are principled.** Non-centered reparametrisations
  for geometry, funnel-avoiding sigma priors, tiered QCEW noise, gated
  covariates — and a vmap padding scheme that is *proven* posterior-invariant
  via log-density equality, not just MCMC agreement. The CPU speedup was
  measured and reported honestly.
- **The parity machinery is statistically literate.** MCSE z-tests, ESS-derived
  bounds, and a kurtosis-aware SD-ratio escape hatch whose rationale (the
  reference's centered-GRW scales mix poorly) is documented.
- **The written design record is unusually disciplined.** `plans/` gate logs
  with concrete numbers, per-package `CLAUDE.md` maps, and `specs/`/`archive/`
  make the system legible and record hazards explicitly.

### The systemic weakness (the pattern under the findings)

**The most important invariants are protected by documentation and discipline,
not by code.** The store-rebuild footgun (C-1), the unenforced parity contract
(H-2), and the silent covariate vanishing (H-4) are all *known* — they are
written down in the very files that describe the system — yet *unmitigated
mechanically*. The codebase is remarkable at *recording* its hazards and
surprisingly willing to leave them *armed*. The corollary is a recurring
**silent-degradation** posture at data boundaries: missing indicators → no
`φ₃`; parse drift → wrong calendar; `build` → data loss. The system tends to
emit a valid-but-wrong artifact rather than fail loud — which is striking,
because *inside* the censoring selection it already fails loud beautifully. The
fix pattern is consistent across nearly every high finding: **turn the hard
rules in CLAUDE.md into guards, assertions, and CI checks.** A secondary,
cosmetic pattern is *port residue* (dead arrays, dead config, sketch drift).

### The 3–5 highest-leverage changes, in order

1. **Guard `build_store` against the canonical store (C-1).** One small change
   neutralises the single most dangerous operation in the repo and stops the two
   store writers from disagreeing about safety. Do this first.
2. **Make the parity contract executable in CI (H-2).** A committed,
   credential-free fixture running `build_model_data`-shaped input →
   `fit_model(light)` → `collect_parity_arrays` against a small committed golden
   closes the gap between "parity is the definition of done" and "CI can't see
   parity." Without this, every other invariant rides on manual runs.
3. **Extend the fail-loud discipline outward to the input boundaries (H-4).**
   Assert non-empty cyclical load when `as_of` post-dates the lag horizon; assert
   calendar-parse cardinality; give censored cyclicals a NaN sentinel. This is
   the same instinct the censoring selection already embodies — just applied to
   what feeds it.
4. **Quarantine or drop the dead BD covariate arrays (H-3)** and bump
   `SCHEMA_VERSION`, so the snapshot contract stops implying covariates that the
   model ignores.
5. **Reconcile the version/dependency contract (M-8):** pin every package to
   `>=3.12` and align the numpy bounds, so declared floors match what is built
   and tested.

**Bottom line.** This is a strong, carefully engineered research codebase whose
*architecture and hard math are its best features* and whose *operational safety
rails are its weakest* — not because the author is unaware of the hazards (they
are documented with unusual honesty), but because the hazards have been written
down rather than wired shut. The highest-leverage work is almost entirely
"convert known invariants from prose into code," and most of it is small.
