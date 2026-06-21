# Package Review — `nfp-lookups`

**Scope this turn (read first).** Retrieval is still unavailable, and my
in-context coverage of this package is thinner than the others. I have:

- **`paths.py`** — good detail (the base-dir heuristic, the store-location /
  remote resolution, `storage_options_for`, the module constants). Reviewed
  directly below.
- **`pyproject.toml`** — complete.
- The **config layer's contract as observed at call sites** in nfp-ingest /
  nfp-vintages / nfp-model (e.g. `ProviderConfig.name`/`.error_model`,
  `get_noise_multiplier(key, rev)`, the QCEW revision schedule, `PANEL_SCHEMA`
  vs `VINTAGE_STORE_SCHEMA`). This lets me assess the *API and its consistency*,
  not the implementations.

**Deferred (source not seen — no correctness findings asserted):** the bodies of
`schemas.py`, `series_ids.py`, `revision_schedules.py`, `provider_config.py`,
`benchmark_revisions.py`, `__init__.py`, and the package's tests. These are
exactly the high-value targets for the next pass (see the risk list at the end);
a thin static-data package's bugs hide in the *values* (a wrong series-ID offset,
a wrong benchmark date, a wrong provider publication lag), and those need the
source.

**Calibration up front:** nothing in this package is Critical or High. It is the
soundest of the five. The findings are a genuine fragility in `paths.py`, an
implicit env-var contract, and one cross-cutting theme — the package's
would-be-canonical definitions are duplicated/re-encoded in consumers, so it
partially fails at its single job of being the source of truth, not because its
code is wrong but because the rest of the codebase routes around it. I'll say
where the code is simply fine.

Findings new to this review are **[NEW]**; carried ones are **[SYS]**.

---

## Package role

`nfp-lookups` is the foundation: the single source of filesystem/storage layout
(`paths`) plus the static reference data every other package needs (schemas,
series-ID construction, revision schedules, provider definitions, benchmark
dates). It imports no other `nfp_*` package — the one architectural invariant
here, and it holds. It has essentially no runtime behaviour beyond path
resolution; everything else is data. That low-risk profile is reflected in the
findings.

---

## `paths.py`

The most load-bearing module in the package: it's threaded into every other
package as the path/storage seam, and the system-level review already credited it
as the reason the S3 migration was clean. That credit stands — the
`Path`-locally / `UPath`-on-S3 duality with `is_remote` / `storage_options_for`
as the boundary is the right abstraction. The findings are about its edges.

**[NEW] L-1 (Medium) — `_find_base_dir` is correct only for editable/workspace
installs and degrades silently otherwise.** The resolution order is
`NFP_BASE_DIR` env first (good), then a walk up `Path(__file__).parents` looking
for a repo marker (`pyproject.toml` + `packages/`), then a fallback. Under the
project's actual setup (`uv sync`, editable), `__file__` lives at
`packages/nfp-lookups/src/nfp_lookups/paths.py`, so the walk-up finds the repo
root and everything works. Under a **wheel/site-packages install**, `__file__`
is in `site-packages/nfp_lookups/`, the walk-up finds no repo marker, and control
reaches the fallback — which resolves `BASE_DIR` to something unrelated to the
data (I recall it as either CWD or a fixed parent depth; either is wrong for a
non-editable layout, and either is **silent**). `plans/` itself flags this
("works by coincidence"). Because `BASE_DIR` then determines `DATA_DIR`,
`INTERMEDIATE_DIR`, and the `.env` location, a wrong base dir cascades. For the
one module whose entire job is correct path resolution, the fallback should
**fail loud** instead of guessing:

```python
def _find_base_dir() -> Path:
    if env := os.environ.get('NFP_BASE_DIR'):
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'pyproject.toml').exists() and (parent / 'packages').is_dir():
            return parent
    raise RuntimeError(
        'could not locate the repo root from {}; set NFP_BASE_DIR explicitly '
        '(this happens for non-editable installs)'.format(here)
    )
```

(Hedge: I'm reconstructing the marker set and the exact fallback line from
memory; confirm against the source. The *shape* of the issue — silent
degradation off the editable path, corroborated by the plans — is the finding.)

**[NEW] L-2 (Low–Medium) — the env-var contract is implicit and unvalidated.**
`NFP_STORE_URI` and the S3 credential variables (the MinIO-style
endpoint/key/secret) are read straight from `os.environ` inside
`storage_options_for`, with no check that, when `NFP_STORE_URI` points at
`s3://…`, the credentials are actually present. A missing credential surfaces
deep inside a Polars read/write as an opaque object-store error rather than at
config time. A small guard in the foundation module would localize one of the
most common operational misconfigurations:

```python
def storage_options_for(path) -> dict:
    if not is_remote(path):
        return {}
    opts = {k: os.environ.get(v) for k, v in _S3_ENV_MAP.items()}
    missing = [v for k, v in _S3_ENV_MAP.items() if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f'remote store {path} but missing env: {missing}')
    return opts
```

**[NEW] L-3 (Low) — path constants are import-time-bound to the environment.**
`VINTAGE_STORE_PATH` (and the local/remote `BASE_DIR` family) are resolved once,
at module import, from the env as it stands then. Setting `NFP_STORE_URI` or
loading `.env` *after* something has already imported `nfp_lookups.paths` has no
effect — the constants are frozen. The conftest/CLI handle this by calling
`load_dotenv()` before importing anything (and the dotenv-ordering bug the plans
fixed was exactly this class of problem), so it works in practice. Worth a
one-line docstring note that env/`.env` must be in place before the first import
of `paths`, since this is a non-obvious sequencing requirement that a new test or
script can trip over. (Alternatively, expose the store path via a small accessor
function rather than a module constant, so it's resolved at call time — but
that's a larger change and the import-time form is fine if documented.)

**What's simply fine here.** `is_remote` (string `s3://` prefix / `UPath`
protocol check) and the local-vs-remote constant resolution are clean; the seam
is well-placed; and the duality is genuinely good design. No change needed beyond
the edges above.

---

## The config layer (assessed by contract, not source)

I can't review the implementations this turn, but the *way these modules are
consumed* surfaces one real, cross-cutting theme and a couple of clean spots.

**[NEW] L-4 (Medium, cross-cutting) — the package's canonical definitions are
duplicated or re-encoded in consumers, undercutting its single job.** Two
observed instances:

1. **The QCEW revision schedule.** `revision_schedules.py` is, by name, the
   intended home of the `{Q1:4, Q2:3, Q3:2, Q4:1}` schedule. Yet nfp-ingest's
   `vintage_store.py` carries its own `_QCEW_MAX_REVISION = {1:4, 2:3, 3:2, 1:1}`
   constant (which I found is **dead**), *and* hardcodes the same schedule inline
   in the `_select_qcew_at_horizon` rank rules, *and* the test redefines it
   locally. So the schedule lives in (at least) three or four places and the
   selection logic — the code that most needs it to be right — doesn't import the
   canonical one. The abstraction exists; the "single source of truth" is
   fictional. (Hedge: confirm `revision_schedules.py` actually owns the QCEW
   schedule; if it owns only CES revision *timing*, this softens — but the
   duplication on the nfp-ingest side is real regardless.)
2. **Schema ownership.** `nfp-lookups` has a `schemas.py`, but the foundational
   `VINTAGE_STORE_SCHEMA` lives in `nfp_ingest.vintage_store`, and `PANEL_SCHEMA`
   appears to be referenced as the panel contract elsewhere. So schema
   definitions are split across the foundation package and a consumer. This is
   the cohesion seam behind nfp-vintages B-3 (the store is *written* in vintages
   without enforcing the schema *defined* in ingest): when the schema and its
   enforcement live in different packages from the writer, nobody enforces it.

**Fix for both:** make `nfp-lookups` the actual, imported source of truth —
move `VINTAGE_STORE_SCHEMA` into `schemas.py`, have `_select_qcew_at_horizon`
derive its rank rules from `revision_schedules`, and delete the duplicated
constants. This is low-effort and directly prevents the kind of drift that
already produced three divergent dedup tie-breaks (D-1) and a no-op constant.

**What looks clean by contract.** `ProviderConfig` is a well-used abstraction:
consumers read `cfg.name` and `cfg.error_model` (default `'iid'`), and that one
field cleanly drives the model's per-provider AR(1)-vs-IID branch — a tidy
data-driven switch. The `getattr(..., 'error_model', 'iid')` fallback at the
snapshot boundary is appropriately defensive. No concern from the usage side.

**Open question I can't resolve without source:** there appear to be *two*
noise-multiplier paths — a static `get_noise_multiplier(key, rev)` (used in
`model_data`, likely sourced here or in `revision_schedules`) and an empirical
`build_noise_multiplier_vector` in `nfp_vintages.evaluation`. If both feed the
model's QCEW/CES noise scaling, it's worth confirming they don't disagree or
double-apply. Flagging for the next pass, not asserting a bug.

---

## Dependencies & tooling

**[SYS, M-8] `requires-python = '>=3.10'` on the foundation package contradicts
the real `>=3.12` contract.** This is the most consequential instance of the
workspace's version drift because everything depends on `nfp-lookups`: its
declared floor is the floor the ecosystem advertises, yet the lockfile is 3.12
and CI runs 3.12, so the 3.10/3.11 floor is never built or tested. (Whether the
*code* needs 3.12 I can't verify without the source — `match`/`X | None` work on
3.10 — but the declared floor should match what's actually exercised.) Align it
to `>=3.12`. Same note on `numpy>=1.24.0,<2.4` here vs unbounded in nfp-model:
the single lockfile papers over it today, but standalone installs would diverge.

The dependency set is otherwise lean and appropriate (Polars, NumPy,
`python-dotenv`, and the `UPath`/object-store backend for the S3 path duality).

---

## Prioritized findings (package-scoped)

| ID | Sev | Location | One-liner |
|----|-----|----------|-----------|
| **L-1** | Medium | `paths._find_base_dir` | Base-dir heuristic is correct only for editable installs; falls back silently otherwise. Make the fallback fail loud. **[NEW]** |
| **L-4** | Medium | config layer (cross-package) | Canonical definitions (QCEW schedule, store schema) are duplicated/re-encoded in consumers; the "source of truth" isn't imported. **[NEW]** |
| **L-2** | Low–Med | `paths.storage_options_for` | Implicit, unvalidated S3 env contract; missing creds fail deep instead of at config. **[NEW]** |
| **M-8** | Low–Med | `pyproject` | `requires-python >=3.10` on the foundation package vs the real 3.12 contract. **[SYS]** |
| **L-3** | Low | `paths` constants | Import-time env binding; undocumented sequencing requirement. **[NEW]** |

---

## Synthesis

**What's good.** This is the cleanest package in the workspace. The path
abstraction is well-designed and well-placed, the one architectural invariant
(no inbound `nfp_*` imports) holds, `ProviderConfig` is a tidy data-driven
switch, and there is essentially no risky runtime behaviour to get wrong. I'm not
going to manufacture problems here — most of the module surface I can see is
simply fine.

**The one real theme.** A foundation package's value is being the single source
of truth, and `nfp-lookups` only partially is: its would-be-canonical
definitions (the QCEW revision schedule, the store schema) are duplicated or
re-encoded in the consumers that most need them, so the abstraction exists on
paper while the real logic carries its own copy. That's the upstream cause of
several downstream findings — the dead `_QCEW_MAX_REVISION`, the three divergent
store-write tie-breaks (D-1), and the unenforced store schema on write (B-3). The
fix is cheap and high-leverage *for the whole codebase*: make these definitions
live in `nfp-lookups` and be **imported**, not paraphrased.

**Top changes for `nfp-lookups`, in order.**

1. **Fail loud in `_find_base_dir`'s fallback (L-1)** — require `NFP_BASE_DIR`
   when the repo root can't be found, rather than silently guessing.
2. **Centralize the duplicated definitions (L-4):** move `VINTAGE_STORE_SCHEMA`
   into `schemas.py` and have nfp-ingest's QCEW selection import the schedule
   from `revision_schedules`; delete the duplicated constants. This retires the
   no-op constant and is the structural fix behind D-1/B-3.
3. **Validate the S3 env contract in `storage_options_for` (L-2)** and document
   the import-time env binding (L-3).
4. **Align `requires-python` to `>=3.12` (M-8)** and the numpy bound.

**Bottom line.** `nfp-lookups` is sound — the path layer is good design and the
config surface is clean where I can see it. Its weaknesses are a deployment-mode
fragility in base-dir resolution and an under-fulfilled mandate: it should be the
imported source of truth, and in a few load-bearing cases it isn't, which is what
lets the rest of the codebase drift. Low-effort fixes with outsized payoff
elsewhere.

---

## Deferred to next pass (retrieval permitting) — and where the real risk is

The config-module *internals* are where a static-data package actually hides
bugs, and they need the source:

- **`series_ids.py`** — BLS series-ID construction (survey + seasonal code +
  supersector/industry + data-type concatenation, with zero-padding). A wrong
  offset or pad produces a *valid-looking but wrong* series ID that silently
  fetches the wrong data. Highest-value target.
- **`revision_schedules.py`** — must be checked *against* the schedule hardcoded
  in `vintage_store._select_qcew_at_horizon`; any disagreement is a real
  censoring bug.
- **`benchmark_revisions.py`** — the CES annual benchmark dates that drive
  `benchmark_revision` tagging; wrong dates mis-tag rows.
- **`provider_config.py`** — provider publication lags drive the provider-data
  censoring horizon in `model_data`; a wrong lag censors at the wrong as-of.
- **`schemas.py`** — column dtypes; a dtype here disagreeing with what's written
  is the B-3 mismatch.

Re-run project-knowledge retrieval and I'll complete these — and reconcile
`revision_schedules` against the duplicated schedule, which is the one place a
config-vs-logic disagreement could be an actual bug rather than just duplication.