# Porting alt-nfp to Bloomberg — Operator Runbook

**Purpose.** A step-by-step, ordered checklist for standing up `alt-nfp` on Bloomberg
compute (real S3 + co-located GPU). The reader is the maintainer — every step names the
exact file, env var, and copy-pasteable command, plus a one-line **verify** check. The
current localhost MinIO (`AWS_ENDPOINT_URL=http://127.0.0.1:9000`) is a transitional S3
stand-in; on Bloomberg it becomes a real S3-compatible endpoint with the store and GPU
co-located.

**Principles.**
- **Build-here / validate-on-port.** Numeric work and code changes happen in this repo;
  Bloomberg is where you *validate* against the real provider store and GPU. Don't treat a
  green local run as a correctness certificate.
- **Parity ≠ correctness.** The frozen reference repo (`~/Projects/alt_nfp`) is a
  port-fidelity floor, not truth. Validate against external ground truth (published BLS /
  ALFRED), not the reference.
- **No GitHub Actions on Bloomberg.** `ci.yml` was removed; the only workflow left
  (`docs.yml`) is `workflow_dispatch`-only and irrelevant here. **Every gate** (ruff,
  pytest, interrogate, mkdocs) runs **manually** — see the [Manual gates appendix](#manual-gates-no-ci).
  Docs deploy is a local `mkdocs gh-deploy --force` to github.com.

**The spine: import-time env resolution.** `NFP_STORE_URI`, `NFP_DATA_URI`, and
`NFP_PROVIDERS_URI` resolve to `UPath` instances **at first `nfp_*` import**
(`paths.py:152`, `:214`, `:217`) — credentials are baked into the filesystem instance at
that moment. Only `NFP_SNAPSHOTS_URI` is resolved lazily (at use time,
`snapshots.py:41-62`). This is why **Step 1 (env in the container before any import) is a
hard precondition for every later step.** If a `*_URI` is unset, the resolver silently
falls back to local `./data`, which is **not writable** in the Bloomberg container.

---

## Prerequisites

| Requirement | Detail |
|---|---|
| Bloomberg compute account | GitHub mirror has **no Actions** — all gates manual. |
| Python ≥ 3.12 + `uv` | `requires-python = ">=3.12"` (`pyproject.toml:5`); 3.11 is rejected by uv at install. |
| JAX with CUDA | Install `jax[cuda12]` **before** `nfp-model` (see Step 4). The repo pins only `jax>=0.4.38` (CPU) — no CUDA extra exists. |
| GPU with native fp64 | Compute capability ≥ 6.0 (Pascal+). The model **requires** float64 (A3 parity contract); a card that throttles fp64 silently breaks correctness. |
| S3 bucket(s) + `AWS_*` creds | `s3://alt-nfp` (store + data + snapshots) and a **separate** provider store (Step 3d). |
| FRED API key | `FRED_API_KEY` — required to download cyclical indicators (claims, jolts). |
| BLS contact email | `BLS_CONTACT_EMAIL` (non-`github.com`) — Akamai 403s a bare UA on bulk LABSTAT files. Optional `BLS_API_KEY` for higher rate limits. |

---

## Ordered port checklist

### Step 1 — Secrets / `.env` and the four `*_URI` roots

**Files:** `conftest.py:12` (test-only `load_dotenv`), `paths.py:98-219`. The full env-var set
is enumerated in the block below (on Bloomberg set them as container env, not via a file).

Production library + CLI code does **not** call `load_dotenv` — only the test bootstrap
(`conftest.py:12`) and the two scratch scripts do. So on Bloomberg the env vars **must be
injected into the container environment** (container secrets / k8s env), not merely placed
in a `.env`. Set them all before the process starts any `nfp_*` import.

```bash
# Object-storage roots (all four required on Bloomberg)
export NFP_STORE_URI=s3://alt-nfp/store
export NFP_SNAPSHOTS_URI=s3://alt-nfp/snapshots
export NFP_DATA_URI=s3://alt-nfp
export NFP_PROVIDERS_URI=s3://alt-nfp-providers   # DECISION: confirm real Bloomberg URI (Step 3d)

# Credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
# AWS_ENDPOINT_URL — DECISION (see below)
export AWS_REGION=us-east-1                        # set if the endpoint is region-strict

# Data-acquisition keys
export FRED_API_KEY=...
export BLS_CONTACT_EMAIL=alt-nfp@example.com       # must NOT be a github.com address
# export BLS_API_KEY=...                            # optional, higher rate limits
```

**DECISION — `AWS_ENDPOINT_URL`:**
- **Real AWS-native S3** → leave `AWS_ENDPOINT_URL` **unset**; boto/s3fs use the default
  endpoint resolution.
- **Bloomberg-hosted S3-compatible endpoint** → set it (HTTPS). `aws_allow_http` is only
  auto-set for `http://` endpoints (`paths.py:145-146`), so an HTTPS endpoint needs no
  extra flag. `AWS_REGION` is passed to Polars object_store but **not** to the UPath/s3fs
  instance (`paths.py:135`) — if the endpoint is region-strict, also export
  `AWS_DEFAULT_REGION` so s3fs picks it up.
- If Bloomberg uses IAM role / instance credentials, `AWS_ACCESS_KEY_ID`/`SECRET` may be
  left unset — they're only added to the options dict when non-None (`paths.py:139-141`),
  and s3fs falls back to the boto credential chain.

**Verify:**
```bash
python -c "from nfp_lookups import paths; print(paths.VINTAGE_STORE_PATH, paths.PROVIDERS_DIR); \
print(getattr(paths.VINTAGE_STORE_PATH,'protocol',None))"
# Expect s3 URIs and protocol == 's3', NOT a local data/store path.
```

---

### Step 2 — Stand up the vintage store on S3 (rebuild → promote)

**Files:** `scripts/bootstrap_store.py`, `rebuild_store.py:212-289` (`write_rebuild_store`),
`build_store.py:54-60` + `mirror_store.py:38-44` (`is_canonical_store` guards),
`paths.py:71-95`. CLAUDE.md:82-96.

On Bloomberg the store is **rebuilt fresh** from public BLS APIs (CES triangular CSVs +
QCEW live CEW API) — it is **not** copied from any pre-existing MinIO snapshot. All raw
downloads route to `tempfile` (`bootstrap_store.py:224`); only the rebuilt store on the
`--scratch` S3 prefix survives the run. Promotion to canonical is a **copy-then-delete
cutover** (filenames encode vintage ranges, so an overwrite-mirror would leave both old-
and new-named files and silently corrupt the store).

**The guard that matters:** the write target is the `--scratch` **argument**, not
`NFP_STORE_URI`. `is_canonical_store()` returns True for any remote URI ending in `/store`
(`paths.py:71-95`) and refuses it as `--scratch` (`bootstrap_store.py:216-219`). So
`--scratch` must end in something **other than** `/store` (use `…/store-rebuild`).

**Build to scratch only (inspect before promoting):**
```bash
uv run python scripts/bootstrap_store.py \
  --scratch s3://alt-nfp/store-rebuild \
  --canonical s3://alt-nfp/store \
  --no-promote
```

**Rebuild + promote in one shot (the documented command):**
```bash
uv run python scripts/bootstrap_store.py \
  --scratch s3://alt-nfp/store-rebuild \
  --canonical s3://alt-nfp/store
```

> **Note — backup-before-promote is NOT container-safe out of the box.**
> `bootstrap_store.py`'s promote does **not** snapshot canonical first, and the obvious
> fallback `scripts/_t8_promote.py backup` writes the copy to a **local**
> `data/canonical_backup_<DATE>/` (`_t8_promote.py:41,81,96`) — which violates the
> read-only `./data` contract from Step 1 and will fail/block on a Bloomberg container.
> Take the safety copy with a **pure-S3** copy of the canonical prefix instead, e.g.
> `aws s3 cp --recursive s3://alt-nfp/store s3://alt-nfp/store-prev-<DATE>` (or an `s3fs`
> copy), before promoting. (First port = empty canonical = nothing to back up; this only
> bites when rebuilding *over* an existing canonical.)

**Verify:**
```bash
uv run alt-nfp status --store s3://alt-nfp/store
# Expect coverage report with CES/QCEW partitions populated, no "uncaptured" alarm gap.
```

---

### Step 3 — Seed the non-store inputs

Seven inputs must exist **outside** the vintage store. The store (Step 2) is the only one
this repo reconstructs end-to-end. Two paths exist for the S3 artifacts:

- **Greenfield (recommended on Bloomberg):** with `NFP_DATA_URI` set, the generators write
  **straight to S3**. Cleaner than staging locally.
- **Local → S3 migration:** `scripts/seed_data_s3.py --apply` uploads an existing local
  `DATA_DIR` tree (indicators/, competitors/, intermediate dates) to S3. It **excludes
  providers** by design and refuses any key under `store*`. Use this only if you already
  have the artifacts locally.

| # | Input | Env / path | How to produce |
|---|---|---|---|
| 3a | Cyclical indicators (claims ICNSA, jolts JTSJOL) | `NFP_DATA_URI/indicators/*.parquet` | `alt-nfp update --as-of … --only indicators` (needs `FRED_API_KEY`) |
| 3b | Release + vintage schedules | `NFP_DATA_URI/intermediate/{release,vintage}_dates.parquet` | Written by `advance_release_calendar()` inside `alt-nfp update` (default on; suppress with `--no-refresh-calendar`) |
| 3c | Consensus survey median | `NFP_CONSENSUS_PATH` or `NFP_DATA_URI/competitors/consensus.parquet` | Maintainer-supplied (Bloomberg street forecast); conform to `specs/bloomberg_consensus.md §1` |
| 3d | Provider parquets | `NFP_PROVIDERS_URI` (separate store) | **Pre-exist**; not seeded by this repo (Step 3d below) |
| 3e | RIF intervention priors | `government.py:36-46` (source constant) | **Edit source** — PLACEHOLDER (Step 3e below) |

**3a — Indicators.** Refresh from FRED; must be re-run each release cycle (claims weekly,
jolts monthly w/ 2-month pub-lag). Missing file → `read_indicator` returns None and the
model's covariate gating drops the array (runs without that covariate, no crash).
```bash
uv run alt-nfp update --as-of 2026-01-12 --only indicators
# Verify: parquet appears at s3://alt-nfp/indicators/claims.parquet + jolts.parquet
```
> **Note:** `--only indicators` still refreshes the release/vintage calendar unless you also
> pass `--no-refresh-calendar`; `--as-of` does not change which FRED series are fetched.

**3b — Schedules.** There is **no** `alt-nfp calendar` subcommand. The four CLI commands
are `update / status / watch / snapshot` (`__main__.py`, `@app.command()` ×4). Calendar
refresh is folded into `update` and writes both `release_dates.parquet` and
`vintage_dates.parquet` to `NFP_DATA_URI/intermediate/`. The `www.bls.gov` scrape requires
curl_cffi Chrome impersonation (plain httpx → 403).
```bash
uv run alt-nfp update --as-of 2026-01-12   # refresh-calendar is ON by default
# Verify: s3://alt-nfp/intermediate/{release_dates,vintage_dates}.parquet exist + non-empty
```
> **Caveat (local-only fallback):** `revision_schedules._load_vintage_dates`
> (`revision_schedules.py:36`) reads from the **hardcoded local `INTERMEDIATE_DIR`**, not
> `NFP_DATA_URI` — on Bloomberg this path is always missing and it falls back to a
> lag-based approximation. All **other** callers (`tagger`, `releases`, `ces_triangular`,
> `qcew_bulk`) respect `storage_options_for(VINTAGE_DATES_PATH)` and read from S3. Confirm
> the lag-based approximation is acceptable for `get_default_calendar()` consumers, or
> stage the parquet locally for that one caller.

**3c — Consensus.** Total-NFP street median; scored on **Track B (Total)**, not the
private track. Absent file → `load_consensus` returns None → consensus column renders `—`
(pipeline continues). `ref_month` uses month-start (day=1); `survey_date < release_date`.
```bash
export NFP_CONSENSUS_PATH=s3://alt-nfp/competitors/consensus.parquet
# Verify: python -c "from nfp_vintages.competitors.consensus import load_consensus; print(load_consensus() is not None)"
```

**3d — Provider store.** `NFP_PROVIDERS_URI` is a **separate** bucket/store from
`alt-nfp`, **not seeded by this repo** (`paths.py:202-211`, comment). The parquets
(`providers/g/g_provider.parquet`, `providers/g/g_births.parquet`) must **pre-exist** at
that URI; `ProviderConfig.file` paths are joined to the root and must match the store's
layout exactly. Missing files degrade **gracefully**: `read_provider_table` → None,
`ingest_provider` → `empty_panel()`, model runs **providerless**. This is the dropped
turning-point edge a real validation needs — not a crash-blocker but a correctness lever.
- **DECISION:** the placeholder default is `s3://alt-nfp-providers` (the Step-1 env block above),
  but the real Bloomberg URI is **maintainer-supplied** and not committed anywhere. Confirm it and
  that the parquets are present.
```bash
# Verify presence (with NFP_PROVIDERS_URI set):
python -c "from nfp_ingest.payroll import load_provider_series; from nfp_lookups.provider_config import PROVIDERS_DEFAULT; \
print([load_provider_series(c) is not None for c in PROVIDERS_DEFAULT])"
```

**3e — RIF intervention priors (HIGHEST-RISK).** `KNOWN_INTERVENTIONS` in
`packages/nfp-lookups/src/nfp_lookups/government.py:36-46` is **hardcoded in source** and
carries an explicit PLACEHOLDER comment: the single `federal_rif_2025` entry has
`magnitude_k=-50.0`, `magnitude_sd_k=25.0`, `announcement_date=2025-02-11`,
`source_url="PLACEHOLDER …"`. These flow **unconditionally** into the wedge model's
intervention-coefficient prior whenever `as_of >= 2025-02-11` (`wedge_data.py` builds
`X_intervention` + the per-intervention `(magnitude_k, magnitude_sd_k)`, consumed by
`wedge_model` in `wedge.py` as `mu_t = drift + season + X_intervention @ coef`). The source
comment states: *"must be replaced before any accuracy claim."*
- **TODO before any Track B accuracy claim:** replace `magnitude_k` with the real announced
  permanent-separation count (signed thousands); set `magnitude_sd_k` from
  `calibrate_intervention_sd(observed_federal_change, baseline_sd)`
  (`wedge_diagnostics.py:22` — **no production caller**, only a unit test exercises it, so
  it's a manual step);
  set the real `source_url`; confirm `announcement_date`. Also verify the
  `GOVERNMENT_INDICATORS` FRED IDs (`government.py:88-89`, "PLAN-SIDE VERIFICATION required")
  are fetchable before relying on them.

---

### Step 4 — Model / GPU bring-up

**Files:** `nfp-model/pyproject.toml:7-11`, `__init__.py:7-9,21` (float64 latch),
`data.py:25-93` (ModelData contract), `batch.py:341-349`, `config.py:99-116`.

**4a — Install the CUDA JAX wheel first.** `nfp-model` itself needs no GPU build; numpyro
resolves on whatever jax is present. Do **not** let `pip install nfp-model` resolve the CPU
jax wheel first.
```bash
pip install -U 'jax[cuda12]'      # or 'jax[cuda12_local]' against a pinned driver
uv sync                           # then install the workspace
# If a CPU jaxlib is already present, uninstall it — two jaxlibs => silent CPU dispatch.
```
- **OPEN:** no CUDA wheel version is pinned anywhere. One-time pilot: confirm a
  `jax[cuda12]` release compatible with `jax>=0.4.38` **and** the Bloomberg CUDA driver.

**4b — float64 is automatic.** Importing `nfp_model` fires `numpyro.enable_x64()` at
import time (`__init__.py:21`). **Do not** set `JAX_ENABLE_X64=0` or call
`jax.config.update('jax_enable_x64', False)` anywhere — either reverts precision and breaks
the A3 parity contract. Import `nfp_model` before any other JAX computation in the process
(the latch is one-way per process).

**4c — Select the GPU** via env before process start (no code change):
```bash
export JAX_PLATFORM_NAME=cuda
export CUDA_VISIBLE_DEVICES=0      # pin a single GPU
```

**4d — ModelData contract.** The model never hits the network — it consumes frozen
content-hashed `.npz` snapshots (built by `alt-nfp snapshot` or the backtest `snapshot`
mode, which call `nfp_ingest` against the S3 store). Feed `load_snapshot()` →
`from_snapshot()` → `model_inputs()` → `fit_model_batch()`. Provider entries with **zero**
observations crash `pad_model_inputs()` — drop the provider/date or use `A5_NO_PROVIDERS=1`.

**Verify (device AND precision — correctness, not just perf):**
```bash
python -c "import nfp_model, jax, jax.numpy as jnp; print(jax.devices(), jnp.zeros(1).dtype)"
# Expect: [CudaDevice(...)] and float64.  CpuDevice or float32 => STOP (broken correctness).
```
> **Caveats (no in-repo probe):** the dtype check confirms float64 is *enabled* but **not**
> that the GPU runs it at native rate — a consumer card can report `float64` while emulating
> it slowly. Confirm un-throttled fp64 out-of-band (card spec / a timed matmul) during the
> one-time pilot. Separately, if you ever **migrate** pre-existing `.npz` snapshots instead
> of rebuilding them, check `snapshots.py` `SCHEMA_VERSION` (currently 3) **yourself** —
> `from_snapshot` does **not** validate it (no version check; a stale-schema snapshot loads
> silently and can mis-shape the model inputs). Rebuilt snapshots (the assumed path) are
> always in-version.

---

### Step 5 — Wire script / CLI output roots (no `./data` writes)

**Files:** `paths.output_root()` (`paths.py:176`); the backtest scripts import it
(`run_a5_backtest.py:71`, `run_a4_backtest.py:49`) and resolve `output_root(sys.argv[2])`
inside `main()` (`run_a5_backtest.py:547`).

The production CLI (`update/status/watch/snapshot`) routes through
`VINTAGE_STORE_PATH`/`NFP_STORE_URI` (with `--store` override) and `NFP_SNAPSHOTS_URI` —
none writes under `./data` on Bloomberg. The two backtest scripts take the **output root as
a positional argv arg** resolved by `output_root(arg)`: an `s3://` URI → credentialed
`UPath`; anything else → a local `Path`. **There is no env-var fallback** — Bloomberg must
always pass an `s3://` URI as `sys.argv[2]`.

```bash
uv run python scripts/run_a5_backtest.py snapshot s3://alt-nfp/backtests/a5
uv run python scripts/run_a4_backtest.py snapshot s3://alt-nfp/backtests/a4
```
> **Dev-only `./data` writer:** `scripts/_05_convergence_fit.py` hardcodes a local
> `data/05_convergence_baseline/` write (`:61`) and asserts an S3 store. It is an untracked
> scratch/dev gate (the '05 convergence gate is already CLOSED/PASS) — not a production
> script; leave it off the Bloomberg hot path.

**Verify:** after a backtest `snapshot` run, confirm `grid_manifest.json` + `snapshots/`
appear under the `s3://…/backtests/a5` prefix and **nothing** new appears under `./data`.

---

### Step 6 — Run the manual gates

No CI. Run the full gate set manually (see [appendix](#manual-gates-no-ci)). Minimum
before declaring the port healthy: `ruff check .` clean, and the non-network suite green.
Store-dependent tests self-skip without `@pytest.mark.real_store` (`conftest.py:69-85`
blanks store creds for every non-`real_store` test).

---

### Step 7 — First validation run (build-here / validate-on-port)

This is the workload that moves to the GPU: the **A5 first-print backtest** (private '05'
nowcast scoreboard vs published BLS first prints). Use the `light` preset (the A5 default);
`fit_model_batch` forces `chain_method='vectorized'` automatically (`batch.py:348-349`).

```bash
# 1. Build snapshots from the S3 store, 2. fit batched on GPU, 3. score
uv run python scripts/run_a5_backtest.py snapshot s3://alt-nfp/backtests/a5
uv run python scripts/run_a5_backtest.py batched  s3://alt-nfp/backtests/a5
uv run python scripts/run_a5_backtest.py score    s3://alt-nfp/backtests/a5
# Track B (Total = private '05' ⊕ government wedge) vs consensus:
uv run python scripts/run_a5_backtest.py total    s3://alt-nfp/backtests/a5
```
- **`A5_NO_PROVIDERS=1`** builds the public-only skeleton (avoids the placeholder provider
  edge); **`A5_N_BACKTEST`** (default 24) sets the window.
- **'05 target — no operator action needed:** the A5 harness **hardwires** the private
  target, `HEADLINE_INDUSTRY = "05"` (`run_a5_backtest.py:43`), threaded through
  `cmd_snapshot`'s `panel_to_model_data(..., industry_code=HEADLINE_INDUSTRY)` (`:185`).
  There is **no flag** and no way for an A5 run to score '00'. The `'00'` default only
  affects the *generic* `build_model_data` entry point, which A5 deliberately bypasses — so
  only ad-hoc `build_model_data` calls outside the A5 path inherit the '00 default.

**Verify:** `a5_results.parquet` + `a5_report.md` (and `total_scores.json` for `total`)
land under the S3 prefix; sanity-check the report's MAE/coverage against the documented
public-skeleton baseline (beats naive on normal months; loses on turning points until the
real provider store is wired).

---

## Known gaps / decisions before go-live

- **RIF priors are a PLACEHOLDER** (`government.py:36-46`: `-50k ± 25k`). They flow into the
  Track B wedge prior unconditionally for `as_of ≥ 2025-02-11`. **Must** be replaced with
  real 2025 values (magnitude, calibrated sd via `calibrate_intervention_sd` — which has no
  production caller, only a unit test, source_url) **before any Track B accuracy claim.**
- **Provider store is not seeded by this repo.** `NFP_PROVIDERS_URI` points to a separate
  Bloomberg store whose URI is maintainer-supplied and uncommitted; the parquets must
  pre-exist with matching `ProviderConfig.file` paths. Missing → providerless run (graceful,
  but it's the dropped turning-point edge).
- **'05 vs '00 default — A5 is already correct.** The A5 harness forces `industry_code='05'`
  (`run_a5_backtest.py:43`), so Track A validation scores the right private target with no
  action. Only the generic `build_model_data` entry point defaults to '00'; A5 bypasses it.
- **Consensus is Track B (Total).** It's a total-NFP street median, scored on the Total
  product, not the private track. Absent → column renders `—`, pipeline continues.
- **`AWS_ENDPOINT_URL` is a deployment decision** (real AWS S3 → unset; Bloomberg
  S3-compatible → set, HTTPS) and `revision_schedules._load_vintage_dates` reads
  **local-only** `INTERMEDIATE_DIR` (always missing on Bloomberg → lag-based approximation);
  confirm both are acceptable.

**Open questions to close with the maintainer:**
- Real AWS-native S3 vs a Bloomberg-hosted S3-compatible endpoint (drives
  `AWS_ENDPOINT_URL` / `aws_allow_http` / region handling)?
- Exact `NFP_PROVIDERS_URI` and the provider-store key layout.
- Has `seed_data_s3.py` (or the greenfield direct-write path) been run against the
  Bloomberg `alt-nfp` bucket yet?
- Which `jax[cuda12]` release is compatible with `jax>=0.4.38` **and** the Bloomberg CUDA
  driver (one-time pilot)?
- Is the GPU's native fp64 confirmed un-throttled (correctness, not perf)?
- Are the `GOVERNMENT_INDICATORS` FRED IDs (`government.py:88-89`) fetchable and used in any
  active inference path, or diagnostics-only?

---

## Manual gates (no CI)

GitHub Actions are unavailable on Bloomberg; run these by hand. Gates 4–5 require the docs
group (`uv sync --group docs`). No `addopts` are configured, so flags are explicit.

```bash
# --- one-time setup ---
uv sync                  # workspace + dev group (pytest, ruff, mypy, black)
uv sync --group docs     # adds mkdocs, mkdocstrings, interrogate (for gates 4-5)

# --- COPY-PASTEABLE GATE SET ---
# 1. Lint
uv run ruff check .

# 2. Fast suite (~30s, skips MCMC smoke) — rapid iteration
uv run pytest -m "not network and not slow" --no-cov

# 3. Full local suite (~3min, includes MCMC smoke)
uv run pytest -m "not network" --no-cov

# 4. Docstring coverage (fail-under = 100)
uv run interrogate -c pyproject.toml packages

# 5. Docs build (--strict promotes warnings to errors)
uv run mkdocs build --strict
# --- END GATE SET ---
```

- `real_store` tests are **not** excluded by `not network` — they need `.env` with `AWS_*`
  + `NFP_STORE_URI`. To exclude them on a credential-less box:
  `-m "not network and not slow and not real_store"`.
- `mkdocs gh-deploy --force` (local) is the docs deploy path to github.com (no Actions).
