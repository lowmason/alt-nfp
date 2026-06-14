# Independent-Audit Remediation Plan (plans/9)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development
> to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> **Task A0 (the IMD-1 ruling) is RESOLVED below.** Plan execution itself is
> currently PAUSED pending maintainer go-ahead (docs committed 2026-06-13).

**Goal:** Remediate the confirmed, parity-neutral findings from the independent
audit ([`specs/audit_independent.md`](../specs/audit_independent.md)), surface the
one substantive value bug for a semantics ruling, and record the parity-faithful
remainder as an explicit backlog.

**Architecture:** Same workspace + parity discipline as plans/8. This plan
**subsumes the still-open "remediate the three new audits" track** (lookups /
vintages / model): the source-grounded independent pass re-adjudicated all of
them with adversarial verification, so there is now one merged track. The
cross-reference table below shows how ~45 raw findings collapse to **2 substantive
fixes + 7 small cleanups + an explicit backlog**.

**Tech stack:** Python 3.12, Polars, NumPyro/JAX, Typer CLI, pytest, uv workspace.

---

## PARITY POLICY (carry-over from plans/8 — unchanged, still binding)

1. **Parity = posterior/array/store parity vs the frozen reference at
   `~/Projects/alt_nfp`**, NOT `content_hash` parity. The fast suite is NOT parity
   evidence.
2. Classify every change against the frozen reference:
   - **(a) parity-neutral** — no effect on store bytes (A0), `build_model_data`
     arrays (A2), or the posterior. → fix freely.
   - **(b) port-bug** — current code diverges from the reference *and* the
     reference is correct. → fix silently (restores parity).
   - **(c) correct-but-divergent** — the "fix" would make us *diverge* from a
     byte-faithful reference. → **STOP and surface for a per-item ruling.**
3. Correct docstrings/comments regardless of class.
4. **Store-write safety (carry-over, hard):** never run a store-writing function
   against the real/canonical store in a test (`tmp_path` only); rebuilds target a
   scratch prefix (`s3://alt-nfp/store-rebuild`), never `s3://alt-nfp/store`.

**Every `parity_intentional` finding in the audit is a class-(c) item** — it is
real but byte-faithful to the frozen reference. Those are all in the **Backlog**
section; do NOT touch them pre-parity.

---

## Cross-reference: ~45 raw findings → this plan

| Source finding(s) | Independent verdict | Disposition here |
|---|---|---|
| vintages B-1/V-1 (Critical store-wipe) | already guarded (C-1/T1) | **done** (plans/8) |
| lookups M-8 / all version+numpy | done (T7) | **done** |
| vintages H-4b parse drift | done (T4) | **done** |
| model MOD-5a AST boundary, L-11 lam_ces | done (T6/T8) | **done** |
| ingest H-3 BD arrays, H-4a, I-2, I-4 | done (T10–T14) | **done** (H-4a/H-2b still MCMC-deferred) |
| lookups L-1 (`_find_base_dir`) | refuted (fallback is correct) | **no action** |
| lookups L-2 / IND-LK-1 (S3 env validation) | refuted | **no action** |
| lookups L-2025-benchmark / IND-LK-2 | refuted (value is correct) | **no action** |
| IND-DL-6, IND-IST-5, IND-MD-2, IND-MD-4 | refuted / already-handled | **no action** |
| **IND-XC-2** mirror_store unguarded write | confirmed (med) | **Task 1 — fix now** |
| **IND-IMD-1** benchmark-level mis-stamp | confirmed (med) | **Task A0 RESOLVED (approved) → Task 9** |
| IND-IMD-2 orphaned `bd_qcew_lag` | confirmed | **Task 2 — fix now** |
| IND-MD-3 dead `births` in `from_snapshot` | confirmed | **Task 3 — fix now** |
| IND-VT-1 snapshot day-12 | confirmed | **Task 4 — fix now** |
| IND-VT-5 `build_store.main()` | confirmed | **Task 5 — fix now** |
| IND-XC-5 A2-golden absence assertion | confirmed | **Task 6 — fix now** |
| IND-IST-3 tie-break comments | confirmed | **Task 7 — fix now** |
| IND-XC-1 parity gates skip in CI | confirmed (med) | **Task 8 — CI note now; smoke = H-2b** |
| LK-3, LK-4, DL-1..5, IST-1, IST-2, IST-4, IMD-3, IMD-4, VT-2, VT-3, VT-4, MD-1, XC-3, XC-4; old D-1/B-2/B-3/B-4/V-1-refactor/V-2; model MOD-1/3/6/7/8 | parity_intentional / deferred | **Backlog** (see end) |

---

## DECISION REQUIRED before any code

### Task A0 — Ruling on IND-IMD-1 (CES benchmark-level mis-stamp) [maintainer]

**The bug (confirmed, MEDIUM).** In
[`packages/nfp-ingest/src/nfp_ingest/releases.py`](../packages/nfp-ingest/src/nfp_ingest/releases.py)
`_fetch_ces_releases` (≈153–174), the join
`raw.join(latest_vdates, on='ref_date', how='left')` fans the single
post-benchmark flat-file level onto **both** the `(rev=2, bmr=1)` reprint row and
the `(rev=1, bmr=0)` ordinary-second-print row. Post-benchmark re-capture writes
the benchmark-revised level (~900k off) onto the rev1/bmr0 track.

**Why it needs a ruling (not an autonomous fix).** The *correct* pre-benchmark
second-print value is **not present in the BLS flat file** (which only carries the
post-benchmark level). So the fix cannot supply the right number — it can only
**stop emitting the wrong one** and let the triangular-revisions pipeline own the
rev1/bmr0 level. That is a semantics decision keyed to
`specs/ces_growth_convention.md` §4(c)-i / §5, and under the parity policy it is a
class-(c) "correct-but-divergent" change that must be ruled on.

**Proposed semantics (for sign-off):** when a `ref_date` has a `bmr≥1` reprint row
in `latest_vdates`, do **not** attach the current flat-file level to that
`ref_date`'s `(rev=1, bmr=0)` row — emit only the reprint track from live capture;
the rev1/bmr0 second-print level is supplied by the triangular revisions pipeline.

**Options:** (a) approve the proposed semantics → execute Task 9; (b) prefer a
different rule (state it); (c) defer IMD-1 to the post-parity backlog (leave the
live-capture path as-is, documented).

**RULING (2026-06-13 — maintainer delegated the call to Claude): (a) APPROVED.**
Rationale: a known-wrong level (~900k off) is strictly worse than none; the
pre-benchmark second print legitimately belongs to the triangular-revisions
pipeline, and per spec §5 the rev1/bmr0 slot is empty for benchmarked months.
Parity-safe (live-capture = new appends, not golden-mastered). **Condition:** Task
9 Step 1 must re-confirm the exact emit-rule against `ces_growth_convention.md`
§4(c)-i/§5 before writing code; if the spec dictates a different rule, follow the
spec. Task 9 is unblocked for whenever plan execution is authorized.

---

## Bucket A — Fix now (parity-neutral, confirmed)

Branch: `audit-independent` off `main`. One commit per task.

### Task 1 — Guard `mirror_store.py` against the canonical store (IND-XC-2)

**Files:**
- Create helper: `packages/nfp-lookups/src/nfp_lookups/paths.py` (add `is_canonical_store`)
- Modify: `scripts/mirror_store.py`
- Modify: `packages/nfp-vintages/src/nfp_vintages/build_store.py` (reuse helper)
- Test: `packages/nfp-lookups/tests/test_paths.py`

Rationale: the canonical-store refusal predicate currently lives inline in
`build_store` only. `mirror_store.py` does a raw `s3fs.put_file()` loop to
`NFP_STORE_URI` with no guard and runs *outside* the pytest safety net. Promote one
shared predicate to the foundation package and apply it at both write doors.

- [ ] **Step 1: Write the failing test** in `test_paths.py`:

```python
from nfp_lookups.paths import is_canonical_store

def test_is_canonical_store_matches_canonical_uri():
    assert is_canonical_store("s3://alt-nfp/store") is True
    assert is_canonical_store("s3://alt-nfp/store/") is True
    assert is_canonical_store("s3://alt-nfp/store-rebuild") is False
    assert is_canonical_store("s3://alt-nfp/store-rebuild/") is False

def test_is_canonical_store_false_for_local_paths(tmp_path):
    assert is_canonical_store(tmp_path) is False
    assert is_canonical_store(str(tmp_path / "store")) is False
```

- [ ] **Step 2: Run it, confirm ImportError/FAIL.**
- [ ] **Step 3: Implement `is_canonical_store` in `paths.py`** (uses existing `is_remote`):

```python
def is_canonical_store(path) -> bool:
    """True iff ``path`` is the remote canonical vintage store (…/store).

    The canonical store is append-only and irreplaceable; writers must refuse it
    unless explicitly forced. A scratch prefix (…/store-rebuild) returns False.
    """
    if not is_remote(path):
        return False
    return str(path).rstrip("/").endswith("/store")
```

- [ ] **Step 4: Run the test, confirm PASS.**
- [ ] **Step 5: Refactor `build_store.py`** to call `is_canonical_store(out_path)`
  in its existing guard (behavior-identical — the inline check already used the
  same `endswith("/store")` logic). Run `test_build_store_guard.py`, confirm PASS.
- [ ] **Step 6: Add the guard to `mirror_store.py`** after it resolves `uri`/`dest`:

```python
from nfp_lookups.paths import is_canonical_store
...
if is_canonical_store(dest) and "--allow-canonical" not in sys.argv:
    raise SystemExit(
        f"refusing to mirror onto the canonical store {dest!r} — it is "
        "append-only and irreplaceable; target a scratch prefix or pass "
        "--allow-canonical"
    )
```

- [ ] **Step 7:** `uv run ruff check .` + `uv run pytest -m "not network and not slow" --no-cov` (lookups + vintages). Commit.

### Task 2 — Remove orphaned `bd_qcew_lag` config field (IND-IMD-2)

**Files:** `packages/nfp-ingest/src/nfp_ingest/model_data.py:57`

Dead since the BD covariates were dropped (SCHEMA_VERSION=3); no reader anywhere
(grep-confirmed). parity_gate = none.

- [ ] **Step 1:** `grep -rn "bd_qcew_lag\b" packages/` to re-confirm zero readers (only the definition).
- [ ] **Step 2:** Delete the `bd_qcew_lag: int = 6` line from `ModelDataConfig`.
- [ ] **Step 3:** `uv run pytest packages/nfp-ingest -m "not network and not slow" --no-cov` (snapshot/model-data tests green). Commit.

### Task 3 — Drop dead `births`/`births_obs` from `from_snapshot` (IND-MD-3)

**Files:** `packages/nfp-model/src/nfp_model/data.py:88-89`; verify `tests/test_data.py`.

`model_inputs` already strips providers to `{name, error_model, g_pp, pp_obs}`, so
these never reach the model; the A2 golden generator emits them at the array level
(untouched). Round-trip test asserts only the 4 consumed keys. parity_gate = none.

- [ ] **Step 1:** Confirm `test_data.py` round-trip asserts only `name/error_model/g_pp/pp_obs` (no `births`).
- [ ] **Step 2:** Remove the two `.get(f"{name}__births*")` entries from the `from_snapshot` provider dict.
- [ ] **Step 3:** `uv run pytest packages/nfp-model -m "not network and not slow" --no-cov`. Commit.

### Task 4 — Enforce `snapshot --as-of` day-12 convention (IND-VT-1)

**Files:** `packages/nfp-vintages/src/nfp_vintages/__main__.py` (snapshot cmd ≈218–247); Test: `packages/nfp-vintages/tests/` (CLI test).

The single `--as-of` path passes the raw date through while the `--grid-end` path
snaps to the 12th — inconsistent. parity_gate = none (snapshots are downstream of
the store).

- [ ] **Step 1: Failing test** — invoking `snapshot --as-of 2026-01-05` (no grid) raises `typer.BadParameter` (use Typer's `CliRunner`, assert non-zero exit + message).
- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement** — after `start = _date.fromisoformat(as_of)`, in the single-date branch:

```python
if grid_end is None and start.day != 12:
    raise typer.BadParameter("--as-of must fall on the 12th (day-12 convention)")
```

- [ ] **Step 4: Run, confirm PASS.** Commit.

### Task 5 — Harden `build_store.main()` against silent `--releases` no-op (IND-VT-5)

**Files:** `packages/nfp-vintages/src/nfp_vintages/build_store.py:130-145`

`--releases` with no following token silently falls back to the default. parity_gate = none.

- [ ] **Step 1:** `grep -rn "build_store" --include=*.md --include=*.yml --include=*.py .` to check whether `python -m nfp_vintages.build_store` has any caller (CI/docs/scripts).
- [ ] **Step 2a (no callers):** delete `main()` + its `if __name__ == "__main__"` block in favor of the Typer `alt-nfp build` command.
- [ ] **Step 2b (callers exist):** make a malformed `--releases` (flag present, no value) `raise SystemExit("--releases requires a path")` instead of silently using the default.
- [ ] **Step 3:** `uv run ruff check .` + vintages tests. Commit.

### Task 6 — Assert dropped BD keys are absent in the A2 golden gate (IND-XC-5)

**Files:** `packages/nfp-ingest/tests/test_model_data_golden.py` (≈121–127)

Today the gate *skips* `birth_rate`/`bd_proxy`/`bd_qcew_lagged`; it doesn't assert
`build_model_data` stopped *emitting* them. Parity-safe (asserts absence only).

- [ ] **Step 1:** Add, right after `data` is built from `build_model_data`:

```python
assert not (_DROPPED & set(data)), (
    f"build_model_data re-emitted dropped BD arrays: {_DROPPED & set(data)}"
)
```

- [ ] **Step 2:** Run the test (skips without store, or passes with it). Confirm no syntax/collection error via `uv run pytest packages/nfp-ingest/tests/test_model_data_golden.py --collect-only`. Commit.

### Task 7 — Tighten append/compact tie-break comments (IND-IST-3)

**Files:** `packages/nfp-ingest/src/nfp_ingest/vintage_store.py` (append ≈696, compact ≈796)

Comment-only. State precisely: append keeps the **first-written** row for a ukey;
compact keeps **min(vintage_date)**; note they coincide only for
chronologically-ordered appends.

- [ ] **Step 1:** Replace the misleading "matches append" comment with the precise statement above (no code change).
- [ ] **Step 2:** `uv run ruff check .`. Commit.

### Task 8 — CI parity-gate transparency note (IND-XC-1, partial)

**Files:** `.github/workflows/ci.yml`

The A1/A2/A3 golden gates self-skip without store creds, so a green CI check is
**not** parity confirmation. The credential-free *value* smoke is the deferred
**H-2b** item (needs frozen-reference MCMC to mint the golden) — keep it deferred.
Do the achievable part now: make the gap explicit in CI.

- [ ] **Step 1:** Add a comment/echo step in `ci.yml` near the pytest run, e.g.
  `echo "NOTE: A1/A2/A3 parity goldens self-skip without store creds — green here is NOT parity confirmation."`
- [ ] **Step 2:** Confirm `ci.yml` still parses (yamllint or a dry inspection). Commit.

---

## Bucket A* — Gated on Task A0

### Task 9 — Fix IND-IMD-1 per the approved semantics [BLOCKED on Task A0]

**Files:** `packages/nfp-ingest/src/nfp_ingest/releases.py` (`_fetch_ces_releases`);
Test: `packages/nfp-ingest/tests/test_new_ingest.py`.

- [ ] **Step 0 (pre-req):** Task A0 answered. If "defer" → skip Task 9, move IMD-1
  to Backlog. If "fix" → proceed.
- [ ] **Step 1: Independently re-confirm the mechanism** — read `_fetch_ces_releases`
  + `_latest_ces_vintage_dates` and verify the `on='ref_date'`-only join fans one
  level onto the two `(rev,bmr)` rows (don't take the audit on faith; it's the only
  substantive code change and it's in the irreplaceable-data path).
- [ ] **Step 2: Failing regression** — capture a benchmarked December twice (rev-1
  release, then post-benchmark) and assert the stored `(rev=1, bmr=0)` **level**
  equals the pre-benchmark second print (or is absent, per the approved rule), NOT
  the benchmark reprint. Use `tmp_path`/synthetic frames only — never the real store.
- [ ] **Step 3: Run, confirm FAIL** on current code.
- [ ] **Step 4: Implement the approved semantics** (proposed: don't attach the
  flat-file level to a `(rev=1,bmr=0)` row whose `ref_date` also has a `bmr≥1`
  reprint).
- [ ] **Step 5: Run, confirm PASS.** Re-run the full nfp-ingest suite (metadata-triple
  regressions must stay green).
- [ ] **Step 6 (optional verification):** rebuild to a **scratch** prefix
  (`NFP_STORE_URI=s3://alt-nfp/store-rebuild`) and spot-check the Dec-2025 rows.
  Never target the canonical store. Commit.

---

## Backlog — do NOT fix pre-parity (class-(c) / parity-faithful / deferred)

Each is **real but byte-faithful to the frozen reference**, so fixing it diverges
from parity; revisit in the post-parity novelty phase. Grouped:

- **Store/censoring (A0/A2-sensitive):** IND-IST-1 (CES fallback benchmark leak =
  I-2), IND-IST-2 (validator benchmark assertion — pairs with IST-1), IND-IST-4
  (validator check-#7 comment), IND-VT-2 (build_store dedup determinism = old B-2),
  IND-VT-3 (build_store schema cast = old B-3), IND-XC-3 (`VINTAGE_STORE_SCHEMA`
  duplication = old L-4), IND-XC-4 (QCEW schedule duplication = old L-4); old D-1
  (unify store-write surface), old B-4 (reads ignore storage_options).
- **Download (parity-faithful):** IND-DL-1..5 (silent-empty fetch, no-write bulk,
  FRED stale-5xx, `max_retries=0`, stale `end_year`).
- **Model (parity-faithful / documented design):** IND-MD-1 (gate z-test power =
  old MOD-4), IND-MD-2 (intake asserts = old MOD-2, unreachable), model MOD-3 (phi
  clip), MOD-6 (batch common-axis doc), MOD-7 (silent cyclical gate — also H-4a
  MCMC-deferred), MOD-8 (dead BD reconstruction — resolved by H-3/T13; verify).
- **Lookups (dead/parity-faithful):** IND-LK-3 (en_series_id dead+invalid),
  IND-LK-4 (validate_panel dead constants).
- **Vintages (dead/parity-faithful):** IND-VT-4 (views.py dead code — delete or docstring).
- **Cosmetic / cross-package (old audit, not re-raised as actionable):** old V-1
  refactor (factor `_work()` fns — danger already neutralized), V-2 (CLAUDE.md
  "idempotent" claim), MOD-1 (fit-manifest reproducibility — *not* corroborated by
  the source-grounded model pass; assess seed handling in `sampling.py` before
  treating as a gap), MOD-5b (x64-import doc), L-3 (paths import-time-binding doc),
  IND-IMD-3 (unused `date_list`), IND-IMD-4 (compositing weight lookahead — only
  cell-level providers; default 'G' national unaffected).
- **MCMC-deferred (from plans/8, unchanged):** H-4a NaN sentinel (model-side),
  H-2b model parity-in-CI smoke (= the value half of IND-XC-1).

---

## No action — refuted / already-handled

IND-LK-1, IND-LK-2, IND-DL-6, IND-IST-5 (confirms T2), IND-MD-2 (unreachable),
IND-MD-4. The memory-based audits' Critical/High items (vintages B-1, lookups L-1,
the parsers, nowcast arithmetic) all verified clean/parity-faithful — see
`specs/audit_independent.md` "What the memory-based audits got wrong."

---

## Sequencing

1. **Task A0 first** — get the IMD-1 ruling (blocks Task 9 only; Bucket A can run in parallel).
2. Bucket A in order **1 → 8** (Task 1 is the highest-value: store-safety). Each is
   independent and parity-neutral; one commit each, full suite green between.
3. Task 9 only after A0 = "fix".
4. Final whole-branch review; do **not** merge/push without explicit maintainer request.

## Self-review (run before declaring done)

- [ ] Every Bucket-A change is class-(a) parity-neutral (no A0/A2/posterior effect) — re-verify, don't assume.
- [ ] No test writes to a real/remote store (`tmp_path` / synthetic only); `real_store` markers respected.
- [ ] `uv run ruff check .` clean; `uv run pytest -m "not network" --no-cov` green (incl. MCMC smoke).
- [ ] Task 1's `is_canonical_store` refactor of `build_store` is behavior-identical (guard test still passes).
- [ ] Task 9 (if executed) carries a regression that asserts the **stamped level**, and was verified against a scratch prefix only.
- [ ] No Backlog item was touched.
