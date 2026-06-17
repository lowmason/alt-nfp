# Rebuilt-store model-data drop-in — Implementation Plan (plans/11)

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **Builds target the SCRATCH prefix only** (`NFP_STORE_URI=s3://alt-nfp/store-rebuild`) — never the canonical store — exactly as `plans/10`.

**Design record:** [`specs/store_rebuild_model_data_dropin.md`](../specs/store_rebuild_model_data_dropin.md). Unblocks `plans/10` T7 (goldens) + T8 (promotion).

**Goal:** Add the two published series the NSA-only rebuild omitted — the **SA CES** triangles and the **QCEW `00` total** — so `build_model_data(as_of=D)` at the current `'00'` target is non-degenerate against the rebuilt store, **without changing the model**.

**Architecture:** Same uv workspace + data-package chain. CES builder also reads the `tri_*_SA.csv` triangles (emit `seasonally_adjusted=True`); the QCEW acquire/crosswalk also keeps the published `own_code=0` total row (→ CES `'00'`, `ownership='total'`). SA rows are **parallel** to the NSA hierarchy (null size, excluded from the §10 NSA gates). The B work (private-hierarchy modeling, government, from-scratch composition) stays deferred.

**Tech stack:** Python 3.12, Polars, pytest, uv workspace.

---

## POLICY (carried from plans/10)

1. **Scratch-only builds (hard).** Every build writes `s3://alt-nfp/store-rebuild`. The `is_canonical_store` guard refuses `…/store`.
2. **Store-write test safety (hard).** Never run a store-writing fn against the canonical store in a test — `tmp_path`/synthetic frames only; the root `conftest.py` `_block_live_store` autouse fixture severs s3fs for unmarked tests.
3. **No frozen-reference parity.** Acceptance is the §10 gate set + the drop-in checks here; A1/A2 are re-baselined in `plans/10` T7 (not here).
4. **Public data only.** `data/` is gitignored; the `cesvinall` triangles + provider/indicator data are proprietary.

---

## T0 — SA vintage-structure spike (read-only) `[depends: none]` — ✅ DONE → **Decision A**

**Finding (2026-06-17): SA `(rev,bmr)` mirrors NSA exactly — proceed with the mirror (A).** Verified against the canonical store + the `_SA` triangle:
- Canonical SA and NSA `00` carry the **identical** cohort set `{(0,0),(1,0),(2,0),(2,1)}`, identical row counts (1088 each), **max 4 rows per ref-month** (no extra SA-only vintages).
- Jun-2023 `00`: SA and NSA have the **same four `vintage_date`s** (2023-07-07, -08-07, -09-07, 2024-02-02) — only the *values* differ — prints SA 156204/156155/156075 vs NSA 156963/156945/156905, and the 2024-02-02 `(2,1)` benchmark SA 156027 vs NSA 156842. *(Corrected after T1's primary-source read: the earlier `…/155880` quoted SA's 2026-benchmark cell at vintage 2026-02-11, which a 2026-01 frontier filters out — not the 2024-02-02 value.)* The Feb benchmark re-seasonal-adjustment lands on the **same** `(2,1)` vintage as NSA; it does **not** introduce non-benchmark vintage steps.
- `tri_000000_SA.csv` is structurally identical to `_NSA` (1047 cols, same column grid).

⟹ **T1 reuses `_diagonals` unchanged**; only the file suffix (`_SA`) + the `seasonally_adjusted=True` flag change. No SA-specific diagonal path needed.

*(Original concern, now retired:)* The NSA builder assigns `(rev,bmr)` by reading down each ref-month column: first three prints → `(0,0)/(1,0)/(2,0)`; subsequent January-vintage benchmark restatements → per-benchmark `(2,1)`. The worry was that SA's February re-seasonal-adjustment might change an SA column *without* a benchmark — but the canonical store shows it does not (SA revises on the same vintage cadence as NSA).

- [ ] Read one `tri_000000_SA.csv` triangle and compare its diagonal/benchmark structure to the canonical store's SA rows (`read_vintage_store(source='ces', seasonally_adjusted=True)` against `s3://alt-nfp/store`): how does the canonical pipeline assign `(revision, benchmark_revision)` to SA, and do SA columns step at non-benchmark vintages?
- [ ] **Decide + record** (in this file under T0) one of:
  - **(A)** SA `(rev,bmr)` mirrors NSA exactly → T1 reuses `_diagonals` unchanged, only the file suffix + `seasonally_adjusted` flag change.
  - **(B)** SA needs adapted handling (e.g. seasonal-factor revisions collapse into the print sequence, or `(2,1)` keys differ) → T1 carries an SA-specific diagonal path; document the rule.
- [ ] **Acceptance:** a written go decision (A or B) with the canonical-SA evidence; no code yet.

---

## T1 — SA CES builder (`nfp-ingest`) `[depends: T0]` — ✅ DONE (eb7cd7f + docs 051dffb)

**Done (2026-06-17).** `build_ces_panel` loops over `(("NSA", False), ("SA", True))`, reuses `_diagonals` unchanged (Decision A), carries `seasonally_adjusted` per-part, adds it to the sort key. Regression guard (NSA subset == NSA-only build) + SA anchor cross-check (SA `00` 2023-06-12: 156204/156155/156075, `(2,1)` {(156027,2024-02-02),(155871,2025-02-07)}) both green. Spec ✅ + code-quality ✅ (12 tests, ruff clean).

**Files:**
- Modify: `packages/nfp-ingest/src/nfp_ingest/ces_builder.py` (the `tri_*_NSA.csv` glob + the `seasonally_adjusted=pl.lit(False)` literal)
- Test: `packages/nfp-ingest/tests/test_ces_builder.py`

Today `build_ces_panel` globs `tri_*_NSA.csv` and hard-stamps `seasonally_adjusted=False`. Generalize it to build **both** adjustments.

- [ ] **Step 1 — failing test (NSA unchanged + SA produced).** In `test_ces_builder.py`, add a test that points `build_ces_panel` at a fixture dir holding a synthetic `tri_000000_NSA.csv` **and** `tri_000000_SA.csv` and asserts: the result has rows with `seasonally_adjusted=True` **and** `False` for `00`; the SA rows carry the same `(rev,bmr)` cohorts T0 decided; SA rows have null `size_class_*`; NSA output is byte-identical to the NSA-only build (regression guard).

```python
def test_build_ces_panel_emits_sa_and_nsa(tmp_path):
    _write_tri(tmp_path / "tri_000000_NSA.csv", nsa_cols)   # existing helper
    _write_tri(tmp_path / "tri_000000_SA.csv", sa_cols)
    out = build_ces_panel(tmp_path, as_of=date(2026, 6, 1))
    sa = out.filter((pl.col("industry_code") == "00") & pl.col("seasonally_adjusted"))
    nsa = out.filter((pl.col("industry_code") == "00") & ~pl.col("seasonally_adjusted"))
    assert sa.height > 0 and nsa.height > 0
    assert sa["size_class_type"].is_null().all()
```

- [ ] **Step 2 — run, expect FAIL** (`seasonally_adjusted` is `False`-only today).
  Run: `uv run pytest packages/nfp-ingest/tests/test_ces_builder.py -k sa_and_nsa -v`
- [ ] **Step 3 — implement.** Replace the single NSA glob/stamp with a loop over both adjustments. Sketch (adapt to the T0 decision):

```python
for adj_suffix, is_sa in (("NSA", False), ("SA", True)):
    for csv_path in sorted(path.glob(f"tri_*_{adj_suffix}.csv")):
        code6 = csv_path.stem[len("tri_"):-len(f"_{adj_suffix}")]
        ...  # existing per-file logic (entry lookup, _diagonals, taxonomy)
        parts.append(diag.with_columns(
            industry_type=..., industry_code=..., ownership=...,
            seasonally_adjusted=pl.lit(is_sa, pl.Boolean),
        ))
# the final .select(...) carries seasonally_adjusted from the column, not a literal
```
  If T0 chose **(B)**, branch the `_diagonals` call on `is_sa`.

- [ ] **Step 4 — run, expect PASS** (+ the existing NSA tests stay green).
  Run: `uv run pytest packages/nfp-ingest/tests/test_ces_builder.py -v`
- [ ] **Step 5 — offline cross-check.** Against the local `cesvinall`, assert the SA `00` print for a known ref-month matches the canonical store's SA `00` value to the unit (mirrors the existing NSA anchor check).
- [ ] **Step 6 — commit.** `feat(ingest): build_ces_panel also emits the SA triangles (seasonally_adjusted=True)`
- [ ] **Acceptance:** SA + NSA both emitted; NSA output unchanged; SA `00` matches canonical SA to the unit; `ruff` clean.

---

## T2 — QCEW `00` total (`nfp-vintages` acquire + `nfp-ingest` crosswalk) `[depends: none]` — ✅ DONE (d904a73 + cq 552c7f5)

**Done (2026-06-17).** `_prep_area_raw` keeps `own_code ∈ {'5','0'}` (drops govt 1/2/3); `industry.py` carries `QCEW_OWN_TOTAL='0'` + `QCEW_TOTAL_PULL` (industry='10', agglvl='10' — **primary-source verified**: `own_code=0` is exactly one area row at those coords, Jan-2024 = 152,393,725 persons). `build_qcew_panel` emits a parallel `('total','00','total')` track (÷1000, NSA, bmr=0) via shared `_cast_raw`/`_explode_monthly`; `ownership` now flows from a column. Dual-context safe: size-path (own_code=5-only) → empty total, private tree byte-identical (regression guard green). No sort-key change. Spec ✅ + code-quality ✅ (397 tests, ruff clean).

**Files:**
- Modify: `packages/nfp-vintages/src/nfp_vintages/rebuild_store.py` (`_prep_area_raw` — the `own_code == "5"` filter)
- Modify: `packages/nfp-ingest/src/nfp_ingest/qcew_crosswalk.py` (`build_qcew_panel` — own/area filter + add the `00` total mapping)
- Modify: `packages/nfp-lookups/src/nfp_lookups/industry.py` (a pull entry for the QCEW total → `00`)
- Test: `packages/nfp-ingest/tests/test_qcew_crosswalk.py`

QCEW publishes the total as one `own_code=0` row (`industry '10'`, agglvl 10) — verified on the area endpoint (`own_code ∈ {0,1,2,3,5,8,9}`). Keep it and map it to CES `'00'`/`ownership='total'`, NSA, `bmr=0`. **No government.**

- [ ] **Step 1 — failing test.** In `test_qcew_crosswalk.py`, feed a synthetic raw frame containing the private rows **and** a single `(own_code='0', industry_code='10', agglvl_code='10')` total row; assert `build_qcew_panel` emits a `(industry_type='total', industry_code='00', ownership='total')` monthly series (÷1000), distinct from the private `05`.

```python
def test_qcew_total_maps_to_00():
    raw = _raw_rows(private=..., total_own0=_row(own_code="0",
                    industry_code="10", agglvl_code="10", emp=160_000_000))
    out = build_qcew_panel(raw.with_columns(pl.lit(0).alias("revision")))
    tot = out.filter((pl.col("industry_code") == "00") & (pl.col("ownership") == "total"))
    assert tot.height == 3  # one per month of the quarter
    assert tot["employment"][0] == pytest.approx(160_000.0)  # persons -> thousands
```

- [ ] **Step 2 — run, expect FAIL** (`build_qcew_panel` drops everything but `own_code=5` today).
  Run: `uv run pytest packages/nfp-ingest/tests/test_qcew_crosswalk.py -k total_maps_to_00 -v`
- [ ] **Step 3 — implement.**
  - `_prep_area_raw`: keep `own_code` ∈ `{"5", "0"}` (was `== "5"`).
  - `industry.py`: add `QCEW_TOTAL_PULL = (own_code='0', industry_code='10', agglvl='10') -> ('total','00','total')`.
  - `build_qcew_panel`: after the private path, pull the total row by `QCEW_TOTAL_PULL`, tag `industry_type='total'`, `industry_code='00'`, `ownership='total'`, run the same monthly explode (÷1000, per-vintage). Do **not** let it enter the private nesting sums.
- [ ] **Step 4 — run, expect PASS** (+ existing crosswalk tests green: private nesting `05=06+08`, supersectors sum — the `00` total must NOT perturb them).
  Run: `uv run pytest packages/nfp-ingest/tests/test_qcew_crosswalk.py -v`
- [ ] **Step 5 — commit.** `feat: map the QCEW own_code=0 total to CES 00 (ownership=total)`
- [ ] **Acceptance:** QCEW `00` total emitted, ÷1000, NSA, `bmr=0`; private hierarchy unchanged; `ruff` clean.

---

## T3 — Compose SA + gate updates (`nfp-ingest`, `nfp-vintages`) `[depends: T1, T2]` — ✅ DONE (c8a1cc1 + cq 6ce2697)

**Done (2026-06-17).** Added shared `_nsa_only` guard. **Filter-vs-key by gate intent:** SA *excluded* from the summing/NSA-compare rails (`gate_history_consistency`, `gate_gap_fill`/`_check_additive_nesting`, `gate_reconstruction_accuracy` CES side — SA doesn't nest, QCEW is NSA); SA *added to the identity key* in `gate_ces_fidelity` (SA↔SA, NSA↔NSA — the SA rail; fixes a false HARD fail from the shared-vintage_date 1×2 fan-out) and `gate_vintage_integrity` dup key. Added a **PROVISIONAL** `'00'` residual band to all three reconstruction constants (T4 calibrates). `compose_rebuild_panel` needed **no change** (unions CES wholesale; §7 is QCEW-size-only) — verified by `TestComposeCarriesSaAndTotal`. `_row` test default flipped True→False so the filters are genuine no-ops on existing NSA frames; negative-control tests prove each filter load-bearing. Spec ✅ + code-quality ✅ (148 vintages tests, ruff clean).

**Files:**
- Verify/Modify: `packages/nfp-vintages/src/nfp_vintages/rebuild_store.py` (`compose_rebuild_panel`)
- Modify: `packages/nfp-vintages/src/nfp_vintages/rebuild_gates.py` (`_EXPECTED_QCEW_CES_RESIDUAL` + SA-exclusion)
- Test: `packages/nfp-vintages/tests/test_rebuild_store.py`, `packages/nfp-vintages/tests/test_rebuild_gates.py`

- [ ] **Step 1 — compose test.** `compose_rebuild_panel(ces_with_sa, qcew_with_00, size)` carries SA CES rows through unchanged (the §7 Q1 size override is QCEW-only; SA CES has null size so it is never anti-joined). Assert SA rows present in the output and untouched.
- [ ] **Step 2 — run/implement.** `compose_rebuild_panel` already unions `ces` wholesale; expect this to PASS without change (the test is the regression guard). If the §7 override touches CES rows (it should not — it joins on the size frame), fix to scope strictly to `source='qcew'` size `'0'` rows.
- [ ] **Step 3 — gate test: SA excluded from NSA nesting.** Add a `gate_gap_fill` / `gate_reconstruction_accuracy` unit test feeding a frame with SA rows and assert they do **not** enter the additive-nesting sums or the QCEW≤CES residual (the gates must pre-filter `seasonally_adjusted=False`). Implement the pre-filter where missing.
- [ ] **Step 4 — gate test: `00` reconstruction band.** Add a `'00'` key to `_EXPECTED_QCEW_CES_RESIDUAL` + `_QCEW_CES_RESIDUAL_BAND`; unit-test that a QCEW `00` total within band passes and out-of-band fails. (Set the expected band in T4 from the observed real-store residual; leave a clearly-marked provisional value here that T4 calibrates.)
- [ ] **Step 5 — `gate_ces_fidelity` SA rail.** Extend the real-store `test_ces_fidelity_real` to also assert rebuilt **SA** CES == `build_ces_panel(cesvinall)` SA rows to the unit.
- [ ] **Step 6 — run all rebuild unit suites + ruff.**
  Run: `uv run pytest packages/nfp-vintages/tests/test_rebuild_store.py packages/nfp-vintages/tests/test_rebuild_gates.py -q --no-cov`
- [ ] **Step 7 — commit.** `feat(rebuild): compose SA CES + 00 QCEW; gates exclude SA, add 00 band`
- [ ] **Acceptance:** SA composed through; NSA gates ignore SA; `00` band present; unit suites + ruff green.

---

## T4 — Scratch rebuild + drop-in verification (maintainer-run) `[depends: T1, T2, T3]`

**Files:** none (run + record). Network + scratch write.

**Pre-handoff (controller, 2026-06-17): T1–T3 code-complete + locally de-risked.**
Full fast suite green (661 passed, all packages). Local drop-in seam confirmed
**without network/S3**: real `build_ces_panel(CESVINALL_DIR)` + a synthetic QCEW
(`own_code=0` total + a private sector) through `compose_rebuild_panel` yields a
composed frame carrying `source=ces` **SA** `00` (741 rows), `source=ces` NSA `00`
(562), and `source=qcew` `00`/`total` (3). `00` cohort breakdown: prints
`(0,0)/(1,0)/(2,0)` **identical** SA↔NSA (109/108/107); the SA surplus is entirely
in `(2,1)` (SA 417 vs NSA 238) — expected (SA re-bases each February). So the
maintainer's scratch build will write exactly the two previously-missing series;
Steps 1/3/4 below are the operational confirmation on real S3 data.

- [ ] **Step 1 — rebuild to scratch.** `NFP_STORE_URI=s3://alt-nfp/store-rebuild uv run alt-nfp build-rebuild`. Confirm CES row count ≈ doubles (SA + NSA) and a QCEW `00` total appears; canonical untouched (guard holds).
- [ ] **Step 2 — calibrate the `00` band (T3 Step 4).** Read the rebuilt `00` QCEW vs CES `00` SA residual at benchmark months; set `_EXPECTED_QCEW_CES_RESIDUAL['00']` + band from the observed value; commit the calibration.
- [ ] **Step 3 — drop-in check.** `NFP_STORE_URI=s3://alt-nfp/store-rebuild` →
  `build_model_data(as_of=date(2023,7,12))` returns `qcew_obs` non-empty **and** `g_ces_sa` populated (contrast the pre-fix `qcew_obs=0`/`ces_sa=0`). Repeat for the A1/A2 as-of dates ≥ 2020.
- [ ] **Step 4 — gates green.** `NFP_STORE_URI=…store-rebuild NFP_LEGACY_STORE_URI=…store uv run pytest packages/nfp-vintages/tests/test_rebuild_gates.py::TestGatesAgainstRealStore -m real_store --no-cov` — all 7 wrappers pass (now incl. the SA `ces_fidelity` rail + the `00` band).
- [ ] **Step 5 — record + commit.** Update `plans/10` T7 from BLOCKED → unblocked (drop-in verified, date); note the store now carries SA+NSA+QCEW-`00`.
- [ ] **Acceptance:** `build_model_data` non-degenerate against the rebuilt store at `'00'`; all 7 real-store gates green; `plans/10` T7/T8 unblocked.

---

## Deferred (explicitly NOT this plan)

The "B" architecture (modeling the private NSA hierarchy + composing the SA `00` from private + government), the government ownership axis (`own_code` 1/2/3; codes `90`–`93`), from-scratch seasonal adjustment, and the A1/A2 golden re-baseline itself (`plans/10` T7) + promotion (`plans/10` T8). See `specs/store_rebuild.md` §11.

## Open risks

- **T0 unresolved SA `(rev,bmr)`** — if SA seasonal-factor revisions don't fit the print/benchmark model, T1 grows an SA-specific path; the `gate_ces_fidelity` SA rail (T3 Step 5) is the backstop that catches a mis-assignment to the unit.
- **QCEW `00` definitional band** — total-covered vs total-nonfarm; calibrated from data in T4 Step 2, not asserted equal.
- **Store size doubles** (SA + NSA) — confirm write/read performance on the scratch Hive store is acceptable; partition by `seasonally_adjusted` already exists.
