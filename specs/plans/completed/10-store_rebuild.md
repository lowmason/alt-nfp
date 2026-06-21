# Vintage-Store Rebuild Implementation Plan (plans/10)

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **Execution targets a SCRATCH prefix only** (`s3://alt-nfp/store-rebuild`) — never the canonical store — until the §10 acceptance gates pass and the maintainer approves promotion (T8).

**Design record:** [`specs/completed/store_rebuild.md`](../specs/completed/store_rebuild.md) (the 12-section spec; consistency- and coherence-verified). Companions: `ces_qcew_industry.md`, `size_classes.md`, `ces_growth_convention.md`, `bloomberg_consensus.md`, `store_audit_findings.md`.

**Goal:** Rebuild the national vintage store cleanly from public BLS triangular + bulk files: a **NSA**, **vintage-aware** store carrying the **private industry hierarchy** (`ownership='private'`) plus the **`00` total-nonfarm scoring anchor** (`ownership='total'`, stored-not-modeled), with the new `ownership` axis, the `size_class` cross-product, and the provenance-keyed `(rev, bmr)` convention. Build to scratch, validate against acceptance gates, then promote.

**Architecture:** Same uv workspace + linear data-package chain (`nfp-lookups → nfp-download → nfp-ingest → nfp-vintages`). The rebuild is a deliberate **divergence** from the frozen reference — there is **no frozen-reference parity gate** here; acceptance is the §10 gate set, and A1/A2 goldens are **re-baselined** against the rebuilt store (T7), not held.

**Tech stack:** Python 3.12, Polars, Typer CLI, pytest, uv workspace; store I/O via `nfp_lookups.paths` (`storage_options_for`, `is_canonical_store`).

------------------------------------------------------------------------

## Status (2026-06-16)

Builder + gate **code** is complete (T0–T6), and **T6 acceptance is DONE — all gates green against the rebuilt store, with the Q1 defect FIXED and verified.** The §7 **Q1 all-sizes undercount** found in the first run (rebuilt Q1 headline light vs the published un-suppressed BLS area total) is resolved by a value-override in `compose_rebuild_panel` (the Q1 `'0'` headline now carries the area-levels total; metadata/vintage untouched). **Rebuilt to scratch with the fix + all 7 `real_store` gates re-run green** (2026-06-16): `qcew_fidelity` un-scoped to **all four quarters** reproduces the area endpoint to the unit (the former Q1 divergences are gone — 2024-Q1 15→0, 2025-Q1 75→0; 2025-Q1 `total/05` corrected from 106.4M / −18% to the published 129.7M exactly), and `reconstruction`/`ces_fidelity`/`history`/`gap_fill`/`q1_continuity`/`vintage_integrity` all stay green (the override moved March residuals toward the band without overshooting positive). Two §10 gate premises (history, reconstruction) were recalibrated against verified-correct rebuilt data with primary-source evidence (legitimate; see T6). **T7 (re-baseline goldens) was BLOCKED** on `build_model_data` being degenerate against the rebuilt store (0 QCEW observations vs 131 against canonical, because the rebuild omitted the SA CES + QCEW `00` series the model reads) — **[plans/11](11-model_data_dropin.md) resolved it 2026-06-17** by adding those two **published** series so the **unchanged** model reads the store non-degenerately (`qcew_obs=71`, `g_ces_sa=77`; 7 `real_store` gates green; `00` band calibrated). **T7 is now DONE** (re-baselined 2026-06-17 — A1 11/11 + A2 9/9 green on scratch goldens, manifests rewritten; the Oct-2025 shutdown moved the ref_date-gap expected-failure to `2026-01-12`; [plans/12](12-goldens_rebaseline.md)); the full SA `00` composition from the private hierarchy (spec §11) was not needed and stays deferred. **T8 (promotion) is DONE — cutover executed 2026-06-18** (copy-then-delete `store-rebuild`→`store` + goldens; prior canonical backed up at `…-prev-20260618` + local; A1 11/11 + A2 9/9 + 7 gates + model smoke green on the promoted canonical; PR #4 merged). See T7/T8.

| Task | State | Notes |
|---|---|---|
| **T0** Acquisition spike | ✅ **resolved** (local, 2026-06-15) | 3 unknowns resolved — [`store_rebuild_acquire.md`](../specs/store_rebuild_acquire.md). `cesvinall` reconstructs `(rev,bmr)` (verified vs store); QCEW size + levels via API slices. |
| **T0.5** QCEW vintaging | ✅ **resolved (A)** | Per-industry QCEW has **no** published revision history — rev-0 only (verified: only national `00` has rev 0–4; `qcew-revisions.csv` & the BLS revisions page are total-level). **Decision A** (2026-06-15): store rev-0, carry revision uncertainty model-side (`QCEW_REVISIONS` noise). No reconstruction. |
| **T1** Schema & grammar | ✅ **done** (`bc932ba`) | ownership axis, `national` retired, taxonomy + remap, `55` two-level, schema dedup (IND-XC-3), tolerant reader. |
| **T2** CES builder | ✅ **done** (`e662329`) | `nfp_ingest.ces_builder.build_ces_panel`: `cesvinall` → `(0,0)/(1,0)/(2,0)` + **per-benchmark** `(2,1)` (value+date same benchmark, no lookahead), ownership taxonomy, pure `as_of`. Spec + code-quality reviews passed; 10 tests, store anchors verified (Jun-2023 `00`). |
| **T3** QCEW crosswalk | ✅ **done** (`f399cc5`) | `qcew_crosswalk.build_qcew_panel`; agglvl 13/14/15/16 pull tables in lookups; synthetic tests green. |
| **T4** Size-class cross-product | ✅ **done** (`a28de4e`) | `size_class.build_size_class_panel` + `all_sizes_predicate`; `size_classes.py` scheme; `size_class_*` schema cols. |
| **T5** Build orchestration | ✅ **done** (`fa5168a`; run 2026-06-16) | Compose + guarded write + `build-rebuild` CLI + httpx API-slice acquire (area per-qtr + size per-Q1; size crosswalk reuses `build_qcew_panel` via agglvl −10 remap; drops `disclosure_code='N'`; excludes the 61–64 dup). **Full rebuild ran to scratch** (CES 16,408 / QCEW 17,880 rows, 2017+) and is verified faithful to published QCEW (Other Services 80 == published QCEW NAICS 81 to the unit); canonical untouched. |
| **T6** Acceptance-gate validator | ✅ **DONE — all gates green, Q1 defect fixed** | 6 gates + new `gate_ces_fidelity` + 65 unit tests; 7 `real_store` wrappers pass against the rebuilt scratch (+ canonical for history). History gate **recalibrated** (benchmark-free vs legacy HARD; benchmark cohorts SOFT) after cesvinall cross-check proved the rebuild faithful and the legacy splice buggy; `gate_ces_fidelity` is the new HARD CES accuracy rail (rebuilt==cesvinall). **§7 Q1 all-sizes undercount FIXED** (compose value-override → Q1 `'0'` = area total); `qcew_fidelity` un-scoped to **all four quarters**, 0 divergences after the fix-rebuild. |
| **T7** Re-baseline goldens | ✅ **DONE 2026-06-17 ([plans/12](12-goldens_rebaseline.md))** | A1 **11/11** + A2 **9/9** green vs rebuilt store + scratch goldens (`…/golden/a1-rebuild`, `…/a2-rebuild`); manifests re-baselined (2017+, BD arrays dropped, shutdown EF `2026-02-12`→`2026-01-12`); frozen goldens untouched. PR held for T8. See T7. |
| **T8** Promotion | ✅ **DONE 2026-06-18** | Cutover executed (copy-then-delete `store-rebuild`→`store` + goldens; prior canonical backed up at `…-prev-20260618` + local `data/canonical_backup_20260618/`); A1 11/11 + A2 9/9 + 7 §10 gates + model smoke green on the promoted canonical; PR #4 merged. Follow-ups: A3 goldens + `snapshots/` now stale. |

Full non-network suite green (513 passed; only the 2 pre-existing `claims`/`jolts` indicator env-failures, unrelated to the rebuild). `ruff check .` clean.

**Acquire gaps for T0/T5 to confirm + fix** (found reading `nfp_download.bls.bulk`):

-   `download_qcew_bulk._WANTED_AGGLVL = {10,11,14,15,50,51,54,55}` **excludes agglvl 13** (supersector pulls `1012`–`1027`, used by T3) **and 16** (Logging `1133`); `_KEEP_COLUMNS` **omits `size_code`** (used by T4). Rebuild acquire must widen the filter to `{13,16,<size-agglvls>}` and keep `size_code`.
-   QCEW size data is **inside** the `{year}_qtrly_singlefile.zip` (not a separate file), Q1 only, at size agglvl codes the spike must enumerate.

------------------------------------------------------------------------

## POLICY (rebuild-specific — replaces the plans/8-9 frozen-reference parity)

1.  **Scratch-only builds (hard).** Every build writes to `NFP_STORE_URI=s3://alt-nfp/store-rebuild`. The `is_canonical_store` guard (on `main`) must refuse `…/store`. The canonical store only ever takes the explicit, post-validation promotion in T8 (`--allow-canonical`).
2.  **Store-write safety (hard, carry-over).** Never run a store-writing function against the canonical store in a test — `tmp_path` / synthetic frames only. The root `conftest.py` auto-loads live prod creds and the `_block_live_store` autouse fixture severs s3fs for unmarked tests; do not defeat it. (A red-phase guard test once wiped the canonical store — recovered from the reference.)
3.  **Parity retired → acceptance gates.** Promotion is gated on §10, not on byte-parity vs `~/Projects/alt_nfp`. Re-baseline A1/A2 (T7); document the divergence in the goldens manifest.
4.  **Public data only.** `data/` is gitignored; provider + consensus data are proprietary and never committed. Store tests self-skip without store env.

------------------------------------------------------------------------

## T0 — Acquisition spike — ✅ DONE (2026-06-15) → [`specs/store_rebuild_acquire.md`](../specs/store_rebuild_acquire.md)

All three unknowns resolved (read-only: cached `cesvinall` + live QCEW API slices, cross-checked vs the store):

-   [x] **CES triangular coverage** — GO. 113 NSA codes (full hierarchy). `cesvinall` reconstructs `(0,0)/(1,0)/(2,0)/(2,1)` by itself; verified vs store to the unit (Jun-2023 `00` NSA = 156963/156945/156905/156701). `(2,1)` is **per-benchmark** (each Feb re-basing); the "bulk benchmarked file" is just the triangle's latest vintage row.
-   [x] **QCEW size-class coverage** — GO. Size endpoint `/{year}/1/size/{1-9}.csv` (Q1 only), national = private (`own_code=5`), `size_code` 1–9, agglvls `{21–28}`, includes supersector pulls + sectors.
-   [x] **NAICS vintage** — GO (low risk). NAICS-2022-for-all OK at supersector/sector aggregation; spot-check 3-digit durable/nondurable.
-   [x] **Acceptance note** written: `specs/store_rebuild_acquire.md`, with go/no-go per task.

**Bonus:** the QCEW acquire can use targeted API slices (US000 area per-qtr carries agglvl 13/16; size per-Q1) instead of the 280 MB singlefiles.

## T0.5 — QCEW historical vintaging — ✅ RESOLVED (decision A, 2026-06-15)

Investigation outcome (corrects the earlier "reference reconstructs per-industry
vintages" read): **per-industry QCEW vintages do not exist.** BLS publishes
revision history **only at the national total** — `qcew-revisions.csv` and the
whole `bls.gov/cew/revisions/` page are area×field with no industry/size breakdown
(verified by fetch), the open-data API serves current values only, and historical
singlefiles are overwritten. In the existing store this shows up cleanly: only
`industry_code='00'` carries rev 0–4; every per-industry private code is rev-0.
The `qcew_vintages.parquet` / `load_qcew_vintages` / `ingest_qcew` path that would
have held per-industry revisions is a **dead stub** — nothing writes it, no live
callers. The live pipeline (`nfp_vintages/processing/qcew_bulk.py`) makes
per-industry rev-0 (bulk) + national-`00` rev 0–4 (revisions CSV).

-   [x] **Decision A** — store per-industry QCEW as a single `revision=0` row
    (current value); carry revision uncertainty **model-side** via the
    `QCEW_REVISIONS` noise schedule. No per-industry reconstruction.
    (Rejected **B**: proportional synthesis from total-level revision ratios —
    manufactures data BLS doesn't publish, assumes uniform per-industry revision.)
    Spec §5 corrected to match. T5's QCEW path is therefore just **acquire current
    (API slices) → T3 crosswalk → rev-0**; see `store_rebuild_acquire.md`.

------------------------------------------------------------------------

## T1 — Schema & grammar (`nfp-lookups`) `[depends: T0]` — ✅ DONE (`bc932ba`)

-   [x] Add `ownership` (str) to `VINTAGE_STORE_SCHEMA`; values `{private, total}` (reserve `government`, not yet written).
-   [x] Retire `industry_type='national'`; set the enum to `{total, domain, supersector, sector}`. Encode the §3 taxonomy table (`industry_type × ownership → code`) as the canonical mapping, including the `00`=`(total,total)` anchor and `05`=`(total,private)` root.
-   [x] Add an **old→new `industry_type` remap** helper for the ≤2023 history join (`national/00`→`(total,total)`, `domain/05`→`(total,private)`, supersectors/sectors unchanged) — used by T6.
-   [x] Update the series-ID grammar / hierarchy helpers; ensure **code `55`** is representable at both `supersector` and `sector` levels (the cross-level collision the keys must survive).
-   [x] **Acceptance:** unit tests for the taxonomy map, the remap, and the `55` two-level representation; `ruff` clean; no upward imports. → `nfp_lookups.{INDUSTRY_TAXONOMY, ownership_for, codes_for, industry_types_for_code, remap_industry_type}`; tests in `test_industry_taxonomy.py` (21). Also deduped the duplicate `VINTAGE_STORE_SCHEMA` (ingest imports from lookups; IND-XC-3) and made `read_vintage_store` tolerant of legacy stores (`missing_columns="insert"`).

------------------------------------------------------------------------

## T2 — CES builder (`nfp-ingest`) `[depends: T1]` — ✅ DONE (`e662329`)

`nfp_ingest.ces_builder.build_ces_panel(cesvinall_dir=None, *, as_of=None)`. Note:
T0 showed `cesvinall` alone reconstructs every `(rev,bmr)` — including the
benchmark `(2,1)` (the triangle's February column-steps) — so the builder is
**triangle-sourced** (no separate bulk file), and `(2,1)` is **per-benchmark**
(decision this session) rather than the spec's original single-bulk row.

-   [x] Triangular `cesvinall` → `(0,0)/(1,0)/(2,0)` prints, `bmr=0`, ownership taxonomy (`00`→`total/total`, `05`→`total/private`, `06/08`→`domain/private`, supersectors/sectors→`private`); `07/90–93` dropped.
-   [x] Per-benchmark `(2,1)`: one row per distinct annual-benchmark basis (value + `vintage_date` both from the same `(Y,1)` release — no lookahead; unchanged later benchmarks skipped). Pure `as_of` frontier filter.
-   [x] Day-12 `ref_date`; NSA; thousands; 2017+; `nfp_lookups`-only imports; no store I/O.
-   [x] **Acceptance:** 10 tests (synthetic + `as_of` gating + dedup) + offline `cesvinall` cross-check — Jun-2023 `00` NSA: prints `156963/156945/156905`, `(2,1)`={`156842`@2024-02-02, `156701`@2025-02-07}. Spec-compliance + code-quality reviews passed.

------------------------------------------------------------------------

## T3 — QCEW crosswalk + monthly explode (`nfp-ingest`) `[depends: T1]` — ✅ DONE (`f399cc5`)

-   [x] Crosswalk per `ces_qcew_industry.md`: `own_code=='5'`→`ownership='private'`, `area_fips=='US000'`, aggregate `(industry_code, agglvl)` cells into the CES private codes (§3). Apply the structural sums (`10`=`21`+Logging `1133`; Durable/Nondurable `31`/`32`). Drop raw-NAICS provenance.
-   [x] Explode `month1/2/3_emplvl` → monthly rows, ÷1000. Sum the measure, never a rate. **Per-vintage aggregation:** never cross a QCEW `(rev, vintage_date)`.
-   [x] `vintage_date` via `revision_schedules.get_qcew_vintage_date`; depth Q1=4/Q2=3/Q3=2/Q4=1; `bmr=0` always.
-   [x] **Acceptance:** tests for the crosswalk sums, the ÷1000 units, per-vintage isolation, and additive nesting (`05 = 06 + 08`, supersectors sum, sectors sum) on a synthetic frame. → `nfp_ingest.qcew_crosswalk.build_qcew_panel`; pull tables in `nfp_lookups` (`QCEW_SECTOR_PULLS/SUPERSECTOR/DOMAIN/AGGLVL/OWN_PRIVATE/AREA_NATIONAL`); `test_qcew_crosswalk.py` (10). Supersectors use the agglvl-13 direct pull (`10` sums its sectors); domains/`05` roll up from supersectors.

------------------------------------------------------------------------

## T4 — Size-class cross-product (`nfp-ingest`) `[depends: T1, T3]` — ✅ DONE (`a28de4e`)

Spec §8.

-   [x] Q1 only (ref-month ∈ {01,02,03}): ingest native `size_code` 1–9 (`large`), derive `small`/`medium` via `size_class_members` rollup, `total`(`'0'`) by summing natives. Never join `small`/`medium` to raw QCEW.
-   [x] Cross-product `industry_code × size_class_type`; rows inherit the parent's `(rev, vintage_date)` and `ownership='private'`.
-   [x] On Q1 emit the all-sizes level as `total`/`'0'` **only** — no null-size row (avoids the §7 `IS NULL OR size_class_code='0'` double-count). Null `size_class_*` for CES + QCEW Q2/Q3/Q4.
-   [x] **Acceptance:** tests for the rollup, the Q1-only rule, the no-null-row invariant, and the all-sizes selector returning one row per Q1 month. → `nfp_ingest.size_class.{build_size_class_panel, all_sizes_predicate}`; scheme in `nfp_lookups.size_classes`; `size_class_{type,code}` added to `VINTAGE_STORE_SCHEMA`; tests `test_size_classes.py` (7) + `test_size_class.py` (18).

------------------------------------------------------------------------

## T5 — Build orchestration → scratch (`nfp-vintages` CLI) `[depends: T2, T3, T4]` — ✅ DONE (`fa5168a`; run 2026-06-16)

-   [x] **Compose glue (done, unit-tested):** `compose_rebuild_panel(ces, qcew_levels, size=None)` unions the three panels via `diagonal_relaxed` (null-fills the size cols `build_qcew_panel` omits) and enforces §7: for Q1, drop a `qcew_levels` null-size row **only** where the size frame has a `total`/`'0'` (all-sizes) row for that 6-col series identity (`geo_type, geo_code, ownership, industry_type, industry_code, ref_date`) — a conditional anti-join, **not** a month filter, so partial-coverage industries keep their null-size level. Q2–Q4 use the T3 null-size rows. Tests cover no-double-emit (exactly one `all_sizes_predicate` row), partial coverage (both branches), and non-Q1 never dropped.
-   [x] **Guard (done):** `write_rebuild_store(panel, store_path=None, *, allow_canonical=False)` raises before any I/O when `is_canonical_store(store_path)` and not `allow_canonical`; mirrors `build_store`'s Hive write (untouched). Null-`vintage_date` partitions fail loud (no `v_None_None.parquet`).
-   [x] **CLI (wired):** `alt-nfp build-rebuild [--allow-canonical]` wires `build_ces_panel()` → acquire-QCEW → acquire-size → compose → guarded write. The two acquire steps are `NotImplementedError` seams (`_acquire_qcew_levels`, `_acquire_qcew_size_native`) pointing to `store_rebuild_acquire.md`.
-   [x] **Acquire layer (done):** httpx API-slice fetchers — `_acquire_qcew_levels` (area per-qtr `…/api/{y}/{q}/area/US000.csv`, all 4 qtrs) + `_acquire_qcew_size_native` (size per-Q1 `…/api/{y}/1/size/{1-9}.csv`), `own_code=5`, 2017+, `revision=0`, 404-tolerant. **Transport = plain httpx** (`create_client`) — `data.bls.gov` is not Akamai-fingerprinted (only www.bls.gov is); the singlefile/`_WANTED_AGGLVL` path stays untouched. Disclosure: drop `disclosure_code='N'` (withheld); the 61–64 duplicate family is excluded by the 21–28 filter.
-   [x] **QCEW size crosswalk (done — no new pull maps needed):** the size tree = the area tree shifted **+10 agglvl** (verified live: 23=supersector…26=4-digit incl. `1133`). So `_size_raw_to_native` remaps agglvl **−10** and reuses the T3-tested `build_qcew_panel`, run **once per `size_code`** (its grouping has no size axis — a combined call would sum across size classes), then re-tags `size_code` → `native`. Suppression contained to sectors `31`/`32`/`11` (3/4-digit); hard-gate levels exact. Tests: per-size_code independence, 61–64 exclusion, disclosure null-safety, round-trip through `build_size_class_panel`.
-   [x] **Acceptance run (done, maintainer, 2026-06-16):** `alt-nfp build-rebuild` ran to `s3://alt-nfp/store-rebuild` — CES 16,408 / QCEW 17,880 rows (2017+); structure verified (ownership axis, four `(rev,bmr)`, size cross-product, `05=06+08` exact) and **faithful to published QCEW** (Other Services `80` == published QCEW NAICS `81` to the unit @ 2024-06). Canonical untouched (guard held). Known frontier-lag: 2025-Q1 size/sector-detail tables hadn't published (SOFT-warned by the reconstruction gate; fills on the next rebuild).

------------------------------------------------------------------------

## T6 — Acceptance-gate validator (§10) `[depends: T5]` — ✅ DONE: all gates green on the real stores (2026-06-16)

Gap-collector gates in `nfp_vintages.rebuild_gates` (+60 unit tests). Key on `industry_type + industry_code + ownership + (rev,bmr)` via the T1 remap (code `55` unambiguous). Run via the `@real_store` wrappers in `test_rebuild_gates.py::TestGatesAgainstRealStore`: `NFP_STORE_URI=s3://alt-nfp/store-rebuild` (rebuilt) and, for history, `NFP_LEGACY_STORE_URI=s3://alt-nfp/store` (legacy). **All 7 wrappers pass.**

-   [x] **History consistency** (`gate_history_consistency`): **RECALIBRATED + ✅ GREEN** (dual-store). The original premise — "rebuilt reproduces the legacy store to 0.5k on all four `(rev,bmr)` cohorts" — proved wrong for the **benchmark-bearing** cohorts. Primary-source check vs `cesvinall` (2026-06-16): rebuilt `(2,0)` *and* `(2,1)` reproduce the **literal triangle cells to the unit** (incl. the per-benchmark `(2,1)` fan-out, e.g. Jun-2019 `00` = {151739@2020, 151716@2021, 151714@2023, 151713@2025}); the **legacy store deviates** there — its `(2,1)` mis-stamped the *latest* benchmark value under the *earliest* `vintage_date` (a lookahead bug). The benchmark-**free** `(0,0)`/`(1,0)` prints reproduce the legacy store **exactly** (0/2520 diverge). So: HARD on `(0,0)`/`(1,0)` vs legacy + the four-combo `(rev,bmr)` population; the `(2,0)`/`(2,1)`-vs-legacy divergence is **SOFT** (legacy splice ≠ cesvinall; the rebuild diverges *toward* ground truth).
-   [x] **CES fidelity** (`gate_ces_fidelity`, **new**): the HARD CES accuracy rail (CES analogue of `gate_qcew_fidelity`) the history recalibration relies on — rebuilt CES `==` a fresh `build_ces_panel(cesvinall)` to the unit on the full per-vintage key, so a real benchmark-walk regression the legacy comparison no longer catches fails HERE. **✅ GREEN** against the rebuilt store.
-   [x] **Gap fill** (`gate_gap_fill`): HARD `05`+supersectors-to-frontier + Dec `(2,1)`; SOFT additive nesting. **✅ GREEN** (0 HARD; 4 SOFT nesting drifts ≤7 jobs at the unsettled 2025 frontier). Wrapper derives `frontier_ref_date`/`dec_cohort_years` from the store.
-   [x] **Reconstruction accuracy** (`gate_reconstruction_accuracy`): **RECALIBRATED** (`1fba229`) — per-series QCEW≤CES bands `_EXPECTED_QCEW_CES_RESIDUAL` ±`_QCEW_CES_RESIDUAL_BAND`; SOFT-warns the unsettled frontier + COVID. **✅ GREEN** (0 HARD). `specs/ces_qcew_industry.md` §8 corrected.
    -   *Q1 continuity (T5 carry-over):* `gate_q1_continuity` — temporal-neighbour proxy; SOFT/diagnostic-only. **✅ GREEN** (SOFT surfaced).
-   [x] **QCEW fidelity** (`gate_qcew_fidelity`): rebuilt-QCEW vs the **area endpoint** (the store's own source) to the unit. The real wrapper was broken (CSV inferred `area_fips` as i64 → choked on `"C1010"`; also fetched the wrong product, the `_qtrly_singlefile`). **Fixed**: reference = `_acquire_qcew_levels` of `/api/{y}/{q}/area/US000.csv`. **✅ GREEN over all four quarters** after the §7 Q1 fix-rebuild (the former Q1 divergences are gone — 2024-Q1 15→0, 2025-Q1 75→0).
-   [x] **Vintage integrity** (`gate_vintage_integrity`): no dups, one vintage per series-month, no null/zero. **✅ GREEN**.
-   [x] **Acceptance:** all 7 `real_store` wrappers pass against the rebuilt scratch (+ canonical-for-history), **`qcew_fidelity` over all four quarters**. Complete.

**✅ RESOLVED — §7 Q1 all-sizes undercount (fixed + verified 2026-06-16).** The first run's store had a **Q1 all-sizes** headline (`size_class_code='0'`) that was **light vs the published, un-suppressed BLS area total**: the §7 compose substituted the size-cross-product `total/'0'` = sum of the *available* native buckets, which undercounts whenever buckets are missing (suppression **or** unpublished frontier tables). Quantified (Q1, vs the live area endpoint, `under = published − store`): settled 2017–2024 headline `total/05` max ~3.9k (0.003%) but sector `11` up to **8.5%** / sector `32` ~1% (~45k); **frontier 2025-Q1 catastrophic** — `total/05` **23.4M / ~18% too low**, supersectors `20`/`30`/`10`/`50` **81–93% under** (the partial-size-table case turned a benign "detail not yet published" into a malignant "headline 18% wrong").

**Fix (Option A2 — compose value-override):** `compose_rebuild_panel` now overrides **only** the Q1 `'0'` row's *employment* to the area-levels total (left-join area on `_SERIES_IDENTITY_KEY`, `coalesce`-fallback to the bucket-sum if no area row), leaving the bucket rows + all `(rev, vintage_date)`/metadata untouched (so vintage-integrity is unchanged; area totals nest by BLS construction, so `05=06+08` additive closure is restored at Q1). 5 CI unit tests (`TestComposeQ1HeadlineCarriesAreaTotal`); specs §8/§9 + `size_classes.md` + the `gate_q1_continuity` docstring corrected; `qcew_fidelity` un-scoped to all four quarters. **Rebuilt to scratch + all 7 gates re-run green; the former Q1 divergences are eliminated** (2024-Q1 15→0; 2025-Q1 75→0, `total/05` 106.4M→129.7M = published exactly). The adversarial-review workflow's flagged risk — `gate_reconstruction_accuracy` March residual overshooting positive — did **not** materialize (still 0 HARD). The §7 fix was a peer-session hand-off; this session owned the implementation + the scratch-rebuild verification. *(The 2025-Q1 frontier lag was already noted under T5 as "fills on the next rebuild"; this finding shows it also corrupts the headline until §7 is fixed.)*

------------------------------------------------------------------------

## T7 — Re-baseline A1/A2 goldens `[depends: T6 passing]` — ✅ **DONE 2026-06-17** (plans/11 drop-in + [plans/12](12-goldens_rebaseline.md) re-baseline; A1 11/11 + A2 9/9 green on scratch goldens; PR held for T8)

**Unblocked (2026-06-17) by [plans/11](11-model_data_dropin.md)** — the *simpler* way this note anticipated. The model-data layer is **unchanged**; plans/11 added the two **published** series the NSA-only rebuild omitted: **SA CES** (`build_ces_panel` now emits the `_SA` triangles) and the **QCEW `00` total** (`own_code=0` → CES `00`/`ownership='total'`). Verified read-only against the scratch rebuild: `build_model_data(as_of=2023-07-12)` is now **non-degenerate** — `qcew_obs=71`, `g_ces_sa=77` (was `0`/`0`); likewise at 2021/2024 as-of dates. **All 7 `real_store` gates green** (incl. the new SA `ces_fidelity` rail + the calibrated `00` reconstruction band, median −0.0182). `qcew_obs=71` vs canonical 131 is the documented **2017+ truncation**, not a regression. The full private-hierarchy SA `00` composition + government axis (spec §11) stays deferred and was **not** needed. T7 (re-baseline goldens) + T8 (promotion) now proceed as scoped below.

*(Historical BLOCKED finding, 2026-06-16, retained for the record:)*

**Finding (2026-06-16): the rebuilt store is not yet a consumable substrate for `build_model_data`, so re-baselining A1/A2 now would pin a *degenerate* configuration.** Verified empirically (read-only, `NFP_STORE_URI=s3://alt-nfp/store-rebuild`):

-   `build_panel(as_of_ref=D)` runs against the rebuilt store but yields an **NSA-only** panel (4,699 rows vs the frozen reference's 11,020) with **0 CES `00` and 0 QCEW `00` rows** — the model-data layer does not surface the rebuilt store's NSA `00` anchor / ownership hierarchy as the CES/QCEW **target** series it reads.
-   `build_model_data(as_of=D)` therefore returns a **degenerate** dict: `qcew_obs` is shape `(0,)` (**zero** QCEW observations), `g_qcew` is all-NaN, `qcew_is_m2`/`qcew_noise_mult` empty. Only the **local-file** inputs (provider `G`, cyclical `claims_c`/`jolts_c`) are populated, because those don't come from the store. Contrast: the **canonical** store gives `qcew_obs` shape `(131,)`, `g_qcew` 131/137 non-NaN.

The model-data layer still consumes the **canonical SA schema**; it has **not** been adapted to the rebuilt NSA + ownership store. That adaptation is the **deferred downstream** (spec §11: "the downstream `00` SA composition" — and the NSA→SA consumption path). **T7 (and T8) are gated on it.** Pinning A1/A2 goldens against a 0-QCEW-observation input would baseline a broken modeling path, so no fixtures were generated (also: never overwrite the frozen-reference goldens at `s3://…/golden/a1|a2` — re-baseline must target a scratch golden prefix + staging manifests, promoted *alongside* T8, to keep the parity tests on `main` intact and the diff auditable).

Two divergences the original T7 note did not list, to fold in once unblocked: **2017+ history truncation** (rebuilt store starts 2017 vs `START_YEAR=2012`) and **which store validates the goldens** (generated from scratch, but the tests read `VINTAGE_STORE_PATH` = canonical until T8 — re-check `AS_OF_DATES`/`EXPECTED_FAILURE_DATES` against a 2017+ store).

-   [x] **Prereq — RESOLVED 2026-06-17 via [plans/11](11-model_data_dropin.md):** the model-data layer is **unchanged**; instead of adapting it, plans/11 added the two **published** series the rebuild omitted (SA CES + QCEW `00` total), so the existing `build_panel`/`panel_to_model_data`/`build_model_data` reads the rebuilt store non-degenerately (`qcew_obs` non-empty, `g_ces_sa` populated). The full SA `00` composition from the private hierarchy (spec §11) was **not** needed and stays deferred.
-   [x] **DONE 2026-06-17 ([plans/12](12-goldens_rebaseline.md)):** regenerated A1/A2 fixtures from the rebuilt store, staged to **scratch** prefixes `s3://alt-nfp/golden/a1-rebuild` + `…/a2-rebuild` (frozen `…/golden/a1|a2` untouched); committed manifests rewritten (`start_year` 2012→2017, BD arrays dropped, EF `2026-02-12`→**`2026-01-12`** for the Oct-2025 shutdown). **A1 11/11 + A2 9/9 green** with `NFP_STORE_URI`/`NFP_GOLDEN_URI` pointed at the rebuilt store + scratch goldens. Divergences (2017+ truncation, ownership/`00`/NSA schema normalization, ~31–53% row counts, shutdown EF swap) documented in the manifest `provenance.divergence` + plans/12 T4. Branch `goldens-rebaseline`; PR held for T8.

------------------------------------------------------------------------

## T8 — Promotion runbook (scratch → canonical) `[depends: T6, T7; MAINTAINER GO]` — ✅ **DONE 2026-06-18**

-   [x] **Cutover executed 2026-06-18** (maintainer GO). `mirror_store --allow-canonical` was deliberately NOT used: it is overwrite-only, and the partition filenames encode vintage ranges (`v_2003-…` canonical vs `v_2017-…` rebuilt), so a plain copy would leave BOTH files per partition and corrupt the store on read. Instead a guarded **copy-then-delete-old** cutover (per partition: copy the rebuilt file in, then delete the old-named orphan, then verify). Promoted `store-rebuild`→`store`, `golden/a1-rebuild`→`golden/a1`, `golden/a2-rebuild`→`golden/a2`.
-   [x] **Backups (retained).** Prior canonical snapshotted before any overwrite — **local** `data/canonical_backup_20260618/` (store + golden/a1 + golden/a2, 8.9 MB) **and S3** `…/store-prev-20260618`, `…/golden/a1-prev-20260618`, `…/golden/a2-prev-20260618`. The local copy is the durable one (survives MinIO loss); keep until the new store is confirmed downstream in the model.
-   [x] **Acceptance MET.** Object fidelity (canonical == rebuild content, no orphans); A1 **11/11** + A2 **9/9** green with **default env** (canonical `NFP_STORE_URI` + default `golden/a1|a2`); **7 §10 gates** green (promoted canonical vs the `-prev-` legacy backup); **model smoke** green (`build_model_data(2026-02-12)`→`fit_model`: T=109, qcew_obs=101, ces_sa_obs=108, 25 finite posterior sites, 0 divergences). PR #4 merged; root CLAUDE.md + `is_canonical_store` "irreplaceable/never-rebuild" framing retired.
-   **Rollback** (if the new store misbehaves downstream) — **copy-then-delete, NOT `mirror_store`/overwrite.** The same dual-filename trap applies in reverse: the backup's `v_2003-…` files and the live `v_2017-…` files have *different names*, so an overwrite-only mirror would leave BOTH per partition and corrupt the store. Correct procedure, per prefix (`store`, `golden/a1`, `golden/a2`): **(1)** copy every object from `…-prev-20260618` into the canonical prefix (same relative paths); **(2)** delete every canonical object whose key is NOT in the `-prev-` set (removes the rebuilt `v_2017-…` files); **(3)** verify the prefix now equals the `-prev-` set. Source of truth if the S3 `-prev-` prefixes are also lost: the durable local `data/canonical_backup_20260618/`. The one-time `scripts/_t8_promote.py rollback` (kept local, untracked) automates exactly this from `backup_manifest.json` — but the manual copy-then-delete above is script-independent.

**Follow-ups (NOT blockers):** the **A3 posterior goldens** (`s3://…/golden/a3`) still pin the *frozen* baseline — they now diverge from the rebuilt store, and the A3 spot-stem `2026-01-12` is the shutdown **expected-failure**, so A3 needs its own re-baseline before the parity gate runs green again. `s3://…/snapshots/` holds ModelData built from the **old** store — now stale; regenerate as needed.

------------------------------------------------------------------------

## Deferred (explicitly NOT this plan)

Government (`ownership='government'`, codes `07`/`90`–`93`, QCEW `own_code` 1/2/3) and the downstream `00` SA composition; live capture (BLS feed cron); geography beyond national; seasonal adjustment; NAICS vintage-aware crosswalks; births-deaths. See spec §11.

## Open risks

-   T0 may show `cesvinall` lacks some sector/benchmark rows → narrows T6's reconstruct-and-validate scope (sectors stay best-effort, per §10).
-   The reconstruction-accuracy tolerance (T6) is a judgement call with no reference number — set it from observed benchmark-month residuals, document it.
-   **Confirmed (reading `nfp_download.bls.bulk`):** the current QCEW acquire drops the agglvl `13`/`16` rows and `size_code` that T3/T4 consume — a small acquire fix (T5), but the existing `qcew_bulk.parquet` cannot feed the rebuild until the filter is widened. T0 pins the exact size agglvl codes.