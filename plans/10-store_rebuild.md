# Vintage-Store Rebuild Implementation Plan (plans/10)

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **Execution targets a SCRATCH prefix only** (`s3://alt-nfp/store-rebuild`) — never the canonical store — until the §10 acceptance gates pass and the maintainer approves promotion (T8).

**Design record:** [`specs/store_rebuild.md`](../specs/store_rebuild.md) (the 12-section spec; consistency- and coherence-verified). Companions: `ces_qcew_industry.md`, `size_classes.md`, `ces_growth_convention.md`, `bloomberg_consensus.md`, `store_audit_findings.md`.

**Goal:** Rebuild the national vintage store cleanly from public BLS triangular + bulk files: a **NSA**, **vintage-aware** store carrying the **private industry hierarchy** (`ownership='private'`) plus the **`00` total-nonfarm scoring anchor** (`ownership='total'`, stored-not-modeled), with the new `ownership` axis, the `size_class` cross-product, and the provenance-keyed `(rev, bmr)` convention. Build to scratch, validate against acceptance gates, then promote.

**Architecture:** Same uv workspace + linear data-package chain (`nfp-lookups → nfp-download → nfp-ingest → nfp-vintages`). The rebuild is a deliberate **divergence** from the frozen reference — there is **no frozen-reference parity gate** here; acceptance is the §10 gate set, and A1/A2 goldens are **re-baselined** against the rebuilt store (T7), not held.

**Tech stack:** Python 3.12, Polars, Typer CLI, pytest, uv workspace; store I/O via `nfp_lookups.paths` (`storage_options_for`, `is_canonical_store`).

------------------------------------------------------------------------

## Status (2026-06-15)

Pure-code, locally-verifiable tasks are done; data-dependent tasks (BLS network / store creds) await the maintainer's local runs.

| Task | State | Notes |
|---|---|---|
| **T0** Acquisition spike | ✅ **resolved** (local, 2026-06-15) | 3 unknowns resolved — [`store_rebuild_acquire.md`](../specs/store_rebuild_acquire.md). `cesvinall` reconstructs `(rev,bmr)` (verified vs store); QCEW size + levels via API slices. |
| **T0.5** QCEW vintaging | ✅ **resolved (A)** | Per-industry QCEW has **no** published revision history — rev-0 only (verified: only national `00` has rev 0–4; `qcew-revisions.csv` & the BLS revisions page are total-level). **Decision A** (2026-06-15): store rev-0, carry revision uncertainty model-side (`QCEW_REVISIONS` noise). No reconstruction. |
| **T1** Schema & grammar | ✅ **done** (`bc932ba`) | ownership axis, `national` retired, taxonomy + remap, `55` two-level, schema dedup (IND-XC-3), tolerant reader. |
| **T2** CES builder | ✅ **done** (`e662329`) | `nfp_ingest.ces_builder.build_ces_panel`: `cesvinall` → `(0,0)/(1,0)/(2,0)` + **per-benchmark** `(2,1)` (value+date same benchmark, no lookahead), ownership taxonomy, pure `as_of`. Spec + code-quality reviews passed; 10 tests, store anchors verified (Jun-2023 `00`). |
| **T3** QCEW crosswalk | ✅ **done** (`f399cc5`) | `qcew_crosswalk.build_qcew_panel`; agglvl 13/14/15/16 pull tables in lookups; synthetic tests green. |
| **T4** Size-class cross-product | ✅ **done** (`a28de4e`) | `size_class.build_size_class_panel` + `all_sizes_predicate`; `size_classes.py` scheme; `size_class_*` schema cols. |
| **T5** Build orchestration | 🟡 **core done** (`523f808`) | Compose + guarded scratch write + `build-rebuild` CLI wiring **done** (`nfp_vintages.rebuild_store.{compose_rebuild_panel, write_rebuild_store}`; §7 anti-join; 20 tests; spec + code-quality reviews passed). **Owed (maintainer / network):** the API-slice fetchers (area per-qtr + size per-Q1), the QCEW size NAICS→CES crosswalk (needs **new** agglvl-21–28 pull maps absent from lookups), and the actual small-window scratch run. Acquire seams are `NotImplementedError` stubs pointing to `store_rebuild_acquire.md`. |
| **T6** Acceptance-gate validator | ⬜ depends T5 | Gate *functions* are T0-independent and can be pre-built on synthetic frames. |
| **T7** Re-baseline goldens | ⬜ depends T6 | Needs scratch store. |
| **T8** Promotion | ⬜ depends T6/T7 + GO | Needs store + maintainer approval. |

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

## T5 — Build orchestration → scratch (`nfp-vintages` CLI) `[depends: T2, T3, T4]` — 🟡 CORE DONE (`523f808`)

-   [x] **Compose glue (done, unit-tested):** `compose_rebuild_panel(ces, qcew_levels, size=None)` unions the three panels via `diagonal_relaxed` (null-fills the size cols `build_qcew_panel` omits) and enforces §7: for Q1, drop a `qcew_levels` null-size row **only** where the size frame has a `total`/`'0'` (all-sizes) row for that 6-col series identity (`geo_type, geo_code, ownership, industry_type, industry_code, ref_date`) — a conditional anti-join, **not** a month filter, so partial-coverage industries keep their null-size level. Q2–Q4 use the T3 null-size rows. Tests cover no-double-emit (exactly one `all_sizes_predicate` row), partial coverage (both branches), and non-Q1 never dropped.
-   [x] **Guard (done):** `write_rebuild_store(panel, store_path=None, *, allow_canonical=False)` raises before any I/O when `is_canonical_store(store_path)` and not `allow_canonical`; mirrors `build_store`'s Hive write (untouched). Null-`vintage_date` partitions fail loud (no `v_None_None.parquet`).
-   [x] **CLI (wired):** `alt-nfp build-rebuild [--allow-canonical]` wires `build_ces_panel()` → acquire-QCEW → acquire-size → compose → guarded write. The two acquire steps are `NotImplementedError` seams (`_acquire_qcew_levels`, `_acquire_qcew_size_native`) pointing to `store_rebuild_acquire.md`.
-   [ ] **Owed — acquire layer (maintainer / network):** implement the QCEW **API-slice** fetchers (area per-qtr `…/api/{y}/{q}/area/US000.csv` + size per-Q1 `…/api/{y}/1/size/{1-9}.csv`, `own_code=5`, 2017+; tag `revision`). **The old "widen `download_qcew_bulk._WANTED_AGGLVL`" idea is superseded** — that filters the 280 MB *singlefiles*, which the API-slice path bypasses entirely (T0). Do not touch it.
-   [ ] **Owed — QCEW size crosswalk (maintainer):** NAICS→CES mapping that preserves `size_code`, producing the `native` frame `build_size_class_panel` consumes. Needs **new agglvl-21–28 pull maps** — the lookups pulls (`QCEW_SECTOR_PULLS`, `_QCEW_AGGLVL`) cover only the area endpoint's 13/14/15/16. Validate against real size rows before trusting it.
-   [ ] **Owed — acceptance run:** a small-window build to scratch (`NFP_STORE_URI=s3://alt-nfp/store-rebuild`) succeeds. (The canonical-refusal guard is already proven by `test_raises_for_canonical_store`.)

------------------------------------------------------------------------

## T6 — Acceptance-gate validator (§10) `[depends: T5]`

Implement the four gates; key on `industry_type + industry_code + ownership + (rev,bmr) + values` using the T1 remap (so code `55` stays unambiguous).

-   [ ] **History consistency:** rebuilt `source=ces` matches the current store ≤2023 (private hierarchy + `00` anchor); four-combo `(rev,bmr)` reproduces.
-   [ ] **Gap fill (priority):** *hard* — `05` + supersectors current to frontier, 2024-12/2025-12 `(2,1)` complete; *reconstruct-and-validate* — `06`/`08`/sectors refilled, additive nesting validated where present (missing sector-month does not block).
-   [ ] **Reconstruction accuracy:** QCEW vs published CES at benchmark months / annual averages; `81/80/08/05` residual small and **non-negative**. **Set the numeric `|residual|` tolerance here** (owed by this plan per §10) — choose and justify a bound; do not require exact equality.
-   [ ] **Vintage integrity:** `_validate_censored_selection`-style checks on an as-of slice (no dups, no cross-vintage sums, no nulls/zeros).
-   [ ] **Acceptance:** validator runs against the scratch store and reports pass/fail per gate; gates are tests, not prose.

------------------------------------------------------------------------

## T7 — Re-baseline A1/A2 goldens `[depends: T6 passing]`

-   [ ] Regenerate A1 (censored panels) and A2 (`build_model_data` arrays) fixtures from the **scratch** store; update the goldens manifest. Document the divergence from the frozen reference (ownership axis, `00` anchor, NSA, QCEW-mapped) in the manifest/readme so the change is auditable, not silent.
-   [ ] **Acceptance:** A1/A2 gates green against the re-baselined fixtures; the diff vs old fixtures is explained.

------------------------------------------------------------------------

## T8 — Promotion runbook (scratch → canonical) `[depends: T6, T7; MAINTAINER GO]`

-   [ ] With all §10 gates green and maintainer approval: cut over deliberately — repoint `NFP_STORE_URI`, or copy scratch→canonical via the explicit `--allow-canonical` escape hatch (`scripts/mirror_store.py`). Snapshot the prior canonical first; keep it until the new store is confirmed in the model.
-   [ ] **Acceptance:** canonical store serves the rebuilt schema; a post-cutover read reproduces the §10 gate results; rollback path documented.

------------------------------------------------------------------------

## Deferred (explicitly NOT this plan)

Government (`ownership='government'`, codes `07`/`90`–`93`, QCEW `own_code` 1/2/3) and the downstream `00` SA composition; live capture (BLS feed cron); geography beyond national; seasonal adjustment; NAICS vintage-aware crosswalks; births-deaths. See spec §11.

## Open risks

-   T0 may show `cesvinall` lacks some sector/benchmark rows → narrows T6's reconstruct-and-validate scope (sectors stay best-effort, per §10).
-   The reconstruction-accuracy tolerance (T6) is a judgement call with no reference number — set it from observed benchmark-month residuals, document it.
-   **Confirmed (reading `nfp_download.bls.bulk`):** the current QCEW acquire drops the agglvl `13`/`16` rows and `size_code` that T3/T4 consume — a small acquire fix (T5), but the existing `qcew_bulk.parquet` cannot feed the rebuild until the filter is widened. T0 pins the exact size agglvl codes.