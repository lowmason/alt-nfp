# Rebuilt-store model-data drop-in — SA CES + QCEW `00` total

**Status:** design (2026-06-17). Companion to [`store_rebuild.md`](store_rebuild.md)
(the rebuild) and `plans/10-store_rebuild.md` (T7/T8, blocked on this). Becomes
`plans/11` once approved.

**One-line goal:** make the rebuilt store (`s3://alt-nfp/store-rebuild`) a
**drop-in substrate** for the *current* `build_model_data` — by adding the two
published series the NSA-only rebuild left out — so T7 (re-baseline goldens) and
T8 (promotion) unblock **without changing the model**.

---

## 1. Problem

The rebuild deliberately built a **NSA, private-hierarchy** store (spec §6: NSA is
required because the private hierarchy's additive closure `05 = 06 + 08` only
holds in NSA — SA series don't sum). Run against it, `build_model_data(as_of=D)`
is **degenerate** because the model-data layer consumes two things the rebuild
omitted:

| model-data input | source it reads | rebuilt store has it? |
|---|---|---|
| CES **SA** target (`source='ces'`, `seasonally_adjusted=True` → panel `ces_sa`) | published BLS SA series | **No** — rebuild wrote `seasonally_adjusted=False` only |
| QCEW at the target industry | `source='qcew'`, `industry_code='00'` | **No** at `'00'` (private-only); **yes** at `'05'` |

Verified (2026-06-17, `NFP_STORE_URI=s3://alt-nfp/store-rebuild`):

- `panel_to_model_data(..., industry_code='00')` → `qcew_obs=0`, `ces_sa=0`,
  `ces_nsa=77`.
- `panel_to_model_data(..., industry_code='05')` → `qcew_obs=72`, `ces_sa=0`,
  `ces_nsa=77` (QCEW private already feeds; SA still missing).
- Canonical store → `qcew_obs=131`, both CES SA and NSA populated.

So the rebuilt store is **close**: NSA CES and private QCEW already work. The gap
is the **SA series** and the **QCEW `'00'` total**. Both are **published by BLS**
and were always ingested by the canonical pipeline — this is *re-adding dropped
inputs*, not new modeling.

## 2. Goal / non-goals

**Goal.** The rebuilt store carries SA+NSA CES (full hierarchy + `00` anchor +
`05` root) and a QCEW `00` total, so `build_model_data` at the current `'00'`
target is non-degenerate and the store can be promoted to replace the canonical
store with the **model unchanged**.

**Non-goals (stay deferred — spec §11 / the "B" product vision).**

- Modeling the private NSA hierarchy directly and **composing** the `00` total
  from private + government (the supersector "why" architecture).
- The **government** ownership axis (`own_code` 1/2/3; codes `90`–`93`).
- From-scratch seasonal adjustment. The SA series is **published**; we ingest it,
  we do not compute it.
- Re-targeting the model from `'00'` to `'05'` (its providers are national `'00'`;
  a retarget is a modeling change, out of scope here).

## 3. Design

### Component 1 — SA CES series

The CES builder (`nfp_ingest/ces_builder.build_ces_panel`) currently reads only
`tri_{code6}_NSA.csv` (`# NSA only — ignore _SA companions`). The 113
`tri_{code6}_SA.csv` triangles are already on disk and carry the same triangular
revision structure.

- Build the SA series by the **same diagonal logic** over the `_SA` triangles,
  emitting rows with `seasonally_adjusted=True` for **all 113 codes** (mirror NSA
  — consistent and cheap), same ownership taxonomy (`00`→`total/total`,
  `05`→`total/private`, etc.).
- SA rows are **parallel** to the NSA hierarchy: they carry **null
  `size_class_*`** (no size cross-product — that is a QCEW Q1 product) and do
  **not** participate in additive nesting. The §10 NSA gates read
  `seasonally_adjusted=False` and are therefore **untouched**.
- The compose step (`compose_rebuild_panel`) unions the SA CES rows like the NSA
  CES rows; the §7 Q1 size override is QCEW-only and unaffected.

**Open implementation detail (pin in the plan, T-task #1):** SA vintage
semantics. The annual February benchmark **re-seasonally-adjusts** the whole
series, so the SA triangle's `(revision, benchmark_revision)` cohorts may not map
1:1 to the NSA convention (where `(2,1)` is the per-benchmark re-basing). Verify
the SA triangle's diagonal/benchmark structure against the canonical store's SA
rows **before** mirroring the NSA `_diagonals` logic blindly; adapt the
`(rev,bmr)` assignment if SA differs.

### Component 2 — QCEW `'00'` total

QCEW publishes the total-covered-employment as a single `own_code=0` row
(`industry '10'`, agglvl 10) on the same area endpoint the rebuild already uses —
verified: `own_code` values present are `{0,1,2,3,5,8,9}`; `own_code=0` is one
published total row. **No government hierarchy is needed.**

- Extend the QCEW acquire (`_acquire_qcew_levels` / `_prep_area_raw`) to also keep
  `own_code=0`, and the crosswalk (`qcew_crosswalk.build_qcew_panel`) to map the
  QCEW total (`industry '10'`, agglvl 10, `own_code=0`) → CES `'00'`
  (`ownership='total'`), NSA, `benchmark_revision=0`, one row per month per
  quarter-vintage. This mirrors how the private `own_code=5` rows are handled
  today, just for the one additional total row.
- Size cross-product is unaffected (`'00'` total is not part of the size product).

**Open implementation detail (pin in the plan, T-task #2):** the QCEW
total-covered (`own_code=0`) vs CES total-nonfarm (`'00'`) **definitional gap**
(QCEW counts UI-covered employment incl. agriculture; CES is nonfarm). This is the
same direction the reconstruction gate already models for `05/06/08/80`; add a
`'00'` band to `_EXPECTED_QCEW_CES_RESIDUAL` from the observed residual rather
than asserting equality.

### Component 3 — gate updates (in scope)

- A `'00'` entry in `gate_reconstruction_accuracy`'s `_EXPECTED_QCEW_CES_RESIDUAL`
  / band (the QCEW-total vs CES-nonfarm gap).
- An explicit check that SA rows (`seasonally_adjusted=True`) are **excluded**
  from the NSA nesting/reconstruction gates (they must not be summed).
- The `gate_ces_fidelity` rail extends naturally to the SA rows (rebuilt SA ==
  `build_ces_panel(_SA)` to the unit) — add the SA real-store assertion.

**Boundary.** This plan delivers Components 1–3 (the drop-in + gate updates) and
verifies `build_model_data` is non-degenerate against a fresh scratch rebuild.
That **unblocks** `plans/10` **T7** (re-baseline A1/A2 goldens to a scratch golden
prefix — never overwriting the frozen reference; documenting the schema divergence
+ **2017+ history truncation** + QCEW-total shift) and **T8** (promotion), which
proceed in `plans/10` as already scoped. Golden regeneration and promotion are
**not** re-scoped here.

## 4. Acceptance criteria

1. `build_model_data(as_of=D)` against a fresh scratch rebuild (default `'00'`
   target) returns a **non-degenerate** dict for the A1/A2 as-of dates (all
   postdate 2017): `qcew_obs` non-empty and `ces_sa` populated. (Panel *history*
   is still truncated to 2017+ — the calendar is shorter than the frozen
   reference's 2012+; that is documented, not a failure.)
2. The rebuilt store carries `seasonally_adjusted ∈ {True, False}` for CES, with
   SA row counts comparable to NSA (per code/ref-month), and a QCEW `'00'` total.
3. All §10 NSA gates stay green (SA rows excluded; QCEW `'00'` band added);
   `gate_ces_fidelity` green for SA as well as NSA.
4. A rebuild to scratch + the 7 `real_store` gate wrappers re-run green.

This plan ends here (the drop-in verified). `plans/10` T7 (goldens) and T8
(promotion) are then unblocked and proceed there.

## 5. Files touched (orientation)

- `nfp_ingest/ces_builder.py` — SA build over `_SA` triangles (Component 1).
- `nfp_vintages/rebuild_store.py` — `_acquire_qcew_levels`/`_prep_area_raw` keep
  `own_code=0`; compose unions SA CES (Components 1–2).
- `nfp_ingest/qcew_crosswalk.py` — map QCEW total → `'00'` (Component 2).
- `nfp_vintages/rebuild_gates.py` — `'00'` reconstruction band; SA-exclusion check
  (Component 3).
- Scripts/tests for A1/A2 golden regeneration to the scratch prefix (T7).

## 6. Risks

- **SA vintage structure** differs enough from NSA that mirroring the diagonal
  logic mis-assigns `(rev,bmr)` (Component 1 open detail). Mitigation: verify
  against the canonical SA rows first; the `gate_ces_fidelity` SA rail catches a
  mismatch to the unit.
- **QCEW `'00'` definitional** residual lands outside a naive band (Component 2
  open detail). Mitigation: set the band from the observed residual, as the
  existing per-series bands were.
- **History truncation (2017+)** makes the re-baselined goldens cover less than
  the frozen reference. Mitigation: document explicitly in the goldens manifest;
  it is inherent to the rebuild, not introduced here.
