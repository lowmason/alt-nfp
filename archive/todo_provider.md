# TODO: Provider Representativeness Correction — alt_nfp Integration

## Context

This TODO covers **only** the work implemented in the **`alt_nfp`** (this) repo for the provider representativeness correction pipeline. The private provider microdata repo is responsible for pseudo-establishment construction, geoclustering, cell assignment, frozen panel construction, and exporting cell-level employment parquets. Here we implement:

- Loading QCEW from the vintage store and computing cell-level employment shares (with carry-forward for recent months).
- QCEW-weighted national compositing from cell-level provider parquets.
- Weight redistribution for missing or low-coverage cells.
- Weight staleness tracking and reporting.
- Integration so the existing model receives a single national growth series per provider (unchanged `pp_data` interface).

**Spec reference:** `specs/provider_spec.md` §5 (QCEW-Weighted National Composite), §6.3 (Downstream Evaluation), §7.5 (Phase 5: Integration), §9.2 (alt_nfp Configuration).

**Prerequisite:** Cell-level provider parquets conforming to §1.4.1 (44 cells: 11 supersectors × 4 regions; schema: `geographic_type`, `geographic_code`, `industry_type`, `industry_code`, `ref_date`, `n_pseudo_estabs`, `employment`) are placed at `data/providers/<name>/<name>_provider.parquet`. The vintage store (`data/store/`) contains QCEW data with `source='qcew'`, `geographic_type='region'`, `industry_type='supersector'`.


---


## Task A — QCEW Weight Loading Module

**Priority: High. Spec §5.3.1, §5.3.2, §5.3.3.**

Create a new module that loads QCEW employment at supersector × region level from the vintage store, computes cell-level national employment shares per ref_date, implements carry-forward for months beyond the QCEW frontier, and records weight staleness.

### A.1  New module `src/alt_nfp/ingest/compositing.py`

Add a module that provides:

1. **Load QCEW region × supersector employment from the store**

   - Read the Hive-partitioned store at `STORE_DIR` (from `config.py`).
   - Filter: `source == 'qcew'`, `geographic_type == 'region'`, `industry_type == 'supersector'`.
   - Restrict `ref_date` to a range that overlaps the provider (e.g. min/max ref_date from the provider parquet).
   - Ensure cell key is `(geographic_code, industry_code)` with `geographic_code` ∈ {'1','2','3','4'} and `industry_code` in the 11 supersector codes ('10','20','30','40','50','55','60','65','70','80','90').
   - Aggregate to one row per `(ref_date, geographic_code, industry_code)` with `employment` (sum if multiple vintages/rows per cell; typically one row per cell per ref_date).

2. **Compute employment shares per ref_date**

   For each QCEW `ref_date`, compute:
   `w_c = employment_c / Σ_c employment_c` over the 44 cells. Output a structure keyed by ref_date (e.g. dict of ref_date → array or DataFrame of cell weights).

3. **Carry-forward for provider ref_dates with no QCEW**

   Given the sorted list of provider `ref_date`s and the set of QCEW ref_dates:
   - For each provider ref_date, if QCEW exists for that date, use those weights.
   - Otherwise, use the weights from the most recent QCEW ref_date that is strictly before (or equal to) the provider ref_date.
   - If no QCEW exists for any date ≤ provider ref_date, use the earliest available QCEW vintage (forward-fill from past).

4. **Weight staleness metadata**

   For each composite month (provider ref_date), record:
   - `qcew_weight_ref_date`: the ref_date of the QCEW observation used.
   - `weight_staleness_months`: number of months between composite ref_date and `qcew_weight_ref_date` (0 when current, positive when carried forward).

Expose a function such as:

```python
def load_qcew_weights(
    store_path: Path,
    min_ref: date,
    max_ref: date,
    provider_ref_dates: Sequence[date],
) -> tuple[dict[date, np.ndarray], pl.DataFrame]:
    """Return (ref_date -> weight vector for 44 cells), and a DataFrame with
    columns ref_date, qcew_weight_ref_date, weight_staleness_months.
    """
```

Cell order must be canonical (e.g. (region 1..4 × supersector 10,20,...,90) so that provider cell data can be aligned.

Files: **New** `src/alt_nfp/ingest/compositing.py`

### A.2  Dependencies and constants

- Use `config.STORE_DIR` (or `config.DATA_DIR / 'store'`) for the store path.
- Use `lookups/industry.get_supersector_codes()` or the explicit list of 11 supersector codes for validation.
- Use `lookups/geography.REGION_NAMES` / region codes '1','2','3','4' for geography.

Files: `src/alt_nfp/ingest/compositing.py`, `src/alt_nfp/config.py` (if STORE_DIR not already exposed)


---


## Task B — Weight Redistribution for Missing or Low-Coverage Cells

**Priority: High. Spec §5.2.**

Implement the weight redistribution algorithm in `compositing.py`.

### B.1  Redistribution algorithm

- **Inputs:** (1) Set of cells with valid provider coverage and sufficient `n_pseudo_estabs` (`C_covered`). (2) Raw QCEW weight vector for all 44 cells (or the subset present in QCEW).
- **Steps:**
  1. For each cell not in `C_covered`, that cell’s weight must be redistributed.
  2. For missing cell `(s, r)` (supersector `s`, region `r`):  
     a. Add its weight to covered cells in the same supersector `s`, proportionally to their current weights (preserve industry margin).  
     b. If no covered cell in supersector `s`, distribute to covered cells in region `r`, proportionally (preserve geography margin).  
     c. If neither exists, distribute uniformly over all covered cells.
  3. Renormalize so that the final weight vector sums to 1.0.

Implement as a pure function, e.g.:

```python
def redistribute_weights(
    weights_44: np.ndarray,
    covered_cell_indices: np.ndarray,
    cell_to_supersector: np.ndarray,
    cell_to_region: np.ndarray,
) -> np.ndarray:
    """Return a weight vector of length 44 that sums to 1, with weight
    from missing cells redistributed over covered cells per spec §5.2.
    """
```

Use a canonical cell index mapping (e.g. 0..43) so that `covered_cell_indices` and the supersector/region mappings are well-defined.

Files: `src/alt_nfp/ingest/compositing.py`


---


## Task C — National Composite Computation

**Priority: High. Spec §5.1.**

Compute the representativeness-corrected national growth series from cell-level provider data and QCEW weights.

### C.1  Cell-level growth from provider parquet

- Read the provider parquet (all rows; no filter to national).
- For each cell `(geographic_code, industry_code)`, sort by ref_date and compute log-difference growth: `y_{c,t} = ln(E_{c,t}) - ln(E_{c,t-1})`.
- Mark cells with `n_pseudo_estabs < MIN_PSEUDO_ESTABS` as invalid for that month (excluded from `C_covered`).

### C.2  Composite formula

For each ref_date `t`:

1. Get QCEW weight vector for `t` (from Task A, including carry-forward).
2. Redistribute weights for missing/low-coverage cells (Task B) → `w_redist`.
3. Compute `y_{p,t}^G = Σ_c w_redist[c] * y_{p,c,t}^G` over covered cells (growth in uncovered cells is not used; their weight was redistributed).

The composite is a **growth rate** series. The downstream model and panel expect growth, so the compositing layer outputs growth directly. If a level is needed for panel rows (e.g. for consistency with employment_level in panel schema), accumulate growth into a level index with a base of 100 or use the first month’s national employment from QCEW as scale.

### C.3  Output shape and staleness

Return:

- A flat series: `ref_date` and composite **growth** (and optionally employment level for panel) for each month.
- A DataFrame or structure with `ref_date`, `qcew_weight_ref_date`, `weight_staleness_months` for validation.

Expose a single entry point, e.g.:

```python
def compute_provider_composite(
    provider_cell_df: pl.DataFrame,
    store_path: Path,
    min_pseudo_estabs: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (composite_df with ref_date, growth, [employment_level]),
    staleness_df with ref_date, qcew_weight_ref_date, weight_staleness_months).
    """
```

Files: `src/alt_nfp/ingest/compositing.py`


---


## Task D — Integration into Provider Loading

**Priority: High. Spec §7.5.**

Wire cell-level parquets through the compositing layer so that the rest of the pipeline still sees one national series per provider.

### D.1  Detect cell-level provider data

In `load_provider_series()` in `ingest/payroll.py`:

- After reading the parquet, check whether the data is cell-level: e.g. presence of many rows per ref_date, or explicit check that `geographic_type` (or `geographic_type` if that is the column name in the file) equals `'region'` and `industry_type == 'supersector'`.
- If cell-level: do **not** filter to a single geography/industry; pass the full table to the compositing module. Call `compute_provider_composite(provider_cell_df, store_path, min_pseudo_estabs)` and use the returned composite growth series (and ref_date) to build the same return shape as today: DataFrame with `ref_date`, `employment` (use composite **level** if panel expects level for growth computation elsewhere, or already supply growth if that’s what the model uses — see current `load_provider_series` return and `ingest_provider` usage).
- If national (single row per ref_date or filter to national): keep current behavior (filter by config, return ref_date + employment).

**Note:** Currently the model uses **growth** from the panel; `ingest_provider` computes growth from levels. For cell-level, compositing produces growth directly. So either: (1) compositing returns levels (e.g. from cumulative growth) and existing `ingest_provider` flow computes growth from levels, or (2) compositing returns growth and the panel/ingest path for cell-level providers builds panel rows with that growth. Align with how `panel_adapter` and `model` consume provider series (they need `g_pp` = growth). So the composite output should feed into the same path that currently yields growth (e.g. composite growth → panel rows with that growth, or composite level → existing level→growth logic).

### D.2  Schema alignment

- Provider parquet schema uses `geographic_type`, `geographic_code` (spec §1.4.1). The existing `ProviderConfig` and `payroll.py` use `geography_type` / `geography_code` for **filter** keys. Ensure the compositing path and payroll reader use the same column names as the parquet (e.g. `geographic_type` in parquet; if the code filters by column name, use the name that exists in the file). Update payroll.py to use the column names present in the cell-level parquet so that detection and grouping work.

### D.3  Panel and panel_adapter

- `ingest_provider()` must still produce panel rows with one national growth per provider per month. If we use compositing, `ingest_provider` can call the same compositing entry point (or get pre-aggregated national series from `load_provider_series`). Ensure `load_provider_series()` when used from `panel_adapter` returns a DataFrame that leads to the same `pp_data` shape: ref_date and a series the adapter can turn into `g_pp`. No change to `panel_adapter`’s expected `pp_data` structure; only the source of the provider series changes when the file is cell-level.

Files: `src/alt_nfp/ingest/payroll.py`


---


## Task E — Config and Constants

**Priority: Medium. Spec §9.2.**

### E.1  Minimum pseudo-establishments

Add a constant for the minimum number of pseudo-establishments per cell (e.g. 5):

```python
MIN_PSEUDO_ESTABS_PER_CELL: int = 5
```

Use it in compositing and, if desired, allow override via `ProviderConfig` (e.g. optional field `min_pseudo_estabs: int | None = None`).

Files: `src/alt_nfp/config.py`

### E.2  ProviderConfig and PROVIDERS

- Ensure `ProviderConfig` documents that when `geography_type='region'` (or the parquet contains cell-level data), the loader will perform QCEW-weighted compositing.
- Add an example provider entry in `PROVIDERS` for the cell-level case (e.g. `file='providers/G/g_provider.parquet'`) so that once the parquet exists, the pipeline runs without further code change. Leave PROVIDERS as empty list until a provider is ready, or add a commented example.

Files: `src/alt_nfp/config.py`


---


## Task F — Tests

**Priority: High.**

### F.1  Unit tests for compositing

Add `tests/test_compositing.py`:

- **QCEW weight loading:** With a small fixture or mock store (QCEW rows for a few ref_dates and 44 cells), assert that employment shares sum to 1 per ref_date and that cell keys are correct.
- **Carry-forward:** Provide QCEW for ref_dates [2020-01, 2020-02]; request weights for provider ref_dates [2020-01, 2020-02, 2020-03, 2020-04]. Assert 2020-03 and 2020-04 use 2020-02 weights and that `weight_staleness_months` is 1 and 2 respectively.
- **Weight redistribution:** Given a weight vector and a set of covered cells (e.g. 40 of 44), assert redistributed weights sum to 1, zero weight on missing cells, and weights on covered cells follow the spec (redistribution within supersector or region).
- **National composite:** With synthetic provider cell DataFrame (e.g. two months, a few cells with employment and n_pseudo_estabs), and mock QCEW weights, assert composite growth is a weighted sum of cell growths and that output has ref_date and growth columns.

### F.2  Integration test with synthetic cell-level parquet

- Build a small parquet that matches §1.4.1 (columns geographic_type, geographic_code, industry_type, industry_code, ref_date, n_pseudo_estabs, employment) for 2–3 ref_dates and a subset of cells.
- Run `load_provider_series()` with a ProviderConfig pointing at this file and `geography_type='region'` (or equivalent so the loader treats it as cell-level). Assert that the result is a single national series (one row per ref_date) and that no exception is raised. If the vintage store is not available in the test env, mock the store or use a fixture store with minimal QCEW rows.

Files: **New** `tests/test_compositing.py`


---


## Task G — Downstream Validation and Weight Staleness Reporting

**Priority: Medium. Spec §6.3.**

### G.1  Weight staleness in diagnostics

- When provider data is sourced from cell-level compositing, the compositing layer produces staleness metadata (`qcew_weight_ref_date`, `weight_staleness_months`). Pass this through to a place where diagnostics can report it (e.g. attach to `pp_data` as optional `weight_staleness_df` per provider, or write to a small artifact in `output/`).
- In `diagnostics.py` (or a dedicated validation script): add a short report that, for each provider using compositing, prints or writes:
  - Count of months with current vs carried-forward weights.
  - Summary of `weight_staleness_months` (min, max, mean).

### G.2  Stratify metrics by staleness

- Optionally stratify posterior/nowcast metrics by weight staleness (e.g. report α_p and σ_{G,p} and nowcast RMSE for months with staleness 0 vs 1–6). This can be a follow-up or a small addition to the backtest/validation output once the composite is in use.

Files: `src/alt_nfp/diagnostics.py`, and optionally `src/alt_nfp/backtest.py` or a validation script that consumes `pp_data` and staleness metadata.


---


## Implementation Order

| Step | Task | Description |
|------|------|-------------|
| 1 | A | QCEW weight loading, carry-forward, staleness in `ingest/compositing.py` |
| 2 | B | Weight redistribution algorithm in `compositing.py` |
| 3 | C | National composite computation in `compositing.py` |
| 4 | E | Config: MIN_PSEUDO_ESTABS, ProviderConfig note |
| 5 | D | Payroll loader: detect cell-level, call compositing, return national series |
| 6 | F | Tests: test_compositing.py unit + integration |
| 7 | G | Diagnostics: weight staleness reporting |


## Files Touched (Summary)

| File | Tasks |
|------|-------|
| **New** `src/alt_nfp/ingest/compositing.py` | A, B, C |
| `src/alt_nfp/ingest/payroll.py` | D |
| `src/alt_nfp/config.py` | A (if needed), E |
| `src/alt_nfp/diagnostics.py` | G |
| **New** `tests/test_compositing.py` | F |


## Risks

| Risk | Mitigation |
|------|------------|
| Vintage store has no region-level QCEW | Ensure QCEW processing (e.g. aggregate.py) produces region rows in the store; document dependency |
| Cell key mismatch (provider vs QCEW) | Use canonical (geographic_code, industry_code) and same 11 supersector codes everywhere |
| Performance when loading store for every provider | Load QCEW once per pipeline run and reuse weight lookup for all providers; cache by ref_date range |
