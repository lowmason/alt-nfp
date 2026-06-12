# Representativeness Correction Implementation Spec

**Alt-NFP Nowcasting System — Provider Signal Representativeness Correction**

Version: 2.0 \| Date: 2026-03-02 \| Author: Lowell Mason

------------------------------------------------------------------------

## 1. Overview

This spec covers the representativeness correction pipeline that transforms raw provider microdata into cell-level (supersector × Census region) employment parquets suitable for consumption by the alt-NFP state space model. The pipeline spans two repositories:

- **Private provider repo:** pseudo-establishment construction, geoclustering, cell assignment, frozen panel construction, and cell-level parquet export.
- **`alt_nfp` (this repo):** QCEW weight loading, carry-forward logic, national compositing, growth rate computation, weight staleness tracking, and downstream model integration.

### 1.1 Problem

A raw aggregate provider growth signal reflects the compositional mix of the provider's client base, which differs from the national employment distribution by industry and geography. When the composition of national growth shifts — e.g., healthcare surging while manufacturing contracts — providers with different client portfolios produce divergent aggregate signals. The national model's single latent state interprets this divergence as measurement noise rather than informative compositional heterogeneity, attenuating signal extraction during periods of heterogeneous sectoral growth.

### 1.2 Solution

Construct a provider-specific national composite that reweights cell-level continuing-units growth to QCEW employment shares, so that each provider's signal approximates what it would measure if its client base matched the national employment distribution. The cell granularity is **supersector × Census region** — coarse enough that classification errors are rare, fine enough to remove the dominant sources of composition bias.

### 1.3 Pipeline Reuse

The cell-level signals produced here serve all three releases through a **single parquet interface**. The private repo always exports the same cell-level (supersector × Census region) employment parquet; the `alt_nfp` public repo is responsible for QCEW weighting, compositing, and any release-specific aggregation.

| Release | `alt_nfp` Consumption | What `alt_nfp` Does with the Parquet |
|------------------|--------------------|-----------------------------------|
| 1 (National Accuracy) | Full 44-cell parquet | QCEW-weighted composite → single national growth rate per provider |
| 2 (Industry Decomposition) | Full 44-cell parquet | Aggregate cells by supersector → 11 supersector-level signals per provider |
| 3 (Geographic × Industry) | Full 44-cell parquet | Use cell-level signals directly in cell-level measurement equations |

This design keeps the QCEW weighting logic auditable in the public repo and avoids any interface changes between releases.

### 1.4 Interface Contract with `alt_nfp`

The private repo exports a single **cell-level employment parquet** per provider. The `alt_nfp` public repo consumes this file via `ProviderConfig.file` and performs all downstream aggregation (QCEW weighting, national compositing, growth rate computation) internally.

#### 1.4.1 Employment Parquet Schema

One row per provider × cell × month. The file covers the full 11 × 4 = 44 cell grid; cells with no provider coverage in a given month are omitted (implicit missing).

The provider parquet is placed at `data/providers/<name>/<name>_provider.parquet` within the `alt_nfp` data directory (e.g. `data/providers/G/g_provider.parquet`).

```
xxx_provider.parquet
─────────────────────────────────────────────────────────
geographic_type  : str        — 'region'
geographic_code  : str        — '1' (Northeast), '2' (Midwest), '3' (South), '4' (West)
industry_type    : str        — 'supersector'
industry_code    : str        — BLS supersector code: '10', '20', '30', '40', '50',
                                '55', '60', '65', '70', '80', '90'
ref_date         : date       — reference month, always YYYY-MM-12
                                (12th of the month, matching CES/QCEW pay-period convention)
n_pseudo_estabs  : uint32     — pseudo-establishment count in this cell × month
employment       : float64    — continuing-units employment in thousands (000s)
```

The `employment` column is the **level** of continuing-units employment from the frozen measurement panel (not a growth rate). The `alt_nfp` repo log-differences this column to compute month-over-month growth rates, consistent with the existing `ProviderConfig` interface where `employment` is log-differenced.

**Conventions:**

-   `ref_date` uses the 12th to align with CES/QCEW reference period conventions (pay period including the 12th).
-   `employment` is expressed in thousands (e.g., 1,234.5 = 1,234,500 employees) to match BLS reporting units.
-   `geographic_type` uses `'region'` to match the vintage store's geographic type vocabulary (see `ingest/base.py` and `ingest/aggregate.py`).
-   `n_pseudo_estabs` enables the public repo to apply minimum coverage thresholds without accessing microdata.
-   Cells below the minimum pseudo-establishment threshold (§4.2) should still be included in the parquet with `n_pseudo_estabs` reported; the public repo decides the threshold.

#### 1.4.2 Birth-Rate File (Optional, National Only)

Birth rates remain a **national total** per provider per month — not cell-decomposed. Business formation events observed by a payroll provider are sparse enough that splitting across 44 cells would produce many zero-count months, adding noise without meaningful signal. The BD model in Release 1 is national-level, so a single series is the right granularity.

```
xxx_births.parquet
─────────────────────────────────────────────────────────
ref_date         : date       — reference month, YYYY-MM-12
birth_rate       : float      — provider-observed business formation rate (national)
```

> **Release 2 note:** When industry decomposition introduces supersector-level BD modeling, birth rates could be aggregated to 11 supersector-level totals. This is a coarse enough granularity to have sufficient mass per cell. Supersector × region (44 cells) is almost certainly too fine for births.

#### 1.4.3 ProviderConfig Mapping

``` python
ProviderConfig(
    name='G',
    file='providers/G/g_provider.parquet',
    error_model='iid',
)
```

The `ProviderConfig` schema supports `geography_type` / `geography_code` / `industry_type` / `industry_code` filter fields. When these are set to their defaults (`'national'` / `'00'`), the model expects a pre-aggregated national row. When the provider file contains cell-level data (detectable by `geographic_type='region'`), the model's data-loading layer performs QCEW-weighted compositing before passing observations to the measurement equation.

### 1.5 Repository Responsibilities

The pipeline spans two repositories. The private provider repo produces parquets conforming to §1.4; this repo (`alt_nfp`) never sees microdata — it only consumes those parquets and the existing vintage store.

**Private provider repo:**

| Responsibility | Spec reference |
|---|---|
| Pseudo-establishment pipeline: establishment-likeness diagnostics, geoclustering, degraded-location fallbacks, cell assignment | §3 |
| Frozen panel construction at cell level: panel mechanics, cell-level continuing-units growth, panel refresh | §4 |
| Cell-level employment parquet export: aggregate to (supersector × region) × month, employment in thousands, ref_date convention, `n_pseudo_estabs` | §1.4.1 |
| Birth-rate parquet export (optional) | §1.4.2 |
| Private-repo validation: composition bias diagnostics, internal QCEW-weighted composite for signal quality checks, schema compliance | §6.1, §6.2 |

**`alt_nfp` (this repo):**

| Responsibility | Spec reference |
|---|---|
| Read provider parquet from `data/providers/<name>/<name>_provider.parquet` via `ProviderConfig.file` | §1.4.3 |
| Load QCEW from the vintage store (`data/store`, Hive-partitioned): filter `source='qcew'`, `geographic_type='region'`, `industry_type='supersector'`, `ref_date` in provider range | §5.3 |
| QCEW-weighted compositing: weight computation (including carry-forward for months beyond QCEW frontier), weight redistribution for missing/low-coverage cells, national composite growth | §5.1, §5.2, §5.3 |
| Weight staleness tracking: record which QCEW vintage was used per `ref_date`, compute staleness in months | §5.3.2 |
| Log-difference provider cell-level `employment` to get growth; apply min `n_pseudo_estabs` threshold; feed composite series into the model | §5.1 |
| Downstream validation: compare `α_p`, `σ_{G,p}`, nowcast RMSE; report weight vintage and staleness in validation outputs | §6.3 |

------------------------------------------------------------------------

## 2. Cell Definition

### 2.1 Dimensions

Cells are defined at **supersector × Census region**:

-   **Supersectors (11):** The canonical BLS supersector codes are **10** (Mining & Logging), **20** (Construction), **30** (Manufacturing), **40** (Trade, Transportation & Utilities), **50** (Information), **55** (Financial Activities), **60** (Professional & Business Services), **65** (Education & Health Services), **70** (Leisure & Hospitality), **80** (Other Services), **90** (Government).
-   **Census regions (4):** Northeast (1), Midwest (2), South (3), West (4).

Total cells: **44** (11 × 4). In practice, some cells may be empty for a given provider (e.g., supersector 90 for private-sector-only providers), yielding ~36–40 active cells per provider.

### 2.2 Why This Granularity

Supersector boundaries are wide enough that NAICS misclassification almost never crosses them — retail workers at a retail chain are in retail regardless of store location. Census regions are large enough that geographic assignment errors from employee geocoding are negligible (unlike state-level assignment, where headquarter bias is problematic for multi-establishment firms).

Finer granularity (e.g., sector × state) would produce many sparse cells with unreliable growth signals and noisier composites, while barely improving the composition bias correction.

------------------------------------------------------------------------

## 3. Pseudo-Establishment Pipeline *[Private repo]*

### 3.1 Motivation

Payroll provider data are organized at the **client** level, where a client is a legal entity that may operate multiple establishments across different states and industries. Client-level NAICS codes and headquarters locations do not reliably identify where employment is situated, particularly for large multi-establishment firms. However, employee-level geocodes are available for all clients.

Pseudo-establishments transform client-level records into establishment-like units suitable for cell assignment.

### 3.2 Establishment-Likeness Diagnostics

Before clustering, classify each client as establishment-like, enterprise-like, or ambiguous using employee geographic dispersion:

-   **Distance distributions:** Percentile distances of employees from client address.
-   **Multimodality detection:** Mixture models on employee geocodes to detect distinct worksites.
-   **Industry- and metro-specific commuting priors:** A retail client with employees in 50 zip codes is enterprise-like; a professional services firm with employees within commuting distance of one office is establishment-like.
-   **Temporal stability:** Dispersion patterns that are stable over time are more reliable than transient patterns.

**Output:** A latent classification per client per panel window: `establishment_like`, `enterprise_like`, or `ambiguous`.

Establishment-like clients skip the geoclustering step and are assigned to their headquarters cell directly.

### 3.3 Geoclustering

For enterprise-like and ambiguous clients, cluster employees into geographically coherent groups:

```
E_{c,k,t} = Σ_{i ∈ C_{c,k}} 1{i employed in p(t)}
```

where `C_{c,k}` is the set of employees assigned to cluster `k` within client `c`.

**Clustering method:** DBSCAN or HDBSCAN on employee geocodes, with industry-specific distance parameters (tighter for retail/food service, looser for professional services). Each cluster becomes a pseudo-establishment.

**Assignment:**

-   **Geography:** Cluster centroid (or population-weighted medoid) determines Census region assignment. For supersector × Census region cells, centroid-based assignment is almost always correct.
-   **Industry:** Inherited from the client's NAICS code. Reliable at the supersector level — NAICS codes almost never cross supersector boundaries.

### 3.4 Degraded Location Information

Different providers offer different levels of location information. The pipeline must handle a cascade of information regimes:

#### 3.4.1 Full Geocodes (Best Case)

Full pseudo-establishment pipeline as described above. Geographic assignment is reliable at all granularities.

#### 3.4.2 Zip Codes Only

Zip code centroids (preferably population-weighted Census centroids) provide sufficient resolution for supersector × Census region assignment. Zip codes straddling state lines are negligible in employment terms; Census region assignment is almost never wrong.

Clustering for enterprise-like detection remains effective for the cases that matter most (national chains with employees in many zip codes). Loss is primarily for mid-size firms with 2–3 worksites sharing a zip code.

#### 3.4.3 Client State Only

Only the headquarter state is known. This is informative for small firms (client ≈ establishment) but systematically misleading for large multi-establishment firms.

**Hybrid allocation:**

-   Clients below a size threshold (calibrated per industry using QCEW MWR data on multi-establishment rates): accept headquarters state assignment.
-   Clients above the threshold: redistribute employment across states using QCEW state × industry shares as allocation weights.

```
Ê_{c,g,t} = E_{c,t} * s_{g,n_c}^{QCEW}
```

where `s_{g,n_c}^{QCEW}` is state `g`'s share of national employment in industry `n_c` from the most recent QCEW vintage.

**Size threshold calibration:** QCEW MWR data show that 4.7% of employers operate multiple establishments, accounting for 45% of national employment. The threshold varies by industry — retail and food service firms are multi-establishment at smaller sizes than professional services firms. Use the provider's own located clients (if any exist) to calibrate the size–dispersion relationship.

#### 3.4.4 No Location Information

If neither employee geocodes, zip codes, nor client state are available, all clients are allocated geographically using QCEW shares. This is the weakest regime — every client is treated as if its geographic footprint mirrors the national distribution for its industry. Perform allocation at the finest available NAICS level and aggregate upward to preserve geographic specificity.

### 3.5 Coverage Ratios Exceeding Unity

Under degraded location information, cell-level coverage ratios (payroll employment / QCEW employment) may exceed 1.0. This is the **expected** outcome for large multi-establishment firms under headquarters-state assignment, not an anomaly.

**Remedies:**

-   **Aggregate to coarser cells.** At supersector × Census region, cells are large enough that ratios above 1 are rare.
-   **Iterative proportional fitting (raking).** Treat allocation as a two-dimensional margins problem with known provider employment totals by industry and QCEW state × industry population totals. Raking produces an internally consistent allocation respecting both dimensions.
-   **Two-stage approach.** Stage 1: small clients contribute directly to cell-level signals. Stage 2: large clients above the size threshold are redistributed and then raked to QCEW margins.

------------------------------------------------------------------------

## 4. Frozen Panel Construction at Cell Level *[Private repo]*

### 4.1 Panel Mechanics

The same rotating, frozen measurement panel used for the raw national index applies at the cell level. At fixed refresh intervals (e.g., quarterly):

1.  Define an eligible set of pseudo-establishments that have satisfied a stabilization rule (≥K consecutive reference periods or change-point-based stabilization).
2.  Freeze this set for the panel window. Do not add new pseudo-establishments mid-window.
3.  Within the frozen window, compute month-to-month growth using matched pseudo-establishment observations only.
4.  Pseudo-establishments that administratively exit during the window are removed without contributing negative change.

### 4.2 Cell-Level Growth Signal

Within each cell `c`, continuing-units growth is:

```
y_{p,c,t}^G = matched-panel growth of pseudo-establishments in cell c
```

**Minimum coverage requirement:** If a cell has fewer than a configurable minimum number of pseudo-establishments (e.g., 5), the cell signal is flagged as unreliable and excluded from the composite. The cell's QCEW weight is redistributed per §5.2.

### 4.3 Panel Refresh and Cell Stability

When the panel refreshes, pseudo-establishment compositions within cells may shift. The panel refresh procedure should:

-   Re-run geoclustering on the current employee snapshot.
-   Re-assign pseudo-establishments to cells.
-   Apply stabilization rules to pseudo-establishments in each cell independently.
-   Smoothly transition between panel windows (the existing panel rotation mechanics handle this).

------------------------------------------------------------------------

## 5. QCEW-Weighted National Composite *[alt_nfp]*

### 5.1 Core Formula

The provider's representativeness-corrected national signal is:

```
y_{p,t}^G = Σ_c w_{c,t}^{QCEW} * y_{p,c,t}^G
```

where:

-   `y_{p,c,t}^G` is provider `p`'s continuing-units growth in cell `c` at time `t`, computed as the log-difference of the `employment` column in the provider parquet: `y_{p,c,t}^G = ln(E_{p,c,t}) - ln(E_{p,c,t-1})`.
-   `w_{c,t}^{QCEW}` is cell `c`'s share of national employment from the most recent available QCEW vintage (see §5.3).
-   The summation is over all cells `c` with non-missing provider coverage and sufficient pseudo-establishment counts.

### 5.2 Weight Redistribution for Missing Cells

For cells where a provider has no coverage (or coverage below the minimum threshold), the weight is redistributed proportionally across covered cells **within the same supersector or region**, preserving the marginal industry and geographic distributions as closely as possible.

**Algorithm:**

1.  Let `C_covered` = set of cells with valid provider signals.
2.  Let `C_missing` = set of cells without coverage.
3.  For each missing cell `(s, r)` (supersector `s`, region `r`):
    a.  Distribute its weight proportionally across covered cells in supersector `s` (preserving industry distribution).
    b.  If no covered cells exist in supersector `s`, distribute across covered cells in region `r` (preserving geographic distribution).
    c.  If neither produces a match (extremely rare at this granularity), distribute uniformly across all covered cells.
4.  Renormalize so weights sum to 1.0.

### 5.3 QCEW Weight Source and Updates

#### 5.3.1 Loading QCEW from the Vintage Store

QCEW employment shares are loaded from the `alt_nfp` vintage store (`data/store/`, Hive-partitioned):

```python
qcew_region_ss = (
    pl.read_parquet('data/store', hive_partitioning=True)
    .filter(
        pl.col('source') == 'qcew',
        pl.col('geographic_type') == 'region',
        pl.col('industry_type') == 'supersector',
        pl.col('ref_date').is_between(MIN_PROVIDER_DATE, MAX_PROVIDER_DATE),
    )
)
```

For each QCEW `ref_date`, compute cell-level employment shares:

```
w_{c,t}^{QCEW} = employment_{c,t}^{QCEW} / Σ_c employment_{c,t}^{QCEW}
```

where `c` indexes the 44 supersector × region cells.

#### 5.3.2 Carry-Forward for Recent Months

QCEW has a ~5–6 month publication lag, so for recent months the provider has cell-level employment data but no corresponding QCEW exists. For these months, the pipeline uses the **last available QCEW vintage** to compute cell-level national employment shares.

Concretely, for each provider `ref_date`:

1.  If QCEW data exist for that `ref_date`, use them to compute `w_{c,t}^{QCEW}`.
2.  If no QCEW data exist for that `ref_date`, use the shares from the most recent QCEW `ref_date` that is available.

Employment shares move slowly enough that carry-forward over 5–6 months is reasonable. The implementation must record which QCEW `ref_date` was used for each composite month (see §5.3.3).

#### 5.3.3 Weight Staleness Tracking

For each provider composite month, the implementation records:

-   **`qcew_weight_ref_date`**: the `ref_date` of the QCEW observation used to compute the weights.
-   **`weight_staleness_months`**: the difference in months between the composite `ref_date` and the QCEW weight `ref_date`. Zero when QCEW is current; positive (typically 1–6) when weights are carried forward.

This metadata is stored alongside the composite and made available to validation (§6.3) and diagnostics. It enables stratified evaluation of composite quality by weight freshness.

#### 5.3.4 Weight Update Cadence

QCEW weights are updated as new QCEW vintages become available (~quarterly with ~5 month lag). Between updates, the most recent available weights are used (carry-forward per §5.3.2). All months — whether using current or carried-forward weights — produce a valid composite via the formula in §5.1.

### 5.4 Residual Composition Effects

Representativeness correction at the supersector × Census region level does not eliminate all composition effects. Within-cell heterogeneity (e.g., size class skew, sub-industry concentration) still creates provider-specific deviations from national growth. These residual effects are absorbed by the provider-specific bias `α_p` and noise `σ_{G,p}` in the downstream measurement equation. The correction reduces the **time-varying** component of composition bias — the part that matters for nowcasting accuracy — while leaving the time-invariant component to the existing parameter structure.

------------------------------------------------------------------------

## 6. Validation

### 6.1 Composition Bias Diagnostics *[Private repo]*

For each provider, compute:

1.  **Cell-level coverage ratios:** `payroll_emp_{p,c,t} / QCEW_emp_{c,t}` for each cell and month. Flag cells with ratios >1.0 or <0.01.
2.  **Industry concentration index:** Herfindahl of provider employment across supersectors, compared to QCEW Herfindahl. The correction should bring the provider's effective Herfindahl closer to QCEW's.
3.  **Geographic concentration index:** Same as above for Census regions.
4.  **Time-varying composition shift:** Track how the provider's effective industry/geography mix changes over time. Large shifts indicate composition drift that the correction should address.

### 6.2 Signal Quality Diagnostics *[Private repo]*

1.  **Composite vs. raw comparison.** For each provider, internally compute a QCEW-weighted composite from the cell-level parquet and compare against the raw aggregate index. Compute the correlation, mean divergence, and periods of maximum divergence. Divergence should be largest during months of heterogeneous sectoral growth.
2.  **QCEW tracking.** Compare the internally-computed composite against QCEW national growth (lagged). The corrected signal should track QCEW more closely than the raw signal, especially at turning points.
3.  **Coverage share time series.** Track the QCEW employment share of covered cells over time. Declining coverage indicates the provider is losing representation in important cells.
4.  **Cell employment stability.** Check that cell-level employment levels in the parquet are smooth across panel refresh boundaries — artificial jumps indicate geoclustering instability.

Private-repo validation should use the same weight carry-forward logic as production (§5.3.2) when computing internal composites, so that signal quality diagnostics reflect real-world conditions.

### 6.3 Downstream Evaluation *[alt_nfp]*

After integrating the cell-level parquets into the `alt_nfp` model (where QCEW-weighted compositing is performed):

1.  Does the posterior `α_p` (provider bias) shrink toward zero? The correction should absorb composition bias that was previously captured by the static bias term.
2.  Does the posterior `σ_{G,p}` (provider noise) decrease? Less time-varying composition bias means less unexplained variance.
3.  Does nowcast RMSE improve? This is the ultimate test.
4.  **Weight staleness reporting.** Report which months used current vs. carried-forward QCEW weights (§5.3.3). Stratify bias, noise, and RMSE metrics by weight staleness to detect whether composite quality degrades for months with stale weights. If degradation is found, it motivates tighter QCEW update cadence or QCEW share forecasting (Release 3).

------------------------------------------------------------------------

## 7. Implementation Plan

### 7.1 Phase 1: Pseudo-Establishment Pipeline *[Private repo]*

**Inputs:** Provider microdata with employee-level records, geocodes (or degraded location info), client NAICS codes.

**Steps:**

1.  Implement establishment-likeness diagnostics (§3.2).
2.  Implement geoclustering for enterprise-like clients (§3.3).
3.  Implement degraded-location fallbacks appropriate to each provider's data (§3.4).
4.  Assign pseudo-establishments to supersector × Census region cells.
5.  Validate cell assignments against known multi-establishment firms.

**Output:** A pseudo-establishment table with columns: `pseudo_estab_id`, `client_id`, `supersector`, `census_region`, `cell`, `employment`, `period`.

### 7.2 Phase 2: Cell-Level Signal Construction *[Private repo]*

**Inputs:** Pseudo-establishment table, frozen panel definitions.

**Steps:**

1.  Apply the frozen panel methodology at the cell level (§4).
2.  Compute cell-level continuing-units growth for each provider × cell × month.
3.  Aggregate pseudo-establishment employment to cell level (for parquet export).
4.  Apply minimum coverage thresholds and flag unreliable cells.

**Output:** A cell-level growth table with columns: `provider`, `cell`, `period`, `growth`, `employment`, `n_pseudo_estabs`, `is_reliable`.

### 7.3 Phase 3: Cell-Level Employment Parquet Export *[Private repo]*

**Inputs:** Cell-level signal table from Phase 2, pseudo-establishment employment levels.

**Steps:**

1.  Aggregate pseudo-establishment employment to cell (supersector × Census region) × month level.
2.  Express employment in thousands (000s).
3.  Set `ref_date` to the 12th of each reference month.
4.  Map cells to `geographic_type='region'`, `geographic_code` ∈ {'1','2','3','4'} and `industry_type='supersector'`, `industry_code` ∈ {'10','20','30','40','50','55','60','65','70','80','90'}.
5.  Include `n_pseudo_estabs` per cell × month for downstream threshold filtering.
6.  Export as `xxx_provider.parquet` per the schema in §1.4.1.
7.  Separately, compute national birth rates and export as `xxx_births.parquet` per §1.4.2 (if birth data available).

**Output:** The cell-level employment parquet and (optionally) national birth-rate parquet, per the interface contracts in §1.4.

### 7.4 Phase 4: Validation *[Private repo]*

**Steps:**

1.  Run composition bias diagnostics (§6.1) on the exported cell-level parquet.
2.  Internally compute a QCEW-weighted national composite from the parquet to run signal quality diagnostics (§6.2) — this composite is for validation only, not exported.
3.  Verify parquet schema compliance (ref_date format, employment units, cell coverage, column names and types per §1.4.1).

### 7.5 Phase 5: Integration *[alt_nfp]*

**Steps:**

1.  Place the provider parquet at `data/providers/<name>/<name>_provider.parquet`.
2.  Implement or verify the QCEW-weighted compositing logic in `alt_nfp`'s data-loading layer:
    -   Load QCEW employment shares at supersector × region level from the vintage store (§5.3.1).
    -   Apply carry-forward logic for months beyond the QCEW frontier (§5.3.2).
    -   Record `qcew_weight_ref_date` and `weight_staleness_months` per composite month (§5.3.3).
    -   Apply minimum `n_pseudo_estabs` threshold (configurable in `ProviderConfig` or model constants).
    -   Redistribute weights for missing/below-threshold cells per §5.2.
    -   Compute weighted national composite growth rate per §5.1.
3.  Run the `alt_nfp` model with the cell-sourced composites and compare against previous results (§6.3), including weight staleness reporting.

------------------------------------------------------------------------

## 8. Data Requirements

### 8.1 Provider Microdata *[Private repo]*

| Data | Description | Used in |
|---|---|---|
| Employee records | Employee ID, geocode/zip/state, client ID, pay period, employment status | Phase 1 |
| Client records | Client ID, NAICS code, headquarters address, employment count | Phase 1 |
| Frozen panel definitions | Panel window dates, eligible client sets | Phase 2 |

### 8.2 External Data (Already Available)

| Data | Source | Used in |
|---|---|---|
| QCEW employment by supersector × region | BLS QCEW via `alt_nfp` vintage store | Phase 4 validation (private repo); Phase 5 compositing (`alt_nfp`) |
| Census region ↔ state mapping | Census Bureau | Phase 1, 3, 5 |
| Supersector ↔ NAICS mapping | BLS (see `lookups/industry.py`) | Phase 1 |
| QCEW MWR multi-establishment rates | BLS QCEW | Phase 1 (threshold calibration) |

------------------------------------------------------------------------

## 9. Configuration

### 9.1 Private Repo Configuration

Each provider's pseudo-establishment pipeline is configured with location-information metadata:

``` python
@dataclass
class ReprCorrectionConfig:
    """Provider-specific representativeness correction settings."""

    provider_name: str
    location_regime: str  # 'geocode', 'zipcode', 'state_only', 'none'

    # Geoclustering
    cluster_method: str = 'hdbscan'  # 'dbscan', 'hdbscan'
    cluster_min_samples: int = 3
    cluster_distance_km: float = 50.0  # industry-adjusted

    # Panel mechanics
    min_pseudo_estabs_per_cell: int = 5
    panel_refresh_frequency: str = 'quarterly'

    # Size threshold for degraded-location hybrid allocation
    enterprise_size_threshold: int = 100  # employees; industry-adjusted

    # QCEW weighting
    qcew_weight_vintage: str = 'latest'  # or specific quarter
```

### 9.2 `alt_nfp` Configuration

Provider integration uses the existing `ProviderConfig` in `config.py`:

``` python
ProviderConfig(
    name='G',
    file='providers/G/g_provider.parquet',
    error_model='iid',
)
```

When the data-loading layer detects cell-level data (`geographic_type='region'`), it triggers QCEW-weighted compositing automatically. Compositing parameters (e.g., `min_pseudo_estabs` threshold) are configured in `config.py` or as constants in the compositing module.

------------------------------------------------------------------------

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Geoclustering produces unstable pseudo-establishments across panel refreshes | Cell-level growth signals have artificial volatility at refresh boundaries | Temporal stability checks in diagnostics; smooth transitions between panels |
| Provider has no coverage in key supersectors (e.g., government) | Weight redistribution distorts the composite | Track coverage_share; flag providers with <70% coverage of private-sector QCEW employment |
| Size threshold for hybrid allocation is miscalibrated | Over-redistribution (lose real signal) or under-redistribution (retain HQ bias) | Calibrate per industry using QCEW MWR data; validate against provider's own located clients |
| QCEW weights are stale (5–6 month lag) | Composite lags structural shifts in employment distribution | Carry forward last available weights (§5.3.2); track staleness per month (§5.3.3); stratify validation by staleness (§6.3); future: QCEW share forecasting (Release 3) |
| Different providers have different location regimes | Inconsistent correction quality across providers | Document regime per provider; downstream model absorbs differences via `α_p` and `σ_{G,p}` |
| Sparse cells produce noisy growth signals | Composite quality degrades | Minimum coverage threshold; weight redistribution; coarser cell definition if needed |

------------------------------------------------------------------------

## 11. Success Criteria

1.  **Composition bias reduction:** For each provider, the corrected composite's industry and geographic Herfindahl indices are closer to QCEW's than the raw signal's.

2.  **QCEW tracking improvement:** The corrected composite has lower RMSE against QCEW national growth than the raw signal, measured out-of-sample with QCEW publication lag respected.

3.  **Downstream model improvement:** When the corrected composites are consumed by the `alt_nfp` model, at least one of: (a) posterior `α_p` shrinks, (b) posterior `σ_{G,p}` shrinks, (c) nowcast RMSE improves.

4.  **Coverage adequacy:** Each active provider covers cells representing ≥70% of private-sector QCEW employment by QCEW weight share.

5.  **Pipeline stability:** Cell-level growth signals are stable across panel refreshes (no artificial volatility spikes at refresh boundaries).

6.  **Weight staleness transparency:** Weight vintage and staleness are reported per composite month. Downstream evaluation (§6.3) is stratified by staleness; results should not degrade materially for carried-forward months relative to current-weight months.
