# TOML Configuration for Nowcast Model

**Config Management — Pydantic-validated TOML files for model, sampling, and pipeline settings**

Version: 0.1 \| Date: 2026-03-10

------------------------------------------------------------------------

## 1. Motivation

All model hyperparameters, sampling presets, provider definitions, and pipeline settings currently live as Python-level constants spread across `config.py`, `sampling.py`, `panel_adapter.py`, and `forecast.py`. This works but creates friction in three areas:

1.  **Experiment management** — comparing runs requires editing Python source and remembering to revert. No single artifact records "what configuration produced this output."
2.  **Backtest sweeps** — sensitivity analyses (`sensitivity.py`) monkey-patch `config.LOG_SIGMA_QCEW_*_MU` at runtime then restore, which is fragile.
3.  **Reproducibility** — output directories contain InferenceData and plots but no machine-readable record of the priors, sampling settings, or provider list used.

A TOML config file solves all three: it is human-readable, diffable, versionable, and can be copied into each output directory as a run receipt.

### 1.1 Design Goals

-   **Zero-change default** — running the pipeline with no TOML file produces identical results to the current `config.py` defaults.
-   **Partial override** — a TOML file need only specify the fields being changed; everything else inherits from defaults.
-   **Validation at load time** — Pydantic catches typos, out-of-range values, and schema violations before the sampler starts.
-   **Run receipt** — every output directory gets a copy of the resolved (merged) config as `config.toml`.
-   **No runtime cost** — config is loaded once at startup and threaded through as a frozen dataclass; hot-path code sees plain attributes, not dict lookups.

### 1.2 Non-Goals

-   GUI or interactive config editing.
-   Hot-reloading during sampling.
-   Replacing `ProviderConfig` as a user-facing API — TOML serialises the same fields.
-   Moving lookup tables (`lookups/`) or revision schedules into TOML.

------------------------------------------------------------------------

## 2. TOML Schema

The file is divided into sections that mirror the current module boundaries. All fields are optional — omitted fields use the defaults shown below (which match the current `config.py` / `sampling.py` values exactly).

``` toml
# ── Paths ────────────────────────────────────────────────────────────────
[paths]
data_dir   = "data"           # relative to repo root
output_dir = "output"         # relative to repo root

# ── Latent AR(1) process ─────────────────────────────────────────────────
[model.latent]
log_tau_mu = -4.3428          # log(0.013)
log_tau_sd = 0.5

# ── CES observation noise ───────────────────────────────────────────────
[model.ces]
log_sigma_mu = -6.2146        # log(0.002)
log_sigma_sd = 0.5

# ── QCEW observation noise ──────────────────────────────────────────────
[model.qcew]
nu = 5                        # Student-t degrees of freedom (fixed)
log_sigma_mid_mu    = -7.6009 # log(0.0005), M2 months
log_sigma_mid_sd    = 0.15
log_sigma_boundary_mu = -6.2146  # log(0.002), M3+M1 months
log_sigma_boundary_sd = 0.5

# Post-COVID era multipliers for boundary months (M1+M3).
# Keys are revision numbers (0, 1, 2); default for unlisted revisions = 1.0.
[model.qcew.post_covid_boundary_mult]
0 = 5.0
1 = 3.5
2 = 2.0
default = 1.0

# ── Fourier seasonal ────────────────────────────────────────────────────
[model.fourier]
n_harmonics    = 4
log_sigma_mu   = -8.1117      # log(0.0003), for k=1
log_sigma_sd   = 0.5

# ── Birth / death structural model ──────────────────────────────────────
[model.birth_death]
qcew_lag       = 6            # months
log_sigma_mu   = -5.8091      # log(0.003)
log_sigma_sd   = 0.5

# ── Eras ─────────────────────────────────────────────────────────────────
[model.eras]
breaks = ["2020-01-01"]       # ISO date strings; N_ERAS = len(breaks) + 1

# ── Compositing ──────────────────────────────────────────────────────────
[model.compositing]
min_pseudo_estabs_per_cell = 5

# ── Providers ────────────────────────────────────────────────────────────
[[providers]]
name        = "G"
file        = "providers/g/g_provider.parquet"   # relative to data_dir
error_model = "iid"                              # "iid" | "ar1"
birth_file  = "providers/g/g_births.parquet"     # optional

# ── Cyclical indicators ─────────────────────────────────────────────────
[[indicators]]
name    = "claims"
fred_id = "ICNSA"
freq    = "weekly"       # "weekly" | "monthly"
pub_lag = 1              # publication lag in months

[[indicators]]
name    = "jolts"
fred_id = "JTSJOL"
freq    = "monthly"
pub_lag = 2

# ── Sampling presets ─────────────────────────────────────────────────────
[sampling.default]
draws         = 4000
tune          = 3000
chains        = 4
target_accept = 0.95

[sampling.light]
draws         = 2000
tune          = 2000
chains        = 2
target_accept = 0.95

[sampling.medium]
draws         = 4000
tune          = 3000
chains        = 4
target_accept = 0.95

# ── Backtest ─────────────────────────────────────────────────────────────
[backtest]
n_months         = 24
sampling_preset  = "light"    # which [sampling.*] to use

# ── Forecast ─────────────────────────────────────────────────────────────
[forecast]
end_date = "2026-01-12"       # ISO date, BLS convention day=12

# ── Publication lags (model-level censoring) ─────────────────────────────
[publication_lags]
provider_weeks = 3            # provider data lag in weeks
```

------------------------------------------------------------------------

## 3. Pydantic Models

All models use `pydantic.BaseModel` with `model_config = ConfigDict(frozen=True)` so the resolved config is immutable after loading.

### 3.1 Module: `src/alt_nfp/settings.py`

New module. Replaces `config.py` as the import target for all constants. `config.py` is retained as a thin backwards-compatibility shim (§5.2).

``` python
from __future__ import annotations

import math
import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── Submodels ────────────────────────────────────────────────────────────

class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_dir: str = "data"
    output_dir: str = "output"


class LatentConfig(BaseModel):
    """AR(1) latent growth process."""
    model_config = ConfigDict(frozen=True)

    log_tau_mu: float = math.log(0.013)
    log_tau_sd: Annotated[float, Field(gt=0)] = 0.5


class CESConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_sigma_mu: float = math.log(0.002)
    log_sigma_sd: Annotated[float, Field(gt=0)] = 0.5


class QCEWPostCovidBoundaryMult(BaseModel):
    """Era multipliers keyed by revision number."""
    model_config = ConfigDict(frozen=True, extra="allow")

    default: float = 1.0

    # Pydantic extra fields hold the int-keyed multipliers.
    # Accessor converts to dict[int, float].
    def as_dict(self) -> tuple[dict[int, float], float]:
        mult = {}
        for k, v in self.__pydantic_extra__.items():
            mult[int(k)] = float(v)
        return mult, self.default


class QCEWConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    nu: Annotated[int, Field(ge=2)] = 5
    log_sigma_mid_mu: float = math.log(0.0005)
    log_sigma_mid_sd: Annotated[float, Field(gt=0)] = 0.15
    log_sigma_boundary_mu: float = math.log(0.002)
    log_sigma_boundary_sd: Annotated[float, Field(gt=0)] = 0.5
    post_covid_boundary_mult: QCEWPostCovidBoundaryMult = QCEWPostCovidBoundaryMult(
        **{"0": 5.0, "1": 3.5, "2": 2.0, "default": 1.0}
    )


class FourierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_harmonics: Annotated[int, Field(ge=1, le=12)] = 4
    log_sigma_mu: float = math.log(0.0003)
    log_sigma_sd: Annotated[float, Field(gt=0)] = 0.5


class BirthDeathConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    qcew_lag: Annotated[int, Field(ge=1)] = 6
    log_sigma_mu: float = math.log(0.003)
    log_sigma_sd: Annotated[float, Field(gt=0)] = 0.5


class ErasConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    breaks: list[date] = [date(2020, 1, 1)]

    @field_validator("breaks")
    @classmethod
    def breaks_sorted(cls, v: list[date]) -> list[date]:
        if v != sorted(v):
            raise ValueError("era breaks must be in chronological order")
        return v

    @property
    def n_eras(self) -> int:
        return len(self.breaks) + 1


class CompositingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_pseudo_estabs_per_cell: Annotated[int, Field(ge=1)] = 5


class ModelConfig(BaseModel):
    """All model hyperparameters."""
    model_config = ConfigDict(frozen=True)

    latent: LatentConfig = LatentConfig()
    ces: CESConfig = CESConfig()
    qcew: QCEWConfig = QCEWConfig()
    fourier: FourierConfig = FourierConfig()
    birth_death: BirthDeathConfig = BirthDeathConfig()
    eras: ErasConfig = ErasConfig()
    compositing: CompositingConfig = CompositingConfig()


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    file: str
    error_model: Literal["iid", "ar1"] = "iid"
    birth_file: str | None = None
    industry_type: str = "national"
    industry_code: str = "00"
    geography_type: str = "national"
    geography_code: str = "00"


class IndicatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    fred_id: str
    freq: Literal["weekly", "monthly"]
    pub_lag: Annotated[int, Field(ge=0)]


class SamplingPreset(BaseModel):
    model_config = ConfigDict(frozen=True)

    draws: Annotated[int, Field(ge=100)] = 4000
    tune: Annotated[int, Field(ge=100)] = 3000
    chains: Annotated[int, Field(ge=1)] = 4
    target_accept: Annotated[float, Field(gt=0, lt=1)] = 0.95

    def to_pymc_kwargs(self) -> dict:
        return {
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "target_accept": self.target_accept,
            "return_inferencedata": True,
        }


class SamplingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default: SamplingPreset = SamplingPreset()
    light: SamplingPreset = SamplingPreset(draws=2000, tune=2000, chains=2)
    medium: SamplingPreset = SamplingPreset()

    def get_preset(self, name: str) -> SamplingPreset:
        if not hasattr(self, name):
            raise ValueError(f"unknown sampling preset: {name!r}")
        return getattr(self, name)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_months: Annotated[int, Field(ge=1)] = 24
    sampling_preset: str = "light"


class ForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    end_date: date = date(2026, 1, 12)


class PublicationLagsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_weeks: Annotated[int, Field(ge=0)] = 3


# ── Root config ──────────────────────────────────────────────────────────

class NowcastConfig(BaseModel):
    """Root configuration for the entire nowcast pipeline."""
    model_config = ConfigDict(frozen=True)

    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    providers: list[ProviderConfig] = [
        ProviderConfig(
            name="G",
            file="providers/g/g_provider.parquet",
            error_model="iid",
            birth_file="providers/g/g_births.parquet",
        )
    ]
    indicators: list[IndicatorConfig] = [
        IndicatorConfig(name="claims", fred_id="ICNSA", freq="weekly", pub_lag=1),
        IndicatorConfig(name="jolts", fred_id="JTSJOL", freq="monthly", pub_lag=2),
    ]
    sampling: SamplingConfig = SamplingConfig()
    backtest: BacktestConfig = BacktestConfig()
    forecast: ForecastConfig = ForecastConfig()
    publication_lags: PublicationLagsConfig = PublicationLagsConfig()

    @model_validator(mode="after")
    def _check_backtest_preset_exists(self) -> NowcastConfig:
        self.sampling.get_preset(self.backtest.sampling_preset)
        return self

    # ── Resolved path helpers ────────────────────────────────────────────

    def resolve_paths(self, base_dir: Path) -> ResolvedPaths:
        return ResolvedPaths(
            base_dir=base_dir,
            data_dir=base_dir / self.paths.data_dir,
            output_dir=base_dir / self.paths.output_dir,
        )
```

### 3.2 `ResolvedPaths`

``` python
from dataclasses import dataclass

@dataclass(frozen=True)
class ResolvedPaths:
    base_dir: Path
    data_dir: Path
    output_dir: Path

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "store"

    @property
    def downloads_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def intermediate_dir(self) -> Path:
        return self.data_dir / "intermediate"

    @property
    def indicators_dir(self) -> Path:
        return self.data_dir / "indicators"
```

### 3.3 Loading

``` python
def load_config(path: Path | None = None, *, base_dir: Path | None = None) -> NowcastConfig:
    """Load and validate a TOML config, falling back to defaults.

    Parameters
    ----------
    path
        Path to a TOML file.  *None* → pure defaults (current behaviour).
    base_dir
        Repository root.  Defaults to the parent of src/alt_nfp.
    """
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        cfg = NowcastConfig.model_validate(raw)
    else:
        cfg = NowcastConfig()
    return cfg


def save_config(cfg: NowcastConfig, path: Path) -> None:
    """Write the resolved config as TOML (run receipt)."""
    import tomli_w
    path.write_bytes(tomli_w.dumps(cfg.model_dump(mode="python")))
```

------------------------------------------------------------------------

## 4. Validation Rules

Pydantic enforces structural constraints at parse time. Additional domain constraints are expressed as validators on the Pydantic models.

| Field | Constraint | Rationale |
|------------------------|------------------------|------------------------|
| All `*_sd` fields | `> 0` | Standard deviation must be positive |
| `qcew.nu` | `>= 2` | Student-t requires ≥ 2 for finite variance |
| `fourier.n_harmonics` | `[1, 12]` | Max 6 pairs on a 12-month cycle |
| `sampling.*.draws` | `>= 100` | Minimum useful posterior |
| `sampling.*.target_accept` | `(0, 1)` | HMC acceptance probability |
| `eras.breaks` | sorted ascending | Chronological era boundaries |
| `backtest.sampling_preset` | must exist in `[sampling]` | Validated by `model_validator` |
| `providers.*.error_model` | `"iid"` or `"ar1"` | Literal type constraint |
| `indicators.*.freq` | `"weekly"` or `"monthly"` | Literal type constraint |

------------------------------------------------------------------------

## 5. Integration Plan

### 5.1 Config Threading

The `NowcastConfig` object is created once at the top of each entry point (`alt_nfp_estimation_v3.py`, `backtest.run_backtest()`, etc.) and passed explicitly to every function that currently imports from `config`.

Functions that currently do `from .config import LOG_SIGMA_CES_MU, ...` will instead receive a `cfg: NowcastConfig` parameter and access `cfg.model.ces.log_sigma_mu`.

**Affected call signatures** (non-exhaustive):

| Function | New parameter |
|------------------------------------|------------------------------------|
| `build_model(data)` | `build_model(data, cfg)` |
| `panel_to_model_data(panel, providers)` | `panel_to_model_data(panel, cfg)` |
| `sample_model(model, **kw)` | `sample_model(model, cfg, preset="default")` |
| `forecast_and_plot(idata, data)` | `forecast_and_plot(idata, data, cfg)` |
| `run_backtest(...)` | `run_backtest(..., cfg=None)` |
| `run_sensitivity(...)` | accepts list of `NowcastConfig` objects instead of monkey-patching |

### 5.2 Backwards-Compatibility Shim (`config.py`)

During migration, `config.py` is kept as a thin shim that instantiates a default `NowcastConfig()` and re-exports the old module-level constants. This allows incremental migration — modules can be updated one at a time.

``` python
# config.py — backwards-compatibility shim
from .settings import NowcastConfig, ProviderConfig  # noqa: F401

_DEFAULT = NowcastConfig()

# Re-export legacy constants
LOG_SIGMA_QCEW_MID_MU = _DEFAULT.model.qcew.log_sigma_mid_mu
LOG_SIGMA_QCEW_MID_SD = _DEFAULT.model.qcew.log_sigma_mid_sd
# ... etc for all constants ...

PROVIDERS = _DEFAULT.providers
CYCLICAL_INDICATORS = [
    {"name": i.name, "fred_id": i.fred_id, "freq": i.freq}
    for i in _DEFAULT.indicators
]
```

Once all call sites are migrated, `config.py` is removed.

### 5.3 Run Receipt

Every pipeline entry point that writes to an output directory will call `save_config(cfg, output_dir / "config.toml")` after sampling completes. This captures the exact configuration that produced the results.

For backtest runs, the receipt goes to `output/backtest_runs/<timestamp>/config.toml`.

### 5.4 CLI Integration

The runner script and CLI entry points accept an optional `--config` flag:

``` bash
# Use defaults (identical to current behaviour)
python alt_nfp_estimation_v3.py

# Override with a TOML file
python alt_nfp_estimation_v3.py --config experiments/tight_qcew.toml

# Backtest with custom config
python -c "
from pathlib import Path
from alt_nfp.settings import load_config
from alt_nfp.backtest import run_backtest
cfg = load_config(Path('experiments/tight_qcew.toml'))
run_backtest(cfg=cfg)
"
```

The `alt-nfp` vintage pipeline CLI (`vintages/__main__.py`) gains a `--config` option that passes `cfg.paths` to download/process/build steps.

------------------------------------------------------------------------

## 6. Sensitivity Analysis Cleanup

`sensitivity.py` currently monkey-patches `config.LOG_SIGMA_QCEW_*_MU` at runtime. With TOML configs, each sensitivity scenario is a distinct `NowcastConfig` built via `model_copy(update=...)`:

``` python
def build_sensitivity_configs(
    base: NowcastConfig,
    multipliers: list[tuple[str, float, float]],
) -> list[tuple[str, NowcastConfig]]:
    configs = []
    for label, mid_mult, boundary_mult in multipliers:
        new_qcew = base.model.qcew.model_copy(update={
            "log_sigma_mid_mu": base.model.qcew.log_sigma_mid_mu + math.log(mid_mult),
            "log_sigma_boundary_mu": base.model.qcew.log_sigma_boundary_mu + math.log(boundary_mult),
        })
        new_model = base.model.model_copy(update={"qcew": new_qcew})
        new_cfg = base.model_copy(update={"model": new_model})
        configs.append((label, new_cfg))
    return configs
```

No more global state mutation.

------------------------------------------------------------------------

## 7. Dependencies

| Package | Version | Purpose |
|------------------------|------------------------|------------------------|
| `pydantic` | `>=2.0` | Config validation & serialisation |
| `tomli-w` | `>=1.0` | TOML writing (run receipt); `tomllib` (stdlib 3.11+) handles reading |

`tomllib` is available in Python 3.11+. Since the project requires 3.12, no backport is needed for reading. `tomli-w` is needed only for writing the run receipt.

Add to `pyproject.toml`:

``` toml
dependencies = [
    # ... existing ...
    "pydantic>=2.0",
    "tomli-w>=1.0",
]
```

------------------------------------------------------------------------

## 8. File Changes Summary

| File | Action |
|------------------------------------|------------------------------------|
| `src/alt_nfp/settings.py` | **New** — Pydantic models, `load_config`, `save_config` |
| `src/alt_nfp/config.py` | **Modified** → thin shim re-exporting from `settings.py` |
| `src/alt_nfp/model.py` | **Modified** — accept `cfg` parameter |
| `src/alt_nfp/panel_adapter.py` | **Modified** — accept `cfg`, drop `_CYCLICAL_PUBLICATION_LAGS` dict |
| `src/alt_nfp/sampling.py` | **Modified** — presets read from `cfg.sampling` |
| `src/alt_nfp/forecast.py` | **Modified** — end date from `cfg.forecast.end_date` |
| `src/alt_nfp/backtest.py` | **Modified** — accept `cfg`, use `cfg.backtest.*` |
| `src/alt_nfp/sensitivity.py` | **Modified** — accept list of configs, no monkey-patching |
| `src/alt_nfp/diagnostics.py` | **Modified** — accept `cfg` where era/provider info needed |
| `src/alt_nfp/checks.py` | **Modified** — accept `cfg` for era labels |
| `alt_nfp_estimation_v3.py` | **Modified** — `load_config()`, pass `cfg` through pipeline |
| `src/alt_nfp/vintages/__main__.py` | **Modified** — `--config` flag for paths |
| `pyproject.toml` | **Modified** — add `pydantic`, `tomli-w` deps |
| `tests/test_settings.py` | **New** — config loading, validation, round-trip tests |

------------------------------------------------------------------------

## 9. Migration Order

1.  **Add `settings.py`** with all Pydantic models and `load_config` / `save_config`. Add `pydantic` and `tomli-w` to `pyproject.toml`.
2.  **Convert `config.py`** to a backwards-compat shim that instantiates `NowcastConfig()` and re-exports all constants. Run full test suite — must pass with zero changes elsewhere.
3.  **Thread `cfg` through the pipeline**, one module at a time: `model.py` → `panel_adapter.py` → `sampling.py` → `forecast.py` → `backtest.py` → `sensitivity.py` → `diagnostics.py` / `checks.py`. Each step updates the function signature and its callers.
4.  **Update entry points** (`alt_nfp_estimation_v3.py`, CLI) to accept `--config` and call `save_config` on completion.
5.  **Write tests** for config loading, validation errors, TOML round-trip, and receipt writing.
6.  **Remove shim** — delete re-exports from `config.py` once all call sites are migrated. `config.py` can then be deleted or kept for `ProviderConfig` re-export only.

------------------------------------------------------------------------

## 10. Example TOML Files

### 10.1 Tight QCEW experiment

``` toml
# experiments/tight_qcew.toml — halve QCEW boundary noise
[model.qcew]
log_sigma_boundary_mu = -6.9078   # log(0.001)
log_sigma_boundary_sd = 0.3
```

### 10.2 Fast iteration (fewer draws, 2 chains)

``` toml
[sampling.default]
draws  = 1000
tune   = 1000
chains = 2
```

### 10.3 Extended backtest

``` toml
[backtest]
n_months        = 48
sampling_preset = "light"
```