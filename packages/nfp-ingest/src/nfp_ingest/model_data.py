"""Model-ready data extraction: one function answers "what was knowable on D".

Ports the knowability logic that previously lived in the model package's
``panel_adapter`` (layer-2 censoring) into the data side, per Phase A2:

- :func:`build_model_data` — the single entry point: layer-1 rank-based
  panel censoring (``build_panel(as_of_ref=D)``) followed by layer-2
  extraction (vintage cutoff, best-available CES selection, QCEW noise
  multipliers, provider publication-lag censoring, cyclical-indicator
  masking).
- :func:`panel_to_model_data` — layer-2 only, for callers that already
  hold a panel.
- :class:`ModelDataConfig` — the knowability + measurement-metadata knobs,
  with defaults frozen from the reference implementation's settings.

The model layer consumes the returned dict's finished arrays and must not
import acquisition code. Plotting concerns (e.g. provider colors) are
deliberately absent here.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from nfp_lookups.paths import INDICATORS_DIR
from nfp_lookups.provider_config import (
    CYCLICAL_INDICATORS_DEFAULT,
    PROVIDERS_DEFAULT,
    CyclicalIndicator,
    ProviderConfig,
)
from nfp_lookups.revision_schedules import get_noise_multiplier

from nfp_ingest.panel import build_panel
from nfp_ingest.payroll import load_provider_series

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelDataConfig:
    """Knowability + measurement-metadata knobs for model-data extraction.

    Defaults are frozen from the reference implementation's
    ``NowcastConfig`` (verified live against the old repo, 2026-06-12).
    The pydantic/toml settings system remains a model-package concern;
    the data layer only needs these values.
    """

    era_breaks: tuple[date, ...] = (date(2020, 1, 1),)
    bd_qcew_lag: int = 6
    provider_pub_lag_weeks: int = 3
    qcew_post_covid_boundary_mult: dict[int, float] = field(
        default_factory=lambda: {0: 5.0, 1: 3.5, 2: 2.0}
    )
    qcew_post_covid_boundary_mult_default: float = 1.0
    indicators: tuple[CyclicalIndicator, ...] = tuple(CYCLICAL_INDICATORS_DEFAULT)


def _date_to_era(d: date, breaks: list[date]) -> int:
    """Map a date to its era index using the given era breaks."""
    for i, brk in enumerate(breaks):
        if d < brk:
            return i
    return len(breaks)


def _offset_month(d: date, months: int) -> date:
    """Add *months* to a date, returning the 1st of the resulting month."""
    total = d.month + months
    year = d.year + (total - 1) // 12
    month = ((total - 1) % 12) + 1
    return date(year, month, 1)


def build_model_data(
    as_of: date,
    *,
    store_path: Path | None = None,
    providers: list[ProviderConfig] | None = None,
    start_year: int = 2012,
    end_year: int | None = None,
    config: ModelDataConfig | None = None,
    indicators_dir: Path | None = None,
) -> dict:
    """Everything knowable on *as_of*, as model-ready arrays.

    Applies both censoring layers with the same cutoff: layer-1 rank-based
    panel construction (``build_panel(as_of_ref=as_of)``, day-12
    convention) and layer-2 extraction (``panel_to_model_data(as_of=
    as_of)``).

    Parameters
    ----------
    as_of : date
        Universal knowledge cutoff (BLS day-12 reference convention).
    store_path : Path, optional
        Vintage store root; defaults to ``VINTAGE_STORE_PATH``.
    providers : list[ProviderConfig], optional
        Defaults to ``PROVIDERS_DEFAULT``.
    start_year, end_year : int
        Model calendar bounds passed to ``build_panel``.
    config : ModelDataConfig, optional
        Knowability knobs; defaults are the frozen reference values.
    indicators_dir : Path, optional
        Cyclical indicator parquet directory; defaults to
        ``INDICATORS_DIR``.

    Returns
    -------
    dict
        The model data dict (see :func:`panel_to_model_data`).
    """
    if providers is None:
        providers = list(PROVIDERS_DEFAULT)
    panel = build_panel(
        store_path=store_path,
        providers=providers,
        start_year=start_year,
        end_year=end_year,
        as_of_ref=as_of,
    )
    return panel_to_model_data(
        panel,
        providers,
        as_of=as_of,
        config=config,
        indicators_dir=indicators_dir,
    )


def panel_to_model_data(
    panel: pl.DataFrame,
    providers: list[ProviderConfig],
    censor_ces_from: date | None = None,
    as_of: date | None = None,
    *,
    geographic_code: str = "US",
    industry_code: str = "00",
    config: ModelDataConfig | None = None,
    indicators_dir: Path | None = None,
) -> dict:
    """Convert an observation panel to the model data dict.

    Parameters
    ----------
    panel : pl.DataFrame
        Validated observation panel (PANEL_SCHEMA).
    providers : list[ProviderConfig]
        Provider list (e.g. ``PROVIDERS_DEFAULT``).
    censor_ces_from : date, optional
        If set, treat CES SA/NSA as missing from this date onward (for
        backtests). Ignored when *as_of* is provided.
    as_of : date, optional
        Universal censoring cutoff.  When set, observations whose
        ``vintage_date`` exceeds *as_of* are dropped before growth-rate
        extraction, provider birth rates are censored by the provider
        publication lag, and cyclical indicators are masked using their
        respective publication lags.  Supersedes *censor_ces_from*.
    geographic_code : str
        Geographic code for the main series (default 'US'; '00' rows are
        always accepted — vintage store convention).
    industry_code : str
        Industry code for the main series (default '00' = total nonfarm).
    config : ModelDataConfig, optional
        Knowability knobs; defaults are the frozen reference values.
    indicators_dir : Path, optional
        Cyclical indicator parquet directory; defaults to
        ``INDICATORS_DIR``.

    Returns
    -------
    dict
        Dict of finished arrays + metadata consumed by the model layer and
        downstream diagnostics.
    """
    if config is None:
        config = ModelDataConfig()

    era_breaks = list(config.era_breaks)
    bd_qcew_lag = config.bd_qcew_lag
    indicators = list(config.indicators)
    publication_lags = {ind.name: ind.pub_lag for ind in indicators}
    provider_pub_lag_weeks = config.provider_pub_lag_weeks
    qcew_pcb_mult = dict(config.qcew_post_covid_boundary_mult)
    qcew_pcb_default = config.qcew_post_covid_boundary_mult_default
    if indicators_dir is None:
        indicators_dir = INDICATORS_DIR

    if as_of is not None and "vintage_date" in panel.columns:
        panel = panel.filter(
            pl.col("vintage_date").is_null() | (pl.col("vintage_date") <= as_of)
        )
    elif as_of is not None:
        warnings.warn(
            "Panel lacks vintage_date column; as_of censoring is incomplete.",
            stacklevel=2,
        )

    # Restrict to national scope and chosen industry (no fallback 05→00 so we
    # stay private-only when matching legacy; vintage store often has '00' only)
    geo_filter = pl.col("geographic_code").is_in([geographic_code, "00", "US"])
    national = panel.filter(
        (pl.col("geographic_type") == "national")
        & geo_filter
        & (pl.col("industry_code") == industry_code)
    )
    if len(national) == 0:
        raise ValueError(
            f"No national observations for industry_code={industry_code!r}; "
            "panel may be empty or use a different industry_code."
        )

    # Unique sorted periods = model calendar
    dates = sorted(national["period"].unique().to_list())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    T = len(dates)
    month_of_year = np.array([d.month - 1 for d in dates], dtype=int)
    year0 = dates[0].year
    year_of_obs = np.array([d.year - year0 for d in dates], dtype=int)
    n_years = int(year_of_obs.max()) + 1
    era_idx = np.array([_date_to_era(d, era_breaks) for d in dates], dtype=int)

    # Helper: one T-length array per source, filled from panel (final vintage per period)
    def _growth_series(source: str) -> np.ndarray:
        out = np.full(T, np.nan, dtype=float)
        sub = national.filter(pl.col("source") == source)
        if len(sub) == 0:
            return out
        # Prefer is_final; then highest revision (benchmark -1 before 0,1,2)
        by_period = (
            sub.with_columns(
                pl.when(pl.col("revision_number") == -1)
                .then(999)
                .otherwise(pl.col("revision_number"))
                .alias("_rev_sort")
            )
            .sort(pl.col("is_final").fill_null(False), "_rev_sort", descending=[True, True])
            .unique(subset=["period"], keep="first")
        )
        for row in by_period.iter_rows(named=True):
            period = row["period"]
            growth = row["growth"]
            if period in date_to_idx and growth is not None and np.isfinite(growth):
                out[date_to_idx[period]] = float(growth)
        return out

    def _qcew_series_with_meta(
        nat: pl.DataFrame, date_list: list, length: int
    ) -> tuple[np.ndarray, dict]:
        """QCEW growth array and period -> revision_number for selected rows."""
        out = np.full(length, np.nan, dtype=float)
        sub = nat.filter(pl.col("source") == "qcew")
        period_to_rev: dict = {}
        if len(sub) == 0:
            return out, period_to_rev
        by_period = (
            sub.with_columns(
                pl.when(pl.col("revision_number") == -1)
                .then(999)
                .otherwise(pl.col("revision_number"))
                .alias("_rev_sort")
            )
            .sort(pl.col("is_final").fill_null(False), "_rev_sort", descending=[True, True])
            .unique(subset=["period"], keep="first")
        )
        for row in by_period.iter_rows(named=True):
            period = row["period"]
            growth = row["growth"]
            rev = row.get("revision_number")
            if period in date_to_idx and growth is not None and np.isfinite(growth):
                idx = date_to_idx[period]
                out[idx] = float(growth)
                period_to_rev[period] = int(rev) if rev is not None else 0
        return out, period_to_rev

    g_qcew, qcew_period_to_revision = _qcew_series_with_meta(national, dates, T)

    qcew_obs = np.where(np.isfinite(g_qcew))[0]
    # M2 = quarter-interior months (Feb, May, Aug, Nov); boundary = M3 + M1
    qcew_is_m2 = np.array([dates[i].month in (2, 5, 8, 11) for i in qcew_obs])
    qcew_noise_mult = np.array(
        [
            get_noise_multiplier(
                f"qcew_Q{(dates[i].month - 1) // 3 + 1}",
                int(qcew_period_to_revision.get(dates[i], 0)),
            )
            for i in qcew_obs
        ],
        dtype=float,
    )

    # Era-specific boundary multiplier: inflate noise for post-COVID M1+M3
    for j, i in enumerate(qcew_obs):
        if qcew_is_m2[j]:
            continue
        if _date_to_era(dates[i], era_breaks) >= 1:  # Post-COVID
            rev = int(qcew_period_to_revision.get(dates[i], 0))
            era_mult = qcew_pcb_mult.get(rev, qcew_pcb_default)
            qcew_noise_mult[j] *= era_mult

    # CES best-available: one obs per month using the latest print.
    # Track which vintage (0=1st, 1=2nd, 2=final) each obs came from.
    def _ces_best_available(source: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (growth, vintage_idx) arrays of length T.

        growth[t] = best-available growth for month t (NaN if missing).
        vintage_idx[t] = revision number (0/1/2) selected, or -1 if missing.
        """
        growth = np.full(T, np.nan, dtype=float)
        vidx = np.full(T, -1, dtype=int)
        sub = national.filter(pl.col("source") == source)
        if len(sub) == 0:
            return growth, vidx
        by_period = (
            sub.filter(pl.col("revision_number").is_in([0, 1, 2]))
            .sort("revision_number", descending=True)
            .unique(subset=["period"], keep="first")
        )
        for row in by_period.iter_rows(named=True):
            period = row["period"]
            g = row["growth"]
            rev = row.get("revision_number")
            if period in date_to_idx and g is not None and np.isfinite(g):
                idx = date_to_idx[period]
                growth[idx] = float(g)
                vidx[idx] = int(rev) if rev is not None else 2
        return growth, vidx

    g_ces_sa, ces_sa_full_vidx = _ces_best_available("ces_sa")
    g_ces_nsa, ces_nsa_full_vidx = _ces_best_available("ces_nsa")

    if censor_ces_from is not None and as_of is None:
        for i, d in enumerate(dates):
            if d >= censor_ces_from:
                g_ces_sa[i:] = np.nan
                g_ces_nsa[i:] = np.nan
                ces_sa_full_vidx[i:] = -1
                ces_nsa_full_vidx[i:] = -1
                break

    ces_sa_obs = np.where(np.isfinite(g_ces_sa))[0]
    ces_nsa_obs = np.where(np.isfinite(g_ces_nsa))[0]
    ces_sa_vintage_idx_raw = ces_sa_full_vidx[ces_sa_obs]
    ces_nsa_vintage_idx_raw = ces_nsa_full_vidx[ces_nsa_obs]

    # Remap vintage indices to contiguous 0-based range so sigma_ces has
    # only as many elements as there are observed vintages (avoids ghost
    # parameters for vintages with zero observations).
    _all_vintages = sorted(
        set(ces_sa_vintage_idx_raw.tolist()) | set(ces_nsa_vintage_idx_raw.tolist())
    )
    if not _all_vintages:
        _all_vintages = [2]  # fallback: at least Final
    ces_vintage_map: dict[int, int] = {v: i for i, v in enumerate(_all_vintages)}
    n_ces_vintages = len(_all_vintages)

    ces_sa_vintage_idx = np.array(
        [ces_vintage_map[v] for v in ces_sa_vintage_idx_raw], dtype=int
    )
    ces_nsa_vintage_idx = np.array(
        [ces_vintage_map[v] for v in ces_nsa_vintage_idx_raw], dtype=int
    )

    # Provider data
    pp_data: list[dict] = []
    for cfg in providers:
        source_name = cfg.name.lower()
        emp_col = f"{cfg.name.lower()}_employment"
        g_pp = _growth_series(source_name)
        pp_obs = np.where(np.isfinite(g_pp))[0]
        entry: dict = {
            "name": cfg.name,
            "config": cfg,
            "g_pp": g_pp,
            "pp_obs": pp_obs,
            "emp_col": emp_col,
        }

        pp_series = load_provider_series(cfg)
        if pp_series is not None and "birth_rate" in pp_series.columns:
            births_df = pp_series.select(["ref_date", "birth_rate"])
            births_joined = pl.DataFrame({"ref_date": dates}).join(
                births_df, on="ref_date", how="left"
            )
            births_arr = births_joined["birth_rate"].to_numpy().astype(float)
            # Censor birth rate data not yet published as of the as_of date.
            # Provider data is available ~3 weeks after the reference period.
            if as_of is not None:
                lag = timedelta(weeks=provider_pub_lag_weeks)
                for i, d in enumerate(dates):
                    if d + lag > as_of:
                        births_arr[i:] = np.nan
                        break
            entry["births"] = births_arr
            entry["births_obs"] = np.where(np.isfinite(births_arr))[0]
        else:
            entry["births"] = None
            entry["births_obs"] = None
        pp_data.append(entry)

    # BD covariates — restrict averaging to the provider-covered window so we
    # don't nanmean over all-NaN slices for months before coverage begins.
    birth_arrays = [pp["births"] for pp in pp_data if pp["births"] is not None]
    if birth_arrays:
        stacked = np.vstack(birth_arrays)
        any_finite = np.any(np.isfinite(stacked), axis=0)
        birth_rate = np.full(T, np.nan)
        birth_rate[any_finite] = np.nanmean(stacked[:, any_finite], axis=0)
    else:
        birth_rate = np.full(T, np.nan)
    if pp_data:
        g_pp_stack = np.vstack([pp["g_pp"] for pp in pp_data])
        any_finite = np.any(np.isfinite(g_pp_stack), axis=0)
        g_pp_avg = np.full(T, np.nan)
        g_pp_avg[any_finite] = np.nanmean(g_pp_stack[:, any_finite], axis=0)
    else:
        g_pp_avg = np.full(T, np.nan)
    bd_proxy = g_qcew - g_pp_avg
    bd_qcew_lagged = np.full(T, np.nan)
    for t in range(bd_qcew_lag, T):
        if np.isfinite(bd_proxy[t - bd_qcew_lag]):
            bd_qcew_lagged[t] = bd_proxy[t - bd_qcew_lag]
    cyclical = _load_cyclical_indicators(dates, T, indicators, indicators_dir)

    if as_of is not None:
        for ind in indicators:
            key = f"{ind.name}_c"
            arr = cyclical.get(key)
            if arr is None:
                continue
            lag = publication_lags.get(ind.name, 1)
            for i, d in enumerate(dates):
                if _offset_month(d, lag) > as_of:
                    arr[i:] = 0.0
                    break

    # Levels: ref_date + index columns (reconstruct from growth for compatibility)
    levels_df = _build_levels_from_growth(
        dates=dates,
        g_ces_sa=g_ces_sa,
        g_ces_nsa=g_ces_nsa,
        g_qcew=g_qcew,
        pp_data=pp_data,
        national=national,
    )

    logger.info(
        "Model data: T=%s months (%s → %s); CES SA %s obs, CES NSA %s obs, "
        "QCEW %s obs; %s provider(s)",
        T,
        dates[0],
        dates[-1],
        len(ces_sa_obs),
        len(ces_nsa_obs),
        len(qcew_obs),
        len(providers),
    )

    return dict(
        panel=panel,
        levels=levels_df,
        dates=dates,
        T=T,
        month_of_year=month_of_year,
        year_of_obs=year_of_obs,
        n_years=n_years,
        era_idx=era_idx,
        g_ces_sa=g_ces_sa,
        ces_sa_obs=ces_sa_obs,
        ces_sa_vintage_idx=ces_sa_vintage_idx,
        g_ces_nsa=g_ces_nsa,
        ces_nsa_obs=ces_nsa_obs,
        ces_nsa_vintage_idx=ces_nsa_vintage_idx,
        n_ces_vintages=n_ces_vintages,
        ces_vintage_map=ces_vintage_map,
        g_qcew=g_qcew,
        qcew_obs=qcew_obs,
        qcew_is_m2=qcew_is_m2,
        qcew_noise_mult=qcew_noise_mult,
        pp_data=pp_data,
        n_providers=len(providers),
        birth_rate=birth_rate,
        bd_proxy=bd_proxy,
        bd_qcew_lagged=bd_qcew_lagged,
        **cyclical,
    )


def build_obs_sources(data: dict) -> dict:
    """Build ``{var_name: (label, observed_array)}`` used by predictive checks."""
    sources: dict[str, tuple[str, np.ndarray]] = {}

    ces_sa_obs = data["ces_sa_obs"]
    ces_nsa_obs = data["ces_nsa_obs"]
    if len(ces_sa_obs) > 0:
        sources["obs_ces_sa"] = ("CES SA", data["g_ces_sa"][ces_sa_obs])
    if len(ces_nsa_obs) > 0:
        sources["obs_ces_nsa"] = ("CES NSA", data["g_ces_nsa"][ces_nsa_obs])

    sources["obs_qcew"] = ("QCEW", data["g_qcew"][data["qcew_obs"]])
    for pp in data["pp_data"]:
        if len(pp["pp_obs"]) == 0:
            continue
        name = pp["config"].name.lower()
        sources[f"obs_{name}"] = (pp["name"], pp["g_pp"][pp["pp_obs"]])
    return sources


def _load_cyclical_indicators(
    dates: list,
    T: int,
    indicators: list[CyclicalIndicator],
    indicators_dir: Path,
) -> dict:
    """Load cyclical indicators from parquet, align to model dates, and centre.

    Reads from ``{indicators_dir}/<name>.parquet`` (schema: ``ref_date``,
    ``value``).  Weekly series are aggregated to monthly means before
    centering.

    Returns a dict with keys like ``'claims_c'``, ``'jolts_c'`` mapping to
    centred numpy arrays of length *T*.  Missing files are gracefully
    skipped (value set to ``None``).
    """
    result: dict = {}

    for spec in indicators:
        key = f"{spec.name}_c"
        fpath = indicators_dir / f"{spec.name}.parquet"

        if not fpath.exists():
            result[key] = None
            continue

        try:
            raw = pl.read_parquet(fpath).sort('ref_date')
        except (OSError, pl.exceptions.ComputeError) as e:
            logger.warning("Failed to read indicator %s from %s: %s", key, fpath, e)
            result[key] = None
            continue

        if 'value' not in raw.columns:
            result[key] = None
            continue

        if spec.freq == 'weekly':
            raw = raw.with_columns(
                pl.col('ref_date').dt.truncate('1mo').alias('month')
            )
            monthly = raw.group_by('month').agg(
                pl.col('value').mean().alias('value')
            ).sort('month').rename({'month': 'ref_date'})
        else:
            monthly = raw.select(['ref_date', 'value'])

        cal = pl.DataFrame({'ref_date': dates}).with_columns(
            pl.col('ref_date').dt.truncate('1mo').alias('month')
        )
        monthly = monthly.with_columns(
            pl.col('ref_date').dt.truncate('1mo').alias('month')
        )
        joined = cal.join(monthly.select(['month', 'value']), on='month', how='left')
        arr = joined['value'].to_numpy().astype(float)

        if np.any(np.isfinite(arr)):
            mean_val = float(np.nanmean(arr))
            std_val = float(np.nanstd(arr))
            if std_val > 0:
                arr_c = np.where(np.isfinite(arr), (arr - mean_val) / std_val, 0.0)
            else:
                arr_c = np.where(np.isfinite(arr), arr - mean_val, 0.0)
            result[key] = arr_c
        else:
            result[key] = None

    return result


def _build_levels_from_growth(
    dates: list[date],
    g_ces_sa: np.ndarray,
    g_ces_nsa: np.ndarray,
    g_qcew: np.ndarray,
    pp_data: list[dict],
    national: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a levels DataFrame from growth arrays (ref_date + index columns).

    When *national* is provided, also includes ``ces_sa_level`` and
    ``ces_nsa_level`` columns with actual BLS employment levels (thousands)
    for converting index forecasts to jobs-added estimates.
    """
    T = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    base = 100.0

    def cum_level(g: np.ndarray) -> np.ndarray:
        out = np.full(T, np.nan, dtype=float)
        log_level = np.nan
        for i in range(T):
            if np.isfinite(g[i]):
                if np.isnan(log_level):
                    log_level = np.log(base)
                log_level = log_level + g[i]
                out[i] = np.exp(log_level)
            elif not np.isnan(log_level):
                out[i] = np.exp(log_level)
        return out

    ces_sa_index = cum_level(g_ces_sa)
    ces_nsa_index = cum_level(g_ces_nsa)
    qcew_nsa_index = cum_level(g_qcew)

    d: dict = {
        "ref_date": dates,
        "ces_sa_index": ces_sa_index,
        "ces_nsa_index": ces_nsa_index,
        "qcew_nsa_index": qcew_nsa_index,
    }
    for pp in pp_data:
        d[pp["emp_col"]] = cum_level(pp["g_pp"])

    def _emp_level_series(source: str) -> np.ndarray:
        out = np.full(T, np.nan, dtype=float)
        if national is None or "employment_level" not in national.columns:
            return out
        sub = (
            national.filter(pl.col("source") == source)
            .sort(pl.col("is_final").fill_null(False), "revision_number", descending=[True, True])
            .unique(subset=["period"], keep="first")
        )
        for row in sub.iter_rows(named=True):
            period = row["period"]
            level = row["employment_level"]
            if period in date_to_idx and level is not None and np.isfinite(level):
                out[date_to_idx[period]] = float(level)
        return out

    d["ces_sa_level"] = _emp_level_series("ces_sa")
    d["ces_nsa_level"] = _emp_level_series("ces_nsa")

    return pl.DataFrame(d)
