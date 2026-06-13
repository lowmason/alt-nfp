"""A3 parity criteria: compare a posterior against the frozen reference.

Implements the gate from the plan of record: |mean difference| small
relative to pooled posterior SD **and** MCSE (z-test with ESS-derived
MCSE on both sides), SD ratios within band, latent paths within a
fraction of posterior SD pointwise, and nowcast distributions matched.

Both sides are compared in a reduced *fixture schema* (what
``scripts/generate_a3_reference.py`` saves): full draws for scalar/small-
vector sites, mean/SD summaries for the (T,)-length latent paths, and the
nowcast predictive draws at the target index. ``collect_parity_arrays``
produces that schema from a fresh :class:`FitResult`, so comparisons can
run in-process (the pytest spot check) or against persisted npz files
(``scripts/run_a3_parity.py``'s two-phase fit/compare).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpyro.diagnostics import effective_sample_size

from .nowcast import nowcast_summary
from .sampling import FitResult

#: criteria (documented in plans/5-a3_model_parity.md)
SD_FRAC = 0.15          # |Δmean| ≤ this × pooled posterior SD passes outright
Z_MAX = 4.0             # … or the MCSE z-score is within this
SD_RATIO_BAND = (0.80, 1.25)
PATH_FRAC = 0.25        # max_t |Δmean_t| / SD_t
PATH_SD_BAND = (0.70, 1.40)
TINY = 1e-12

PATH_VARS = ("bd", "g_cont", "g_total_sa", "g_total_nsa", "seasonal")

#: scalar/small-vector sites collected from a fit (when present)
SCALAR_VARS = (
    "tau", "phi_raw", "mu_g_era", "mu_g", "phi_0", "phi_3", "sigma_bd",
    "sigma_qcew_mid", "sigma_qcew_boundary", "sigma_fourier",
    "sigma_ces_sa", "sigma_ces_nsa", "alpha_ces", "lambda_ces",
)


@dataclass
class ParityRow:
    name: str
    passed: bool
    detail: str
    kind: str = "scalar"


@dataclass
class ParityReport:
    stem: str
    rows: list[ParityRow] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.rows)

    @property
    def n_failed(self) -> int:
        return sum(not r.passed for r in self.rows)

    def summary(self, *, failures_only: bool = False) -> str:
        status = "PASS" if self.passed else f"FAIL ({self.n_failed}/{len(self.rows)})"
        lines = [f"{self.stem}: {status}"]
        for r in self.rows:
            if failures_only and r.passed:
                continue
            mark = "ok  " if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.name}: {r.detail}")
        return "\n".join(lines)


def collect_parity_arrays(
    fit: FitResult, *, base_index: float, idx_to_level: float, c_idx: int
) -> tuple[dict[str, np.ndarray], dict]:
    """Reduce a fit to the fixture schema (arrays + meta) for comparison."""
    arrays: dict[str, np.ndarray] = {}
    post = fit.posterior
    for var in SCALAR_VARS:
        if var in post:
            arrays[f"draws__{var}"] = post[var]
    for name in post:
        if name.startswith(("alpha_", "lam_", "sigma_pp_", "rho_")) and name not in (
            "alpha_ces",
        ):
            arrays[f"draws__{name}"] = post[name]
    for var in PATH_VARS:
        arrays[f"path_mean__{var}"] = post[var].mean(axis=(0, 1))
        arrays[f"path_sd__{var}"] = post[var].std(axis=(0, 1))

    nc = nowcast_summary(
        post, base_index=base_index, idx_to_level=idx_to_level, c_idx=c_idx
    )
    arrays["nowcast_pred_mean"] = nc["pred_mean"]
    arrays["nowcast_pred_draws"] = nc["pred_draws"]

    phi3 = post.get("phi_3")
    meta = {
        "T": int(post["g_total_sa"].shape[-1]),
        "c_idx": int(nc["c_idx"]),
        "n_cyclical": 0 if phi3 is None else int(phi3.shape[-1]),
        "nowcast_growth": nc["nowcast_growth"],
        "nowcast_change_k": nc["nowcast_change_k"],
        "nowcast_index": nc["nowcast_index"],
        "num_divergences": fit.num_divergences,
        "seed": fit.seed,
        "wall_seconds": round(fit.wall_seconds, 1),
    }
    return arrays, meta


def _ess(draws: np.ndarray) -> float:
    ess = float(np.asarray(effective_sample_size(draws)))
    if not np.isfinite(ess) or ess <= 1.0:
        ess = draws.size / 10.0  # conservative fallback
    return ess


def _mcse(draws: np.ndarray) -> float:
    """Monte Carlo SE of the mean for one scalar site, (chains, draws)."""
    sd = float(draws.std())
    if sd < TINY:
        return 0.0
    return sd / np.sqrt(_ess(draws))


def _log_sd_se(draws: np.ndarray) -> float:
    """Monte Carlo SE of log(SD̂): Var(log ŝ) ≈ (κ−1)/(4·ESS).

    κ is the draws' kurtosis — for a normal posterior this reduces to the
    familiar 1/(2·ESS); heavy-tailed scale posteriors (e.g. the reference's
    centered-GRW sigma_fourier components at ESS ≈ 175–290) have κ well
    above 3, and a fixed SD-ratio band is miscalibrated there.
    """
    x = np.asarray(draws, dtype=float).ravel()
    sd = x.std()
    if sd < TINY:
        return 0.0
    z = (x - x.mean()) / sd
    kurt = float(np.clip(np.mean(z**4), 1.0 + 1e-6, 50.0))
    return float(np.sqrt((kurt - 1.0) / (4.0 * _ess(draws))))


def compare_scalar(name: str, ref: np.ndarray, new: np.ndarray) -> list[ParityRow]:
    """Compare draws for one (possibly vector) site, component-wise."""
    ref = np.asarray(ref)
    new = np.asarray(new)
    if ref.shape[2:] != new.shape[2:]:
        return [
            ParityRow(
                name, False, f"shape mismatch: ref {ref.shape} vs new {new.shape}"
            )
        ]
    if ref.ndim == 2:
        ref = ref[..., None]
        new = new[..., None]
        labels = [name]
    else:
        labels = [f"{name}[{i}]" for i in range(ref.shape[-1])]

    rows = []
    for i, label in enumerate(labels):
        r, n = ref[..., i], new[..., i]
        d = abs(float(n.mean()) - float(r.mean()))
        pooled_sd = float(np.sqrt((r.std() ** 2 + n.std() ** 2) / 2))
        pooled_mcse = float(np.sqrt(_mcse(r) ** 2 + _mcse(n) ** 2))
        if pooled_sd < TINY:
            ok = d < 1e-9
            rows.append(ParityRow(label, ok, f"degenerate site, |Δ|={d:.2e}"))
            continue
        frac = d / pooled_sd
        z = d / pooled_mcse if pooled_mcse > 0 else (0.0 if d == 0 else np.inf)
        ratio = float(n.std() / max(r.std(), TINY))
        mean_ok = frac <= SD_FRAC or z <= Z_MAX
        # SD criterion: fixed band, or within MC error of the SD estimates
        # themselves (kurtosis-aware — see _log_sd_se).
        sd_se = float(np.sqrt(_log_sd_se(r) ** 2 + _log_sd_se(n) ** 2))
        sd_z = abs(np.log(max(ratio, TINY))) / sd_se if sd_se > 0 else 0.0
        sd_ok = SD_RATIO_BAND[0] <= ratio <= SD_RATIO_BAND[1] or sd_z <= Z_MAX
        rows.append(
            ParityRow(
                label,
                mean_ok and sd_ok,
                f"Δ={d:.2e} ({frac:.2f}·SD, z={z:.1f}), "
                f"SD ratio {ratio:.3f} (z={sd_z:.1f})",
            )
        )
    return rows


def compare_path(
    name: str,
    ref_mean: np.ndarray,
    ref_sd: np.ndarray,
    new_mean: np.ndarray,
    new_sd: np.ndarray,
) -> list[ParityRow]:
    """Pointwise path comparison on (T,) mean/SD summaries."""
    if ref_mean.shape != new_mean.shape:
        return [
            ParityRow(
                f"path:{name}", False,
                f"length mismatch: ref {ref_mean.shape} vs new {new_mean.shape}",
                kind="path",
            )
        ]
    sd_floor = np.maximum(ref_sd, TINY)
    frac = np.abs(new_mean - ref_mean) / sd_floor
    worst = int(np.argmax(frac))
    ratio = new_sd / sd_floor
    mean_ok = bool(frac.max() <= PATH_FRAC)
    sd_ok = bool((ratio >= PATH_SD_BAND[0]).all() and (ratio <= PATH_SD_BAND[1]).all())
    return [
        ParityRow(
            f"path:{name}",
            mean_ok and sd_ok,
            f"max|Δ|/SD={frac.max():.3f} @t={worst}, "
            f"SD ratio [{ratio.min():.2f}, {ratio.max():.2f}]",
            kind="path",
        )
    ]


def compare_reduced(
    ref_arrays: dict,
    fx: dict,
    new_arrays: dict,
    new_meta: dict,
    provenance: dict,
) -> ParityReport:
    """Full A3 comparison of one fixture: reduced schema on both sides.

    *fx* is the reference manifest entry; *provenance* carries
    ``base_index`` / ``idx_to_level`` shared by every date.
    """
    stem = f"asof_{fx['as_of']}_{fx['preset']}"
    report = ParityReport(stem=stem)

    n_cyc_ref = len(fx["cyclical_in_model"])
    report.rows.append(
        ParityRow(
            "structure:cyclical",
            new_meta["n_cyclical"] == n_cyc_ref,
            f"phi_3 dim {new_meta['n_cyclical']} "
            f"(ref {n_cyc_ref}: {fx['cyclical_in_model']})",
            kind="structure",
        )
    )
    report.rows.append(
        ParityRow(
            "structure:T",
            new_meta["T"] == int(fx["T"]) and new_meta["c_idx"] == int(fx["c_idx"]),
            f"T={new_meta['T']} (ref {fx['T']}), "
            f"c_idx={new_meta['c_idx']} (ref {fx['c_idx']})",
            kind="structure",
        )
    )

    for key in sorted(ref_arrays):
        if not key.startswith("draws__"):
            continue
        var = key.removeprefix("draws__")
        if f"draws__{var}" not in new_arrays:
            report.rows.append(
                ParityRow(var, False, "missing from new posterior", kind="scalar")
            )
            continue
        report.rows += compare_scalar(var, ref_arrays[key], new_arrays[key])

    for var in PATH_VARS:
        mk, sk = f"path_mean__{var}", f"path_sd__{var}"
        if mk in ref_arrays and mk in new_arrays:
            report.rows += compare_path(
                var, ref_arrays[mk], ref_arrays[sk], new_arrays[mk], new_arrays[sk]
            )

    # Nowcast distribution at c_idx + the point nowcast in jobs-added space.
    report.rows += [
        ParityRow(f"nowcast:{r.name}", r.passed, r.detail, kind="nowcast")
        for r in compare_scalar(
            "pred_draws",
            ref_arrays["nowcast_pred_draws"],
            new_arrays["nowcast_pred_draws"],
        )
    ]
    ref_change = float(fx["nowcast_change_k"])
    new_change = float(new_meta["nowcast_change_k"])
    level = float(new_meta["nowcast_index"]) * float(provenance["idx_to_level"])
    pooled_mcse = float(
        np.sqrt(
            _mcse(np.asarray(ref_arrays["nowcast_pred_draws"])) ** 2
            + _mcse(np.asarray(new_arrays["nowcast_pred_draws"])) ** 2
        )
    )
    bound = max(Z_MAX * level * pooled_mcse, 1.0)
    d = abs(new_change - ref_change)
    report.rows.append(
        ParityRow(
            "nowcast:change_k",
            d <= bound,
            f"ref {ref_change:+,.0f}k vs new {new_change:+,.0f}k "
            f"(|Δ|={d:.1f}k, bound {bound:.1f}k)",
            kind="nowcast",
        )
    )
    return report


def compare_fixture(
    ref_arrays: dict, fx: dict, fit: FitResult, provenance: dict
) -> ParityReport:
    """In-process comparison of one fixture against a fresh fit."""
    new_arrays, new_meta = collect_parity_arrays(
        fit,
        base_index=float(provenance["base_index"]),
        idx_to_level=float(provenance["idx_to_level"]),
        c_idx=int(fx["c_idx"]),
    )
    return compare_reduced(ref_arrays, fx, new_arrays, new_meta, provenance)
