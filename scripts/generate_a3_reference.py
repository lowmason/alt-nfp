"""A3 reference-posterior generator — run with the OLD repo's interpreter.

Usage::

    ~/Projects/alt_nfp/.venv/bin/python scripts/generate_a3_reference.py data/golden_a3_staging

Fits the frozen PyMC/nutpie reference model at each A3 date with the
CORRECTED indicators config (the A2 finding: default config silently drops
claims/jolts) and serializes the parity-comparable posterior content:

- ``ref_asof_<D>_<preset>.npz`` — full draws for every scalar/small-vector
  parameter, mean/SD paths for the latent deterministics, the CES-SA
  posterior-predictive mean path, and its full draws at the nowcast index.
- ``a3_manifest.json`` — dates, sampler settings, seeds, nowcast scalars,
  gating inventory, provenance.

Restartable: dates whose npz already exists are skipped. Read-only with
respect to the old repo. See ``plans/5-a3_model_parity.md``.
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

DEFAULT_DATES = [date(2023, 7, 12), date(2026, 1, 12)]
WINDOW_DATES = [
    date(2025, 2, 12), date(2025, 3, 12), date(2025, 4, 12), date(2025, 5, 12),
    date(2025, 6, 12), date(2025, 7, 12), date(2025, 8, 12), date(2025, 9, 12),
    date(2025, 10, 12), date(2025, 11, 12), date(2025, 12, 12), date(2026, 1, 12),
]
START_YEAR = 2012
END_YEAR = 2026
BASE_SEED = 20260612

PATH_VARS = ["bd", "g_cont", "g_total_sa", "g_total_nsa", "seasonal"]
GLOBAL_DRAW_VARS = [
    "tau", "phi_raw", "mu_g_era", "mu_g", "phi_0", "phi_3", "sigma_bd",
    "sigma_qcew_mid", "sigma_qcew_boundary", "sigma_fourier",
    "sigma_ces_sa", "sigma_ces_nsa", "alpha_ces", "lambda_ces",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import nfp_ingest
    import numpy as np
    import polars as pl
    from nfp_ingest.panel import build_panel
    from nfp_models.config import PROVIDERS
    from nfp_models.model import build_model
    from nfp_models.panel_adapter import panel_to_model_data
    from nfp_models.sampling import sample_model
    from nfp_models.settings import NowcastConfig, PathsConfig

    old_repo = Path(nfp_ingest.__file__).resolve().parents[4]

    # A2 finding: default config resolves indicators_dir against the model
    # package dir (no data/) and silently drops claims/jolts. The reference
    # posterior must include the phi_3 block, so route data_dir back to the
    # repo root through the old code's own relative-path mechanism.
    cfg = NowcastConfig(paths=PathsConfig(data_dir="../../data"))

    provenance = {
        "generator": "scripts/generate_a3_reference.py",
        "generated_on": date.today().isoformat(),
        "old_repo": str(old_repo),
        "old_repo_commit": subprocess.run(
            ["git", "-C", str(old_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "numpy_version": np.__version__,
        "polars_version": pl.__version__,
        "pymc_version": __import__("pymc").__version__,
        "nutpie_version": __import__("nutpie").__version__,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "base_seed": BASE_SEED,
        "providers": [p.name for p in PROVIDERS],
        "corrected_indicators_config": True,
    }

    # Uncensored panel once: base_index / idx_to_level exactly as backtest.py.
    panel_full = build_panel(
        providers=PROVIDERS, start_year=START_YEAR, end_year=END_YEAR
    )
    data_full = panel_to_model_data(panel_full, PROVIDERS, cfg=cfg)
    levels = data_full["levels"]
    ces_sa_index = levels["ces_sa_index"].to_numpy().astype(float)
    base_index = float(ces_sa_index[0])
    base_row_idx = int(np.argmin(np.abs(ces_sa_index - 100.0)))
    ces_sa_base_level = float(levels["ces_sa_level"].to_numpy().astype(float)[base_row_idx])
    idx_to_level = ces_sa_base_level / 100.0
    provenance["base_index"] = base_index
    provenance["idx_to_level"] = idx_to_level

    jobs = [(d, "default") for d in DEFAULT_DATES] + [(d, "light") for d in WINDOW_DATES]

    manifest_path = out_dir / "a3_manifest.json"
    fixtures: dict = {}
    if manifest_path.exists():
        fixtures = json.loads(manifest_path.read_text()).get("fixtures", {})

    for i, (d, preset) in enumerate(jobs):
        stem = f"asof_{d.isoformat()}_{preset}"
        npz_path = out_dir / f"ref_{stem}.npz"
        if npz_path.exists():
            print(f"[{i + 1}/{len(jobs)}] {stem}: exists, skipping", flush=True)
            continue
        print(f"[{i + 1}/{len(jobs)}] {stem}: building data...", flush=True)
        seed = BASE_SEED + i

        try:
            panel = build_panel(
                providers=PROVIDERS, start_year=START_YEAR, end_year=END_YEAR, as_of_ref=d
            )
            data = panel_to_model_data(panel, PROVIDERS, as_of=d, cfg=cfg)
        except Exception as e:  # noqa: BLE001 — record unbuildable dates, keep going
            print(f"  UNBUILDABLE: {e}", flush=True)
            fixtures[stem] = {"as_of": d.isoformat(), "preset": preset, "error": str(e)}
            continue

        # Same gating the model applies: present and not all-zero.
        cyclical_in_model = [
            f"{ind.name}_c" for ind in cfg.indicators
            if data.get(f"{ind.name}_c") is not None
            and np.any(data[f"{ind.name}_c"] != 0.0)
        ]

        model = build_model(data, cfg=cfg)
        kwargs = cfg.sampling.get_preset(preset).to_pymc_kwargs()
        kwargs["random_seed"] = seed
        print(f"  sampling ({preset}: {kwargs['draws']}d/{kwargs['tune']}t/"
              f"{kwargs['chains']}c, seed {seed})...", flush=True)
        t0 = time.time()
        idata = sample_model(model, sampler_kwargs=kwargs)
        wall = time.time() - t0

        post = idata.posterior
        arrays: dict = {}
        for var in GLOBAL_DRAW_VARS:
            if var in post:
                arrays[f"draws__{var}"] = post[var].values
        for pp in data["pp_data"]:
            name = pp["config"].name.lower()
            for var in (f"alpha_{name}", f"lam_{name}", f"sigma_pp_{name}", f"rho_{name}"):
                if var in post:
                    arrays[f"draws__{var}"] = post[var].values
        for var in PATH_VARS:
            vals = post[var].values  # (chains, draws, T)
            arrays[f"path_mean__{var}"] = vals.mean(axis=(0, 1))
            arrays[f"path_sd__{var}"] = vals.std(axis=(0, 1))

        # Nowcast extraction exactly as backtest.py: CES observation-equation
        # transform, posterior-mean growth path, last state as the proxy for
        # the target month (as_of itself is never in the censored calendar).
        g_sa_post = post["g_total_sa"].values
        alpha_post = post["alpha_ces"].values
        lambda_post = post["lambda_ces"].values
        g_ces_pred = alpha_post[:, :, None] + lambda_post[:, :, None] * g_sa_post
        g_sa_mean = np.nanmean(g_ces_pred, axis=(0, 1))

        censored_dates = data["dates"]
        c_idx = censored_dates.index(d) if d in censored_dates else len(censored_dates) - 1

        series = np.empty(len(g_sa_mean) + 1)
        series[0] = base_index
        for s in range(len(g_sa_mean)):
            series[s + 1] = series[s] * np.exp(g_sa_mean[s])
        nowcast_growth = float(g_sa_mean[c_idx])
        nowcast_change_k = float((series[c_idx + 1] - series[c_idx]) * idx_to_level)

        arrays["nowcast_pred_mean"] = g_sa_mean
        arrays["nowcast_pred_draws"] = g_ces_pred[:, :, c_idx]

        np.savez(npz_path, **arrays)

        try:
            n_div = int(idata.sample_stats["diverging"].values.sum())
        except Exception:  # noqa: BLE001
            n_div = -1

        fixtures[stem] = {
            "as_of": d.isoformat(),
            "preset": preset,
            "seed": seed,
            "sampler": kwargs,
            "T": int(data["T"]),
            "c_idx": int(c_idx),
            "dates_first": censored_dates[0].isoformat(),
            "dates_last": censored_dates[-1].isoformat(),
            "n_ces_vintages": int(data["n_ces_vintages"]),
            "cyclical_in_model": cyclical_in_model,
            "n_divergences": n_div,
            "wall_seconds": round(wall, 1),
            "nowcast_growth": nowcast_growth,
            "nowcast_change_k": nowcast_change_k,
            "array_names": sorted(arrays),
            "sha256_npz": _sha256(npz_path),
        }
        print(f"  done in {wall:.0f}s: T={data['T']}, c_idx={c_idx}, "
              f"divergences={n_div}, nowcast {nowcast_change_k:+,.0f}k", flush=True)

        manifest = {"provenance": provenance, "fixtures": fixtures}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        del idata

    manifest = {"provenance": provenance, "fixtures": fixtures}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {len(fixtures)} fixtures + a3_manifest.json to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
