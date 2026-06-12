"""A2 golden-master generator — run with the OLD repo's interpreter.

Usage::

    ~/Projects/alt_nfp/.venv/bin/python scripts/generate_a2_golden.py data/golden_a2_staging

For each A1 as-of date, runs the old repo's full two-layer pipeline
(``build_panel(as_of_ref=D)`` → ``panel_to_model_data(PROVIDERS, as_of=D)``
with default settings) and serializes the comparable content:

- ``model_data_asof_<D>.npz`` — every numeric array (global, cyclical, and
  per-provider with ``<name>__`` prefixes)
- ``levels_asof_<D>.parquet`` / ``panel_asof_<D>.parquet`` — the two frames
- ``a2_manifest.json`` — dates, scalars, vintage maps, provider/cyclical
  inventory, provenance

Read-only with respect to the old repo. See ``plans/4-a2_seams_snapshots.md``.
"""

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

AS_OF_DATES = [
    date(2020, 5, 12),
    date(2023, 7, 12),
    date(2024, 9, 12),
    date(2024, 12, 12),
    date(2025, 2, 12),
    date(2025, 3, 12),
    date(2025, 7, 12),
    date(2025, 11, 12),
    date(2026, 1, 12),
]
START_YEAR = 2012
END_YEAR = 2026

GLOBAL_ARRAYS = [
    "month_of_year", "year_of_obs", "era_idx",
    "g_ces_sa", "ces_sa_obs", "ces_sa_vintage_idx",
    "g_ces_nsa", "ces_nsa_obs", "ces_nsa_vintage_idx",
    "g_qcew", "qcew_obs", "qcew_is_m2", "qcew_noise_mult",
    "birth_rate", "bd_proxy", "bd_qcew_lagged",
]
SCALARS = ["T", "n_years", "n_ces_vintages", "n_providers"]


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
    from nfp_models.panel_adapter import panel_to_model_data
    from nfp_models.settings import NowcastConfig, PathsConfig

    old_repo = Path(nfp_ingest.__file__).resolve().parents[4]

    # The old settings refactor resolves indicators_dir relative to the model
    # *package* dir (resolve_paths(parents[2]) in panel_adapter), where no
    # data/ exists — so default-config runs silently drop cyclical indicators
    # (claims_c/jolts_c = None). Masters must pin the *intended* behavior:
    # point data_dir back at the repo root through the old code's own
    # relative-path mechanism.
    cfg = NowcastConfig(paths=PathsConfig(data_dir="../../data"))
    provenance = {
        "generator": "scripts/generate_a2_golden.py",
        "generated_on": date.today().isoformat(),
        "old_repo": str(old_repo),
        "old_repo_commit": subprocess.run(
            ["git", "-C", str(old_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "polars_version": pl.__version__,
        "numpy_version": np.__version__,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "providers": [p.name for p in PROVIDERS],
    }

    fixtures = {}
    for d in AS_OF_DATES:
        panel = build_panel(
            providers=PROVIDERS, start_year=START_YEAR, end_year=END_YEAR, as_of_ref=d
        )
        data = panel_to_model_data(panel, PROVIDERS, as_of=d, cfg=cfg)

        cyclical_keys = sorted(
            k for k in data if k.endswith("_c")
        )
        arrays: dict = {}
        for k in GLOBAL_ARRAYS:
            arrays[k] = np.asarray(data[k])
        for k in cyclical_keys:
            if data[k] is not None:
                arrays[k] = np.asarray(data[k])
        pp_meta = []
        for pp in data["pp_data"]:
            name = pp["name"]
            arrays[f"{name}__g_pp"] = np.asarray(pp["g_pp"])
            arrays[f"{name}__pp_obs"] = np.asarray(pp["pp_obs"])
            has_births = pp["births"] is not None
            if has_births:
                arrays[f"{name}__births"] = np.asarray(pp["births"])
                arrays[f"{name}__births_obs"] = np.asarray(pp["births_obs"])
            pp_meta.append({"name": name, "emp_col": pp["emp_col"], "has_births": has_births})

        stem = f"asof_{d.isoformat()}"
        npz_path = out_dir / f"model_data_{stem}.npz"
        np.savez(npz_path, **arrays)
        data["levels"].write_parquet(out_dir / f"levels_{stem}.parquet")
        data["panel"].write_parquet(out_dir / f"panel_{stem}.parquet")

        fixtures[stem] = {
            "as_of_ref": d.isoformat(),
            "scalars": {k: int(data[k]) for k in SCALARS},
            "dates_first": data["dates"][0].isoformat(),
            "dates_last": data["dates"][-1].isoformat(),
            "ces_vintage_map": {str(k): v for k, v in data["ces_vintage_map"].items()},
            "cyclical_present": [k for k in cyclical_keys if data[k] is not None],
            "cyclical_none": [k for k in cyclical_keys if data[k] is None],
            "providers": pp_meta,
            "array_names": sorted(arrays),
            "panel_rows": data["panel"].height,
            "sha256_npz": _sha256(npz_path),
            "sha256_levels": _sha256(out_dir / f"levels_{stem}.parquet"),
            "sha256_panel": _sha256(out_dir / f"panel_{stem}.parquet"),
        }
        print(f"{stem}: T={data['T']}, {len(arrays)} arrays, panel {data['panel'].height:,} rows")

    manifest = {"provenance": provenance, "fixtures": fixtures}
    (out_dir / "a2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {len(fixtures)} fixture sets + a2_manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
