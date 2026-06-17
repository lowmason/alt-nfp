"""Re-baseline A1/A2 goldens from THIS repo + the rebuilt store (plans/12 / plans/10 T7).

Usage (controller, with .env creds; reads the SCRATCH store, writes LOCAL staging)::

    NFP_STORE_URI=s3://alt-nfp/store-rebuild uv run python scripts/regen_golden_rebuild.py data/golden_rebuild_staging

START_YEAR is 2017 (rebuilt store history starts there; frozen-ref was 2012+).
This value is written into a1_manifest.json/a2_manifest.json provenance and is read
back by the tests via MANIFEST["provenance"]["start_year"], so it is load-bearing.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

# --- .env BEFORE any nfp_* import (paths reads env at import time) ---
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

os.environ.setdefault("NFP_STORE_URI", "s3://alt-nfp/store-rebuild")  # default to scratch

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
EXPECTED_FAILURE = (date(2026, 2, 12), "ref_date gap")
START_YEAR, END_YEAR = 2017, 2026
GLOBAL_ARRAYS = [
    "month_of_year",
    "year_of_obs",
    "era_idx",
    "g_ces_sa",
    "ces_sa_obs",
    "ces_sa_vintage_idx",
    "g_ces_nsa",
    "ces_nsa_obs",
    "ces_nsa_vintage_idx",
    "g_qcew",
    "qcew_obs",
    "qcew_is_m2",
    "qcew_noise_mult",
]
SCALARS = ["T", "n_years", "n_ces_vintages", "n_providers"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import polars as pl
    from nfp_ingest.model_data import build_model_data
    from nfp_ingest.panel import build_panel
    from nfp_ingest.payroll import ingest_provider
    from nfp_lookups.paths import VINTAGE_STORE_PATH
    from nfp_lookups.provider_config import PROVIDERS_DEFAULT

    prov = {
        "generator": "scripts/regen_golden_rebuild.py",
        "generated_on": date.today().isoformat(),
        "store_uri": str(VINTAGE_STORE_PATH),
        "repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "polars_version": pl.__version__,
        "numpy_version": np.__version__,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "providers": [p.name for p in PROVIDERS_DEFAULT],
        "divergence": (
            "rebuilt store (s3://alt-nfp/store-rebuild): 2017+ history (vs frozen-ref 2012+), "
            "QCEW reconstructed crosswalk values, ownership/00-anchor/NSA store schema normalized "
            "to PANEL_SCHEMA by transform_to_panel (panel columns unchanged; values + row counts differ)."
        ),
    }

    # ---- A1: censored panels + provider + expected-failure ----
    a1: dict = {}
    for d in AS_OF_DATES:
        panel = build_panel(providers=[], start_year=START_YEAR, end_year=END_YEAR, as_of_ref=d)
        fn = f"panel_asof_{d.isoformat()}.parquet"
        panel.write_parquet(out / fn)
        a1[fn] = {
            "kind": "censored_panel",
            "as_of_ref": d.isoformat(),
            "rows": panel.height,
            "columns": panel.columns,
            "sha256": _sha256(out / fn),
        }

    for cfg in PROVIDERS_DEFAULT:
        df = ingest_provider(cfg)
        fn = f"provider_{cfg.name}.parquet"
        df.write_parquet(out / fn)
        a1[fn] = {
            "kind": "provider_panel",
            "rows": df.height,
            "columns": df.columns,
            "sha256": _sha256(out / fn),
            "provider_config": dataclasses.asdict(cfg),
        }

    ef_date, ef_msg = EXPECTED_FAILURE
    try:
        build_panel(providers=[], start_year=START_YEAR, end_year=END_YEAR, as_of_ref=ef_date)
        raise SystemExit(
            f"FAIL: expected ValueError({ef_msg!r}) at {ef_date}, but build_panel succeeded"
        )
    except ValueError as e:
        assert ef_msg in str(e), f"expected-failure message changed: {e!r}"

    a1_manifest = {
        "provenance": prov,
        "fixtures": a1,
        "expected_failures": [{"as_of_ref": ef_date.isoformat(), "error_contains": ef_msg}],
    }
    (out / "a1_manifest.json").write_text(json.dumps(a1_manifest, indent=2) + "\n")

    # ---- A2: build_model_data arrays + levels/panel frames ----
    a2: dict = {}
    for d in AS_OF_DATES:
        data = build_model_data(
            d, providers=list(PROVIDERS_DEFAULT), start_year=START_YEAR, end_year=END_YEAR
        )
        cyc = sorted(k for k in data if k.endswith("_c"))
        arrays = {k: np.asarray(data[k]) for k in GLOBAL_ARRAYS}
        for k in cyc:
            if data[k] is not None:
                arrays[k] = np.asarray(data[k])
        pp_meta = []
        for pp in data["pp_data"]:
            n = pp["name"]
            arrays[f"{n}__g_pp"] = np.asarray(pp["g_pp"])
            arrays[f"{n}__pp_obs"] = np.asarray(pp["pp_obs"])
            hb = pp["births"] is not None
            if hb:
                arrays[f"{n}__births"] = np.asarray(pp["births"])
                arrays[f"{n}__births_obs"] = np.asarray(pp["births_obs"])
            pp_meta.append({"name": n, "emp_col": pp["emp_col"], "has_births": hb})
        stem = f"asof_{d.isoformat()}"
        npz = out / f"model_data_{stem}.npz"
        np.savez(npz, **arrays)
        data["levels"].write_parquet(out / f"levels_{stem}.parquet")
        data["panel"].write_parquet(out / f"panel_{stem}.parquet")
        a2[stem] = {
            "as_of_ref": d.isoformat(),
            "scalars": {k: int(data[k]) for k in SCALARS},
            "dates_first": data["dates"][0].isoformat(),
            "dates_last": data["dates"][-1].isoformat(),
            "ces_vintage_map": {str(k): v for k, v in data["ces_vintage_map"].items()},
            "cyclical_present": [k for k in cyc if data[k] is not None],
            "cyclical_none": [k for k in cyc if data[k] is None],
            "providers": pp_meta,
            "array_names": sorted(arrays),
            "panel_rows": data["panel"].height,
            "sha256_npz": _sha256(npz),
            "sha256_levels": _sha256(out / f"levels_{stem}.parquet"),
            "sha256_panel": _sha256(out / f"panel_{stem}.parquet"),
        }
        print(f"{stem}: T={data['T']}, {len(arrays)} arrays, panel {data['panel'].height:,} rows")

    (out / "a2_manifest.json").write_text(
        json.dumps({"provenance": prov, "fixtures": a2}, indent=2) + "\n"
    )
    print(f"\nWrote A1 ({len(a1)}) + A2 ({len(a2)}) fixtures + manifests to {out}")


if __name__ == "__main__":
    main()
