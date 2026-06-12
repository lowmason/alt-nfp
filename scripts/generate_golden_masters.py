"""A1 golden-master generator — run with the OLD repo's interpreter.

Usage::

    ~/Projects/alt_nfp/.venv/bin/python scripts/generate_golden_masters.py data/golden_staging

Builds the censored CES+QCEW panels (one per as-of date) and one fixture per
configured payroll provider using the **old** repo's code and local store,
then writes parquets + ``a1_manifest.json`` into the staging dir given as
argv[1]. Read-only with respect to the old repo. See
``plans/3-golden_masters.md``.
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

AS_OF_DATES = [
    date(2020, 5, 12),   # COVID era break: Apr-2020 collapse at rev-0
    date(2023, 7, 12),   # mid-sample control
    date(2024, 9, 12),   # QCEW Q1 max-revision rule (Q1-2024 pub 2024-08-21)
    date(2024, 12, 12),  # QCEW Q2 rule (Q2-2024 pub 2024-11-20)
    date(2025, 2, 12),   # January benchmark print (pub 2025-02-07)
    date(2025, 3, 12),   # QCEW Q3 rule (Q3-2024 pub 2025-02-19)
    date(2025, 7, 12),   # QCEW Q4 rule (Q4-2024 pub 2025-06-04)
    date(2025, 11, 12),  # stale-provider month (behavior gated in A2)
    date(2026, 1, 12),   # frontier + shutdown rev-fallback (doubled rev-1)
]
# Horizons that must REFUSE to build: the 2025 shutdown left Oct/Nov-2025
# supersector detail unpublished until the 2026-02-16 make-up print, so any
# as-of in [2025-12-12*, 2026-02-16) sees a ref-month gap and the fail-fast
# validator raises. (*2026-02-12 verified; the test pins this behavior.)
EXPECTED_FAILURE_DATES = [date(2026, 2, 12)]
START_YEAR = 2012
END_YEAR = 2026


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
    import polars as pl  # the old venv's polars
    from nfp_ingest.panel import build_panel
    from nfp_ingest.payroll import ingest_provider
    from nfp_models.config import PROVIDERS  # old repo only (model layer)

    # editable install: .../<repo>/packages/nfp-ingest/src/nfp_ingest/__init__.py
    old_repo = Path(nfp_ingest.__file__).resolve().parents[4]
    provenance = {
        "generator": "scripts/generate_golden_masters.py",
        "generated_on": date.today().isoformat(),
        "old_repo": str(old_repo),
        "old_repo_commit": subprocess.run(
            ["git", "-C", str(old_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "polars_version": pl.__version__,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
    }

    fixtures = {}

    for d in AS_OF_DATES:
        panel = build_panel(
            providers=[], start_year=START_YEAR, end_year=END_YEAR, as_of_ref=d
        )
        fname = f"panel_asof_{d.isoformat()}.parquet"
        panel.write_parquet(out_dir / fname)
        fixtures[fname] = {
            "kind": "censored_panel",
            "as_of_ref": d.isoformat(),
            "rows": panel.height,
            "columns": panel.columns,
            "sha256": _sha256(out_dir / fname),
        }
        print(f"{fname}: {panel.height:,} rows")

    for cfg in PROVIDERS:
        df = ingest_provider(cfg)
        fname = f"provider_{cfg.name}.parquet"
        df.write_parquet(out_dir / fname)
        fixtures[fname] = {
            "kind": "provider_panel",
            "provider_config": asdict(cfg),
            "rows": df.height,
            "columns": df.columns,
            "sha256": _sha256(out_dir / fname),
        }
        print(f"{fname}: {df.height:,} rows")

    # Negative masters: verify the old repo refuses these horizons, then pin.
    expected_failures = []
    for d in EXPECTED_FAILURE_DATES:
        try:
            build_panel(providers=[], start_year=START_YEAR, end_year=END_YEAR, as_of_ref=d)
            raise SystemExit(f"expected ValueError for as_of_ref={d}, but build succeeded")
        except ValueError as e:
            expected_failures.append({"as_of_ref": d.isoformat(), "error_contains": "ref_date gap"})
            print(f"expected failure confirmed for {d}: {str(e)[:80]}")

    manifest = {
        "provenance": provenance,
        "fixtures": fixtures,
        "expected_failures": expected_failures,
    }
    (out_dir / "a1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {len(fixtures)} fixtures + a1_manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
