# scripts/run_tier1_diagnostics.py
"""Tier 1 diagnostics: Aruoba revision regression + Mincer-Zarnowitz + gate.

Usage:
    uv run python scripts/run_tier1_diagnostics.py data/backtests/a5

Reads the A5 results parquet (for the model's MZ) and the vintage store (for the
Aruoba LHS + design), writes tier1_diagnostics.md / .parquet, prints the gate.
Provider-ablation is forward-looking (Bloomberg-only) and self-skips locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from nfp_ingest.indicators import read_indicator
    from nfp_vintages.diagnostics import (
        GateConfig,
        aruoba_regression,
        build_aruoba_design,
        build_revision_table,
        gate_decision,
        mincer_zarnowitz,
    )

    rev_tbl = build_revision_table()
    ref = [r["ref_date"] for r in rev_tbl.iter_rows(named=True)]
    rev = np.array([r["revision_k"] for r in rev_tbl.iter_rows(named=True)], dtype=float)

    # ---- regressors (skeleton: claims momentum, JOLTS, lagged revision) ----
    def _monthly(name: str) -> dict:
        df = read_indicator(name)
        if df is None or df.is_empty():
            return {}
        m = (df.with_columns(pl.col("ref_date").dt.truncate("1mo").alias("m"))
             .group_by("m").agg(pl.col("value").mean().alias("v")).sort("m"))
        return {r["m"]: r["v"] for r in m.iter_rows(named=True)}

    claims = read_indicator("claims")
    claims_mom = {}
    if claims is not None and not claims.is_empty():
        mm = (claims.with_columns(pl.col("ref_date").dt.truncate("1mo").alias("m"))
              .group_by("m").agg(pl.col("value").mean().alias("v")).sort("m")
              .with_columns((pl.col("v") - pl.col("v").shift(3)).alias("mom3")))
        claims_mom = {r["m"]: r["mom3"] for r in mm.iter_rows(named=True)}
    lagged_rev = {ref[i]: rev[i - 1] for i in range(1, len(ref))}
    regressors = {"claims_mom": claims_mom, "jolts": _monthly("jolts"),
                  "lagged_revision": lagged_rev}

    X, names, used = build_aruoba_design(ref, regressors)
    pooled = aruoba_regression(rev, X, names)
    skeleton = sorted(set(regressors) - set(used))

    # ---- per-month-type Aruoba (reuse the Tier 0 classifier) ----
    from nfp_vintages.scoreboard import MonthTypeConfig, classify_month_types

    claims_arr = np.array([claims_mom.get(m, np.nan) / 1000.0 for m in ref], dtype=float)
    mtypes = classify_month_types(ref, np.abs(rev), claims_arr, MonthTypeConfig())
    r2_by_type: dict[str, float] = {}
    for mt in ["normal", "large_revision", "turning_point", "benchmark_window"]:
        idx = [i for i, m in enumerate(ref) if mtypes[m] == mt]
        if len(idx) > X.shape[1] + 2:  # enough dof
            r2_by_type[mt] = aruoba_regression(rev[idx], X[idx], names).r2

    # ---- Mincer-Zarnowitz on the model nowcast ----
    mz_lines = []
    results_path = root / "a5_results.parquet"
    if results_path.exists():
        df = pl.read_parquet(results_path)
        mrows = df.filter((pl.col("competitor") == "model")
                          & pl.col("error_k").is_not_null())
        if mrows.height > 5:
            actual = mrows["actual_first_print_k"].to_numpy()
            pred = mrows["pred_change_k"].to_numpy()
            mz = mincer_zarnowitz(actual, pred)
            mz_lines = [f"- model: alpha={mz.alpha:+.1f}k, beta={mz.beta:.3f}, "
                        f"joint p(alpha=0,beta=1)={mz.joint_p:.3f}, n={mz.n}"]
        # consensus MZ only if consensus predictions are present
        crows = df.filter((pl.col("competitor") == "consensus")
                          & pl.col("error_k").is_not_null())
        if crows.height > 5:
            mzc = mincer_zarnowitz(crows["actual_first_print_k"].to_numpy(),
                                   crows["pred_change_k"].to_numpy())
            mz_lines.append(f"- consensus: alpha={mzc.alpha:+.1f}k, beta={mzc.beta:.3f}, "
                            f"joint p={mzc.joint_p:.3f}, n={mzc.n}")
        else:
            mz_lines.append("- consensus: — (no consensus predictions present; "
                            "Bloomberg file not landed)")

    gate = gate_decision(r2_by_type, GateConfig())

    # ---- provider-ablation (forward-looking; self-skip locally) ----
    ablation_note = ("- provider-ablation: skipped (public-only venue; no provider "
                     "data). Forward-looking to the Bloomberg regime — spec section 4.")

    lines = ["# Tier 1 diagnostics", "",
             f"Venue: **{'full' if not skeleton else 'public-only (skeleton)'}**. "
             f"Regressors used: {used}. Missing (full-regime only): {skeleton}.", "",
             "## Aruoba revision regression (revision = alpha + gamma'.X)",
             f"- pooled: intercept (bias) = {pooled.intercept_k:+.1f}k, "
             f"R² (forecastable share) = {pooled.r2:.3f}, n={pooled.n}",
             "- R² by month type: " + ", ".join(f"{k}={v:.3f}" for k, v in r2_by_type.items()),
             "", "## Mincer-Zarnowitz efficiency", *mz_lines,
             "", "## Provider ablation", ablation_note,
             "", "## Gate decision", f"- {gate['rationale']}",
             f"- fund first-release-vintage rebuild (section 7): {gate['fund_first_release_rebuild']}",
             f"- fund turning-point birth/death (section 6): {gate['fund_tier3_bd']}", ""]
    (root / "tier1_diagnostics.md").write_text("\n".join(lines) + "\n")

    pl.DataFrame([{"month_type": k, "aruoba_r2": v} for k, v in r2_by_type.items()] +
                 [{"month_type": "pooled", "aruoba_r2": pooled.r2}]).write_parquet(
        root / "tier1_diagnostics.parquet")
    print((root / "tier1_diagnostics.md").read_text())
    print(f"[gate] {gate['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
