"""A5 backtest — model vs competitors on the first print, at T−7 and T−1.

    uv run python scripts/run_a5_backtest.py snapshot data/backtests/a5
    uv run python scripts/run_a5_backtest.py batched  data/backtests/a5
    uv run python scripts/run_a5_backtest.py score    data/backtests/a5

Reuses the A4 batched harness verbatim (``fit_model_batch``); only the as-of
dates differ (release(M) − {7,1}). Snapshots live under ``<root>/<regime>/``.
"""
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

N_BACKTEST = int(os.environ.get("A5_N_BACKTEST", "24"))
END_YEAR = 2026
PRESET = "light"
BATCH_SEED = 9100
REGIMES = {"t7": 7, "t1": 1}  # name -> days_before release

# Months delayed/distorted by the 2025 government shutdown — flagged, not pooled
# (see memory ces-oct2025-shutdown; specs/model_improvements.md section 3).
SHUTDOWN_FLAGGED = frozenset({date(2025, 10, 1), date(2025, 9, 1)})


def _claims_momentum_k() -> dict[date, float]:
    """3-month change in monthly initial claims (thousands), keyed by month-start.

    Returns {} if the claims indicator is absent locally (skeleton venue)."""
    from nfp_ingest.indicators import read_indicator

    df = read_indicator("claims")
    if df is None or df.is_empty():
        return {}
    import polars as pl

    monthly = (
        df.with_columns(pl.col("ref_date").dt.truncate("1mo").alias("m"))
        .group_by("m").agg(pl.col("value").mean().alias("v"))
        .sort("m")
        .with_columns((pl.col("v") - pl.col("v").shift(3)).alias("mom3"))
    )
    return {r["m"]: (r["mom3"] / 1000.0 if r["mom3"] is not None else float("nan"))
            for r in monthly.iter_rows(named=True)}


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _write_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def cmd_snapshot(root: Path) -> None:
    from nfp_ingest.first_print import first_print_changes
    from nfp_ingest.model_data import (
        PROVIDERS_DEFAULT,
        levels_provenance,
        panel_to_model_data,
    )
    from nfp_ingest.panel import build_panel
    from nfp_ingest.snapshots import load_snapshot, snapshot_model_data
    from nfp_lookups.paths import VINTAGE_STORE_PATH
    from nfp_vintages.a5 import near_release_asof

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "grid_manifest.json"
    manifest: dict = _read_json(manifest_path) if manifest_path.exists() else {"regimes": {}}

    print("Building uncensored panel (truth side)...", flush=True)
    panel_full = build_panel(end_year=END_YEAR)
    data_full = panel_to_model_data(panel_full, list(PROVIDERS_DEFAULT))
    dates = data_full["dates"]
    levels = data_full["levels"]
    ces_sa_index = levels["ces_sa_index"].to_numpy().astype(float)
    base_index, idx_to_level = levels_provenance(levels)

    fp = first_print_changes()  # ref_date -> first_print_change_k
    fp_map = dict(
        fp.select(["ref_date", "first_print_change_k"]).iter_rows()
    )

    T = len(dates)
    target_indices = list(range(T - N_BACKTEST, T))
    manifest["provenance"] = {
        "base_index": base_index,
        "idx_to_level": idx_to_level,
        "end_year": END_YEAR,
        "preset": PRESET,
        "n_backtest": N_BACKTEST,
    }

    for rname, days_before in REGIMES.items():
        snap_dir = root / rname
        reg = manifest["regimes"].setdefault(rname, {"days_before": days_before, "targets": {}})
        for n, t_idx in enumerate(target_indices):
            target = dates[t_idx]
            key = target.isoformat()
            as_of = near_release_asof(
                target, days_before=days_before, store_path=VINTAGE_STORE_PATH
            )
            hits = sorted((snap_dir / f"asof={as_of.isoformat()}").glob("model_data_*.npz"))
            path = hits[0] if hits else None
            if path is None:
                print(f"[{rname} {n + 1}/{N_BACKTEST}] target {key} as_of {as_of}: building", flush=True)
                try:
                    path, _ = snapshot_model_data(as_of, out_root=snap_dir, end_year=END_YEAR)
                except Exception as e:  # noqa: BLE001 — A1 negative-master pattern
                    print(f"  UNBUILDABLE: {e}", flush=True)
                    reg["targets"][key] = {"error": str(e), "as_of": as_of.isoformat()}
                    _write_json(manifest_path, manifest)
                    continue
            _, meta = load_snapshot(path)
            cdates = [date.fromisoformat(d) for d in meta["dates"]]
            c_idx = cdates.index(target) if target in cdates else len(cdates) - 1
            actual_index = float(ces_sa_index[t_idx])
            prev_index = float(ces_sa_index[t_idx - 1])
            reg["targets"][key] = {
                "t_idx": t_idx,
                "as_of": as_of.isoformat(),
                "T": len(cdates),
                "c_idx": int(c_idx),
                "content_hash": meta["content_hash"],
                "snapshot": str(path.relative_to(root)),
                # first_print is a *monthly* series keyed to month-start (day=1);
                # ``target`` rides the model's daily axis (CES ref day, the 12th).
                # Bucket the lookup to the month so the monthly value joins.
                "first_print_change_k": fp_map.get(target.replace(day=1)),
                "best_avail_change_k": (actual_index - prev_index) * idx_to_level,
                # prev_index required by cmd_score calibration (change_draws_k).
                "prev_index": prev_index,
                # n_providers from the snapshot scalars (0 locally → public-only venue).
                "n_providers": meta["scalars"]["n_providers"],
            }
            _write_json(manifest_path, manifest)
    print(f"Grid built under {root}")


def cmd_batched(root: Path) -> None:
    from nfp_ingest.snapshots import load_snapshot
    from nfp_model import fit_model_batch, from_snapshot, model_inputs, pad_model_inputs

    manifest = _read_json(root / "grid_manifest.json")
    prov = manifest["provenance"]

    def _data(snap_rel: str):
        path = root / snap_rel
        arrays, meta = load_snapshot(path)
        return from_snapshot(arrays, meta)

    for rname in REGIMES:
        reg = manifest["regimes"][rname]
        targets = [(k, t) for k, t in sorted(reg["targets"].items()) if "error" not in t]
        if not targets:
            continue
        print(f"[{rname}] loading {len(targets)} snapshots...", flush=True)
        inputs = [model_inputs(_data(t["snapshot"])) for _, t in targets]
        c_idx = [int(t["c_idx"]) for _, t in targets]
        bi = pad_model_inputs(inputs, c_idx=c_idx)
        t0 = time.time()
        batch = fit_model_batch(
            bi,
            settings=PRESET,
            seed=BATCH_SEED,
            base_index=float(prov["base_index"]),
            idx_to_level=float(prov["idx_to_level"]),
        )
        print(f"[{rname}] batched fit {batch.wall_seconds / 60:.1f} min "
              f"({time.time() - t0:.0f}s wall)", flush=True)
        entries: dict = {}
        for i, (key, _t) in enumerate(targets):
            arrays, meta = batch.date_arrays(i)
            np.savez(root / f"{rname}_batched_{key}.npz", **arrays)
            entries[key] = meta
        _write_json(root / f"{rname}_batched_manifest.json",
                    {"entries": entries, "batch_wall_seconds": round(batch.wall_seconds, 1)})
    print("Batched fits complete.")


def cmd_score(root: Path) -> int:
    import numpy as np
    import polars as pl
    from nfp_ingest.first_print import first_print_changes
    from nfp_vintages.a5 import score
    from nfp_vintages.competitors.consensus import Consensus, load_consensus
    from nfp_vintages.competitors.naive import RandomWalk, TrailingMean
    from nfp_vintages.diagnostics import build_revision_table
    from nfp_vintages.scoreboard import (
        MonthTypeConfig,
        change_draws_k,
        classify_month_types,
        crps_sample,
        interval_coverage,
        venue_for,
    )

    manifest = _read_json(root / "grid_manifest.json")
    prov = manifest["provenance"]
    idx_to_level = float(prov["idx_to_level"])
    fp = first_print_changes()
    fp_hist = fp.select(["ref_date", "first_print_change_k", "vintage_date"])
    consensus = Consensus(load_consensus())  # None until Bloomberg file lands → "—"
    naive_rw, naive_mean = RandomWalk(fp_hist), TrailingMean(fp_hist, window=12)

    # Month-type inputs (skeleton-safe: empty maps degrade to "normal"/"benchmark").
    rev_tbl = build_revision_table()  # [ref_date, first_print_change_k, later_change_k, revision_k]
    rev_months = [r["ref_date"] for r in rev_tbl.iter_rows(named=True)]
    rev_abs = np.array([abs(r["revision_k"]) if r["revision_k"] is not None else np.nan
                        for r in rev_tbl.iter_rows(named=True)], dtype=float)
    mom = _claims_momentum_k()
    claims_arr = np.array([mom.get(m, np.nan) for m in rev_months], dtype=float)
    month_type = classify_month_types(rev_months, rev_abs, claims_arr, MonthTypeConfig())

    rows = []
    for rname, _days_before in REGIMES.items():
        reg = manifest["regimes"][rname]
        batched = _read_json(root / f"{rname}_batched_manifest.json")["entries"]
        for key, t in sorted(reg["targets"].items()):
            if "error" in t or key not in batched:
                continue
            ref = date.fromisoformat(key)
            as_of = date.fromisoformat(t["as_of"])
            actual = t["first_print_change_k"]
            if actual is None:
                continue
            model = batched[key]["nowcast_change_k"]
            # Predictive draws for calibration (model only) from the persisted npz.
            cov80 = cov90 = crps = None
            npz_path = root / f"{rname}_batched_{key}.npz"
            if npz_path.exists() and actual is not None:
                with np.load(npz_path) as z:
                    if "nowcast_pred_draws" in z:
                        prev_index = float(t["prev_index"])
                        cd = change_draws_k(
                            z["nowcast_pred_draws"],
                            prev_index=prev_index, idx_to_level=idx_to_level,
                        )
                        cov80 = interval_coverage(cd, actual, 0.80)
                        cov90 = interval_coverage(cd, actual, 0.90)
                        crps = crps_sample(cd, actual)
            providers_present = bool(t.get("n_providers", 0))
            # month_type keys are month-start (day=1) — ref is the day-12 model
            # date, so normalize before lookup (same day-12-vs-day-1 alignment the
            # harness already does for fp_map at run_a5_backtest.py:111).
            mtype = month_type.get(ref.replace(day=1), "normal")
            preds = {
                "model": model,
                "consensus": consensus.predict(ref, as_of=as_of),
                "naive_rw": naive_rw.predict(ref, as_of=as_of),
                "naive_mean": naive_mean.predict(ref, as_of=as_of),
            }
            for comp, pred in preds.items():
                rows.append({
                    "regime": rname,
                    "ref_month": ref,
                    "month_type": mtype,
                    "venue": venue_for(providers_present=providers_present),
                    "shutdown_flag": ref in SHUTDOWN_FLAGGED,
                    "competitor": comp,
                    "pred_change_k": pred,
                    "actual_first_print_k": actual,
                    "error_k": None if pred is None else actual - pred,
                    # calibration only meaningful for the model row
                    "coverage_80": cov80 if comp == "model" else None,
                    "coverage_90": cov90 if comp == "model" else None,
                    "crps_k": crps if comp == "model" else None,
                })

    if not rows:
        (root / "a5_report.md").write_text(
            "# A5 backtest report\n\nNo scoreable targets "
            "(all unbuildable or missing first prints).\n"
        )
        print("No scoreable targets — wrote empty a5_report.md")
        return 0

    df = pl.DataFrame(rows)
    scored = df.filter(
        pl.col("error_k").is_not_null()
        & ~pl.col("ref_month").dt.year().is_in([2020, 2021])
        & ~pl.col("shutdown_flag")
    )
    df.write_parquet(root / "a5_results.parquet")

    venues = sorted({v for v in df["venue"].unique() if v is not None})
    lines = ["# A5 backtest report", "",
             "Model vs competitors on the CES **first print**, at T−7 and T−1, "
             "decomposed by month type.",
             "Consensus is T−1-only and renders `—` until the Bloomberg file lands.",
             f"Venue(s) in this run: **{', '.join(venues) or 'public-only'}** — a "
             "`public-only` run scores a providerless skeleton (spec section 10).",
             "COVID (2020–2021) and shutdown-flagged months excluded from metrics.", ""]
    order = ["normal", "large_revision", "turning_point", "benchmark_window"]
    for rname in REGIMES:
        lines += [f"## Regime {rname}", ""]
        for mtype in order:
            sub = scored.filter((pl.col("regime") == rname) & (pl.col("month_type") == mtype))
            n_months = sub.select(pl.col("ref_month").n_unique()).item()
            lines += [f"### {mtype} ({n_months} months)", "",
                      "| competitor | n | ME | MAE | RMSE |", "|---|---|---|---|---|"]
            for comp in ["model", "consensus", "naive_rw", "naive_mean"]:
                e = sub.filter(pl.col("competitor") == comp)["error_k"].to_numpy()
                m = score(e)
                if m["n"] == 0:
                    lines.append(f"| {comp} | 0 | — | — | — |")
                else:
                    lines.append(
                        f"| {comp} | {m['n']} | {m['me']:+,.0f}k | {m['mae']:,.0f}k "
                        f"| {m['rmse']:,.0f}k |")
            # Model calibration row for this bucket.
            mc = sub.filter(pl.col("competitor") == "model")
            cov80 = mc["coverage_80"].drop_nulls().mean()
            cov90 = mc["coverage_90"].drop_nulls().mean()
            crps = mc["crps_k"].drop_nulls().mean()
            if cov80 is not None:
                lines += ["",
                          f"model calibration — 80% coverage: {cov80:.0%}, "
                          f"90% coverage: {cov90:.0%}, mean CRPS: {crps:,.0f}k", ""]
            else:
                lines.append("")
    (root / "a5_report.md").write_text("\n".join(lines) + "\n")
    print((root / "a5_report.md").read_text())

    # ---- Second scoreboard: model & ADP vs QCEW-settled truth ----
    from nfp_vintages.diagnostics import qcew_settled_changes
    try:
        qcew = {r["ref_date"]: r["qcew_settled_change_k"]
                for r in qcew_settled_changes().iter_rows(named=True)}
    except Exception as exc:  # store unavailable locally
        qcew = {}
        print(f"[qcew scoreboard] skipped: {exc}")
    if qcew:
        qlines = ["", "## Truth scoreboard (vs QCEW-settled change)", "",
                  "Fair target for QCEW-anchored competitors (model, ADP). "
                  "ADP renders `—` until Bloomberg data lands.",
                  "| regime | competitor | n | ME | MAE | RMSE |",
                  "|---|---|---|---|---|---|"]
        model_rows = df.filter(pl.col("competitor") == "model")
        for rname in REGIMES:
            sub = model_rows.filter(pl.col("regime") == rname)
            errs = []
            for r in sub.iter_rows(named=True):
                # qcew keys are month-start; ref_month rows are the day-12 model date.
                truth = qcew.get(r["ref_month"].replace(day=1))
                if truth is not None and r["pred_change_k"] is not None:
                    errs.append(truth - r["pred_change_k"])
            mm = score(np.array(errs, dtype=float))
            cell = (f"| {rname} | model | {mm['n']} | {mm['me']:+,.0f}k "
                    f"| {mm['mae']:,.0f}k | {mm['rmse']:,.0f}k |") if mm["n"] else \
                   f"| {rname} | model | 0 | — | — | — |"
            qlines.append(cell)
            qlines.append(f"| {rname} | adp | 0 | — | — | — |")  # Bloomberg-only
        with (root / "a5_report.md").open("a") as fh:
            fh.write("\n".join(qlines) + "\n")

    return 0


def main() -> None:
    mode, root_arg = sys.argv[1], sys.argv[2]
    root = Path(root_arg).resolve()
    {"snapshot": cmd_snapshot, "batched": cmd_batched}.get(mode, lambda r: None)(root)
    if mode == "score":
        raise SystemExit(cmd_score(root))
    elif mode not in ("snapshot", "batched"):
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
