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
    import polars as pl
    from nfp_ingest.first_print import first_print_changes
    from nfp_vintages.a5 import score
    from nfp_vintages.competitors.consensus import Consensus, load_consensus
    from nfp_vintages.competitors.naive import RandomWalk, TrailingMean

    manifest = _read_json(root / "grid_manifest.json")
    fp = first_print_changes()
    fp_hist = fp.select(["ref_date", "first_print_change_k", "vintage_date"])
    consensus = Consensus(load_consensus())  # None until Bloomberg file lands → "—"
    naive_rw, naive_mean = RandomWalk(fp_hist), TrailingMean(fp_hist, window=12)

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
                    "competitor": comp,
                    "pred_change_k": pred,
                    "actual_first_print_k": actual,
                    "error_k": None if pred is None else actual - pred,
                })

    if not rows:
        (root / "a5_report.md").write_text(
            "# A5 backtest report\n\nNo scoreable targets "
            "(all unbuildable or missing first prints).\n"
        )
        print("No scoreable targets — wrote empty a5_report.md")
        return 0

    df = pl.DataFrame(rows)
    # Exclude COVID (2020–2021) from headline metrics (decided-questions rule)
    scored = df.filter(
        pl.col("error_k").is_not_null()
        & ~pl.col("ref_month").dt.year().is_in([2020, 2021])
    )
    df.write_parquet(root / "a5_results.parquet")

    lines = ["# A5 backtest report", "",
             "Model vs competitors on the CES **first print**, at T−7 and T−1.",
             "Consensus is T−1-only and renders `—` until the Bloomberg file lands.",
             "COVID (2020–2021) excluded from metrics.", ""]
    for rname in REGIMES:
        lines += [f"## Regime {rname}", "", "| competitor | n | ME | MAE | RMSE |",
                  "|---|---|---|---|---|"]
        for comp in ["model", "consensus", "naive_rw", "naive_mean"]:
            e = scored.filter(
                (pl.col("regime") == rname) & (pl.col("competitor") == comp)
            )["error_k"].to_numpy()
            m = score(e)
            if m["n"] == 0:
                lines.append(f"| {comp} | 0 | — | — | — |")
            else:
                lines.append(
                    f"| {comp} | {m['n']} | {m['me']:+,.0f}k | {m['mae']:,.0f}k "
                    f"| {m['rmse']:,.0f}k |"
                )
        lines.append("")
    (root / "a5_report.md").write_text("\n".join(lines) + "\n")
    print((root / "a5_report.md").read_text())
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
