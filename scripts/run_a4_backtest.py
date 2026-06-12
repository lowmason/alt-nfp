"""A4 vmapped backtest — snapshot grid, serial baseline, batched run, compare.

Four phases (all restartable; default root ``data/backtests``):

    uv run python scripts/run_a4_backtest.py snapshot data/backtests
    uv run python scripts/run_a4_backtest.py serial   data/backtests
    uv run python scripts/run_a4_backtest.py batched  data/backtests
    uv run python scripts/run_a4_backtest.py compare  data/backtests

``snapshot`` builds the as-of grid (last ``N_BACKTEST`` months of the
uncensored panel, day-12 convention — the reference backtest's window) as
hash-pinned ModelData snapshots under ``<root>/snapshots/``, plus
dual-track actuals (first print and best-available revision, with
big-disagreement rows flagged) and nowcast provenance into
``grid_manifest.json``.

``serial`` is the baseline the gate compares against: one A3-proven
``fit_model`` per date (light preset), reduced to the fixture schema.
``batched`` pads the whole grid and fits it in a single vmapped NUTS
program. ``compare`` applies the A3 parity criteria per date (serial as
reference, batched as candidate), assembles the backtest results table
(actual vs nowcast, MAE/RMSE for both runs), and writes
``a4_report.md`` + ``a4_results.parquet``. Exit 1 on any parity failure.

Snapshots live locally under the run root (not the canonical
``NFP_SNAPSHOTS_URI``) so the experiment is self-contained.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

N_BACKTEST = 24
END_YEAR = 2026
PRESET = "light"
SERIAL_SEED0 = 7000
BATCH_SEED = 9000
SPLICE_K = 150.0  # |best-available − first-print| change_k flagging threshold


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _snapshot_path(snap_dir: Path, as_of: str) -> Path | None:
    hits = sorted((snap_dir / f"asof={as_of}").glob("model_data_*.npz"))
    return hits[0] if hits else None


def _snapshot_data(snap_dir: Path, as_of: str) -> dict:
    from nfp_ingest.snapshots import load_snapshot

    path = _snapshot_path(snap_dir, as_of)
    if path is None:
        raise FileNotFoundError(f"no snapshot for {as_of} under {snap_dir}")
    arrays, meta = load_snapshot(path)
    from nfp_model import from_snapshot

    return from_snapshot(arrays, meta)


def _targets(manifest: dict) -> list[tuple[str, dict]]:
    return [
        (d, t) for d, t in sorted(manifest["targets"].items()) if "error" not in t
    ]


# =========================================================================
# snapshot
# =========================================================================


def cmd_snapshot(root: Path) -> None:
    import polars as pl
    from nfp_ingest.model_data import PROVIDERS_DEFAULT, panel_to_model_data
    from nfp_ingest.panel import build_panel
    from nfp_ingest.snapshots import load_snapshot, snapshot_model_data

    snap_dir = root / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "grid_manifest.json"
    manifest: dict = (
        _read_json(manifest_path)
        if manifest_path.exists()
        else {"targets": {}}
    )

    print("Building uncensored panel (truth side)...", flush=True)
    panel_full = build_panel(end_year=END_YEAR)
    data_full = panel_to_model_data(panel_full, list(PROVIDERS_DEFAULT))

    # First-print (rev-0) growth per period — the dual-track actual. The two
    # conventions disagree materially where revisions are large (the 2025-08
    # May/June revision, annual benchmarks) and where store growth rows cross
    # vintage bases (rev-0 growth differences against the *prior* vintage's
    # previous-month level, so a same-day revision shifts it off the headline;
    # 2026-02 benchmark vintages leak the level shift into best-available
    # growth). Score against both, flag big disagreements.
    fp_map = dict(
        panel_full.filter(
            (pl.col("source") == "ces_sa")
            & (pl.col("geographic_type") == "national")
            & (pl.col("industry_code") == "00")
            & (pl.col("revision_number") == 0)
        )
        .select(["period", "growth"])
        .iter_rows()
    )

    dates = data_full["dates"]
    g_ces_sa_actual = np.asarray(data_full["g_ces_sa"], dtype=float)
    levels = data_full["levels"]
    ces_sa_index = levels["ces_sa_index"].to_numpy().astype(float)
    base_index = float(ces_sa_index[0])
    base_row_idx = int(np.argmin(np.abs(ces_sa_index - 100.0)))
    ces_sa_base_level = float(levels["ces_sa_level"].to_numpy().astype(float)[base_row_idx])
    idx_to_level = ces_sa_base_level / 100.0

    T = len(dates)
    target_indices = list(range(T - N_BACKTEST, T))
    print(
        f"Panel: T={T}, {dates[0]} … {dates[-1]}; window = last {N_BACKTEST} "
        f"targets {dates[target_indices[0]]} … {dates[target_indices[-1]]}",
        flush=True,
    )

    manifest["provenance"] = {
        "base_index": base_index,
        "idx_to_level": idx_to_level,
        "end_year": END_YEAR,
        "preset": PRESET,
        "n_backtest": N_BACKTEST,
        "panel_T": T,
        "panel_last": dates[-1].isoformat(),
    }

    for n, t_idx in enumerate(target_indices):
        target = dates[t_idx]
        key = target.isoformat()
        path = _snapshot_path(snap_dir, key)
        if path is None:
            print(f"[{n + 1}/{N_BACKTEST}] {key}: building snapshot...", flush=True)
            try:
                path, _ = snapshot_model_data(target, out_root=snap_dir, end_year=END_YEAR)
            except Exception as e:  # noqa: BLE001 — record, move on (A1 negative-master pattern)
                print(f"  UNBUILDABLE: {e}", flush=True)
                manifest["targets"][key] = {"error": str(e)}
                _write_json(manifest_path, manifest)
                continue
        else:
            print(f"[{n + 1}/{N_BACKTEST}] {key}: snapshot exists", flush=True)
        _, meta = load_snapshot(path)
        digest = meta["content_hash"]
        censored_dates = [date.fromisoformat(d) for d in meta["dates"]]
        c_idx = (
            censored_dates.index(target)
            if target in censored_dates
            else len(censored_dates) - 1
        )

        actual_index = float(ces_sa_index[t_idx])
        prev_index = float(ces_sa_index[t_idx - 1])
        actual_change_k = (actual_index - prev_index) * idx_to_level
        fp_growth = fp_map.get(target)
        fp_change_k = (
            None if fp_growth is None
            else float(np.expm1(fp_growth) * prev_index * idx_to_level)
        )
        manifest["targets"][key] = {
            "t_idx": t_idx,
            "T": len(censored_dates),
            "c_idx": int(c_idx),
            "content_hash": digest,
            "snapshot": str(path.relative_to(root)),
            "actual_growth": float(g_ces_sa_actual[t_idx]),
            "actual_change_k": actual_change_k,
            "first_print_growth": None if fp_growth is None else float(fp_growth),
            "first_print_change_k": fp_change_k,
            "splice": (
                fp_change_k is not None
                and abs(actual_change_k - fp_change_k) > SPLICE_K
            ),
        }
        _write_json(manifest_path, manifest)
        t = manifest["targets"][key]
        fp_str = "n/a" if fp_change_k is None else f"{fp_change_k:+,.0f}k"
        print(
            f"  T={len(censored_dates)}, c_idx={c_idx}, actual best {actual_change_k:+,.0f}k / "
            f"rev0 {fp_str}{'  ** SPLICE' if t['splice'] else ''}, hash {digest[:12]}",
            flush=True,
        )

    n_ok = len(_targets(manifest))
    print(f"\nGrid: {n_ok}/{N_BACKTEST} buildable. Manifest: {manifest_path}")


# =========================================================================
# serial baseline
# =========================================================================


def cmd_serial(root: Path) -> None:
    from nfp_model import ModelPriors, fit_model, model_inputs
    from nfp_model.batch import active_cyclicals
    from nfp_model.parity import collect_parity_arrays

    manifest = _read_json(root / "grid_manifest.json")
    prov = manifest["provenance"]
    out_path = root / "serial_manifest.json"
    entries: dict = _read_json(out_path) if out_path.exists() else {}

    targets = _targets(manifest)
    for n, (key, t) in enumerate(targets):
        npz_path = root / f"serial_{key}.npz"
        if npz_path.exists():
            print(f"[{n + 1}/{len(targets)}] {key}: exists, skipping", flush=True)
            continue
        seed = SERIAL_SEED0 + n
        print(f"[{n + 1}/{len(targets)}] {key}: fitting ({PRESET}, seed {seed})...", flush=True)
        data = _snapshot_data(root / "snapshots", key)
        fit = fit_model(data, settings=PRESET, seed=seed)
        arrays, meta = collect_parity_arrays(
            fit,
            base_index=float(prov["base_index"]),
            idx_to_level=float(prov["idx_to_level"]),
            c_idx=int(t["c_idx"]),
        )
        meta["cyclical_in_model"] = list(active_cyclicals(model_inputs(data), ModelPriors()))
        # source availability at the nowcast index (reference report fields)
        c = int(t["c_idx"])
        meta["sources"] = sorted(
            (["CES"] if c in np.asarray(data["ces_sa_obs"]) else [])
            + (["QCEW"] if c in np.asarray(data["qcew_obs"]) else [])
            + [pp["name"] for pp in data["pp_data"] if c in np.asarray(pp["pp_obs"])]
        )
        np.savez(npz_path, **arrays)
        entries[key] = meta
        _write_json(out_path, entries)
        print(
            f"  done in {meta['wall_seconds']:.0f}s: div={meta['num_divergences']}, "
            f"nowcast {meta['nowcast_change_k']:+,.0f}k "
            f"(actual {t['actual_change_k']:+,.0f}k)",
            flush=True,
        )

    total_wall = sum(e["wall_seconds"] for e in entries.values())
    print(f"\nSerial baseline: {len(entries)} fits, {total_wall / 60:.1f} min sampling wall.")


# =========================================================================
# batched run
# =========================================================================


def cmd_batched(root: Path) -> None:
    from nfp_model import fit_model_batch, model_inputs, pad_model_inputs

    manifest = _read_json(root / "grid_manifest.json")
    prov = manifest["provenance"]
    targets = _targets(manifest)

    print(f"Loading {len(targets)} snapshots...", flush=True)
    inputs = [model_inputs(_snapshot_data(root / "snapshots", key)) for key, _ in targets]
    c_idx = [int(t["c_idx"]) for _, t in targets]
    bi = pad_model_inputs(inputs, c_idx=c_idx)
    print(
        f"Padded batch: n={bi.n_dates}, T_max={bi.static['T']}, "
        f"T_real {bi.T_real.min()}–{bi.T_real.max()}, "
        f"providers {bi.pp_names}, cyclical {bi.static['cyclical_active']}",
        flush=True,
    )

    t0 = time.time()
    batch = fit_model_batch(
        bi,
        settings=PRESET,
        seed=BATCH_SEED,
        base_index=float(prov["base_index"]),
        idx_to_level=float(prov["idx_to_level"]),
    )
    wall_total = time.time() - t0
    print(
        f"Batched fit: {batch.wall_seconds / 60:.1f} min sampling "
        f"({wall_total / 60:.1f} min with data prep), "
        f"divergences per date: {batch.arrays['num_divergences'].astype(int).tolist()}",
        flush=True,
    )

    entries: dict = {}
    for i, (key, _t) in enumerate(targets):
        arrays, meta = batch.date_arrays(i)
        np.savez(root / f"batched_{key}.npz", **arrays)
        entries[key] = meta
    _write_json(
        root / "batched_manifest.json",
        {
            "entries": entries,
            "batch_wall_seconds": round(batch.wall_seconds, 1),
            "n_dates": batch.n_dates,
            "seed": BATCH_SEED,
            "preset": PRESET,
            "T_max": int(bi.static["T"]),
        },
    )
    print(f"Saved {batch.n_dates} per-date reductions + batched_manifest.json")


# =========================================================================
# compare + report
# =========================================================================


def cmd_compare(root: Path) -> int:
    import polars as pl
    from nfp_model.parity import compare_reduced

    manifest = _read_json(root / "grid_manifest.json")
    prov = manifest["provenance"]
    serial = _read_json(root / "serial_manifest.json")
    batched = _read_json(root / "batched_manifest.json")

    reports = []
    rows = []
    skipped = []
    for key, t in _targets(manifest):
        if key not in serial or key not in batched["entries"]:
            skipped.append(key)
            continue
        s_meta, b_meta = serial[key], batched["entries"][key]
        fx = {
            "as_of": key,
            "preset": PRESET,
            "T": s_meta["T"],
            "c_idx": s_meta["c_idx"],
            "cyclical_in_model": s_meta["cyclical_in_model"],
            "nowcast_change_k": s_meta["nowcast_change_k"],
        }
        report = compare_reduced(
            _load_npz(root / f"serial_{key}.npz"),
            fx,
            _load_npz(root / f"batched_{key}.npz"),
            b_meta,
            prov,
        )
        reports.append(report)
        print(report.summary(failures_only=True), flush=True)
        fp = t.get("first_print_change_k")
        rows.append(
            {
                "date": date.fromisoformat(key),
                "actual_first_print_k": fp,
                "actual_best_avail_k": t["actual_change_k"],
                "splice": bool(t.get("splice", False)),
                "serial_change_k": s_meta["nowcast_change_k"],
                "batched_change_k": b_meta["nowcast_change_k"],
                "serial_error_fp_k": None if fp is None else fp - s_meta["nowcast_change_k"],
                "batched_error_fp_k": None if fp is None else fp - b_meta["nowcast_change_k"],
                "serial_error_best_k": t["actual_change_k"] - s_meta["nowcast_change_k"],
                "batched_error_best_k": t["actual_change_k"] - b_meta["nowcast_change_k"],
                "batched_vs_serial_k": b_meta["nowcast_change_k"] - s_meta["nowcast_change_k"],
                "serial_divergences": s_meta["num_divergences"],
                "batched_divergences": b_meta["num_divergences"],
                "sources": "+".join(s_meta["sources"]),
                "parity": "PASS" if report.passed else "FAIL",
                "criteria_failed": report.n_failed,
                "criteria_total": len(report.rows),
            }
        )

    df = pl.DataFrame(rows)
    df.write_parquet(root / "a4_results.parquet")

    n_fail = sum(r.n_failed for r in reports)
    n_rows = sum(len(r.rows) for r in reports)
    verdict = "PASS" if n_fail == 0 and reports else "FAIL"
    serial_wall = sum(e["wall_seconds"] for e in serial.values())
    batch_wall = float(batched["batch_wall_seconds"])
    splice_dates = [str(r["date"]) for r in rows if r["splice"]]

    def _metrics(col: str, *, exclude_splice: bool = False) -> str:
        sub = df.filter(~pl.col("splice")) if exclude_splice else df
        e = sub[col].drop_nulls().to_numpy()
        if len(e) == 0:
            return "n/a"
        return (
            f"ME {e.mean():+,.0f}k, MAE {np.abs(e).mean():,.0f}k, "
            f"RMSE {np.sqrt((e**2).mean()):,.0f}k (n={len(e)})"
        )

    headline = (
        f"A4 BACKTEST {verdict}: {len(reports)} dates, "
        f"{n_rows - n_fail}/{n_rows} parity criteria passed; "
        f"serial {serial_wall / 60:.1f} min vs batched {batch_wall / 60:.1f} min "
        f"(×{serial_wall / batch_wall:.1f})"
        + (f"; skipped {skipped}" if skipped else "")
    )
    print("\n" + headline)

    lines = [
        "# A4 backtest report",
        "",
        headline,
        "",
        f"Window: {len(reports)} as-of dates, preset `{PRESET}`. Dual-track actuals:",
        "first print (rev-0) and uncensored best-available revision (the reference",
        "convention). † marks rows where the two disagree by more than "
        f"{SPLICE_K:.0f}k —",
        "genuine large revisions (e.g. the 2025-08-01 May/June revision), annual",
        "benchmarks (2024's in the 2025-01 first print; 2025's −911k entering via",
        "2026-02 vintages), and store rev-0 growth computed against the prior",
        "vintage's previous-month level (off-headline in big-revision months).",
        "† rows are excluded from best-available metrics. Defining the scoring",
        "convention is A5's evaluation question. Parity criteria:",
        "`nfp_model.parity` (the A3 gate instrument), serial run as reference.",
        "",
        "| date | rev-0 Δk | best Δk | serial Δk | batched Δk | b−s Δk | div s/b | sources | parity |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fp_str = "—" if r["actual_first_print_k"] is None else f"{r['actual_first_print_k']:+,.0f}"
        lines.append(
            f"| {r['date']} | {fp_str} "
            f"| {r['actual_best_avail_k']:+,.0f}{'†' if r['splice'] else ''} "
            f"| {r['serial_change_k']:+,.0f} "
            f"| {r['batched_change_k']:+,.0f} | {r['batched_vs_serial_k']:+,.1f} "
            f"| {r['serial_divergences']}/{r['batched_divergences']} "
            f"| {r['sources']} | {r['parity']} |"
        )
    lines += [
        "",
        f"- serial vs first print: {_metrics('serial_error_fp_k')}",
        f"- batched vs first print: {_metrics('batched_error_fp_k')}",
        f"- serial vs best-available (non-splice): "
        f"{_metrics('serial_error_best_k', exclude_splice=True)}",
        f"- batched vs best-available (non-splice): "
        f"{_metrics('batched_error_best_k', exclude_splice=True)}",
        f"- batched vs serial: {_metrics('batched_vs_serial_k')}",
        f"- wall: serial {serial_wall / 60:.1f} min sampling; "
        f"batched {batch_wall / 60:.1f} min for the whole grid",
    ]
    if splice_dates:
        lines.append(
            f"- benchmark-splice dates (†, excluded from best-available metrics): "
            f"{', '.join(splice_dates)}"
        )
    lines += [
        "",
        "## Per-date parity detail",
        "",
    ]
    for report in reports:  # passes get the one-liner; failures show failing rows
        lines += ["```", report.summary(failures_only=True), "```", ""]
    (root / "a4_report.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {root / 'a4_report.md'}")
    return 0 if verdict == "PASS" else 1


def main() -> None:
    mode, root_arg = sys.argv[1], sys.argv[2]
    root = Path(root_arg).resolve()
    if mode == "snapshot":
        cmd_snapshot(root)
    elif mode == "serial":
        cmd_serial(root)
    elif mode == "batched":
        cmd_batched(root)
    elif mode == "compare":
        raise SystemExit(cmd_compare(root))
    else:
        raise SystemExit(f"unknown mode {mode!r} (use: snapshot | serial | batched | compare)")


if __name__ == "__main__":
    main()
