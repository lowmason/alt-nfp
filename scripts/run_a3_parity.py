"""A3 parity runner — new-side fits + comparison against reference fixtures.

Two phases (both restartable):

    uv run python scripts/run_a3_parity.py fit data/golden_a3_staging data/a3
    uv run python scripts/run_a3_parity.py compare data/golden_a3_staging data/a3

``fit`` reads the reference manifest, rebuilds the same as-of data through
``nfp_ingest.model_data.build_model_data``, fits the NumPyro model with the
matching preset, and saves the reduced parity schema per date
(``new_<stem>.npz`` + ``new_manifest.json``). ``compare`` applies the
criteria in ``nfp_model.parity`` to every stem present on both sides and
writes ``parity_report.md``. Exit code 1 on any failure.

The reference staging dir may also be the S3 fixture prefix once uploaded
(pass an ``s3://…`` URI; requires the store env). See
``specs/plans/5-a3_model_parity.md``.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

END_YEAR = 2026
SEED_OFFSET = 1000  # independent of the reference seeds (different sampler)


def _location(arg: str):
    if arg.startswith("s3://"):
        import os

        from upath import UPath

        client_kwargs = {}
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        return UPath(
            arg,
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            client_kwargs=client_kwargs,
        )
    return Path(arg).resolve()


def _read_json(path) -> dict:
    return json.loads(path.read_text())


def _load_npz(path) -> dict:
    import io

    with path.open("rb") as f:
        npz = np.load(io.BytesIO(f.read()), allow_pickle=False)
        return {k: npz[k] for k in npz.files}


def _jobs(ref_manifest: dict) -> list[tuple[str, dict]]:
    return [
        (stem, fx)
        for stem, fx in sorted(ref_manifest["fixtures"].items())
        if "error" not in fx
    ]


def cmd_fit(ref_dir, out_dir: Path) -> None:
    from nfp_ingest.model_data import build_model_data
    from nfp_model import fit_model
    from nfp_model.parity import collect_parity_arrays

    ref_manifest = _read_json(ref_dir / "a3_manifest.json")
    prov = ref_manifest["provenance"]
    out_dir.mkdir(parents=True, exist_ok=True)

    new_manifest_path = out_dir / "new_manifest.json"
    entries: dict = {}
    if new_manifest_path.exists():
        entries = _read_json(new_manifest_path)

    jobs = _jobs(ref_manifest)
    for i, (stem, fx) in enumerate(jobs):
        npz_path = out_dir / f"new_{stem}.npz"
        if npz_path.exists():
            print(f"[{i + 1}/{len(jobs)}] {stem}: exists, skipping", flush=True)
            continue
        as_of = date.fromisoformat(fx["as_of"])
        seed = int(fx["seed"]) + SEED_OFFSET
        print(f"[{i + 1}/{len(jobs)}] {stem}: building data + fitting "
              f"({fx['preset']}, seed {seed})...", flush=True)

        data = build_model_data(as_of, end_year=END_YEAR)
        fit = fit_model(data, settings=fx["preset"], seed=seed)
        arrays, meta = collect_parity_arrays(
            fit,
            base_index=float(prov["base_index"]),
            idx_to_level=float(prov["idx_to_level"]),
            c_idx=int(fx["c_idx"]),
        )
        np.savez(npz_path, **arrays)
        entries[stem] = meta
        new_manifest_path.write_text(json.dumps(entries, indent=2) + "\n")
        print(f"  done in {meta['wall_seconds']:.0f}s: divergences="
              f"{meta['num_divergences']}, nowcast {meta['nowcast_change_k']:+,.0f}k "
              f"(ref {fx['nowcast_change_k']:+,.0f}k)", flush=True)


def cmd_compare(ref_dir, out_dir: Path) -> int:
    from nfp_model.parity import compare_reduced

    ref_manifest = _read_json(ref_dir / "a3_manifest.json")
    prov = ref_manifest["provenance"]
    new_entries = _read_json(out_dir / "new_manifest.json")

    reports = []
    lines = [
        "# A3 parity report",
        "",
        f"Reference: {prov['old_repo_commit'][:12]} (pymc {prov['pymc_version']}, "
        f"nutpie {prov['nutpie_version']}), corrected indicators config.",
        "",
    ]
    skipped = []
    for stem, fx in _jobs(ref_manifest):
        if stem not in new_entries:
            skipped.append(stem)
            continue
        ref_arrays = _load_npz(ref_dir / f"ref_{stem}.npz")
        new_arrays = _load_npz(out_dir / f"new_{stem}.npz")
        report = compare_reduced(ref_arrays, fx, new_arrays, new_entries[stem], prov)
        reports.append(report)
        print(report.summary(failures_only=True), flush=True)
        lines += ["## " + stem, "", "```", report.summary(), "```", ""]

    n_fail = sum(r.n_failed for r in reports)
    n_rows = sum(len(r.rows) for r in reports)
    verdict = "PASS" if n_fail == 0 and reports else "FAIL"
    headline = (
        f"A3 PARITY {verdict}: {len(reports)} fixtures, "
        f"{n_rows - n_fail}/{n_rows} criteria passed"
        + (f", skipped (no new fit yet): {skipped}" if skipped else "")
    )
    print("\n" + headline)
    lines.insert(2, headline)
    lines.insert(3, "")
    (out_dir / "parity_report.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {out_dir / 'parity_report.md'}")
    return 0 if verdict == "PASS" else 1


def main() -> None:
    mode, ref_arg, out_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    ref_dir = _location(ref_arg)
    out_dir = Path(out_arg).resolve()
    if mode == "fit":
        cmd_fit(ref_dir, out_dir)
    elif mode == "compare":
        raise SystemExit(cmd_compare(ref_dir, out_dir))
    else:
        raise SystemExit(f"unknown mode {mode!r} (use: fit | compare)")


if __name__ == "__main__":
    main()
