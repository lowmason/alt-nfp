#!/usr/bin/env python3
"""One-time historical store rebuild + promote (NOT a CLI command).

Lifts the **rebuild** lineage (spec cli_production_workflow.md §10), ordered::

    download_ces(data_dir=tmp)                    # extract cesvinall/ triangular CSVs
    advance_release_calendar()                    # vintage_dates.parquet present for overlap parity
    build_ces_panel(cesvinall_dir=tmp/...)        # CES NSA+SA store-schema rows
    acquire_qcew_levels(...)      -> build_qcew_panel(...)
    acquire_qcew_size_native(...) -> build_size_class_panel(...)   # Q1-only
    compose_rebuild_panel(...)
    write_rebuild_store(panel, scratch, allow_canonical=False)     # scratch prefix
    promote(scratch -> canonical)                 # copy-then-delete cutover (_t8_promote flow)

Usage::

    NFP_STORE_URI=s3://alt-nfp/store-rebuild \\
      uv run python scripts/bootstrap_store.py \\
      --scratch s3://alt-nfp/store-rebuild --canonical s3://alt-nfp/store

Scope is national-only, 2017+ (the intended canonical scope). QCEW is fetched
live from the CEW API (not the bulk ZIPs), so only ``download_ces`` is wired.

Container-safety (plans/15): every byproduct that the legacy lineage parked under
``./data`` is routed to a run-scoped ``tempfile.TemporaryDirectory`` here — the
raw ``cesvinall/`` extract and the CES read both point at ``tmp``. The only
artifact that survives the run is the rebuilt **store** on S3 (the scratch
``NFP_STORE_URI`` prefix). ``write_rebuild_store`` is itself container-safe
(``str(path)`` + ``storage_options_for`` + ``is_remote`` mkdir guard).

The promote step copies rebuild files into the canonical prefix then deletes the
old orphans (filenames encode vintage ranges, so a plain overwrite-mirror would
leave both files and corrupt the store — the exact hazard CLAUDE.md warns about).
``scripts/mirror_store.py`` is overwrite-only and is deliberately NOT used here.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# --- .env MUST load before any nfp_* import: nfp_lookups.paths reads
#     NFP_STORE_URI at import time. ---
from dotenv import load_dotenv

load_dotenv(".env")

from nfp_download.bls.bulk import download_ces  # noqa: E402
from nfp_ingest.ces_builder import build_ces_panel  # noqa: E402
from nfp_ingest.qcew_acquire import (  # noqa: E402
    acquire_qcew_levels,
    acquire_qcew_size_native,
)
from nfp_ingest.qcew_crosswalk import build_qcew_panel  # noqa: E402
from nfp_ingest.size_class import build_size_class_panel  # noqa: E402
from nfp_lookups.paths import is_canonical_store  # noqa: E402
from nfp_vintages.calendar import advance_release_calendar  # noqa: E402
from nfp_vintages.rebuild_store import (  # noqa: E402
    compose_rebuild_panel,
    write_rebuild_store,
)


def _is_remote(uri: str) -> bool:
    return uri.startswith(("s3://", "s3a://"))


def _store_path(uri: str):
    """A pathlib-compatible handle for a scratch/canonical store location.

    Local prefixes return a plain ``Path``; ``s3://`` prefixes return a
    credentialed ``UPath`` (the same shape ``write_rebuild_store`` accepts).
    """
    if _is_remote(uri):
        from upath import UPath

        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        client_kwargs = {"endpoint_url": endpoint} if endpoint else {}
        return UPath(
            uri,
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            client_kwargs=client_kwargs,
        )
    return Path(uri)


# ---------------------------------------------------------------------------
# Promote: generalized _t8_promote.py:cutover (copy-then-delete per partition)
# ---------------------------------------------------------------------------


def _s3fs():
    import s3fs

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    return s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint} if endpoint else {},
    )


def _local_keys(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.glob("**/*.parquet"))


def _s3_keys(fs, prefix: str) -> list[str]:
    """Genuine children of *prefix* only (store vs store-rebuild share a head)."""
    return sorted(k for k in fs.find(prefix) if k.startswith(prefix + "/"))


def _promote_local(scratch: Path, canonical: Path) -> None:
    rel_keys = _local_keys(scratch)
    if not rel_keys:
        sys.exit(f"FATAL: scratch store {scratch} is empty — refusing promote.")
    canonical.mkdir(parents=True, exist_ok=True)
    # 1) copy rebuild files in (under their rebuilt names).
    for rel in rel_keys:
        dst = canonical / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((scratch / rel).read_bytes())
    # 2) delete old-named orphans (anything canonical-side not in the new set).
    new_set = set(rel_keys)
    for p in sorted(canonical.glob("**/*.parquet")):
        if p.relative_to(canonical).as_posix() not in new_set:
            p.unlink()
    # 3) verify: canonical == exactly the rebuild set.
    final = set(_local_keys(canonical))
    if final != new_set:
        sys.exit(f"FATAL: post-promote keyset mismatch under {canonical}.")
    print(f"promote (local): +{len(new_set)} files; canonical == rebuild set, verified")


def _promote_remote(scratch_uri: str, canonical_uri: str) -> None:
    fs = _s3fs()
    src = scratch_uri.removeprefix("s3://").rstrip("/")
    dst = canonical_uri.removeprefix("s3://").rstrip("/")
    src_keys = _s3_keys(fs, src)
    if not src_keys:
        sys.exit(f"FATAL: scratch store {scratch_uri} is empty — refusing promote.")
    new_dst = {k.replace(src, dst, 1): k for k in src_keys}  # dst -> src
    # 1) copy rebuild files in (new names).
    for dst_key, src_key in new_dst.items():
        fs.pipe_file(dst_key, fs.cat_file(src_key))
    # 2) delete old-named orphans.
    for k in _s3_keys(fs, dst):
        if k not in new_dst:
            fs.rm(k)
    # 3) verify keyset.
    final = _s3_keys(fs, dst)
    if final != sorted(new_dst):
        sys.exit(f"FATAL: post-promote keyset mismatch under {canonical_uri}.")
    print(f"promote (s3): +{len(new_dst)} files; canonical == rebuild set, verified")


def _promote_scratch_to_canonical(scratch_uri: str, canonical_uri: str) -> None:
    """Copy-then-delete cutover from *scratch* to *canonical* (no overwrite-mirror)."""
    if _is_remote(canonical_uri):
        _promote_remote(scratch_uri, canonical_uri)
    else:
        _promote_local(Path(scratch_uri), Path(canonical_uri))


# ---------------------------------------------------------------------------
# Rebuild orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-time store rebuild + promote.")
    parser.add_argument(
        "--scratch",
        required=True,
        help="Scratch store URI/path (e.g. s3://alt-nfp/store-rebuild). "
        "Must NOT be the canonical store.",
    )
    parser.add_argument(
        "--canonical",
        required=True,
        help="Canonical store URI/path to promote into (e.g. s3://alt-nfp/store).",
    )
    parser.add_argument(
        "--start-year", type=int, default=2017, help="First QCEW reference year."
    )
    parser.add_argument(
        "--end-year", type=int, default=None, help="Last QCEW reference year (inclusive)."
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Build the scratch store but skip the canonical promote.",
    )
    args = parser.parse_args(argv)

    # Guard FIRST — refuse the canonical store as a scratch target (no I/O before this).
    if is_canonical_store(args.scratch):
        sys.exit(
            f"refusing to bootstrap straight to the canonical store ({args.scratch}); "
            "target a scratch prefix (e.g. s3://alt-nfp/store-rebuild)."
        )

    # Container-safety: every ./data byproduct (raw cesvinall extract, scraped HTML)
    # lands under a run-scoped tempdir — only the rebuilt STORE on S3 survives.
    with tempfile.TemporaryDirectory(prefix="altnfp-bootstrap-") as tmp:
        tmp_root = Path(tmp)
        cesvinall_dir = tmp_root / "downloads" / "ces" / "cesvinall"

        print("=== Bootstrap: download CES triangular CSVs (-> tempdir) ===")
        download_ces(data_dir=tmp_root)

        print("=== Bootstrap: advance release calendar (overlap parity) ===")
        advance_release_calendar()

        print("=== Bootstrap: build CES panel (NSA + SA) ===")
        ces = build_ces_panel(cesvinall_dir=cesvinall_dir)
        print(f"  CES: {ces.height:,} rows")

        print(f"=== Bootstrap: acquire QCEW levels ({args.start_year}-{args.end_year}) ===")
        raw_qcew = acquire_qcew_levels(start_year=args.start_year, end_year=args.end_year)
        qcew_levels = build_qcew_panel(raw_qcew)
        print(f"  QCEW levels: {qcew_levels.height:,} rows")

        print(f"=== Bootstrap: acquire QCEW size native ({args.start_year}-{args.end_year}) ===")
        size_native = acquire_qcew_size_native(
            start_year=args.start_year, end_year=args.end_year
        )
        size = build_size_class_panel(size_native) if size_native.height else None
        if size is not None and size.height:
            print(f"  QCEW size: {size.height:,} rows")
        else:
            size = None
            print("  QCEW size: 0 rows (skipped)")

        print("=== Bootstrap: compose panels ===")
        panel = compose_rebuild_panel(ces, qcew_levels, size)
        print(f"  Combined: {panel.height:,} rows")

        print(f"=== Bootstrap: write scratch store ({args.scratch}) ===")
        # panel is positional-first; store_path second (rebuild_store.py:212-217).
        write_rebuild_store(panel, _store_path(args.scratch), allow_canonical=False)

    if args.no_promote:
        print("Done (scratch only; --no-promote set).")
        return

    print(f"=== Bootstrap: promote {args.scratch} -> {args.canonical} ===")
    _promote_scratch_to_canonical(args.scratch, args.canonical)
    print("Done.")


if __name__ == "__main__":
    main()
