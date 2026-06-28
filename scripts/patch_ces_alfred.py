# scripts/patch_ces_alfred.py
"""One-shot ALFRED CES frontier patch.

Dry-run (default) reports what would be appended; --apply writes. Point --store
at a SCRATCH prefix first (validate), then canonical. Never writes ./data.

    uv run python scripts/patch_ces_alfred.py --through 2026-06-12            # dry-run, canonical (read-only)
    uv run python scripts/patch_ces_alfred.py --apply --store s3://alt-nfp/store-rebuild
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")  # noqa: E402

from nfp_ingest.capture import capture_ces_alfred_window  # noqa: E402
from nfp_lookups.paths import VINTAGE_STORE_PATH  # noqa: E402


def _store(uri: str | None):
    """Resolve a --store argument to a Path/UPath (default VINTAGE_STORE_PATH)."""
    if uri is None:
        return VINTAGE_STORE_PATH
    if uri.startswith(("s3://", "s3a://")):
        from upath import UPath

        return UPath(uri)
    return Path(uri)


def main() -> None:
    """Parse args and run the ALFRED CES frontier patch (dry-run unless --apply)."""
    p = argparse.ArgumentParser(description="ALFRED CES frontier patch")
    p.add_argument("--through", type=date.fromisoformat, default=date.today(),
                   help="Upper bound on vintage_date (YYYY-MM-DD); default today.")
    p.add_argument("--store", default=None, help="Store URI/path (default VINTAGE_STORE_PATH).")
    p.add_argument("--apply", action="store_true", help="Write (default: dry-run).")
    args = p.parse_args()

    res = capture_ces_alfred_window(
        through=args.through, store_path=_store(args.store), dry_run=not args.apply
    )
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] appended={res.appended} skipped={res.skipped} "
          f"corrected={len(res.corrected)}")
    for c in res.corrected:
        print(f"  CORRECTED ref={c.ref_date} code={c.industry_code} rev={c.revision} "
              f"stored={c.stored_employment} incoming={c.incoming_employment}")


if __name__ == "__main__":
    main()
