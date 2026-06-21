"""Seed the Tier-A persistent artifacts from local DATA_DIR into S3.

Usage::

    # Dry-run (default, safe — does not connect to S3):
    uv run python scripts/seed_data_s3.py

    # Apply — uploads and verifies each file:
    uv run python scripts/seed_data_s3.py --apply

Uploads ONLY:
  - ``indicators/``     (recursed)
  - ``competitors/``    (recursed)
  - ``intermediate/vintage_dates.parquet``
  - ``intermediate/release_dates.parquet``

EXCLUDED:
  - ``providers/``  — lives on a separate Bloomberg store; not ours to move.

SAFETY GUARD:
  Refuses to write any key whose first path segment is ``store``, ``store-prev``
  (or any ``store-prev*`` variant), or ``store-rebuild``. The vintage store must
  never be touched by this script.

Credentials and endpoint come from ``.env`` (loaded automatically):
  ``NFP_DATA_URI``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
  ``AWS_ENDPOINT_URL``.

The destination bucket is parsed from ``NFP_DATA_URI`` (default ``s3://alt-nfp``).
Keys mirror the local relative path under DATA_DIR:
  ``s3://<bucket>/<relpath>``

This script is idempotent: re-running ``--apply`` overwrites existing keys.
Verification compares local MD5 against the S3 object's MD5 (read-back).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Tier-A artifacts to upload (paths relative to DATA_DIR)
# ---------------------------------------------------------------------------
_RECURSE_DIRS = ["indicators", "competitors"]
_SINGLE_FILES = [
    "intermediate/vintage_dates.parquet",
    "intermediate/release_dates.parquet",
]

# These first-path-segment prefixes must never be written by this script.
_FORBIDDEN_SEGMENTS = {"store", "store-rebuild"}


def _is_forbidden_key(relpath: str) -> bool:
    """Return True if the path-within-bucket starts with a protected store prefix.

    Args:
        relpath: The key path relative to the bucket root (NOT prefixed with the
            bucket name), e.g. ``"store/x.parquet"`` or ``"indicators/foo.parquet"``.
    """
    segment = relpath.lstrip("/").split("/")[0]
    if segment in _FORBIDDEN_SEGMENTS:
        return True
    # also match store-prev* variants
    if segment.startswith("store-prev"):
        return True
    return False


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Collect (local_path, relpath_str) pairs for all Tier-A artifacts."""
    results: list[tuple[Path, str]] = []

    for dir_name in _RECURSE_DIRS:
        d = data_dir / dir_name
        if not d.exists():
            print(f"  [SKIP] {dir_name}/ not found locally")
            continue
        found = sorted(d.rglob("*"))
        files = [p for p in found if p.is_file()]
        if not files:
            print(f"  [SKIP] {dir_name}/ exists but is empty")
            continue
        for f in files:
            results.append((f, f.relative_to(data_dir).as_posix()))

    for rel in _SINGLE_FILES:
        p = data_dir / rel
        if not p.exists():
            print(f"  [SKIP] {rel} not found locally")
            continue
        results.append((p, rel))

    return results


def main() -> None:
    load_dotenv()

    apply_mode = "--apply" in sys.argv

    # Resolve DATA_DIR
    from nfp_lookups.paths import DATA_DIR  # noqa: PLC0415

    data_dir = DATA_DIR

    # Resolve destination bucket from NFP_DATA_URI
    data_uri = os.environ.get("NFP_DATA_URI", "s3://alt-nfp").rstrip("/")
    if not data_uri.startswith("s3://"):
        sys.exit(f"NFP_DATA_URI is not an s3:// URI: {data_uri!r}")
    bucket = data_uri.removeprefix("s3://").split("/")[0]

    print(f"DATA_DIR  : {data_dir}")
    print(f"Bucket    : {bucket}")
    print(f"Mode      : {'APPLY (writing to S3)' if apply_mode else 'DRY-RUN (no S3 connection)'}")
    print()

    # Collect files
    print("Collecting Tier-A artifacts...")
    files = _collect_files(data_dir)
    if not files:
        print("Nothing to upload.")
        return

    print()
    print("Planned uploads:")
    for local_path, relpath in files:
        s3_key = f"{bucket}/{relpath}"
        if _is_forbidden_key(relpath):
            sys.exit(
                f"ABORT: key {s3_key!r} starts with a forbidden store prefix. "
                "This script must not touch the vintage store."
            )
        print(f"  {local_path} -> s3://{s3_key}")

    print()
    print(f"Total: {len(files)} file(s) planned")

    if not apply_mode:
        print()
        print("DRY-RUN complete. Pass --apply to upload.")
        return

    # ------------------------------------------------------------------
    # APPLY mode: connect to S3 and upload
    # ------------------------------------------------------------------
    import s3fs  # noqa: PLC0415

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    fs = s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint} if endpoint else {},
    )

    # Ensure bucket exists
    if not fs.exists(bucket):
        fs.mkdir(bucket)
        print(f"Created bucket: {bucket}")

    print()
    print("Uploading...")
    passed = 0
    failed = 0
    for local_path, relpath in files:
        s3_key = f"{bucket}/{relpath}"
        if _is_forbidden_key(relpath):
            sys.exit(
                f"ABORT: key {s3_key!r} starts with a forbidden store prefix. "
                "This script must not touch the vintage store."
            )
        fs.put_file(str(local_path), s3_key)

        # Verify by MD5 comparison
        local_md5 = _md5(local_path)
        with fs.open(s3_key, "rb") as fobj:
            remote_bytes = fobj.read()
        remote_md5 = hashlib.md5(remote_bytes).hexdigest()

        if local_md5 == remote_md5:
            print(f"  PASS  {relpath}")
            passed += 1
        else:
            print(f"  FAIL  {relpath}  (local={local_md5} remote={remote_md5})")
            failed += 1

    print()
    print(f"Done: {passed} PASS, {failed} FAIL out of {len(files)} file(s)")
    if failed:
        sys.exit(f"{failed} file(s) failed verification")


if __name__ == "__main__":
    main()
