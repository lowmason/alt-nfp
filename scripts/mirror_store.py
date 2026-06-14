"""Mirror a local Hive-partitioned vintage store into the S3 bucket.

Usage::

    uv run python scripts/mirror_store.py [SRC_DIR] [--allow-canonical]

SRC_DIR defaults to the workspace's local ``data/store/``. The destination
and credentials come from the environment (``.env`` is loaded):
``NFP_STORE_URI``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
``AWS_ENDPOINT_URL``. Creates the bucket if missing and uploads every
parquet preserving the hive layout. Idempotent: existing keys are
overwritten. (Kept as a script because no mc/aws CLI is installed.)

Refuses to mirror onto the canonical, append-only store
(``s3://alt-nfp/store``) unless ``--allow-canonical`` is passed; target a
scratch prefix (e.g. ``.../store-rebuild``) instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    uri = os.environ.get("NFP_STORE_URI", "")
    if not uri.startswith("s3://"):
        sys.exit("NFP_STORE_URI is not an s3:// URI; nothing to mirror to.")

    from nfp_lookups.paths import is_canonical_store

    allow_canonical = "--allow-canonical" in sys.argv
    if is_canonical_store(uri) and not allow_canonical:
        sys.exit(
            f"refusing to mirror onto the canonical store {uri!r} — it is "
            "append-only and irreplaceable; target a scratch prefix or pass "
            "--allow-canonical"
        )

    import s3fs
    from nfp_lookups.paths import STORE_DIR

    positionals = [a for a in sys.argv[1:] if a != "--allow-canonical"]
    src = Path(positionals[0]).expanduser() if positionals else STORE_DIR
    files = sorted(src.glob("**/*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {src}")

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    fs = s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint} if endpoint else {},
    )

    dest = uri.removeprefix("s3://").rstrip("/")
    bucket = dest.split("/", 1)[0]
    if not fs.exists(bucket):
        fs.mkdir(bucket)
        print(f"Created bucket: {bucket}")

    for f in files:
        key = f"{dest}/{f.relative_to(src).as_posix()}"
        fs.put_file(str(f), key)
        print(f"  {f.relative_to(src)} -> s3://{key}")

    print(f"Mirrored {len(files)} files from {src} to {uri}")


if __name__ == "__main__":
    main()
