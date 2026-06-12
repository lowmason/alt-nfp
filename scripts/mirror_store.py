"""Mirror a local Hive-partitioned vintage store into the S3 bucket.

Usage::

    uv run python scripts/mirror_store.py [SRC_DIR]

SRC_DIR defaults to the workspace's local ``data/store/``. The destination
and credentials come from the environment (``.env`` is loaded):
``NFP_STORE_URI``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
``AWS_ENDPOINT_URL``. Creates the bucket if missing and uploads every
parquet preserving the hive layout. Idempotent: existing keys are
overwritten. (Kept as a script because no mc/aws CLI is installed.)
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

    import s3fs
    from nfp_lookups.paths import STORE_DIR

    src = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else STORE_DIR
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
