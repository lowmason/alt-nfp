"""Upload the locally-staged rebuilt A1/A2 goldens to the SCRATCH S3 prefixes
(plans/12 T2 step 3). Reads the local staging dir; writes ONLY to
``s3://alt-nfp/golden/a1-rebuild`` + ``…/a2-rebuild``. Hard-refuses the frozen
canonical ``…/golden/a1`` / ``…/golden/a2``.

    NFP_STORE_URI=s3://alt-nfp/store-rebuild uv run python \\
        scripts/stage_goldens_rebuild.py data/golden_rebuild_staging
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

A1_TARGET = "s3://alt-nfp/golden/a1-rebuild"
A2_TARGET = "s3://alt-nfp/golden/a2-rebuild"
# Tripwire: the frozen reference prefixes are read-never-written.
_FORBIDDEN = {"s3://alt-nfp/golden/a1", "s3://alt-nfp/golden/a2"}


def _root(uri: str):
    from upath import UPath

    assert uri not in _FORBIDDEN, f"refusing to write the frozen reference prefix: {uri}"
    assert uri.rstrip("/").rsplit("/", 1)[-1].endswith("-rebuild"), (
        f"target must be a *-rebuild scratch prefix, got {uri}"
    )
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    return UPath(
        uri,
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint} if endpoint else {},
    )


def _upload(local_dir: Path, uri: str) -> int:
    root = _root(uri)
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(local_dir.iterdir()):
        if not f.is_file():
            continue
        (root / f.name).write_bytes(f.read_bytes())
        n += 1
        print(f"  -> {uri}/{f.name}  ({f.stat().st_size:,} B)")
    return n


def main() -> None:
    stage = Path(sys.argv[1]).resolve()
    a1_dir, a2_dir = stage / "a1", stage / "a2"
    assert a1_dir.is_dir() and a2_dir.is_dir(), f"expected a1/ and a2/ under {stage}"

    print(f"A1: {a1_dir}  ->  {A1_TARGET}")
    n1 = _upload(a1_dir, A1_TARGET)
    print(f"\nA2: {a2_dir}  ->  {A2_TARGET}")
    n2 = _upload(a2_dir, A2_TARGET)
    print(f"\nUploaded {n1} A1 + {n2} A2 objects to the scratch golden prefixes.")


if __name__ == "__main__":
    main()
