"""T8 promotion (plans/10 T8): scratch rebuild -> canonical, with backups.

ONE-TIME, dated cutover. Subcommands (run in order; each guarded):

    NFP_STORE_URI=s3://alt-nfp/store uv run python scripts/_t8_promote.py backup
    NFP_STORE_URI=s3://alt-nfp/store uv run python scripts/_t8_promote.py cutover
    NFP_STORE_URI=s3://alt-nfp/store uv run python scripts/_t8_promote.py verify
    # emergency only:
    NFP_STORE_URI=s3://alt-nfp/store uv run python scripts/_t8_promote.py rollback

Bypasses the is_canonical_store write-doors deliberately (this IS the sanctioned
cutover); the local+S3 backups + copy-then-delete-old are the safety net. Reads
creds from .env. The canonical store has been wiped before with no backup but the
frozen reference — so `cutover` REFUSES to run unless `backup` produced a verified
local copy first.
"""

import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    for line in Path(".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

DATE = "20260618"
# LOCAL-ONLY, NOT CONTAINER-SAFE (plans/15 Tier D, by design). This is a
# one-time, dated cutover that was already run locally on 2026-06-18; the local
# backup is the PERSISTENT safety net taken before the canonical store is
# touched, so it must NOT be a tempfile (it has to survive the run). It is not
# meant to run on Bloomberg's footprint-limited container — if it ever must,
# point LOCAL_BACKUP at a persistent S3/host path, never ./data.
LOCAL_BACKUP = Path("data/canonical_backup_" + DATE).resolve()
MANIFEST = LOCAL_BACKUP / "backup_manifest.json"

# (canonical prefix, rebuild source prefix) — bucket-qualified keys (no s3://).
PAIRS = [
    ("alt-nfp/store", "alt-nfp/store-rebuild"),
    ("alt-nfp/golden/a1", "alt-nfp/golden/a1-rebuild"),
    ("alt-nfp/golden/a2", "alt-nfp/golden/a2-rebuild"),
]
# local subdir name per canonical prefix
LOCAL_NAME = {"alt-nfp/store": "store", "alt-nfp/golden/a1": "golden_a1", "alt-nfp/golden/a2": "golden_a2"}


def _fs():
    import s3fs

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    return s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint} if endpoint else {},
    )


def _keys(fs, prefix):
    """Genuine children of *prefix* only — never sibling prefixes (store vs
    store-rebuild vs store-prev all share the 'alt-nfp/store' string head)."""
    return sorted(k for k in fs.find(prefix) if k.startswith(prefix + "/"))


def _size(fs, key):
    return int(fs.info(key).get("size", 0))


def _copy(fs, src, dst):
    fs.pipe_file(dst, fs.cat_file(src))


def backup():
    fs = _fs()
    LOCAL_BACKUP.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for canon, _src in PAIRS:
        keys = _keys(fs, canon)
        if not keys:
            sys.exit(f"FATAL: canonical prefix {canon} is empty — refusing.")
        sub = LOCAL_NAME[canon]
        prev = f"{canon}-prev-{DATE}"
        recs = []
        for k in keys:
            rel = k[len(canon) + 1 :]
            sz = _size(fs, k)
            # local download
            lp = LOCAL_BACKUP / sub / rel
            lp.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(k, str(lp))
            assert lp.stat().st_size == sz, f"local size mismatch {lp}: {lp.stat().st_size} != {sz}"
            # S3-side -prev- backup
            _copy(fs, k, f"{prev}/{rel}")
            recs.append({"rel": rel, "size": sz})
        # verify S3 prev landed
        prev_keys = _keys(fs, prev)
        assert len(prev_keys) == len(keys), f"{prev}: {len(prev_keys)} != {len(keys)}"
        manifest[canon] = {"prev_prefix": prev, "local_subdir": sub, "files": recs}
        print(f"backed up {canon}: {len(keys)} files -> local {sub}/ + s3 {prev}")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nBackup manifest: {MANIFEST}")
    print("Backup complete + verified (local sizes match; S3 -prev- populated).")


def _check_backup():
    if not MANIFEST.exists():
        sys.exit("FATAL: no backup manifest — run `backup` first. Refusing cutover.")
    manifest = json.loads(MANIFEST.read_text())
    for _canon, info in manifest.items():
        for rec in info["files"]:
            lp = LOCAL_BACKUP / info["local_subdir"] / rec["rel"]
            if not lp.exists() or lp.stat().st_size != rec["size"]:
                sys.exit(f"FATAL: backup file missing/short: {lp}. Refusing cutover.")
    return manifest


def cutover():
    _check_backup()
    fs = _fs()
    for canon, src in PAIRS:
        src_keys = _keys(fs, src)
        if not src_keys:
            sys.exit(f"FATAL: rebuild source {src} is empty — refusing.")
        new_dst = {k.replace(src, canon, 1): k for k in src_keys}  # dst -> src
        # 1) copy rebuild files in (new names) — partition never empties
        for dst, k in new_dst.items():
            _copy(fs, k, dst)
        # 2) delete old-named orphans (anything under canon not in the new set)
        existing = _keys(fs, canon)
        orphans = [k for k in existing if k not in new_dst]
        for o in orphans:
            fs.rm(o)
        # 3) verify: canonical == exactly the rebuild set, sizes match
        final = _keys(fs, canon)
        assert final == sorted(new_dst), f"{canon}: post-cutover keyset wrong"
        for dst, k in new_dst.items():
            assert _size(fs, dst) == _size(fs, k), f"size mismatch {dst}"
        print(f"cutover {canon}: +{len(new_dst)} rebuild files, -{len(orphans)} orphans, verified")
    print("\nCutover complete + verified (canonical == rebuild content, no orphans).")


def verify():
    fs = _fs()
    ok = True
    for canon, src in PAIRS:
        c = {k[len(canon) + 1 :]: _size(fs, k) for k in _keys(fs, canon)}
        s = {k[len(src) + 1 :]: _size(fs, k) for k in _keys(fs, src)}
        match = c == s
        ok = ok and match
        print(f"{canon}: {len(c)} files; matches {src}? {match}")
        if not match:
            print("   canon:", sorted(c))
            print("   src:  ", sorted(s))
    print("\nFIDELITY", "OK" if ok else "MISMATCH")
    if not ok:
        sys.exit(1)


def rollback():
    """Emergency: restore canonical prefixes from the -prev- S3 backups."""
    fs = _fs()
    manifest = json.loads(MANIFEST.read_text())
    for canon, info in manifest.items():
        prev = info["prev_prefix"]
        prev_keys = _keys(fs, prev)
        new_dst = {k.replace(prev, canon, 1): k for k in prev_keys}
        for dst, k in new_dst.items():
            _copy(fs, k, dst)
        for k in _keys(fs, canon):
            if k not in new_dst:
                fs.rm(k)
        print(f"rolled back {canon} from {prev}: {len(new_dst)} files")
    print("\nRollback complete (canonical restored from -prev- backups).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"backup": backup, "cutover": cutover, "verify": verify, "rollback": rollback}.get(
        cmd, lambda: sys.exit(f"usage: {sys.argv[0]} backup|cutover|verify|rollback")
    )()
