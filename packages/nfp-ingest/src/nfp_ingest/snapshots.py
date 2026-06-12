"""Serialized ModelData snapshots: the artifact boundary between data and model.

Phase A2: the model layer consumes hash-pinned snapshot files instead of
calling into the data pipeline. A snapshot is a single ``.npz`` holding
every model-ready array plus one JSON metadata entry; its identity is a
content hash over array bytes + canonical metadata — **not** file bytes
(npz is a zip; zips embed timestamps and are not byte-stable).

Layout: ``{snapshots root}/asof=<date>/model_data_<hash12>.npz``, where the
root is ``NFP_SNAPSHOTS_URI`` (e.g. ``s3://alt-nfp/snapshots``) or the
local ``data/snapshots/`` fallback.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from nfp_lookups.paths import DATA_DIR, is_remote

from nfp_ingest.model_data import build_model_data

SCHEMA_VERSION = 1

#: model-data keys serialized into every snapshot (global arrays)
GLOBAL_ARRAY_KEYS = [
    "month_of_year", "year_of_obs", "era_idx",
    "g_ces_sa", "ces_sa_obs", "ces_sa_vintage_idx",
    "g_ces_nsa", "ces_nsa_obs", "ces_nsa_vintage_idx",
    "g_qcew", "qcew_obs", "qcew_is_m2", "qcew_noise_mult",
    "birth_rate", "bd_proxy", "bd_qcew_lagged",
]
SCALAR_KEYS = ["T", "n_years", "n_ces_vintages", "n_providers"]


def snapshots_location() -> Any:
    """Snapshot root: ``NFP_SNAPSHOTS_URI`` env, else local ``data/snapshots``.

    Returns ``Path`` or ``UPath`` (same env contract as the store; see
    ``nfp_lookups.paths``).
    """
    uri = os.environ.get("NFP_SNAPSHOTS_URI")
    if not uri:
        return DATA_DIR / "snapshots"

    from upath import UPath  # deferred: s3fs only needed in remote mode

    client_kwargs = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    return UPath(
        uri,
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs=client_kwargs,
    )


def collect_snapshot(data: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Split a model-data dict into (arrays, metadata) for serialization.

    Frames (``panel``, ``levels``) are excluded by design: the panel is
    golden-mastered and reproducible from the store; the model consumes
    arrays.
    """
    arrays: dict[str, np.ndarray] = {}
    for k in GLOBAL_ARRAY_KEYS:
        arrays[k] = np.asarray(data[k])
    cyclical_present = []
    cyclical_none = []
    for k in sorted(data):
        if not k.endswith("_c"):
            continue
        if data[k] is None:
            cyclical_none.append(k)
        else:
            cyclical_present.append(k)
            arrays[k] = np.asarray(data[k])
    providers_meta = []
    for pp in data["pp_data"]:
        name = pp["name"]
        arrays[f"{name}__g_pp"] = np.asarray(pp["g_pp"])
        arrays[f"{name}__pp_obs"] = np.asarray(pp["pp_obs"])
        has_births = pp["births"] is not None
        if has_births:
            arrays[f"{name}__births"] = np.asarray(pp["births"])
            arrays[f"{name}__births_obs"] = np.asarray(pp["births_obs"])
        providers_meta.append(
            {"name": name, "emp_col": pp["emp_col"], "has_births": has_births}
        )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "scalars": {k: int(data[k]) for k in SCALAR_KEYS},
        "dates": [d.isoformat() for d in data["dates"]],
        "ces_vintage_map": {str(k): v for k, v in data["ces_vintage_map"].items()},
        "providers": providers_meta,
        "cyclical_present": cyclical_present,
        "cyclical_none": cyclical_none,
    }
    return arrays, meta


def content_hash(arrays: dict[str, np.ndarray], meta: dict) -> str:
    """sha256 over canonical array bytes + canonical metadata JSON.

    Deterministic across processes and platforms of the same endianness;
    independent of file encoding (never hash the npz bytes — zip headers
    embed timestamps).
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def save_snapshot(arrays: dict[str, np.ndarray], meta: dict, path) -> str:
    """Write a snapshot npz (arrays + embedded metadata JSON) to *path*.

    *path* may be a local ``Path`` or a ``UPath``. Returns the content hash
    (also embedded in the metadata as ``content_hash``).
    """
    digest = content_hash(arrays, meta)
    meta_out = {**meta, "content_hash": digest}
    buf = io.BytesIO()
    np.savez(buf, __meta__=np.frombuffer(
        json.dumps(meta_out, sort_keys=True).encode(), dtype=np.uint8
    ), **arrays)
    if not is_remote(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(buf.getvalue())
    return digest


def load_snapshot(path) -> tuple[dict[str, np.ndarray], dict]:
    """Read a snapshot npz back into (arrays, metadata).

    Verifies the embedded ``content_hash`` against the loaded arrays and
    raises ``ValueError`` on mismatch (bit rot / partial upload).
    """
    with path.open("rb") as f:
        npz = np.load(io.BytesIO(f.read()), allow_pickle=False)
        loaded = {k: npz[k] for k in npz.files}
    meta = json.loads(bytes(loaded.pop("__meta__")).decode())
    embedded = meta.pop("content_hash")
    digest = content_hash(loaded, meta)
    if digest != embedded:
        raise ValueError(
            f"snapshot content hash mismatch: embedded {embedded[:12]}…, "
            f"recomputed {digest[:12]}…"
        )
    meta["content_hash"] = embedded
    return loaded, meta


def snapshot_model_data(
    as_of: date,
    *,
    out_root=None,
    **build_kwargs,
) -> tuple[Any, str]:
    """Build model data for *as_of* and write the snapshot artifact.

    Returns ``(path, content_hash)``. Path layout:
    ``{out_root}/asof=<date>/model_data_<hash12>.npz``.
    """
    if out_root is None:
        out_root = snapshots_location()
    data = build_model_data(as_of, **build_kwargs)
    arrays, meta = collect_snapshot(data)
    meta["as_of"] = as_of.isoformat()
    digest = content_hash(arrays, meta)
    path = out_root / f"asof={as_of.isoformat()}" / f"model_data_{digest[:12]}.npz"
    save_snapshot(arrays, meta, path)
    return path, digest


__all__ = [
    "GLOBAL_ARRAY_KEYS",
    "SCALAR_KEYS",
    "SCHEMA_VERSION",
    "collect_snapshot",
    "content_hash",
    "load_snapshot",
    "save_snapshot",
    "snapshot_model_data",
    "snapshots_location",
]
