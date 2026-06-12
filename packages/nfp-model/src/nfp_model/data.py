"""ModelData intake: normalize data-side dicts and snapshots for the model.

The model consumes a plain dict of numpy arrays. Two producers exist on the
data side (``nfp_ingest``), and this module makes both look identical
without importing either:

- ``build_model_data(as_of=D)`` dicts — provider configs are dataclasses,
  frames (``panel``/``levels``) and date lists ride along.
- snapshot ``(arrays, meta)`` pairs from ``load_snapshot`` — provider
  configs are plain dicts in the metadata, arrays carry ``{name}__``
  prefixes.

``model_inputs`` strips either form down to exactly what the model reads —
keeping polars frames and other non-array objects out of JAX tracing.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

#: global arrays the model reads (subset of the snapshot schema)
MODEL_ARRAY_KEYS = [
    "month_of_year", "year_of_obs", "era_idx",
    "g_ces_sa", "ces_sa_obs", "ces_sa_vintage_idx",
    "g_ces_nsa", "ces_nsa_obs", "ces_nsa_vintage_idx",
    "g_qcew", "qcew_obs", "qcew_is_m2", "qcew_noise_mult",
]
MODEL_SCALAR_KEYS = ["T", "n_years", "n_ces_vintages"]


def provider_field(config: Any, name: str, default: Any = None) -> Any:
    """Read a provider-config field from a dataclass or a plain mapping."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def model_inputs(data: dict) -> dict:
    """Reduce a model-data dict to the arrays/scalars the model consumes.

    Accepts the output of ``nfp_ingest.model_data.build_model_data`` or of
    :func:`from_snapshot`. Provider entries are normalized to
    ``{"name", "error_model", "g_pp", "pp_obs"}``.
    """
    out: dict = {k: int(data[k]) for k in MODEL_SCALAR_KEYS}
    for k in MODEL_ARRAY_KEYS:
        v = data.get(k)
        out[k] = None if v is None else np.asarray(v)
    for k in sorted(data):
        if k.endswith("_c"):
            out[k] = None if data[k] is None else np.asarray(data[k], dtype=float)
    out["pp_data"] = [
        {
            "name": str(provider_field(pp.get("config"), "name", pp["name"])),
            "error_model": str(provider_field(pp.get("config"), "error_model", "iid")),
            "g_pp": np.asarray(pp["g_pp"], dtype=float),
            "pp_obs": np.asarray(pp["pp_obs"], dtype=int),
        }
        for pp in data["pp_data"]
    ]
    return out


def from_snapshot(arrays: dict[str, np.ndarray], meta: dict) -> dict:
    """Rebuild a model-data-shaped dict from snapshot ``(arrays, meta)``.

    The inverse of ``nfp_ingest.snapshots.collect_snapshot`` for everything
    the model layer needs (frames are not in snapshots by design). Provider
    ``error_model`` defaults to ``"iid"`` for schema-v1 snapshots that
    predate the field.
    """
    data: dict = {k: arrays[k] for k in arrays if "__" not in k}
    data.update({k: int(v) for k, v in meta["scalars"].items()})
    data["dates"] = [date.fromisoformat(d) for d in meta["dates"]]
    data["ces_vintage_map"] = {int(k): v for k, v in meta["ces_vintage_map"].items()}
    pp_data = []
    for pm in meta["providers"]:
        name = pm["name"]
        entry: dict = {
            "name": name,
            "config": {"name": name, "error_model": pm.get("error_model", "iid")},
            "emp_col": pm["emp_col"],
            "g_pp": arrays[f"{name}__g_pp"],
            "pp_obs": arrays[f"{name}__pp_obs"],
            "births": arrays.get(f"{name}__births"),
            "births_obs": arrays.get(f"{name}__births_obs"),
        }
        pp_data.append(entry)
    data["pp_data"] = pp_data
    for k in meta.get("cyclical_none", []):
        data.setdefault(k, None)
    return data
