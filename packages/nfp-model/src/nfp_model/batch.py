"""Batched (vmapped) fitting across as-of dates — the A4 evaluation harness.

``vmap`` needs every batch element to share shapes, so per-date model
inputs are padded to the per-key maximum and padded likelihood slots are
masked out (``numpyro.handlers.mask`` — exactly zero log-prob). Padded
latent timesteps still sample their non-centered z's, which are prior-only
N(0,1) dimensions touching no likelihood: the posterior over every shared
parameter and the real-T latent path is exactly the per-date posterior.

The split that makes tracing work:

- **static** (closed over, concrete): T = T_max, calendar arrays
  (``month_of_year`` / ``year_of_obs`` / ``era_idx`` — identical across
  dates on the overlap because every date starts at the same calendar
  month), vintage-bucket count, the cyclical gating decision, provider
  names/error models.
- **batched** (leading date axis, traced under vmap): padded observation
  values/indices/multipliers, masks, cyclical covariate paths, ``c_idx``.

``fit_model_batch`` vmaps a whole ``MCMC.run`` (traceable with the
progress bar off) with vectorized inner chains, and reduces each date's
posterior *in graph* to the A3 fixture schema — scalar draws, path
mean/SD, nowcast predictive draws at ``c_idx`` — so the high-dimensional
noise sites never materialize batch-wide.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from .config import PRESETS, ModelPriors, SamplerSettings
from .model import DETERMINISTIC_SITES, nfp_model
from .parity import PATH_VARS

#: high-dimensional non-centered noise sites dropped from batched results
NOISE_SITES = ("eps_g", "xi_bd", "fourier_z")

#: per-date scalar outputs that live in meta, not in the arrays dict
_SCALAR_OUTPUTS = (
    "nowcast_growth", "nowcast_index", "nowcast_change_k", "num_divergences",
)


@dataclass(frozen=True)
class BatchedInputs:
    """Padded model inputs for a date batch (see module docstring)."""

    static: dict
    batched: dict
    pp_names: tuple[str, ...]
    pp_error_models: tuple[str, ...]
    T_real: np.ndarray
    c_idx: np.ndarray

    @property
    def n_dates(self) -> int:
        return len(self.T_real)


def _pad_to(arr: np.ndarray, length: int, fill) -> np.ndarray:
    out = np.full((length,), fill, dtype=arr.dtype)
    out[: len(arr)] = arr
    return out


def _mask_for(n_real: int, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    mask[:n_real] = True
    return mask


def active_cyclicals(d: dict, priors: ModelPriors) -> tuple[str, ...]:
    """The model's data-driven gating, evaluated on concrete arrays."""
    return tuple(
        name for name in priors.indicator_names
        if d.get(f"{name}_c") is not None
        and np.any(np.asarray(d[f"{name}_c"]) != 0.0)
    )


def pad_model_inputs(
    inputs: list[dict],
    *,
    priors: ModelPriors | None = None,
    c_idx: list[int] | None = None,
) -> BatchedInputs:
    """Pad a list of ``model_inputs`` dicts into one vmappable batch.

    Structure that cannot be masked must be uniform across the batch and is
    asserted: cyclical gating (changes ``phi_3``'s dimension), the provider
    name/error-model sequence, era handling, and calendar agreement on the
    overlap. ``c_idx`` defaults to the last real timestep per date (the
    nowcast proxy convention).
    """
    if not inputs:
        raise ValueError("empty batch")
    p = priors if priors is not None else ModelPriors()
    n = len(inputs)

    T_real = np.array([int(d["T"]) for d in inputs])
    T_max = int(T_real.max())
    longest = inputs[int(np.argmax(T_real))]

    # --- uniform-structure assertions -------------------------------------
    active = active_cyclicals(inputs[0], p)
    for i, d in enumerate(inputs):
        if active_cyclicals(d, p) != active:
            raise ValueError(
                f"non-uniform cyclical gating in batch: date {i} has "
                f"{active_cyclicals(d, p)}, date 0 has {active} — phi_3's "
                "dimension would differ; batch only dates with matching gating"
            )
    pp_sig = [tuple((pp["name"], pp["error_model"]) for pp in d["pp_data"]) for d in inputs]
    if any(sig != pp_sig[0] for sig in pp_sig):
        raise ValueError(f"non-uniform provider structure in batch: {pp_sig}")
    pp_names = tuple(name for name, _ in pp_sig[0])
    pp_error_models = tuple(em for _, em in pp_sig[0])
    if any("__" in name for name in pp_names):
        raise ValueError(f"provider names must not contain '__': {pp_names}")
    for i, d in enumerate(inputs):
        for pp in d["pp_data"]:
            if len(pp["pp_obs"]) == 0:
                raise ValueError(
                    f"date {i}: provider {pp['name']!r} has no observations — "
                    "its sites would be prior-only for that date; drop the "
                    "provider or the date"
                )
    has_era = [d.get("era_idx") is not None for d in inputs]
    if any(h != has_era[0] for h in has_era):
        raise ValueError("non-uniform era_idx presence in batch")

    # --- shared calendar (every date starts at the same month) ------------
    for key in ("month_of_year", "year_of_obs") + (("era_idx",) if has_era[0] else ()):
        master = np.asarray(longest[key])
        for i, d in enumerate(inputs):
            got = np.asarray(d[key])
            if not np.array_equal(got, master[: len(got)]):
                raise ValueError(
                    f"date {i}: {key} is not a prefix of the longest date's — "
                    "batch members must share the calendar start"
                )

    static: dict = {
        "T": T_max,
        "n_years": int(longest["n_years"]),
        "n_ces_vintages": int(max(d["n_ces_vintages"] for d in inputs)),
        "month_of_year": np.asarray(longest["month_of_year"]),
        "year_of_obs": np.asarray(longest["year_of_obs"]),
        "cyclical_active": active,
    }
    if has_era[0]:
        static["era_idx"] = np.asarray(longest["era_idx"])

    # --- padded batched arrays --------------------------------------------
    def stack(fn) -> np.ndarray:
        return np.stack([fn(d) for d in inputs])

    n_qcew = max(len(d["qcew_obs"]) for d in inputs)
    n_ces_sa = max(len(d["ces_sa_obs"]) for d in inputs)
    n_ces_nsa = max(len(d["ces_nsa_obs"]) for d in inputs)

    batched: dict = {
        "g_qcew": stack(lambda d: _pad_to(np.asarray(d["g_qcew"], dtype=float), T_max, 0.0)),
        "g_ces_sa": stack(lambda d: _pad_to(np.asarray(d["g_ces_sa"], dtype=float), T_max, 0.0)),
        "g_ces_nsa": stack(
            lambda d: _pad_to(np.asarray(d["g_ces_nsa"], dtype=float), T_max, 0.0)
        ),
        "qcew_obs": stack(
            lambda d: _pad_to(np.asarray(d["qcew_obs"], dtype=np.int64), n_qcew, 0)
        ),
        "qcew_is_m2": stack(
            lambda d: _pad_to(np.asarray(d["qcew_is_m2"], dtype=bool), n_qcew, True)
        ),
        "qcew_noise_mult": stack(
            lambda d: _pad_to(np.asarray(d["qcew_noise_mult"], dtype=float), n_qcew, 1.0)
        ),
        "qcew_mask": stack(lambda d: _mask_for(len(d["qcew_obs"]), n_qcew)),
        "ces_sa_obs": stack(
            lambda d: _pad_to(np.asarray(d["ces_sa_obs"], dtype=np.int64), n_ces_sa, 0)
        ),
        "ces_sa_vintage_idx": stack(
            lambda d: _pad_to(np.asarray(d["ces_sa_vintage_idx"], dtype=np.int64), n_ces_sa, 0)
        ),
        "ces_sa_mask": stack(lambda d: _mask_for(len(d["ces_sa_obs"]), n_ces_sa)),
        "ces_nsa_obs": stack(
            lambda d: _pad_to(np.asarray(d["ces_nsa_obs"], dtype=np.int64), n_ces_nsa, 0)
        ),
        "ces_nsa_vintage_idx": stack(
            lambda d: _pad_to(np.asarray(d["ces_nsa_vintage_idx"], dtype=np.int64), n_ces_nsa, 0)
        ),
        "ces_nsa_mask": stack(lambda d: _mask_for(len(d["ces_nsa_obs"]), n_ces_nsa)),
    }
    for name in active:
        batched[f"{name}_c"] = stack(
            lambda d, _n=name: _pad_to(np.asarray(d[f"{_n}_c"], dtype=float), T_max, 0.0)
        )
    for j, name in enumerate(pp_names):
        n_pp = max(len(d["pp_data"][j]["pp_obs"]) for d in inputs)
        batched[f"pp__{name}__g_pp"] = stack(
            lambda d, _j=j: _pad_to(np.asarray(d["pp_data"][_j]["g_pp"], dtype=float), T_max, 0.0)
        )
        batched[f"pp__{name}__pp_obs"] = stack(
            lambda d, _j=j, _n=n_pp: _pad_to(
                np.asarray(d["pp_data"][_j]["pp_obs"], dtype=np.int64), _n, 0
            )
        )
        batched[f"pp__{name}__mask"] = stack(
            lambda d, _j=j, _n=n_pp: _mask_for(len(d["pp_data"][_j]["pp_obs"]), _n)
        )

    if c_idx is None:
        c_arr = T_real - 1  # nowcast proxy: last real state
    else:
        c_arr = np.asarray(c_idx, dtype=np.int64)
        if c_arr.shape != (n,) or (c_arr >= T_real).any() or (c_arr < 0).any():
            raise ValueError("c_idx must be per-date indices into the real calendar")
    batched["c_idx"] = c_arr.astype(np.int64)

    return BatchedInputs(
        static=static,
        batched=batched,
        pp_names=pp_names,
        pp_error_models=pp_error_models,
        T_real=T_real,
        c_idx=c_arr,
    )


def assemble_data(bi: BatchedInputs, sl: dict) -> dict:
    """Merge static structure with one date's (possibly traced) slice."""
    data = dict(bi.static)
    pp_fields: dict[str, dict] = {name: {} for name in bi.pp_names}
    for k, v in sl.items():
        if k == "c_idx":
            continue
        if k.startswith("pp__"):
            _, pname, field_name = k.split("__", 2)
            pp_fields[pname][field_name] = v
        else:
            data[k] = v
    data["pp_data"] = [
        {"name": name, "error_model": em, **pp_fields[name]}
        for name, em in zip(bi.pp_names, bi.pp_error_models, strict=True)
    ]
    return data


@dataclass
class BatchFitResult:
    """Reduced per-date posteriors (leading axis = date) plus run metadata."""

    arrays: dict[str, np.ndarray]
    T_real: np.ndarray
    c_idx: np.ndarray
    pp_names: tuple[str, ...]
    settings: SamplerSettings
    seed: int
    wall_seconds: float
    priors: ModelPriors
    base_index: float
    idx_to_level: float

    @property
    def n_dates(self) -> int:
        return len(self.T_real)

    def date_arrays(self, i: int) -> tuple[dict[str, np.ndarray], dict]:
        """One date in the A3 fixture schema (paths sliced to real T)."""
        T = int(self.T_real[i])
        arrays: dict[str, np.ndarray] = {}
        for k, v in self.arrays.items():
            if k in _SCALAR_OUTPUTS:
                continue
            if k.startswith(("path_mean__", "path_sd__")) or k == "nowcast_pred_mean":
                arrays[k] = v[i][:T]
            else:  # draws__* and nowcast_pred_draws
                arrays[k] = v[i]
        phi3 = self.arrays.get("draws__phi_3")
        meta = {
            "T": T,
            "c_idx": int(self.c_idx[i]),
            "n_cyclical": 0 if phi3 is None else int(phi3.shape[-1]),
            "nowcast_growth": float(self.arrays["nowcast_growth"][i]),
            "nowcast_change_k": float(self.arrays["nowcast_change_k"][i]),
            "nowcast_index": float(self.arrays["nowcast_index"][i]),
            "num_divergences": int(self.arrays["num_divergences"][i]),
            "seed": self.seed,
            "wall_seconds": round(self.wall_seconds, 1),  # whole-batch wall
        }
        return arrays, meta


def _reduce_one(
    samples: dict, dets: dict, divergences, c_idx, *, base_index: float, idx_to_level: float
) -> dict:
    """In-graph reduction of one date's posterior to the fixture schema."""
    out: dict = {}
    for k, v in samples.items():
        if k not in NOISE_SITES:
            out[f"draws__{k}"] = v
    for var in PATH_VARS:
        out[f"path_mean__{var}"] = dets[var].mean(axis=(0, 1))
        out[f"path_sd__{var}"] = dets[var].std(axis=(0, 1))

    # nowcast arithmetic (jnp mirror of nowcast.nowcast_summary)
    alpha = samples["alpha_ces"]
    lam = samples["lambda_ces"]
    g_pred = alpha[:, :, None] + lam[:, :, None] * dets["g_total_sa"]
    pred_mean = g_pred.mean(axis=(0, 1))
    index_path = base_index * jnp.exp(jnp.cumsum(pred_mean))
    nowcast_index = jnp.take(index_path, c_idx)
    prev_index = jnp.where(
        c_idx > 0, jnp.take(index_path, jnp.maximum(c_idx - 1, 0)), base_index
    )
    out["nowcast_pred_mean"] = pred_mean
    out["nowcast_pred_draws"] = jnp.take(g_pred, c_idx, axis=2)
    out["nowcast_growth"] = jnp.take(pred_mean, c_idx)
    out["nowcast_index"] = nowcast_index
    out["nowcast_change_k"] = (nowcast_index - prev_index) * idx_to_level
    out["num_divergences"] = divergences
    return out


def fit_model_batch(
    bi: BatchedInputs,
    priors: ModelPriors | None = None,
    *,
    settings: SamplerSettings | str = "light",
    seed: int = 0,
    base_index: float,
    idx_to_level: float,
) -> BatchFitResult:
    """Fit every date in the batch with one vmapped NUTS program.

    Inner chains run vectorized (forced — sequential chains can't trace
    under vmap); warmup adaptation stays per-date, per-chain. Per-date RNG
    keys come from one split of ``seed``, so results are reproducible for
    a fixed batch composition.
    """
    if isinstance(settings, str):
        settings = PRESETS[settings]
    if settings.chain_method != "vectorized":
        settings = replace(settings, chain_method="vectorized")
    p = priors if priors is not None else ModelPriors()

    def _fit_one(key, sl: dict) -> dict:
        c_idx = sl["c_idx"]
        data = assemble_data(bi, sl)
        mcmc = MCMC(
            NUTS(
                nfp_model,
                target_accept_prob=settings.target_accept,
                max_tree_depth=settings.max_tree_depth,
                init_strategy=init_to_median(num_samples=15),
            ),
            num_warmup=settings.num_warmup,
            num_samples=settings.num_samples,
            num_chains=settings.num_chains,
            chain_method="vectorized",
            progress_bar=False,
        )
        mcmc.run(key, data=data, priors=p, extra_fields=("diverging",))
        samples = mcmc.get_samples(group_by_chain=True)
        flat = mcmc.get_samples()
        dets_flat = Predictive(
            nfp_model, posterior_samples=flat, return_sites=list(DETERMINISTIC_SITES)
        )(jax.random.PRNGKey(0), data=data, priors=p)
        nc, nd = settings.num_chains, settings.num_samples
        dets = {k: v.reshape(nc, nd, *v.shape[1:]) for k, v in dets_flat.items()}
        div = mcmc.get_extra_fields(group_by_chain=True)["diverging"].sum()
        return _reduce_one(
            samples, dets, div, c_idx, base_index=base_index, idx_to_level=idx_to_level
        )

    keys = jax.random.split(jax.random.PRNGKey(seed), bi.n_dates)
    batched_fit = jax.jit(jax.vmap(_fit_one))
    t0 = time.time()
    out = batched_fit(keys, bi.batched)
    out = jax.block_until_ready(out)
    wall = time.time() - t0

    return BatchFitResult(
        arrays={k: np.asarray(v) for k, v in out.items()},
        T_real=bi.T_real.copy(),
        c_idx=bi.c_idx.copy(),
        pp_names=bi.pp_names,
        settings=settings,
        seed=seed,
        wall_seconds=wall,
        priors=p,
        base_index=base_index,
        idx_to_level=idx_to_level,
    )
