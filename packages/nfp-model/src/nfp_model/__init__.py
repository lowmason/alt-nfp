"""JAX/NumPyro state-space model for NFP nowcasting.

Inference layer only: ModelData arrays in, posterior out. Imports nothing
from the data packages — inputs come as ``build_model_data`` dicts or
snapshot ``(arrays, meta)`` pairs (see :mod:`nfp_model.data`).

Importing this package enables JAX float64 globally: the model's parity
contract with the PyMC reference is defined in double precision.
"""

import numpyro as _numpyro

from .config import PRESETS, ModelPriors, SamplerSettings
from .data import from_snapshot, model_inputs
from .model import DETERMINISTIC_SITES, nfp_model
from .nowcast import ces_sa_predictive, nowcast_summary
from .sampling import FitResult, fit_model

_numpyro.enable_x64()

__all__ = [
    "DETERMINISTIC_SITES",
    "PRESETS",
    "FitResult",
    "ModelPriors",
    "SamplerSettings",
    "ces_sa_predictive",
    "fit_model",
    "from_snapshot",
    "model_inputs",
    "nfp_model",
    "nowcast_summary",
]
