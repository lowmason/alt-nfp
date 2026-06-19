import os

import numpy as np
import pytest
from dotenv import load_dotenv

pytestmark = [pytest.mark.slow, pytest.mark.real_store]


@pytest.mark.skipif(not os.environ.get("NFP_STORE_URI"), reason="needs real store")
def test_wedge_fit_converges_on_clean_window():
    load_dotenv(".env")
    from datetime import date

    from nfp_ingest.wedge_data import build_wedge_model_data
    from nfp_model.wedge import fit_wedge

    data = build_wedge_model_data(as_of=None, target_month=date(2026, 1, 1))
    fit = fit_wedge(data, settings="default", seed=0)
    # Build gate: convergence, not accuracy.
    assert fit.num_divergences == 0
    assert np.isfinite(fit.posterior["mu"]).all()
