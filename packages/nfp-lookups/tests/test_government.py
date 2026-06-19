from datetime import date

import numpy as np
from nfp_lookups.government import (
    KNOWN_INTERVENTIONS,
    GovIntervention,
    get_known_interventions_as_of,
    intervention_column,
)

REF = [date(2025, m, 1) for m in range(1, 7)]  # Jan..Jun 2025

def test_as_of_filters_on_announcement_date():
    from datetime import timedelta
    rif = next(i for i in KNOWN_INTERVENTIONS if i.name == "federal_rif_2025")
    assert rif not in get_known_interventions_as_of(rif.announcement_date - timedelta(days=1))
    assert rif in get_known_interventions_as_of(rif.announcement_date)

def test_pulse_is_permanent_level_shift():
    iv = GovIntervention(date(2025, 3, 1), "rif", "pulse", -50.0, 25.0,
                         date(2025, 2, 11), "u")
    col = intervention_column(iv, REF)
    assert col.tolist() == [0, 0, 1, 0, 0, 0]          # one-month change
    assert np.isclose(np.cumsum(col)[-1], 1.0)          # level steps and STAYS

def test_box_is_phased_ramp():
    iv = GovIntervention(date(2025, 3, 1), "rif", "box", -60.0, 30.0,
                         date(2025, 2, 11), "u", box_months=3)
    col = intervention_column(iv, REF)
    assert np.allclose(col, [0, 0, 1/3, 1/3, 1/3, 0])
    assert np.isclose(np.cumsum(col)[-1], 1.0)          # total ramp = 1 unit

def test_tc_peaks_then_decays_back_toward_zero():
    iv = GovIntervention(date(2025, 3, 1), "census", "tc", 400.0, 50.0,
                         date(2025, 1, 1), "u", tc_decay=0.5)
    col = intervention_column(iv, REF)
    assert col[2] == 1.0 and col[3] < 0                 # +peak then giveback
    assert np.cumsum(col)[-1] < 1.0                     # decays back down

def test_missing_ref_month_is_all_zero():
    iv = GovIntervention(date(2099, 1, 1), "future", "pulse", 1.0, 1.0,
                         date(2099, 1, 1), "u")
    assert np.all(intervention_column(iv, REF) == 0)
