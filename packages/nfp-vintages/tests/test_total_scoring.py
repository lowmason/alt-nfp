from datetime import date

import numpy as np
import polars as pl
from nfp_vintages.assembly import score_total
from nfp_vintages.competitors.consensus import Consensus, load_consensus


def test_absent_consensus_renders_none(tmp_path):
    assert load_consensus(tmp_path / "missing.parquet") is None
    c = Consensus(None)
    assert c.predict(date(2025, 4, 1), as_of=date(2025, 5, 1)) is None
    row = score_total(np.full(500, 100.0), first_print_k=110.0, consensus_k=None)
    assert row["consensus_err"] is None                 # column renders "—"


def test_populated_consensus_scores(tmp_path):
    p = tmp_path / "consensus_populated.parquet"
    pl.DataFrame({
        "ref_month": [date(2025, 4, 1)], "consensus_median_change_k": [150.0],
        "survey_date": [date(2025, 5, 1)], "release_date": [date(2025, 5, 2)],
        "source": ["synthetic"],
    }).write_parquet(p)
    c = Consensus(load_consensus(p))
    cons = c.predict(date(2025, 4, 1), as_of=date(2025, 5, 1))
    assert cons == 150.0
    row = score_total(np.full(500, 120.0), first_print_k=110.0, consensus_k=cons)
    assert np.isclose(row["point_err"], 10.0)           # |120 - 110|
    assert np.isclose(row["consensus_err"], 40.0)       # |150 - 110|
    assert 0.0 <= row["cover80"] <= 1.0
