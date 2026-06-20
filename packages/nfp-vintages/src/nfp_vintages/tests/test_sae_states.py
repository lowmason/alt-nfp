"""plans/15 Task 10: SAE scratch paths must live under the system tempdir, not data/.

SAE is disabled (``combine.py`` no longer reads ``sae_revisions.parquet`` — see the
commented-out line there), so both the checkpoint and the revisions output are pure
rebuild scratch. On Bloomberg's small-footprint container nothing may write under
``./data``, so these defaults must resolve under ``tempfile.gettempdir()``.
"""

from __future__ import annotations

import tempfile

from nfp_lookups.paths import INTERMEDIATE_DIR
from nfp_vintages.processing import sae_states


def test_sae_scratch_paths_under_tempdir_not_data():
    tmp = tempfile.gettempdir()
    for path in (sae_states.CHECKPOINT_PATH, sae_states.OUTPUT_PATH):
        assert str(path).startswith(tmp), f"{path} is not under the system tempdir"
        assert not str(path).startswith(str(INTERMEDIATE_DIR)), (
            f"{path} still resolves under the data/ intermediate dir"
        )
