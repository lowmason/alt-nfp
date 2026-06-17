# A1/A2 goldens re-baseline (plans/10 T7) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` for the codeable task (T1, the generator); the controller runs the credentialed steps (T2–T4: read the rebuilt store, stage to S3, verify) since they need `.env` + the scratch prefix. Steps use `- [ ]` checkboxes.

**Parent:** [`plans/10`](10-store_rebuild.md) **T7** (re-baseline goldens) — unblocked by `plans/11`. **Next after this:** `plans/10` **T8** (promotion, maintainer GO).

**Goal:** Re-baseline the A1 (censored-panel) and A2 (`build_model_data`) golden fixtures from **this repo + the rebuilt store** (`s3://alt-nfp/store-rebuild`), staged to a **scratch** golden prefix (never the frozen `s3://alt-nfp/golden/a1|a2`), with the committed manifests rewritten and the divergences documented — so the A1/A2 tests pass against the rebuilt store ahead of T8 promotion.

**Architecture:** The existing generators (`scripts/generate_golden_masters.py`, `scripts/generate_a2_golden.py`) run under the **old repo's** venv against the canonical store — they are the *frozen-reference* baseline. This plan adds ONE current-repo generator that calls the same functions the A1/A2 tests already call (`nfp_ingest.panel.build_panel`, `nfp_ingest.payroll.ingest_provider`, `nfp_ingest.model_data.build_model_data`) against the rebuilt store, writes fixtures + manifests to a local staging dir, then the controller uploads them to the scratch prefix. The tests read `START_YEAR`/`END_YEAR` and per-fixture metadata from the **committed** manifest and the fixtures from `NFP_GOLDEN_URI`/`NFP_GOLDEN_A2_URI`, so they self-adapt once the manifests are rewritten and the env points at the scratch prefix.

**Tech stack:** Python 3.12, Polars, NumPy, uv workspace; S3 via `upath`/`s3fs`.

---

## POLICY / safety

1. **Never touch the frozen reference.** Generation writes a LOCAL staging dir; the upload targets `s3://alt-nfp/golden/a1-rebuild` + `…/a2-rebuild` ONLY. The frozen `…/golden/a1|a2` is read-never-written.
2. **Read-only against the store.** The generator READS `s3://alt-nfp/store-rebuild`; it never writes the store. The only S3 writes are the scratch golden fixtures.
3. **`.env` before `nfp_*` import.** `nfp_lookups.paths` reads the env at import time, so the generator must `load_dotenv()` before importing any `nfp_*` module; the runner sets `NFP_STORE_URI=s3://alt-nfp/store-rebuild` (which `load_dotenv(override=False)` preserves).
4. **Branch, not main.** All work on a `goldens-rebaseline` branch. The A1/A2 tests are `@pytest.mark.real_store` + self-skip without store env, so rewriting the committed manifests does NOT affect main's CI (which skips them); the rebuilt manifests + scratch fixtures promote *together* at T8.

---

## The 9 as-of dates (all valid against the 2017+ store)

`2020-05-12, 2023-07-12, 2024-09-12, 2024-12-12, 2025-02-12, 2025-03-12, 2025-07-12, 2025-11-12, 2026-01-12` — all ≥ 2020, so no date-trimming. Expected-failure date `2026-02-12` ("ref_date gap") is re-verified in T1/T3. `start_year` changes **2012 → 2017** (the rebuilt store's coverage); `end_year` stays 2026.

---

## T1 — Current-repo generator (`scripts/regen_golden_rebuild.py`) `[depends: none]`

**Files:**
- Create: `scripts/regen_golden_rebuild.py`

Mirror the two existing generators but for the current repo + rebuilt store. ONE script, two sections (A1 + A2), writing to a local staging dir passed as `argv[1]`.

- [ ] **Step 1 — write the generator.** Structure (adapt the existing two scripts; full detail below):

```python
"""Re-baseline A1/A2 goldens from THIS repo + the rebuilt store (plans/12 / plans/10 T7).

Usage (controller, with .env creds; reads the SCRATCH store, writes LOCAL staging):
    NFP_STORE_URI=s3://alt-nfp/store-rebuild uv run python scripts/regen_golden_rebuild.py data/golden_rebuild_staging
"""
import hashlib, json, os, subprocess, sys
from datetime import date
from pathlib import Path

# --- .env BEFORE any nfp_* import (paths reads env at import time) ---
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("NFP_STORE_URI", "s3://alt-nfp/store-rebuild")  # default to scratch

AS_OF_DATES = [date(2020,5,12), date(2023,7,12), date(2024,9,12), date(2024,12,12),
               date(2025,2,12), date(2025,3,12), date(2025,7,12), date(2025,11,12), date(2026,1,12)]
EXPECTED_FAILURE = (date(2026,2,12), "ref_date gap")
START_YEAR, END_YEAR = 2017, 2026
# v3 build_model_data omits these (posterior-neutral); do NOT emit them (the A2 test's _DROPPED set).
GLOBAL_ARRAYS = ["month_of_year","year_of_obs","era_idx","g_ces_sa","ces_sa_obs","ces_sa_vintage_idx",
                 "g_ces_nsa","ces_nsa_obs","ces_nsa_vintage_idx","g_qcew","qcew_obs","qcew_is_m2","qcew_noise_mult"]
SCALARS = ["T","n_years","n_ces_vintages","n_providers"]

def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=True)
    import numpy as np, polars as pl
    from nfp_ingest.panel import build_panel
    from nfp_ingest.payroll import ingest_provider
    from nfp_ingest.model_data import build_model_data
    from nfp_lookups.provider_config import PROVIDERS_DEFAULT
    from nfp_lookups.paths import VINTAGE_STORE_PATH

    prov = {"generator": "scripts/regen_golden_rebuild.py", "generated_on": date.today().isoformat(),
            "store_uri": str(VINTAGE_STORE_PATH),
            "repo_commit": subprocess.run(["git","rev-parse","HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
            "polars_version": pl.__version__, "numpy_version": np.__version__,
            "start_year": START_YEAR, "end_year": END_YEAR, "providers": [p.name for p in PROVIDERS_DEFAULT],
            "divergence": "rebuilt store (s3://alt-nfp/store-rebuild): 2017+ history (vs frozen-ref 2012+), "
                          "QCEW reconstructed crosswalk values, ownership/00-anchor/NSA store schema normalized "
                          "to PANEL_SCHEMA by transform_to_panel (panel columns unchanged; values + row counts differ)."}

    # ---- A1: censored panels + provider + expected-failure ----
    a1 = {}
    for d in AS_OF_DATES:
        panel = build_panel(providers=[], start_year=START_YEAR, end_year=END_YEAR, as_of_ref=d)
        fn = f"panel_asof_{d.isoformat()}.parquet"; panel.write_parquet(out/fn)
        a1[fn] = {"kind":"censored_panel","as_of_ref":d.isoformat(),"rows":panel.height,
                  "columns":panel.columns,"sha256":_sha256(out/fn)}
    for cfg in PROVIDERS_DEFAULT:
        df = ingest_provider(cfg); fn = f"provider_{cfg.name}.parquet"; df.write_parquet(out/fn)
        a1[fn] = {"kind":"provider_panel","rows":df.height,"columns":df.columns,
                  "sha256":_sha256(out/fn),
                  "provider_config":{k:getattr(cfg,k) for k in ("name","file","error_model","birth_file",
                     "industry_type","industry_code","geographic_type","geographic_code") if hasattr(cfg,k)}}
    # expected-failure: must raise
    ef_date, ef_msg = EXPECTED_FAILURE
    try:
        build_panel(providers=[], start_year=START_YEAR, end_year=END_YEAR, as_of_ref=ef_date)
        raise SystemExit(f"FAIL: expected ValueError({ef_msg!r}) at {ef_date}, but build_panel succeeded")
    except ValueError as e:
        assert ef_msg in str(e), f"expected-failure message changed: {e!r}"
    a1_manifest = {"provenance": prov, "fixtures": a1,
                   "expected_failures": [{"as_of_ref": ef_date.isoformat(), "error_contains": ef_msg}]}
    (out/"a1_manifest.json").write_text(json.dumps(a1_manifest, indent=2)+"\n")

    # ---- A2: build_model_data arrays + levels/panel frames ----
    a2 = {}
    for d in AS_OF_DATES:
        data = build_model_data(d, providers=list(PROVIDERS_DEFAULT), start_year=START_YEAR, end_year=END_YEAR)
        cyc = sorted(k for k in data if k.endswith("_c"))
        arrays = {k: np.asarray(data[k]) for k in GLOBAL_ARRAYS}
        for k in cyc:
            if data[k] is not None: arrays[k] = np.asarray(data[k])
        pp_meta = []
        for pp in data["pp_data"]:
            n = pp["name"]; arrays[f"{n}__g_pp"]=np.asarray(pp["g_pp"]); arrays[f"{n}__pp_obs"]=np.asarray(pp["pp_obs"])
            hb = pp["births"] is not None
            if hb: arrays[f"{n}__births"]=np.asarray(pp["births"]); arrays[f"{n}__births_obs"]=np.asarray(pp["births_obs"])
            pp_meta.append({"name":n,"emp_col":pp["emp_col"],"has_births":hb})
        stem = f"asof_{d.isoformat()}"
        npz = out/f"model_data_{stem}.npz"; np.savez(npz, **arrays)
        data["levels"].write_parquet(out/f"levels_{stem}.parquet")
        data["panel"].write_parquet(out/f"panel_{stem}.parquet")
        a2[stem] = {"as_of_ref":d.isoformat(), "scalars":{k:int(data[k]) for k in SCALARS},
                    "dates_first":data["dates"][0].isoformat(), "dates_last":data["dates"][-1].isoformat(),
                    "ces_vintage_map":{str(k):v for k,v in data["ces_vintage_map"].items()},
                    "cyclical_present":[k for k in cyc if data[k] is not None],
                    "cyclical_none":[k for k in cyc if data[k] is None],
                    "providers":pp_meta, "array_names":sorted(arrays), "panel_rows":data["panel"].height,
                    "sha256_npz":_sha256(npz), "sha256_levels":_sha256(out/f"levels_{stem}.parquet"),
                    "sha256_panel":_sha256(out/f"panel_{stem}.parquet")}
        print(f"{stem}: T={data['T']}, {len(arrays)} arrays, panel {data['panel'].height:,} rows")
    (out/"a2_manifest.json").write_text(json.dumps({"provenance":prov,"fixtures":a2}, indent=2)+"\n")
    print(f"\nWrote A1 ({len(a1)}) + A2 ({len(a2)}) fixtures + manifests to {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2 — `ruff` clean.** `uv run ruff check scripts/regen_golden_rebuild.py`
- [ ] **Acceptance (code review):** the generator imports ONLY current-repo modules, loads `.env` before any `nfp_*` import, never writes the store, drops the 3 BD arrays, and the provider-config dict matches `ProviderConfig`'s fields (cross-check against `nfp_lookups.provider_config.ProviderConfig`). It does NOT upload to S3 (that's T2). `ruff` clean.

---

## T2 — Generate + stage to the scratch prefix (controller-run) `[depends: T1]`

**Files:** none (run + upload). Reads rebuilt store + local providers/indicators; writes LOCAL staging then SCRATCH S3.

- [ ] **Step 1 — generate locally.** `NFP_STORE_URI=s3://alt-nfp/store-rebuild uv run python scripts/regen_golden_rebuild.py data/golden_rebuild_staging` (gitignored `data/`). Confirm: 9 A1 panels + provider_G + a1_manifest.json + 9 A2 npz + 9 levels + 9 panel + a2_manifest.json; the expected-failure assertion passed (no SystemExit); A2 row counts < the frozen-reference manifest's (2017+ truncation).
- [ ] **Step 2 — sanity vs frozen manifests.** Diff the new `a1_manifest.json`/`a2_manifest.json` against the committed ones: same fixture KEYS (9 panels + provider + 9 A2 stems), `start_year` now 2017, `array_names` LACK `birth_rate`/`bd_proxy`/`bd_qcew_lagged`, panel columns UNCHANGED (PANEL_SCHEMA), row counts smaller. Record the row-count deltas.
- [ ] **Step 3 — upload to the scratch prefix.** Upload `data/golden_rebuild_staging/*` to `s3://alt-nfp/golden/a1-rebuild/` (A1 parquets + a1_manifest.json) and `s3://alt-nfp/golden/a2-rebuild/` (A2 npz/levels/panel + a2_manifest.json), via a `UPath` copy loop (mirror the test's `_golden_root()` UPath construction; `.env` creds). **Never** `…/golden/a1|a2`.
- [ ] **Step 4 — install the committed manifests.** Copy the staged `a1_manifest.json`/`a2_manifest.json` over `packages/nfp-ingest/tests/golden/a1_manifest.json` / `a2_manifest.json` (these ARE committed; the tests read them). Commit: `chore(golden): re-baseline A1/A2 manifests to the rebuilt store (2017+, T7)`.
- [ ] **Acceptance:** scratch prefix populated; committed manifests updated (start_year=2017, BD arrays gone, smaller rows); frozen `…/golden/a1|a2` untouched.

---

## T3 — Verify A1/A2 tests green against the staged fixtures (controller-run) `[depends: T2]`

**Files:** none (run + record).

- [ ] **Step 1 — run A1.** `NFP_STORE_URI=s3://alt-nfp/store-rebuild NFP_GOLDEN_URI=s3://alt-nfp/golden/a1-rebuild uv run pytest packages/nfp-ingest/tests/test_golden_masters.py -m real_store --no-cov -v`. Expect: 9 censored-panel + provider_G + the `2026-02-12` expected-failure all PASS.
- [ ] **Step 2 — run A2.** `NFP_STORE_URI=s3://alt-nfp/store-rebuild NFP_GOLDEN_A2_URI=s3://alt-nfp/golden/a2-rebuild uv run pytest packages/nfp-ingest/tests/test_model_data_golden.py -m real_store --no-cov -v`. Expect: 9 fixtures PASS (arrays NaN-exact, frames identical).
- [ ] **Step 3 — if any fail:** a mismatch means the generator's build path diverged from the test's (or a non-determinism). Diff the failing fixture's actual vs staged; fix the generator (T1) or the cause; re-stage (T2) the affected fixtures; re-run. Do NOT loosen the tests.
- [ ] **Acceptance:** A1 + A2 suites green against the rebuilt store + scratch goldens; record the pass counts + the row-count deltas vs frozen reference.

---

## T4 — Document divergences + close T7 `[depends: T3]`

**Files:**
- Modify: `plans/10-store_rebuild.md` (T7 → DONE)
- Modify: `plans/12-goldens_rebaseline.md` (this file — record results)

- [ ] **Step 1 — record divergences.** The manifests' `provenance.divergence` already states them; add the measured row-count deltas (frozen vs rebuilt) here under T4.
- [ ] **Step 2 — close plans/10 T7.** Flip the T7 checklist item ("regenerate A1/A2 fixtures…") to `[x]` with the result (staged to `…/a1-rebuild`+`…/a2-rebuild`, manifests committed, suites green). Note T8 promotes the scratch goldens → canonical `…/golden/a1|a2` (copy) **alongside** the store cutover, and flips the test env defaults.
- [ ] **Step 3 — PR (held for T8).** Open a PR for the `goldens-rebaseline` branch (manifest changes + generator). It does NOT merge until T8 (promotion), so main's frozen-reference goldens stay authoritative until cutover. State this in the PR.
- [ ] **Acceptance:** T7 closed in plans/10; divergences documented; branch PR open and explicitly gated on T8.

---

## T8 hand-off (NOT this plan)

Promotion (plans/10 T8, maintainer GO) copies `s3://alt-nfp/store-rebuild`→`…/store` and `…/golden/a1-rebuild|a2-rebuild`→`…/golden/a1|a2` (the `--allow-canonical` path / `scripts/mirror_store.py`), snapshots the prior canonical first, merges this branch, and confirms a post-cutover read reproduces the §10 gates + A1/A2 suites. Out of scope here.
