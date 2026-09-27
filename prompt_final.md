# Handoff for the final push to LB 0.991 (27 Sep 2026, written ~12:50 IST) — for a new Claude Code thread

Deadline **27 Sep 2026 23:59 IST**; up to 5 uploads per day, **~2 left today** (used today: 0.980, probe 0.848,
0.979 — confirm with the user). Only the user uploads (one login). Context: [prompt.md](prompt.md) (setup, rules,
E6), [experiments.md](experiments.md) (E2–E11 evidence). Branch `e6-sub02-generator-features`.

## 1. Leaderboard history and what it proved

| Submission (subs/…) | What | Fold-0 F0.5 (India+US) | Public LB |
|---|---|---|---|
| sub01_baseline | TF-IDF + LightGBM | 0.9487 | 0.938 |
| sub02_baseline_gen | + E6 generator features | 0.9648 | 0.957 |
| **sub_v5_noce** (best) | TF-IDF ∪ dense e5 candidates, v5 stage 1/2, E6+E7c+E9 features | **0.98794** | **0.980** |
| probe_france_empty | sub_v5_noce with every France row empty | – | 0.848 |
| sub_v5_fr1 | + France-only vocabulary / address canonicalization (E15) | same | 0.979 |

**Root cause of the offline → LB gap (measured):** LB = 0.85·F(India+US) + 0.15·F(France) (France = 14.98% of
test S1). The France-empty probe gives F(France) ≈ (0.980−0.848)/0.1498 + singleton share (~0.055) ≈ **0.936**, and
then **F(India+US) on test ≈ 0.9878 = offline**. India/US have NO generalization gap; **the whole gap is France**
(unseen in training). The E14 orphan-rate simulator explains only ~0.002 (not the cause).

**France direction (measured):** E15 made France accept MORE pairs (3.416 → 3.549 matches/S1) and LB fell
0.980 → 0.979 ⇒ France's marginal predictions are net false positives: the model is **over-confident on France**.
Expected true matches/S1 ≈ 3.46 (train invariant; India/US test obey it: records/S1 × (1 − orphan share) = 3.46).

**Target arithmetic:** 0.85·0.992 + 0.15·0.975 = 0.9895; 0.85·0.993 + 0.15·0.985 = 0.9918. Need BOTH better
India/US (offline) and France.

## 2. Current state of the files (all on the laptop)

- `artifacts/union/{train,test}/*.parquet` — union candidates + 66 v5 features (`ber.union.FEATURES_V5`).
  `test/France.parquet` is the ORIGINAL (0.980) version again; `France.parquet.orig` = same; the E15 French-feature
  code is still in `ber.features` (country_vocab / fr_canon_addr) — **rebuilding the France union now would apply
  it** (it made LB worse): disable it in `country_vocab` before any France rebuild.
- `artifacts/v5/` — the 0.980 run (stage1 models: 3 folds, 12M rows = 34% of folds-0-2 entities; stage2_noce;
  `p1_test.npy`, `p2_test_noce.npy`, `kept_test_noce.npy`, `keep_test_noce.npy` = the 0.980 test state; `.orig` copies).
  Config: `artifacts/free/pipeline.yaml` (BER_CONFIG for v5 commands on this run).
- **Ready, not uploaded:** `subs/sub_v5_fr_strict060/` = 0.980 predictions with **France-only stage-2 odds ×0.6**
  (France matches −1.9%, India/US bit-identical). The logical next LB probe: if LB > 0.980, France strictness helps;
  sizes for other strengths on 0.980 p2: ×0.8 −0.84%, ×0.45 −2.9%, ×0.3 −4.2%, ×0.2 −5.6% (code in this doc's §4).
- Running when this was written: stage-2 A/B `stage2 --tag big` (255 leaves, lr 0.03, min leaf 100) on the 0.980
  stage 1; result via `python experiments/compare_runs.py artifacts/v5/per_entity_stage2_noce.npy artifacts/v5/per_entity_stage2_big.npy`
  (output of the background job: see the user). **User decision: drop it unless clearly significant (CI > 0 and Δ ≥ 0.002).**
- **Prepared, not run:** `run_final.ps1` — stage 1 **streamed on ALL folds-0-2 rows (~35M, 3× the 0.980 run)**
  via `v5.stage1_stream` (LightGBM Sequence + 2M-row reference; rehearsed on a slice, identical F to the normal
  path), optional `-Stage2Big`, per-entity compare vs the 0.980 run, predict → `subs/sub_v5_final/`. ~2.5 h,
  peak RAM ~8–9 GB (laptop 15.7 GB; close other apps). It re-reads the union test France file (original now).

## 3. Measured dead ends (do not repeat)

stage 3 (competition features from p2): −0.00002 · record-relative chain counts (E10): +0.00006 · record-name
frequency (E11): −0.0021 · joint logistic over E9 descriptors: −0.0011 · address canonical features (E6): negative
with CHAIN · prior-shift correction for orphans: offline −0.0002, not uploaded (India/US have no gap anyway) ·
France-only vocabulary/addresses (E15): LB −0.001 · LLM judge: judged not worth it (decoy errors look like typos
to an LLM; France is over-confidence, not missing semantics).

## 4. Where the remaining points are and ideas for the new thread

Loss budget on LB (from 1.0): France ~0.008 · India/US model errors ~0.009 (ceiling 0.99866 − 0.98794, ×0.85) ·
retrieval ~0.001. Irreducible-ish: empty-address records with chain names (~0.003–0.004 of fold-0).

Ideas, cheapest / most direct first:
1. **France calibration via LB probes** (one parameter, low overfit risk): upload `sub_v5_fr_strict060`; move the
   strength in the winning direction. Code pattern (on the 0.980 arrays in artifacts/v5):
   `q[fr] = p2[fr]*r/(p2[fr]*r+1-p2[fr])`, `keep = V.apply_rule(tk, q, rules)`, `V.write_outputs`, `finalize_submission`.
2. **Offline India/US: `run_final.ps1`** (3× stage-1 data). Judge with `compare_v5` (per-entity bootstrap) and
   upload only if CI > 0. Apply the chosen France strength to its test p2 the same way before uploading.
3. France-specific signals not yet used: acronyms (record token = initials of the S1 name, e.g. `ep` = "ensemble
   parents"; many 2-letter tokens with high edit rate in France), glued names with dropped accents
   (`maisondesant`); pseudo-label self-training on France (confident pairs) to recalibrate France p; France-only
   decision thresholds tuned on pseudo-labels. Any new *feature* needs a union rebuild + retrain (~3 h+): too late
   unless it is decision-level only.
4. Stage-2 XGBoost member (`-Xgb` in after_kaggle / `v5.xgb.enabled`): unmeasured, ~+1 h CPU.

## 5. Key commands (PowerShell, repo root)

```
$env:BER_CONFIG = "D:\ML challenge\Code\artifacts\free\pipeline.yaml"      # the 0.980 run
.venv\Scripts\python experiments\compare_runs.py <per_entity_A.npy> <per_entity_B.npy>
powershell -ExecutionPolicy Bypass -File .\run_final.ps1 [-Stage2Big]
```
Validator runs inside every `finalize_submission` (must print PASS). Never overwrite `subs/sub_v5_noce/` (the best).
