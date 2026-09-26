# Handoff: continue from sub02 (leaderboard 0.957) — context for Claude Code on a teammate's laptop

Read this first, then [experiments.md](experiments.md) E2 / E4 / E6 (evidence) and [README.md](README.md) (setup).
Deadline **27 Sep 2026, 23:59 IST**; 5 leaderboard uploads per day. Written 26 Sep ~23:20 IST.

## 1. Where we are

| Submission | What | Fold-0 macro F0.5 (offline) | Public LB |
|---|---|---|---|
| sub01_baseline | word 1+2-gram TF-IDF candidates (k_rec 3) + LightGBM, 27 features | 0.94867 | 0.938 |
| **sub02_baseline_gen** | same candidates + **13 generator-aware features** (E6), LightGBM on 50% of entities | **0.96482** | **0.957** |

- Offline → LB gap shrank from 0.011 to **0.008**. Candidate ceiling (perfect classifier on our candidates):
  **0.9848** fold-0, so ~0.020 is still lost by the model and ~0.015 by retrieval.
- LB top 3 when we checked (26 Sep): ~0.988–0.991.
- Uploads left on 26 Sep: 3; on 27 Sep: 5. **Only one person uploads** (the rules forbid simultaneous logins):
  send files to the uploader, never log in from a second machine.

**Running elsewhere right now (do not duplicate):**
1. Laptop 1: `run_v5lite.ps1` = v5 two-stage GBDT on the same features (all 5 folds, 72% of entities,
   lr 0.05 / 127 leaves, stage 2 with competition features, expected-F rule). Output `subs\sub03_v5lite\`, ~00:30 IST.
2. Kaggle: bi-encoder (multilingual-e5-base) fine-tune, ETA ~02:30 IST 27 Sep; then dense search on two more
   Kaggle accounts ([KAGGLE.md](KAGGLE.md)); then laptop 1 runs `kaggle\after_kaggle.ps1` (union of TF-IDF +
   dense candidates, which already includes the E6 features, see `ber.union.FEATURES_V5`).

**Submission names:** sub03 is taken by v5-lite. Use `sub04_<short_name>` and up, and keep every upload's
`subs\<name>\meta.json` (`leaderboard_score`): the rules require a version history.

## 2. What the reverse engineering (E6) found

The data is synthetic: S2/S3 records are noisy copies of S1 entities. No ID or row-order leak (checked).
Noise patterns seen in true pairs (sample of 20k fold-0 entities, 69k true pairs):

- Names: typos incl. OCR swaps (`Gl0bal`, `8est`), duplicated words, legal suffix added/dropped, inserted filler
  (`com`, `center`, `services`, `dba`, `formerly`, `shri`, `dr`, `the`), dropped generic words (`associates`,
  `care`, `clinic`), domain forms (`gomezcharles.com`), junk prefixes (`--`, `***`, `#45459`), accents,
  full Indic-script transliteration, random trade names (`NEXUMBRA`) in ~6% of pairs.
- Addresses: reordered parts, `<NULL>`, state codes / full names / native script (`महाराष्ट्र`), old city
  names (Bombay, Madras), `CDP` / `CITY` suffixes, **house numbers edited on purpose** (zero-padded `003017`,
  digit dropped `2120→120`, digit replaced `5609→8609`, shifted `3041→3025`).
- Decoys: ~70% of false positives are **orphan records** (no S1 parent) that copy an S1's address and swap one
  content word of the name (`guru media` vs `guru energy`), or keep the name and change the house number.

| Pattern | True pairs | False positives | Rejected true pairs |
|---|---|---|---|
| zero-padded house number in record | 5.2% | 3.1% | **20.0%** |
| content word swapped in name | 4.1% | **15.2%** | 5.2% |
| content word inserted | 6.4% | **15.2%** | 8.4% |
| record address empty | 4.4% | **22.0%** | 16.1% |
| house number 1 edit off / conflicting | 4.7% | 11.3% | **26.0%** |

Ambiguity ceiling: 1.5% of true pairs are an empty-address record whose name is shared by several S1s (chains);
best reachable macro F0.5 is about **0.995–0.996**. No two S1s share normalized name + address.

**Features built from this** (`src/ber/features.py`), measured with `experiments/e6_ablation.py`
(same LightGBM; 150k train entities folds 2–4, 50k tune fold 1, 100k report fold 0; paired bootstrap):

| Set | Features | Δ fold-0 F0.5 | Kept |
|---|---|---|---|
| `GEN_FEATURES` (11) | zero-stripped number agreement, min digit edit / relative diff / prefix-suffix of unmatched numbers; content-word ins / del / swap after removing `NOISE_TOKENS`, shared tokens, content coverage, concatenated-name similarity | +0.0112 | yes |
| `CHAIN_FEATURES` (2) | S1s in the country sharing the S1's / the record's exact `name_n` | +0.0039 on top | yes |
| `ADDR_FEATURES` (3) | canonical address coverage (abbreviations, state codes, old city names) | +0.0008 alone, negative with CHAIN | **no** |

Full run (all fold-0 entities, 441k): **+0.0162, CI [+0.0159, +0.0165]**; biggest gains singletons +0.035, k=1 +0.032.
`NOISE_TOKENS` is curated from edit rates mined on **fold 3** (`experiments/e6_mine_vocab.py`), never the report fold.

## 3. Reproduce sub02 on this laptop

```powershell
git fetch; git checkout e6-sub02-generator-features      # this branch
python -m venv .venv; .\.venv\Scripts\python -m pip install -r requirements.txt; .\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m pytest -q                       # 64 tests
# data: data\dataset\train\*.tsv and data\dataset\test\*.tsv (not in git)
```

Artifacts are not in git. Fastest: copy from laptop 1 (USB / drive) into `artifacts\`:
`folds.parquet` (23 MB), `norm\` (3.5 GB), `baseline\` (2.5 GB), `baseline_gen\` (2.9 GB, sub02's models + features).
Otherwise rebuild: `run_baseline.ps1` (~2.5 h, peak ~5 GB RAM) then `run_gen.ps1 -TrainFrac 0.5` (~45 min).
Windows note: Smart App Control blocks PyTorch DLLs; the CPU pipeline does not need torch.

Key entry points:
- `python -m ber.gen_augment --split train|test` — adds GEN + CHAIN columns to copies of baseline feature files.
- `BER_RUN=<run> BER_FEATURES=baseline_gen BER_GEN=genchain|gen python -m ber.baseline train|compare|predict`
  (`ber.baseline.model_features()` picks the feature list; `predict` reuses the list stored in metrics.json).
- `experiments/e6_ablation.py` — the fast A/B harness (~7 min): copy it and add a new arm for any feature idea.

## 4. Experiments I planned but could not run (laptop 1 is busy training) — in priority order

Gate for every idea: ablation Δ with CI > 0 → full run → `compare` KEEP (CI > 0 and Δ ≥ 0.002) → upload.
Never fit anything (vocabulary, thresholds, rules) on fold 0; never use external data or services.

**E7a — are the extra test records orphans?** (script ready, not run: `experiments/e7_test_orphans.py`, ~5 min)
Test has ~23% more records per S1 (17.3 vs 14.0 candidates/S1) yet sub02 predicts the same 3.25 matches/S1 on
test as on train fold 0, which suggests test S1s were subsampled and their records became **orphans**. The script
fits each test record's best-score distribution as a mix of train "matched" and "unmatched" records.
If the test unmatched share is clearly above train's (~26%), the decoy rate is higher on test → go to E7d.

**E7b — what does sub02 still get wrong?** Re-run the E6 error dump on sub02's OOF
(`experiments/e6_dump_errors.py` with `BER_RUN=baseline_gen BER_FEATURES=baseline_gen`; it reads
`artifacts\baseline_gen\oof.npy` + metrics.json decision), then recompute the E6 pattern table and read ~50
examples per error type. Look for the next generator rule the features still miss. Remaining weakest buckets:
k=1 (0.907), singletons (0.956).

**E7c — raise the candidate ceiling (never-retrieved = 0.015 of F).** Misses are concentrated in: Indic-script
names (63% of India S2 misses), empty addresses (22–49%), chain names, domain / garbled names. For fold-0 missed
records, measure whether alternative retrieval views find the parent in top-k and at what cost in candidates/S1:
transliterated `name_tr + addr_tr` TF-IDF, name-only for empty-address records, address-only for trade names,
char 3/4-grams (`ber.blocking.retrieve`, `ber.blocking.word_retrieval`; `configs/pipeline.yaml` → `blocking`).
Keep only views with a large recall gain per added candidate; the Kaggle dense retriever targets the same misses,
so compare with its recall gate (`artifacts\neural\eval.json`, due ~02:30 IST) before building anything.

**E7d — close the offline→LB gap (0.008).** In v5-lite stage 1, `n_cand_s1_src` carries 55% of the gain and
`rec_name_freq` 24%; both depend on density, which differs on test (see E7a; US test has half the S1s of train).
Options, cheapest first: (1) density-normalized versions (count / country-source mean records-per-S1, or
within-S1 ranks) as an ablation arm; (2) a density-matched training universe: drop `v5.stress_fraction` (~19%)
of train S1 entities *before* retrieval so their records become orphans as in test, rebuild candidates + features,
retrain (needs a small change in `ber.baseline.build`; ~3 h). `python -m ber.gap stress` only re-scores thresholds
and does not recompute features, so it cannot measure this.

**E7e — smaller ideas.** Number-conflict features (`STRUCT_FEATURES`) are not in `baseline_gen` (only the v5
union stage computes them); France-specific normalization (unseen in train: check `C2_france` stats);
`train_frac` 1.0 if this laptop has ≥ 32 GB RAM.

## 5. Files changed in this push (vs eb761c7)

- `src/ber/features.py` (GEN / CHAIN / ADDR features, `NOISE_TOKENS`), `src/ber/gen_augment.py`,
  `src/ber/baseline.py` (`model_features()`, `BER_GEN`), `src/ber/union.py` (E6 features in `FEATURES_V5`)
- `run_gen.ps1` (sub02), `run_v5lite.ps1` (sub03 candidate), `experiments/e6_*.py`, `experiments/e7_test_orphans.py`
- Free-GPU path: `KAGGLE.md`, `kaggle/` (bundle builder, Kaggle runner with OOM fallback, laptop follow-up);
  `ber.neural`: `n_pairs` is now applied in training, dense search holds one source at a time on the GPU
- `experiments.md` E6, `subs/sub02_baseline_gen/meta.json` (LB 0.957), `subs/sub01_baseline/meta.json` (LB 0.938)
