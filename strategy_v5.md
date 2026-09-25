# strategy_v5.md — Combined plan A + B + C (bi-encoder · stronger GBDT · LB-gap)

**Status:** 2026-09-26 ~03:30 IST. Supersedes the E3–E6 ordering in `experiments.md` (E2 verdict) and
merges `plan_neural.md` parts A, B, C with three corrections (leakage-safe folds, streaming dense search,
candidate-size control). Deadline: **27 Sep 23:59 IST**. Submissions: 5 per day.
**Tags:** [D]/[R] measured (phase0_report.md, experiments.md) · [H] hypothesis · [J] judgement call.

---

## 0. Target and honesty rule

**Target: public LB macro F0.5 ≈ 0.997.** That is a *target used to set gates*, **not an estimate**:
nothing measured so far predicts it, and the LB top is 0.988 today. The plan decomposes it into a loss
budget per component; every component has a measured gate and is dropped if it misses.

| Loss component (fold 0, measured in E2) | Now | Budget for 0.997 | Main lever |
|---|---|---|---|
| Never retrieved (1 − oracle F0.5) | 0.0152 | ≤ 0.0010 | **A** dense retrieval (record-side + S1-side) ∪ TF-IDF |
| Found but rejected | 0.0214 | ≤ 0.0010 | **A** cosine feature, **B** stage-2 competition / sibling features |
| False positives | 0.0148 | ≤ 0.0010 | **A** cosine, **B** stage-2, decision layer |
| Offline → LB gap | 0.011 (0.9487 → 0.938) | ≤ 0.0005 | **C** density-robust thresholds, France diagnosis |
| **Total loss** | **0.051 (LB 0.062)** | **≤ 0.0035** | |

Reality check [J]: reaching 0.997 almost certainly also needs **D (cross-encoder)** from
`plan_neural.md`. A is built so D can start from A's fine-tuned weights. D is scheduled only if A passes
its gate by the evening of 26 Sep.

## 1. Where each step runs

**Everything runs on the GPU machine** (friend: 48 GB VRAM, ~15.7 GB RAM): `run_v5.ps1` executes
A → B → C end to end; setup and inputs are in [GPU_RUNBOOK.md](GPU_RUNBOOK.md). Backup: the GPU stages
A0–A4 on Colab (`colab/run_v5_colab.ipynb`, outputs on Drive), then `run_v5.ps1 -CpuOnly` on the laptop
(GPU_RUNBOOK.md §7). The laptop has no
NVIDIA GPU (and Smart App Control blocks PyTorch there); it only prepared the code and the baseline
artifacts (`artifacts/`, ~7.5 GB: copy them over, or rebuild with `run_baseline.ps1`, ~2.5 h).
The CPU stages (union, GBDT, gap) are memory-capped for 15.7 GB (lean pipeline, `max_train_rows`).

## 2. The leakage rule (applies to everything below)

The encoder is trained on GT pairs, so its cosine is over-confident on those pairs. To keep fold-0
evaluation honest and train/test features consistent:

- **Encoder folds = {3, 4}.** The encoder is trained only on pairs whose S1 entity is in folds 3–4.
- **GBDT (stage 1 and stage 2) is cross-fitted on folds {0, 1, 2}** (3-fold out-of-fold), thresholds
  tuned on folds 1–2, **reported on fold 0**. Folds 3–4 are not used by the GBDT at all.
- Test features come from the same encoder (trained on folds 3–4), so fold 0 and test are both
  "unseen by the encoder": the fold-0 score is a valid estimate for test.
- Stage-2 features are computed from stage-1 **out-of-fold** p on train, and from the mean of the
  stage-1 fold models on test.

This changes the methodology versus the baseline (5-fold → 3-fold GBDT on 60% of entities). The
comparison with the baseline is still on the same fold-0 entities, so it remains a paired comparison.

## 3. Part A — bi-encoder (GPU)

**A0. Environment + throughput probe (first 10 minutes on the GPU box).**
`python -m ber.neural.env_check`: torch/CUDA/GPU/VRAM/bf16; encode 10k texts; time 50 training steps.
The printed per-stage time estimates replace the guesses in `plan_neural.md` (5k texts/s etc. are [H]).

**A1. Training data.**
- Positives: GT pairs (record → its S1 parent) whose parent is in folds 3–4 (~3M pairs available).
  Default: 2M sampled pairs.
- Hard negative per positive: the best-ranked TF-IDF candidate of that record that is *not* its parent
  **and whose S1 is also in folds 3–4** (a negative S1 from folds 0–2 would train on the label of a pair
  that fold-0 evaluation later scores); a random fold-3/4 S1 of the same country if none. Expected share
  with a TF-IDF hard negative ≈ 1 − 0.6² ≈ 64% (≈2 non-parent candidates per record, each in folds 3–4
  with probability 0.4); `pairs.py` prints the measured share.
- Text: normalized `name_n | addr_n` (keeps Indic letters; Latin accents folded), prefix `query: `.
- Orphans are not used in the contrastive loss (no positive). The "no parent" decision stays with the
  GBDT (cosine, margins, stage-2 counts).

**A2. Model and loss.** `intfloat/multilingual-e5-base` (278M, MIT on the model card; recheck). Mean
pooling + L2 normalization (e5 convention). Loss = in-batch softmax over positives and hard negatives
(MultipleNegativesRanking), scale 20, batch 256, lr 2e-5, 1 epoch, max_len 64, bf16/fp16 autocast.
Fallback if too slow: `multilingual-e5-small` (118M).

**A3. Gate A-retrieval** (fold-0 records vs the full S1 universe of their country, 20k-record sample per
country): R@1 / R@5 overall and for Indic-script partners, vs E0b TF-IDF (R@5 0.946 overall, 0.861 Indic
at the 0.5% cap). KEEP if R@5 ≥ TF-IDF overall and ≥ +0.05 on Indic.

**A4. Streaming dense search (per split, per country; nothing stored except results).**
All embeddings of one country stay on the GPU (India train: ~5M × 768 fp16 ≈ 7.7 GB):
- record-side top-`k_rec` S1 (**3**, from E4 rank data; see §7), tiled matmul + `torch.topk`;
- S1-side top-`k_s1` records per source (**5**, the per-source true-match maximum; see §7): recovers
  records buried under chain look-alikes;
- cosine for every existing TF-IDF candidate (aligned with the baseline feature-file row order).
Outputs: `artifacts/dense/<split>/<country>_dense.parquet` (src, doc_row, s1_row, cos, drank_rec,
drank_s1) and `<country>_tfidf_cos.npy`.

**A5. Union + features (CPU).** Union of TF-IDF top-3 and dense candidates → retrieval features for both
retrievers (score/rank/gap for TF-IDF, cos/rank/gap for dense, "found by both") + the existing string
features → `artifacts/union/<split>/<country>.parquet`. **Gate A-ceiling:** oracle F0.5 on fold 0 ≥ 0.995
with mean |C| per S1 reported; if |C| grows much, prune in B3.

## 4. Part B — stronger GBDT (CPU)

- **B1 = E3:** train on all (allowed) entities. Currently being debugged (the first E3 run died silently
  after binning; see experiments.md E3).
- **B2 = E4 stage-2 features** from stage-1 p:
  - record side: best p, 2nd-best p, p minus the best *other* S1's p, rank of p within the record;
  - S1 side, per source: rank of p, max p, number of records with p > 0.5, sum of p (caps: ≤5 S2, ≤6 S3 [R]);
  - sibling: max p of the S1's candidates in the other source.
  Stage-2 LightGBM on [stage-1 p + stage-2 features + original features], cross-fitted on the same folds.
- **B3. Candidate pruning (organizers rank smaller |C| higher):** drop pairs with stage-1 p < τ_prune
  before stage 2. The pruned set becomes `candidate_pairs.tsv` (it is exactly what the final model
  scores). Measured on the baseline (E4): τ = 0.003 cuts **14.0 → 5.05 candidates per S1** at an oracle
  loss of **0.00004**. Stage 1 re-reports the curve on the union features.
- **B4. Decision layer:** keep T_first/T_rest + arbitration; add the expected-F0.5 prefix rule on
  isotonic-calibrated stage-2 p as an A/B (keep only if CI > 0).
- **Gate B:** paired bootstrap vs the previous best on fold 0: CI > 0 and Δ ≥ +0.002.

## 5. Part C — the offline → LB gap (CPU)

- **C1. Density stress:** test has 5.8 records per S1 vs 4.68 in train [R]. Remove a random 20% of S1
  entities (all folds) from the scoring universe, drop their candidate rows, re-run arbitration + decision
  with the same p, score the remaining fold-0 entities. Report F0.5 at the tuned T and the best T under
  stress. If the best T moves, tune T under stress for submissions. (Approximation: p is not recomputed
  after removal; documented.)
- **C2. France diagnostics (label-free):** per country on test vs train OOF: distribution of the record's
  best p, share of S1 predicted empty, predicted matches per S1. France far from US/India → coverage or
  threshold problem.
- **C3. LB probe (1 submission, optional):** the current best with France rows emptied. LB delta =
  0.15 × (F_France − singleton share of France); gives France's score to within that unknown share.

## 6. Order of work and submissions

All on the GPU machine, in `run_v5.ps1` order. Times depend on the A0 probe; fill them in after it runs.

| Step | Stage | Submission |
|---|---|---|
| 1 | copy artifacts + setup, A0 probe (~10 min) | — |
| 2 | A1 pairs → A2 fine-tune → A3 recall gate | — |
| 3 | A4 dense search train + test → A5 union features | — |
| 4 | B stage 1 → prune → stage 2 → decision → compare; C1 stress | — |
| 5 | B predict | **Sub #2 (v5)** if compare CI > 0 |
| 6 | C2 France diagnostics; if C1 says stress-tuned T is better, re-decide with it | Sub #3 (stress-tuned T) |
| 7 | optional D (cross-encoder) if time remains on 27 Sep; freeze by 12:00 IST | Sub #4+: best offline; last = best |

## 7. Hyperparameters from the data, with predictions

Measured in experiments.md **E4** (baseline artifacts, full train data) unless marked [L] (literature /
model-family default) or [J] (judgement). "Prediction" = the expected direction of the effect on fold-0
macro F0.5; the measured quantity bounds how much it can matter. None of these are measured gains yet.

### What the errors are (sets the priorities)

| Measurement (fold 0, baseline) | Value | Consequence |
|---|---|---|
| Found true pairs rejected because **p < T** (won arbitration) | **5.31%** of retrieved true pairs | The main recall loss is low confidence, not competition. A learned similarity (cosine) is the lever |
| Found true pairs lost to another S1 (arbitration) | 0.65% | Competition features are secondary |
| False positives on **orphan records** (no parent anywhere) | **72.2%** of FPs | Must tell "similar" from "same business"; plus S1-side pile-up counts (`s1_n_best`) |
| FPs whose true parent was among the record's candidates | 19.2% | Record margin features (`p_minus_rec_other`) |
| FPs whose parent exists but was not retrieved | 8.6% | Dense retrieval |
| Top LightGBM features by gain | `name_len_ratio` 0.23, `translit_ratio` 0.13, `gap_rec` 0.11, `gap_s1` 0.08 | Length/translit signals dominate: typical of truncation + transliteration noise, which an encoder learns directly |

### Part A — encoder and dense retrieval

| Hyperparameter | Value | Evidence | Prediction |
|---|---|---|---|
| `max_len` | **64** | Tokens (e5 tokenizer, this text format): p99 ≤ 56 in every country/source; > 64 in < 0.05% | 128 would change nothing but double encoding time; 48 would truncate India's p95 (44–50) and lose recall |
| Text | `name_n \| addr_n` | Address carries cross-script pairs (R7: address-Jaccard median 0.65–0.76 when names share no tokens) | Name-only would lose most Indic pairs |
| Training pairs | **all 3.06M** from folds 3–4 | India S2 592k (23% Indic), S3 632k (13%), US S2 886k, US S3 945k | More pairs → higher recall, mostly on Indic/typo strata; cut only if the A0 probe says > 3 h |
| Hard negatives | 1 TF-IDF non-parent per positive (folds 3–4 S1 only) | 36–44% of parents share their name with another S1; 5–10% of records have a non-parent ranked above the parent | Improves precision on chain look-alikes vs in-batch only |
| Batches | **same country** | Cross-country in-batch negatives are separable by script/format | Harder negatives → better ranking |
| Batch / lr / scale / epochs | 256 / 2e-5 / 20 / 1 | [L] e5 + MultipleNegativesRanking defaults | — |
| Dense `k_rec` | **3** | TF-IDF: 96.8% (India) / 97.8% (US) of found parents are at rank 0; ranks 1–2 add 2–3 pp | k > 3 adds little recall; pruning removes the extra cost anyway |
| Dense `k_s1` (per source) | **5** | True matches per S1 per source ≤ 5 for 100% (S2) and 99.87% (S3) | Recovers records buried under chain look-alikes (the FP-parent-not-retrieved 8.6%) |

### Part B — GBDT, pruning, decision

| Hyperparameter | Value | Evidence | Prediction |
|---|---|---|---|
| GBDT folds | 0–2 (3-fold), report 0 | Leakage rule §2 | Honest fold-0 estimate for test |
| `train_frac` | 1.0 of folds 0–2 (cap 20M rows) | 30%-sample baseline stopped at 114–217 trees (signal-limited, not capacity-limited) | More data helps modestly; the new *features* matter more |
| LightGBM | lr 0.05, 127 leaves, ≤ 3000 rounds, early stop 50 | Same evidence [L] | Small gain over lr 0.1 / 63 leaves |
| `prune_tau` | **0.003** | 14.0 → 5.05 candidates/S1, oracle loss 0.00004; 0.01 → 4.61 at loss 0.00013 | Candidate set ~3× smaller at no measurable cost: helps the organizers' |C| ranking |
| Stage-2 features | margins, S1-side counts vs caps, `s1_n_best` | Error anatomy above | Targets the orphan FPs (72%) and the 19% wrong-S1 FPs |
| Decision | threshold (T_first, T_rest) vs expected-F0.5 (isotonic) | Baseline optimum T = 0.65; 5% of positives have p < 0.56 | Expected-F helps singletons/k=1, where one error costs the whole entity; chosen on tuning folds |

### Part C — offline → LB gap

| Hyperparameter | Value | Evidence | Prediction |
|---|---|---|---|
| Density-stress fraction | **0.19** | 1 − train/test records per S1: India 0.196, US 0.188 (4.68 vs 5.82 / 5.76) | If thresholds tuned under stress score higher, the LB gap is partly density; submit stress-tuned T |
| France check | per-country p / predicted-k distributions | France test density 5.53 (between train and US/India test) | A shifted-low best-p distribution for France means coverage (the encoder helps); a different predicted-k means thresholds |

## 8. Code map (new modules)

| Module | Part | What |
|---|---|---|
| `ber/neural/env_check.py` | A0 | environment + throughput probe |
| `ber/neural/train_biencoder.py` | A1–A3 | leakage-safe pair building, contrastive fine-tuning, recall gate |
| `ber/neural/dense_retrieve.py` | A4 | streaming GPU search + cosine for TF-IDF candidates |
| `ber/union.py` | A5 | union candidates + features |
| `ber/stage2.py` | B2–B4 | stage-2 features / model / pruning / decision, submission |
| `ber/gap.py` | C1–C3 | density stress, France diagnostics, France-empty probe file |
