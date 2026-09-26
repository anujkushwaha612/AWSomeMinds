# Experiment log

One entry per experiment: what was tried, the exact setup, the numbers, and the decision.
Newest entries at the bottom. IDs match [plan.md](plan.md) Step 3 (E0, E1, …).
Keep failed experiments too: they are what stops us from re-trying dead ends.

**Conventions**
- *Recall@k (reverse)* = for sampled S2/S3 records with a known S1 parent, share whose parent is in the
  record's top-k S1 list. This is a quick proxy; decisions use the fold-0 oracle macro-F0.5 from
  `ber.blocking.audit` once a configuration is run at full scale.
- *Full-run estimate* = time per query × number of queries, on the laptop (16 threads, 15.7 GB, no GPU)
  unless stated otherwise.
- Verdicts: **KEEP** / **DROP** / **INVESTIGATE**.

---

## E0 — Retrieval speed profile: char 3-gram TF-IDF, name+address (2026-09-25)

**Question.** Why did the 1% India timing run take more than 25 min, and can capping common n-grams
("block purging") fix it?

**Setup.** Train India: 883,188 S1, 2,017,799 S2, 2,115,547 S3. View NA (name + address, normalized).
`TfidfVectorizer(char_wb, 3-grams, min_df=2, sublinear_tf)` fitted on S1+S2+S3; caps drop n-grams whose
document frequency exceeds the cap, then rows are re-normalized. Reverse = 3,000 sampled S2 records
(with a known parent) → top-5 S1. Forward = 300 S1 → top-10 S2. `sparse_dot_topn`, 16 threads.
Script: `scratchpad/prof.py` (to be moved to `phase0/` / `experiments/`).

**Findings**
- Fit + transform: 293 s. Vocabulary only **39,536** distinct 3-grams, ~75 per record.
- Most frequent 3-grams come from "Plot No" / "Private Limited": `" no"` 46.9%, `"no "` 44.5%,
  `" pr"` 42.6%, `"ate"` 38.0%, `"ite"` 36.6%, `"mit"` 35.7% of all records.

| n-gram cap | postings removed | reverse ms/query | full reverse run (S2+S3) | recall@1 | recall@5 | forward ms/query | full forward run |
|---|---|---|---|---|---|---|---|
| none | 0% | 26.5 | 30.4 h | 0.904 | 0.933 | 78.0 | 38.3 h |
| 10% (v4 default) | 31.2% | 16.4 | 18.8 h | 0.904 | 0.934 | 45.4 | 22.3 h |
| 5% | 48.5% | 9.6 | 11.0 h | 0.894 | 0.927 | 27.5 | 13.5 h |
| 2% | 69.7% | 2.9 | 3.3 h | 0.873 | 0.905 | 8.7 | 4.3 h |
| 1% | 80.1% | 0.95 | 1.1 h | 0.833 | 0.874 | 3.1 | 1.5 h |
| 0.5% | 87.7% | 0.25 | 0.3 h | 0.744 | 0.803 | 1.0 | 0.5 h |

**Conclusions**
1. Cause confirmed: a few extremely common 3-grams make every query scan ~half the index.
2. Capping alone trades recall for speed too steeply (1% cap: 28× faster but recall@5 0.933 → 0.874).
3. 3-grams are too generic (tiny vocabulary). A bigger machine does not fix a 30 h job.
4. Recall@1 ≈ recall@5: when found, the parent is usually rank 1, so a small record-side k is realistic.
5. Name+address alone misses ~7% of S2-India parents even at top-5 (likely Indic-script names).

**Verdict.** DROP char-3g + cap as the main retriever. INVESTIGATE longer n-grams, word n-grams, and
SVD + approximate nearest-neighbour search (E0b).

---

## E0b (run 1) — char 3/4-gram representations under n-gram caps (2026-09-25)

**Question.** Does a longer n-gram (more specific features) keep recall under the caps that make search fast?

**Setup.** Train India, index = all 883,188 S1 (name+address). Queries = 3,000 S2 + 3,000 S3 records with a
known parent (seeded; 18.4% have Indic-script names). Hashed features (2^22), sublinear TF × IDF over
S1+S2+S3, cap drops features with df > cap × 5.0M records, L2-normalized, exact top-5 with
`sparse_dot_topn` (16 threads). Script: [experiments/e0b_retrieval_variants.py](experiments/e0b_retrieval_variants.py).
Featurization is parallel (~60 s per representation, vs 293 s single-threaded in E0).

| Representation | Cap | ms/query | Full reverse run (4.13M queries) | R@1 | R@5 | R@5 S2 | R@5 S3 | R@5 Latin | R@5 Indic |
|---|---|---|---|---|---|---|---|---|---|
| char3 (41,855 features) | none | 27.8 | 31.9 h | 0.904 | 0.938 | 0.932 | 0.944 | 0.966 | 0.814 |
| char3 | 2% | 3.35 | 3.9 h | 0.847 | 0.893 | 0.904 | 0.882 | 0.922 | 0.764 |
| char3 | 1% | 1.08 | 1.2 h | 0.789 | 0.843 | 0.868 | 0.818 | 0.876 | 0.693 |
| char3 | 0.5% | 0.44 | 0.5 h | 0.690 | 0.756 | 0.799 | 0.714 | 0.787 | 0.621 |
| char4 (312,314 features) | none | 18.9 | 21.7 h | 0.893 | 0.929 | 0.924 | 0.935 | 0.960 | 0.794 |
| char4 | 2% | 3.39 | 3.9 h | 0.859 | 0.902 | 0.909 | 0.895 | 0.934 | 0.762 |
| char4 | 1% | 1.74 | 2.0 h | 0.833 | 0.881 | 0.898 | 0.865 | 0.913 | 0.742 |
| char4 | 0.5% | 0.66 | 0.75 h | 0.803 | 0.858 | 0.886 | 0.830 | 0.887 | 0.731 |

**Incident.** The main process died silently right after featurizing char5 (1,080,313 features), with no
Python traceback (native crash or memory kill during the search). The remaining variants (word n-grams,
address view, SVD+HNSW) did not run. Output was hidden until the orphaned pool workers were stopped,
because a `grep` filter buffered it. Fix: stream output unfiltered; skip char5.

**Conclusions so far**
1. char4 degrades more gracefully than char3 under caps (R@5 at the 0.5% cap: 0.858 vs 0.756), but no
   capped exact sparse search reaches ~0.93 recall in a feasible time. Exact sparse search at good
   recall is ~20 ms/query ≈ 20 h for India alone.
2. Indic-script partners are the weak stratum everywhere (R@5 ≈ 0.79–0.81 uncapped, vs 0.96 Latin).
   That supports the address-only view.
3. Uncapped, S3 is not worse than S2 for retrieval (0.944 vs 0.932).

**Verdict.** DROP exact capped sparse search as the main retriever. Char5 DROP (crash; 1M features).
INVESTIGATE (run 2): SVD + HNSW, word n-grams, address-only view, NA ∪ A.

---

## E0b (run 2) — word n-grams, address-only view, SVD + HNSW (2026-09-25)

**Setup.** Identical sample, index and metrics to run 1 (6,000 queries, 18.4% Indic-script partners).
First attempt died with `MemoryError` (16 featurization workers + full normalized tables > 16 GB); the
retry used 8 workers, 50k-row chunks, and only the 4 needed columns. Char4 was re-run for the unions and
reproduced run 1 exactly (R@5 0.929 / 0.902 / 0.881 / 0.858). Raw results:
`artifacts/experiments/E0b_run2.json`.

| View | Representation | Cap | ms/query | Full reverse run | R@1 | R@5 | R@5 S2 | R@5 S3 | R@5 Latin | R@5 Indic |
|---|---|---|---|---|---|---|---|---|---|---|
| NA | word 1-gram (821,877 feats, 15 / record) | none | 4.92 | 5.7 h | 0.910 | 0.947 | 0.943 | 0.951 | 0.966 | 0.866 |
| NA | word 1-gram | 2% | 0.59 | 41 min | 0.881 | 0.925 | 0.926 | 0.925 | 0.950 | 0.817 |
| NA | word 1-gram | 1% | 0.25 | 17 min | 0.853 | 0.906 | 0.911 | 0.901 | 0.932 | 0.790 |
| NA | word 1-gram | 0.5% | 0.09 | 6 min | 0.828 | 0.882 | 0.899 | 0.865 | 0.910 | 0.760 |
| NA | **word 1+2-gram** (3.48M feats, 30 / record) | none | 5.40 | 6.2 h | 0.917 | **0.955** | 0.952 | 0.958 | 0.972 | 0.882 |
| NA | **word 1+2-gram** | 2% | 0.87 | 1.0 h | 0.914 | **0.951** | 0.950 | 0.952 | 0.968 | 0.875 |
| NA | **word 1+2-gram** | 1% | 0.39 | 27 min | 0.911 | **0.949** | 0.950 | 0.948 | 0.967 | 0.870 |
| NA | **word 1+2-gram** | **0.5%** | **0.15** | **11 min** | 0.905 | **0.946** | 0.947 | 0.945 | 0.965 | 0.861 |
| A | char4 (201,672 feats) | none | 10.43 | 12.0 h | 0.769 | 0.831 | 0.885 | 0.776 | 0.839 | 0.794 |
| A | char4 | 0.5% | 0.30 | 21 min | 0.695 | 0.779 | 0.838 | 0.721 | 0.789 | 0.737 |
| A | word 1+2-gram (2.71M feats) | none | 4.32 | 5.0 h | 0.806 | 0.881 | 0.909 | 0.854 | 0.884 | 0.868 |
| A | word 1+2-gram | 1% | 0.26 | 18 min | 0.796 | 0.871 | 0.904 | 0.838 | 0.875 | 0.854 |
| A | word 1+2-gram | 0.5% | 0.12 | 8 min | 0.792 | 0.867 | 0.903 | 0.831 | 0.872 | 0.844 |
| NA | char4 → SVD(256) + HNSW ef=64 | – | 0.05 | 4 min | 0.583 | 0.635 | – | – | 0.728 | 0.222 |
| NA | char4 → SVD(256) + HNSW ef=256 | – | 0.18 | 12 min | 0.616 | 0.676 | – | – | 0.773 | 0.247 |
| NA | char4 → SVD(256), exact search | – | – | – | – | 0.710 | – | – | – | – |

Unions (top-5 of NA ∪ top-5 of A, same representation and cap):

| Representation | Cap | NA alone R@5 | NA ∪ A R@5 | Latin | Indic |
|---|---|---|---|---|---|
| char4 | none | 0.929 | 0.932 | 0.960 | 0.808 |
| char4 | 0.5% | 0.858 | 0.866 | 0.891 | 0.752 |
| word 1+2 | none | 0.955 | 0.956 | 0.973 | 0.883 |
| word 1+2 | 1% | 0.949 | 0.950 | 0.968 | 0.871 |
| word 1+2 | 0.5% | 0.946 | 0.946 | 0.965 | 0.863 |

**Conclusions**
1. **Word 1+2-grams beat every char n-gram** on recall *and* speed. Capping costs almost nothing:
   R@5 0.955 uncapped vs 0.946 at a 0.5% cap. Pairs like "no 63" or "gulmohar colony" stay specific
   after common single words are purged. At 0.5% the full India reverse search is ~11 min vs 31.9 h for
   the E0 baseline (char3, uncapped, R@5 0.938): ~175× faster at higher recall.
2. **SVD(256) + HNSW fails on recall.** The compression, not the ANN index, is the limit (exact SVD search
   R@5 0.710). Indic partners collapse (0.22–0.25). A different dense encoder (a multilingual sentence
   model) is a separate question, still [EXP].
3. **The address-only view adds almost nothing** on top of word-1+2 name+address (≤ +0.001 R@5), because
   the NA view already contains the address words. Its own Indic recall (0.868) ≈ its Latin recall
   (0.884): the address does carry cross-script pairs, and NA word-1+2 already captures that.
4. Remaining gap: ~5% of records overall, **~12–14% of Indic-script records**, are not in the top-5.
   That's the target for transliteration features, forward retrieval, or a multilingual dense retriever.
5. Recall@1 is within ~0.04 of recall@5 for word 1+2: the parent is usually rank 1, which supports a small
   record-side candidate list (k_r of 1–3) plus pruning.

**Verdict**
- **KEEP** word 1+2-gram TF-IDF, name+address, record-side (S2/S3 → S1) as the main retriever (R1).
  Cap 0.5–1% (choose on the full-scale oracle-F0.5 / |C| audit).
- **DROP** the char 3/4/5-gram representations, SVD(256)+HNSW, and the address-only view as separate
  retrievers (A may still be useful as a *feature*).
- **Next:** full-scale R1 on train India + US at k_r ≤ 5 → rank table → audit (oracle macro-F0.5 vs
  mean |C|); then test; then the baseline matcher and Sub #1.

---

## E1 — Memory-lean pipeline: audit, store loading, worker count (2026-09-26)

**Why.** Free-tier AWS offers ≤ 8 GB instances, and the first version of `ber.baseline` held everything
in RAM (estimated train peak ≈ 9–11 GB, from reading the code). Goal: run locally on 15.7 GB with
peak ≤ 8 GB, without changing the method.

**Audit of the first version** (derived from the code, not measured): the largest resident objects were
all normalized text columns as pandas objects (~3 GB per country), count matrices plus a TF-IDF copy for
all sources (~2.5–3.5 GB for India), the full feature table concatenated (31M × 36 columns) plus a
float32 copy (3.3 GB), and in `predict` a re-read of every text column just to get ids, plus string
merges for the output.

**Changes (method unchanged)**
- Arrow text store per country (`ber.store`); candidates hold only integer row ids.
- Retrieval in two streaming passes (document-frequency pass, then hash + search per chunk); only the S1
  index and at most `window` hashed chunks are resident.
- String features in 500k-pair chunks, appended to parquet; train/predict stream batches from parquet.
- LightGBM trains on 30% of entities with *all* their candidates (same rule as before; no negative
  subsampling); out-of-fold prediction for every pair, batch by batch.
- Output TSVs streamed per country from compact arrays; matches ⊆ candidates by construction.
- Memory logged at every stage (`ber.memory`).

**Measurement 1: loading one country's text** (train India, 8 columns, Arrow data = 1.08 GB)

| Method | Load time | Process memory after load |
|---|---|---|
| `pq.read_table(filters=country)` (mimalloc pool) | 4 s | 4.07 GB |
| same, system allocator | 3 s | 3.44 GB |
| **batch-wise filter** (200k-row batches) | 6 s | **1.66 GB** |

Whole-file filtered reads materialize every country's rows first, and the freed memory stays in
Arrow's pool. → KEEP batch-wise filtering (in `CountryStore`).

**Measurement 2: hashing worker count** (one pass over all 2,017,799 train-India S2 records, word 1+2-grams,
50k-row chunks, window = 2 × workers; peak = main + workers, sampled every 0.2 s)

| Workers | Time | Throughput | Peak tree memory (above the loaded store) |
|---|---|---|---|
| 4 | 13.5 s | 150k rec/s | +0.83 GB |
| **8** | **11.2 s** | **180k rec/s** | +1.56 GB |
| 12 | 12.2 s | 166k rec/s | +1.85 GB |
| 15 | 13.3 s | 151k rec/s | +2.24 GB |

Beyond 8 workers the main process (slicing texts, accumulating counts) is the bottleneck, so more
workers only add memory. → KEEP `workers: 8`.

**Smoke test** (40k rows per source per country, all stages incl. the official validator): PASS. Peak
main-process memory: build 2.2 GB, train 0.52 GB, predict 0.82 GB. On the slice, record-side top-3
retrieval found 97.0% (India) / 99.2% (US) of the true pairs present in the slice. That's a smaller
universe than the real one, so it's not a full-scale estimate. The smoke F0.5 is meaningless (the
slice is mostly singletons).

**Verdict.** KEEP the lean pipeline. Full-scale memory and timings are still to be measured on the
first real run and recorded here.

**Full-scale run (2026-09-26, laptop):** total ~2 h 15 min; peak main-process memory 4.79 GB (train
build, US). Train: 30.96M pairs (14.0/S1), pair recall India 0.939 / US 0.969. Test: 29.9M pairs
(17.3/S1). Fold-0 macro F0.5 **0.9487** (oracle 0.9848). **Public LB (Sub #1): 0.938.**
LB top 3 at submission time: 0.988419 / 0.988095 / 0.987916.

---

## E2 — Error analysis of the baseline: where are the 0.05 lost? (2026-09-26)

**Row-order leak check (none).** Pearson(row position of S1, row position of partner) = 0.001 (S2) /
−0.0007 (S3); share with |Δ position| < 1% = 0.0199 / 0.0198 (random 0.02); GT file order vs S1 order
0.0003; S2 siblings of one entity are spread across the file like random rows. The data carries no
positional shortcut: the leaderboard gap is methodological.

**Loss decomposition (fold 0, 441,370 entities).** "Without FPs" keeps the same true positives and removes
every false positive; "oracle" = every retrieved true pair, no false positives.

| | Macro F0.5 | Loss |
|---|---|---|
| Actual | 0.9487 | – |
| Without false positives | 0.9634 | **FP: 0.0148** |
| Oracle (all retrieved true pairs) | 0.9848 | **found but rejected: 0.0214** |
| Perfect | 1.0 | **never retrieved: 0.0152** |

By k (loss × share of entities): FP loss concentrates on k=0 singletons (0.0044) and k=2–3; the
"rejected" and "missed" losses spread over k=1–6.

**Profile of never-retrieved true pairs (fold 0)**

| Country / source | Missed | Indic-script name | Empty address | Parent's name shared by >1 S1 (vs all pairs) |
|---|---|---|---|---|
| India S2 | 5.66% | 63.0% | 22.5% | 71.1% (44.4%) |
| India S3 | 6.54% | 29.5% | 24.4% | 65.3% (44.1%) |
| US S2 | 3.37% | 0% | 46.5% | 50.9% (35.8%) |
| US S3 | 2.94% | 0% | 49.4% | 51.8% (35.8%) |

Typical misses (examples printed by the script):
- (a) empty or near-empty address plus a name typo ("ferrari bl0ck company", "dermatology anfcor group");
- (b) Indic-script name with a short address ("कृष्णा बिजनेस प्राइवेट लिमिटेड | s no 861 nashik महाराष्ट्र");
- (c) domain-style names ("lvassignments com", "halldennis com");
- (d) chain names (a shared name plus a weak address);
- (e) garbled names ("vioaria").

Word 1+2-gram tokens can't match (a), (b) or (c) at the character level.

**Verdict.** The model side (FP 0.0148 + rejected 0.0214 = 0.036) is larger than retrieval (0.015),
and even the oracle ceiling (0.985) is below the LB top (0.988). Both sides must improve. Next
experiments: E3 (stronger model: all entities, more capacity), E4 (stage-2 competition / sibling
features), E5 (second-pass char-level retrieval only for hard records).
Script: [experiments/e2_error_analysis.py](experiments/e2_error_analysis.py).

---

## E4 — Measurements that set the v5 hyperparameters (2026-09-26)

Script: [experiments/e4_hparam_evidence.py](experiments/e4_hparam_evidence.py) (read-only over baseline
artifacts; raw numbers in `artifacts/experiments/E4_evidence.json`). The token-length check used the
`intfloat/multilingual-e5-base` tokenizer on 20k sampled texts per country × source.

**TF-IDF record-side rank of the true parent (all train pairs)**

| | India | US |
|---|---|---|
| recall@1 / @2 / @3 | 0.9087 / 0.9301 / 0.9388 | 0.9478 / 0.9624 / 0.9688 |
| share of found parents at rank 0 | 0.968 | 0.978 |

**Anatomy of fold-0 errors (baseline, T = 0.65, arbitration)**: of 1,461,618 retrieved true pairs,
94.04% kept, **5.31% rejected with p < T** although they won arbitration, 0.65% lost arbitration.
Of 24,109 false positives, **72.2% are orphan records** (no parent anywhere), 19.2% have their true parent
among their candidates, 8.6% have a parent that was not retrieved.

**Pruning by stage-1 p (fold 0)**

| τ | oracle F0.5 | candidates / S1 | true pairs lost |
|---|---|---|---|
| 0 | 0.98481 | 14.01 | 0 |
| 0.001 | 0.98479 | 5.58 | 0.006% |
| **0.003** | **0.98477** | **5.05** | 0.011% |
| 0.01 | 0.98468 | 4.61 | 0.044% |
| 0.1 | 0.98238 | 3.76 | 0.75% |

Positive p quantiles: 1% of true pairs have p < 0.127, 5% < 0.556. Negatives: median 0.000, q90 0.020,
q99 0.467.

**Density (S2+S3 per S1):** train 4.680 (India) / 4.674 (US); test 5.824 / 5.756 / France 5.531 →
S1 share to remove for matching density: 0.196 / 0.188.

**Encoder data (folds 3–4 positives):** India S2 592,188 (23.2% Indic), S3 631,602 (13.1%); US S2 886,043,
S3 945,493; parents sharing their name with another S1: 44% India, 36% US; records with a non-parent
TF-IDF candidate ranked above the parent: 8.4% / 9.9% (India), 5.6% / 4.8% (US).

**Tokens (e5 tokenizer, "query: name | address")**: p50/p95/p99/max = India S2 33/46/52/72,
S3 30/44/50/76, S1 33/44/49/68; US S2 23/29/31/38, S3 25/31/34/42, S1 23/28/31/38. Share > 64 tokens
≤ 0.04%.

**Baseline LightGBM**: trees per fold 201/114/217/163/156 (early stopping at lr 0.1); gain share
name_len_ratio 0.234, translit_ratio 0.134, gap_rec 0.111, gap_s1 0.080, n_cand_s1 0.076, score 0.067;
near zero: n_digits_s1, nospace_ratio, house_state, n_digits_rec, rec_addr_empty, rec_indic, src, n_cand_rec.

**Verdict.** These set every v5 default (strategy_v5.md §7): max_len 64, all 3.06M pairs, fold-3/4-only
hard negatives, same-country batches, k_rec 3, k_s1 5, prune τ 0.003, stress fraction 0.19, and stage-2
features aimed at orphan false positives. None of the resulting gains is measured yet; that happens on
the GPU machine run.

Also recorded: PyTorch cannot load on the laptop (Smart App Control blocks `c10.dll`), so the neural
code was verified only for its CPU parts (smoke run of pairs → union → stage 1 → stage 2 → compare →
gap on a 20k-row slice with fake dense outputs; leakage checks passed: positives and hard negatives only
from fold-3/4 S1s, evaluation records only fold 0).

---

## E5 — v5.1: review of an external upgrade list; number-conflict features + cross-encoder (2026-09-26)

**Question.** Which of the proposed upgrades (cross-encoder, digit features, bipartite matching, address
parsing, geohash, graph expansion, in-batch masking, density-robust thresholds) are worth building, given
E2/E4?

**Decision (reasoning in strategy_v5.md §9, before any run):** BUILD the cross-encoder (gray zone
p1 ∈ [prune_tau, 0.995] → stage-2 features `ce`, `ce_minus_rec_other`), BUILD set-level number-conflict
features (`STRUCT_FEATURES`), REFINE the in-batch mask (identical S1 texts). REJECT linear-sum assignment
(S1s have many matches; per-record argmax is already optimal under the only exact constraint), address
parsers / geohash (US-only, LGPL, or needs external data), GT graph expansion (GT is already a complete
star clustering). Density-stress stays diagnostic.

**Verification so far.** Unit tests for the new pieces (58 pass). Smoke run of every stage on a
4,000-rows-per-source slice with a tiny random XLM-R (real e5 tokenizer) on CPU: every stage of run_all
(pairs -> bi-encoder -> dense -> union -> stage1 -> stage2_noce -> cross-encoder pairs/train/score -> stage2 ->
compare -> stress -> predict -> france) completes; the only error is the validator rejecting the slice for missing
S1 rows (expected). Union files carry the num_* columns (stage 1 resolved all features); the cross-encoder loads
from the bi-encoder checkpoint. Slice scores are meaningless (~20 true pairs per country) and are not reported.

**Gates (to fill in from the GPU run):**
- stage-1 gain share of `num_*` features: ___
- `stage2_noce` fold-0 F0.5: ___ | `stage2` (with CE): ___ | ablation delta / CI: ___ → KEEP if CI > 0
- LB: sub_v5_noce ___ | sub_v5 ___
