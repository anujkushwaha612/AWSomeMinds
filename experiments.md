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
