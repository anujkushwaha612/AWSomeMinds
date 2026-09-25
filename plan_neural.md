# plan_neural.md — Learned representations for retrieval and scoring

**Status:** proposal, 2026-09-25. Supersedes the E3–E6 ordering in `experiments.md` (E2 verdict).
**Deadline:** 27 Sep 23:59 IST (18:29 UTC). ~2.5 days, 5 submissions/day.
**Tags:** [D]/[R] measured (see phase0_report.md, experiments.md) · [H] hypothesis · [J] judgement call.

---

## 0. Why change course

The pipeline shape (block → pair score → entity decision) is right and stays. What is weak is that both
stages run on **unlearned representations**: word 1+2-gram TF-IDF for retrieval and 27 rapidfuzz/TF-IDF
features for the matcher. LightGBM is the only learned component, and it can only recombine those features.

Evidence from our own runs:

| Fact | Source | Consequence |
|---|---|---|
| Oracle F0.5 on our candidates = **0.9848**; top-50 needs ≥ 0.985 | E1 | Retrieval caps us out before a single scoring error. Retrieval must change, not just the matcher. |
| Indic-script R@5 0.86–0.88 vs Latin 0.97; 63% of India-S2 misses are Indic | E0b, E2 | Nothing word- or rapidfuzz-based can bridge Devanagari ↔ Latin. A learned encoder trained on our pairs can. |
| Misses are typos, glued words, empty addresses | E2 | Character/subword-level learned similarity, not more hand-written ratios. |
| Offline 0.9487 → LB 0.938 (**−0.011**) | E1 | Unmeasured. If it persists, the offline target for LB 0.985 is ≈ **0.995**. Must be measured (§3 C). |
| 7.64M labelled positive pairs, program-generated noise | phase0 | Ideal regime for a fine-tuned transformer; the model learns the generator's noise model. |
| E0b "dense" failure (char4 SVD256 + HNSW, R@5 0.64) | E0b | Not evidence against dense retrieval: it was unsupervised SVD. A fine-tuned bi-encoder is a different object. |

## 1. On using an LLM (≤ 8B) — allowed, but not the best use of the budget

Rules [C]: final model MIT/Apache-2.0, ≤ 8B params. Licence check matters: **Llama-3.x (Llama licence),
Gemma (Gemma licence), Qwen2.5-3B (Qwen Research licence) are NOT allowed.** Apache-2.0 options:
Qwen2.5-0.5B/1.5B/7B, Qwen3-*, Mistral-7B-v0.3. Verify every model card before use.

| Option | Throughput on one 48 GB GPU [H, order of magnitude] | Verdict |
|---|---|---|
| Zero-shot 7B LLM judging (S1, record) pairs | ~100–300 pairs/s with vLLM, short prompt, 1 output token → 30M test pairs = 1–3 days | Only feasible on a ≤ 2M gray zone. Doesn't use our labels. Gray zone is "same chain, other branch" — world knowledge doesn't help there. **Step 5 at most.** |
| Fine-tune 7B (LoRA) as pair classifier | Train slow, inference as above | Dominated by a small encoder in this time window. **No.** |
| Fine-tuned 100–300M multilingual encoder (bi-encoder + cross-encoder) | Encode ~5k texts/s; cross-encode ~3–5k pairs/s at seq 128 → 30M pairs ≈ 2–3 h | Uses the 7.6M labels; fixes cross-script; serves retrieval **and** scoring. **Do this.** |

## 2. Models (all ≤ 300M params; check licence tag on each card before use)

| Role | Candidate | Params | Notes |
|---|---|---|---|
| Bi-encoder (retrieval + cosine feature) | `intfloat/multilingual-e5-base` (MIT per card, verify) | 278M | xlm-roberta-base tokenizer covers all 9 Indic blocks + French. Use `query: ` prefix on both sides (symmetric task). |
| Faster fallback | `intfloat/multilingual-e5-small` | 118M | ~2.5× faster; use if encoding budget is tight. |
| Cross-encoder | `xlm-roberta-base` (MIT) or start from the fine-tuned e5 weights | 278M | Input: `name_s1 [SEP] addr_s1 [SEP] name_rec [SEP] addr_rec`, max_len 128. |

Text fed to the models: the **raw** (not ascii-folded) name + address, lowercased, `"null"` → `""`.
Folding destroys the script information the encoder needs. Keep our normalized views for the GBDT.

## 3. Plan and budget

### A. Bi-encoder (GPU, ~6–8 h wall clock incl. encoding) — highest priority
1. **Pairs.** Positives: all GT pairs (7.64M) — subsample to ~3M for one epoch if time is short, stratified
   by country × source × script. Hard negatives: for each positive, the top-3 TF-IDF S1 candidates of that
   record that are *not* its parent (already in `artifacts/baseline/train/*.parquet`, label 0). Orphan
   records get 3 hard negatives and no positive (teaches "no parent" geometry).
2. **Loss.** MultipleNegativesRanking (in-batch) + the explicit hard negatives, batch 256–512, lr 2e-5,
   1 epoch, fp16. sentence-transformers handles this out of the box.
3. **Validation.** Recall@k of the fold-0 records against the fold-0 S1 universe (same universe as
   E0b's rank table so numbers are comparable), reported by script and by address-empty.
4. **Encode.** Train (2.2M S1 + 10.3M records) and test (1.7M S1 + 10.0M records) ≈ 24M texts.
   At ~5k texts/s ≈ 80 min. Store float16 embeddings to parquet/npy on S3.
5. **Search.** Per country, per source: brute-force cosine top-k (k = 5) on GPU with torch matmul in
   S1 chunks of ~100k rows (1.7M × 10M × 768 fp16 is minutes; no ANN, no recall loss). Record-side, as
   R1 is today. Also emit the S1-side top-3 per source (forward view) — cheap once embeddings exist.
6. **Union** with TF-IDF top-3 → new `candidate_pairs`. Audit: oracle F0.5 vs mean |C| per S1
   (organisers score |C| too — keep k small, prune with the stage-2 model if |C| balloons).
7. **Features** for the GBDT: cosine, dense rank (record-side, S1-side), dense gap to best, retrievers
   agreeing (tfidf ∩ dense), plus everything existing.

Expected effect [H]: Indic R@5 from 0.87 toward Latin's 0.97; typo/glued-word misses drop; ceiling
above 0.99. The cosine feature also shrinks "found but rejected" (0.021) and FP (0.015).

### B. E3 + E4 on CPU, in parallel with A (~2 h)
- E3: `train --train-frac 1.0`, `num_leaves` 127, rounds 1000 with early stopping (`run_e3.ps1` exists).
- E4 stage-2 features from stage-1 OOF p: record's best p vs 2nd-best S1 (competition), per-S1 count of
  records with p > 0.5 per source (cap is ≤5 S2 / ≤6 S3 [R]), sibling evidence (max p of the entity's
  other-source candidates, name/address similarity between the S2 and S3 candidates themselves).

### C. Measure the offline → LB gap (~1–2 h, CPU) — needed to know what offline score to aim for
- **Density stress:** on fold 0, delete a random 20% of S1 entities (their records become orphans) so that
  records/S1 ≈ 5.7 as in test [R]. Re-score with the same T. If F0.5 falls ~0.01, the gap is density and
  the fix is tuning (T_first, T_rest) under the stressed universe (lean strict).
- **France:** label-free check — distribution of best-candidate p and of predicted k for France vs US/India
  on test. If France's p distribution is shifted low, it's a coverage problem the multilingual encoder helps
  with; if its predicted-k distribution looks wrong, it's a threshold problem.
- Optional LB probe (costs 1 of 5 daily submissions): submit with France predictions emptied. Δ tells you
  France's contribution exactly. [J] Do it only if A is delayed and a quota would otherwise be wasted.

### D. Cross-encoder (GPU, ~5–6 h incl. inference) — if A lands by 26 Sep evening
- Train on the union candidates of ~1M entities (all their candidates, natural label ratio), 1 epoch,
  max_len 128, fp16. Validate on fold-0 candidates.
- Inference only where it matters: pairs with stage-1 p in [0.05, 0.95] **or** dense rank ≤ 2. Feed the
  logit as a GBDT feature (stage 2) or average with the GBDT p — pick by paired bootstrap.
- Budget: ≤ 30M pairs × ~128 tokens at 3–5k pairs/s ≈ 2–3 h on the 48 GB card.

### E. Optional: zero-shot 7B (Apache-2.0) on the residual gray zone
Only after D, only on pairs still in [0.3, 0.7] after stage 2 (expect ≤ 1M). Prompt with both records,
ask for yes/no, use the logit as one more feature. Measure on fold 0 before trusting it.

## 4. Decision layer — small but free wins once p is better calibrated
- Isotonic-calibrate stage-2 p on OOF, then pick per entity the prefix of candidates (sorted by p) that
  maximises **expected F0.5** given the per-source caps. Replaces the global (T_first, T_rest) grid.
- Keep hard arbitration (S2/S3 one-to-one is exact in train GT [R]).

## 5. Submission schedule [J]
| When | Submission | Content |
|---|---|---|
| 25 Sep late | Sub #2 | E3 (+E4 if ready). Also the France-empty probe if a slot would be wasted. |
| 26 Sep | Sub #3–4 | Bi-encoder union candidates + cosine features + E4; tuned under density stress. |
| 27 Sep | Sub #5+ | + cross-encoder feature; expected-F0.5 decision. Last submission = best offline. |

Ask the organisers (Google Form): best vs last submission for the private LB — this decides whether
27 Sep experiments are allowed to be risky.

## 6. Deliverable constraints to keep in mind while building
- `candidate_pairs.tsv` must be exactly the post-union, post-prune set scored by the final model.
- The methodology doc must state model names, licences, parameter counts, and why blocking scales
  (per-country partition, record-side top-k, brute-force GPU search cost is O(|S1| · |records| · d) per
  country — state it and show the timing).
- Every pretrained checkpoint used goes in `requirements.txt` / the README with its licence.