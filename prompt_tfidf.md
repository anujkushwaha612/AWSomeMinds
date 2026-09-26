# Task: a better TF-IDF candidate generator (raise the 0.985 ceiling) — for Claude Code on the teammate laptop

Read [prompt.md](prompt.md) first (context, rules, sub02). Deadline **27 Sep 2026 23:59 IST**; have a validated
submission file ready by **~20:00 IST** and hand it to the uploader (only one person logs in to the portal).

## Why

Every model so far scores the same candidate list, so its fold-0 **oracle** (perfect classifier on our
candidates) is stuck at **0.9848** (pair recall 0.9566, 14.0 candidates/S1). sub02 = 0.9648 offline / 0.957 LB,
v5-lite stage 1 = 0.9678; LB top is ~0.988–0.991, above our ceiling. Only more true pairs in the candidate list
can move it. Stage-1 pruning is cheap (fold-0 prune curve: 14.0 → 4.4 candidates/S1 at τ 0.003 costs 0.00001 of
oracle), so a larger candidate list is affordable as long as featurization fits in time and RAM.

## Current retriever (what to improve)

- `src/ber/blocking/word_retrieval.py::retrieve_country`, called by `src/ber/baseline.py::build`.
- Record-side only: every S2/S3 record → top-`k_rec` S1 of the same country by cosine; text =
  `CountryStore.na_text` = `name_n + " " + addr_n`; word 1+2-grams, sublinear TF × IDF (hashing, 2^22),
  n-grams in > `max_df` (1%) of the country's records dropped. Config `configs/pipeline.yaml` → `baseline`
  (`k_rec: 3`, `max_df: 0.01`, `min_score: 0.0`, `workers`, `chunk`, `feature_chunk`).
- Why it misses (experiments.md E2/E4/E6): TF-IDF recall@1/2/3 = 0.909/0.930/0.939 India, 0.948/0.962/0.969 US
  (still rising at rank 3); misses are Indic-script names (63% of India S2 misses: no shared token with the
  Latin S1 name), empty addresses (22–49%), chain names, domain / garbled / trade names; and the generator's
  noise splits tokens: `003017` vs `3017`, `street` vs `st`, `texas` vs `tx`, `bombay` vs `mumbai`, filler words.

## Step A — measure first (fold-0 recall study, ~1–2 h). Build nothing full-scale before this table exists.

Write `experiments/e8_retrieval_variants.py`: per country, index ALL S1 of the country (the real universe),
query the S2/S3 records (all, or a large sample — report which), and for fold-0 entities report
**pair recall, oracle macro F0.5** (`ber.baseline.EntityScorer(...).per_entity(label)` on the candidate labels,
`label = truth_parents(store, truth)[src][doc_row] == s1_row`), **candidates/S1**, and wall time. Variants:

| # | Variant | What it tests |
|---|---|---|
| V0 | current (k 3, `na_text`, max_df 0.01) | must reproduce pair recall ≈ 0.9566 / oracle ≈ 0.9848 (sanity) |
| V1 | k_rec 5 / 10 | depth: how much recall sits at ranks 4–10 |
| V2 | **canonical text** + k 3 / 10: leading zeros stripped from number tokens; address tokens mapped with `ber.features.ADDR_CANON` (street→st, states→codes, old city names); `ber.features.NOISE_TOKENS` dropped from names; `null` dropped | fixes the generator's token splits |
| V3 | **transliterated view** `name_tr + addr_tr` (Latin for every script; both columns exist in `artifacts/norm`), k 5/10, unioned with V2 | Indic-script records |
| V4 | S1-side view: top-5 records per S1 per source (E4: ≤ 5 true matches per source for 100% / 99.9% of S1) | records whose parent is not in their own top-k |
| V5 | max_df 0.02 with V2 | whether the 1% cap now drops useful canonical tokens |
| V6 (optional) | name-only view for empty-address records; address-only view for records whose name shares no token with any candidate | the two weakest regimes |

Report one table (per country and total) and pick the cheapest combination whose **oracle gain per added
candidate/S1** is best. Suggested go/no-go: oracle ≥ 0.988 (+0.003) at ≤ ~30 candidates/S1 — adjust to this
laptop's RAM and to the time left. Leakage: retrieval uses no labels; `NOISE_TOKENS` was mined on fold 3; do not
tune any retrieval knob on fold-0 labels beyond picking among these few variants (report fold 1 too if unsure).

## Step B — full rebuild on the chosen variant (only if Step A passes)

Use a **new run folder** so nothing existing is overwritten (e.g. `BER_RUN=tfidf2`):

1. Implement the chosen variant behind config keys (e.g. `baseline.text_view: na | canonical`,
   `baseline.extra_views: [translit, s1_side]`), default = current behaviour so old runs stay reproducible.
   Several views → one candidate table: outer-join on (src, doc_row, s1_row), keep per-view score / rank columns
   and `n_views`, then the existing `retrieval_features` on the merged set (see `ber.union.merge_candidates` for
   the pattern). Add the new columns to the feature list used by training.
2. `BER_RUN=tfidf2 python -m ber.baseline build --split train` and `--split test` (identical retrieval on both).
   Raise `baseline.workers` / lower `feature_chunk` to match this laptop's cores / RAM.
3. `python -m ber.gen_augment --split train --src-run tfidf2 --dst-run tfidf2_gen` (and `--split test`):
   adds the E6 GEN + CHAIN features.
4. Train: `BER_RUN=tfidf2_gen BER_FEATURES=tfidf2_gen BER_GEN=genchain python -m ber.baseline train --train-frac <x> --num-rounds 1500`
   (use as much data as RAM allows; 50% of 31M rows peaked at 5.2 GB, memory grows with candidates/S1).
   Then the v5 two-stage run on it: copy `run_v5lite.ps1`, point `v5.dirs.union` at `tfidf2_gen` and
   `v5.dirs.v5` at a new folder (add a `-Union` parameter rather than editing the default).
5. **Compare against sub02 per entity.** `ber.baseline compare` needs identical candidate rows, which no
   longer holds. Entity order is identical across runs (S1 row order of the store), so compute per-entity fold-0
   F0.5 for each run with its own keys + decision and run `ber.eval.scorer.paired_bootstrap` on the two aligned
   arrays (add a small `compare_entities(run_a, run_b)` helper). Reference: `baseline_gen` (sub02, 0.96482) and
   `v5_lite` (laptop 1, results in `artifacts/v5_lite/`).
6. Predict: `BER_RUN=tfidf2_gen BER_FEATURES=tfidf2_gen BER_GEN=genchain python -m ber.baseline predict --name sub04_tfidf2`
   (validator must print PASS). Upload only if the per-entity comparison vs the best current run has CI > 0.

## Constraints and coordination

- Do not touch `artifacts/baseline`, `artifacts/baseline_gen`, `artifacts/v5_lite` (laptop 1 depends on them).
- The Kaggle dense path (`after_kaggle.ps1`) is tied to the OLD TF-IDF row order (`*_tfidf_cos.npy` is aligned
  to `artifacts/baseline`), so do not re-point `v5.tfidf_run` at the new run; the two paths stay separate.
- Keep `pytest -q` green; add a test for the canonical-text function (zero stripping, `st`, states, noise tokens).
- Record Step A's table and Step B's result as **E8** in experiments.md; commit on this branch.
- No external data / services; models only MIT/Apache ≤ 8B (TF-IDF needs none).
