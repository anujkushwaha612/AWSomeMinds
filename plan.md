# strategy.md — Execution Plan
## Amazon ML Challenge 2026 — Business Entity Resolution
**Version:** v4 (review-corrected build plan). Supersedes v3.
**Changelog v3 → v4:** adds 9 full-data re-measurements [R] (orphans, per-source cap, train→test density shift, all-Indic scripts, the address carrying cross-script pairs); replaces 70/15/15 slice validation with full-universe OOF; converts candidate audit from pair recall to entity-level oracle F0.5; adds address-only view, reverse retrieval, stage-2 competition/sibling features; drops country features/thresholds and 3:1 subsampling; rewrites gates G1/G4/G5/G6/G7 so each can answer its question; adds guidelines-PDF constraints; moves compute to SageMaker.
**Provenance tags:** [C] challenge PDFs · [D] Phase-0 measurement (phase0_report.md §§1–6) · [R] full-data re-measurement 2026-09-25 (phase0_report.md §8) · [L] literature · [H] hypothesis pending experiment · [J] judgement call (no evidence; chosen in advance).

---

## 0. The objective, exactly

For every entity in `test_source1.tsv` (1,732,544 rows [D]), output the set of matching S2/S3 records:

$$F_{0.5}^{(i)} = \frac{1.25 P_i R_i}{0.25 P_i + R_i} \quad \text{per S1 entity, macro-averaged; singletons score 1.0 iff predicted empty}$$

**Per-decision cost, derived from the formula:**

| Error | Cost |
|---|---|
| 1 FP on k=0 | −1.0 |
| Missing the only match on k=1 | −1.0 |
| 1 FP on k=4 (all 4 found) | −0.167 |
| 1 miss of 4 | −0.0625 |

So precision and the first admission (empty vs non-empty) dominate.

**Hard constraints [C]:**
- Two output files: `matching_results.tsv` (scored) + `candidate_pairs.tsv` (the exact **post-cap** set fed to the final model; matches ⊆ candidates; audited for recall ceiling / reduction ratio).
- Exactly one row per test S1 entity (empty list if singleton). S2/S3 ids only; no duplicates.
- No external data or lookups of any kind (no geocoding, registries, ER APIs, internet augmentation). Code is reviewed; a violation means disqualification.
- Final model: MIT/Apache-2.0, ≤ 8B params. Check the model card of every pretrained model used.
- **Window: 25 Sep 00:00 IST → 27 Sep 23:59 IST; 5 submissions/day.** An unused day's quota is lost.
- **Shortlisting is "based on performance across both leaderboards"** (guidelines PDF), while the problem statement says final ranking is on the private LB. Treat the **public score as mattering too**. **Top 100** teams are shortlisted.
- Deliverables:
  - zip: `output/` + `code/business_entity_resolution/{src, README.md, requirements.txt}` + filled `Documentation_template.md`
  - a **1–2-page summary** (approach, models, experiments, conclusion)
  - source code **with comments describing each function**
  - **version history of all submissions** (shortlisting is based on the submitted solutions)
- The portal is used from one machine per participant (no simultaneous logins).
- **Ask the organizers via the Google Form:** which submission is used for the private LB, the last one or the best one? The Sub #5 strategy (§3) depends on the answer.

**Compute [R]:** all-pairs (~2×10¹³) is impossible, so the cascade is required.
- Work runs on **SageMaker** (credits provided): a high-memory, high-core CPU instance for retrieval, features and GBDT, and a GPU instance only for the optional transformer and dense arms.
- Instance types and credit limits: *TBD, pending the AWS prep blog (JS-only page, not yet read; paste its text to fill this in).*
- Every stage writes parquet to **S3**, because notebook restarts lose local state.
- The local workstation (15.7GB, 16 threads, no GPU) is fine for development on slices.

---

## 1. What the data told us (facts that shape every step)

| Fact | Consequence for the build |
|---|---|
| [D] k histogram (train): 0:5.6% · 1:5.4% · 2:17% · 3:24% · 4:22% · 5:14.6% · 6:7.5% · 7+:4% | Multi-match decisions are the bulk of the mass; k∈{0,1} has the highest per-error leverage. The k-prior is a **diagnostic only** (see R4). Decision layer starts from global T → `T_first`/`T_rest`; per-rank T_r is [H] and cut unless those plateau |
| [R] **Per-source cap: ≤5 S2, ≤6 S3** matches per S1 | Top-K is set **per source**, not over a pooled index |
| [D] S2/S3 one-to-one (7.64M pairs, 0 reuse) | Arbitration permissible. Test it as soft (stage-2 features) vs hard, on full-universe OOF |
| [R] **Orphans: ~26% of S2/S3 records have no S1 parent** | Validation must include them at realistic density. Record-competition features matter |
| [R] **Density shift: 4.68 records per S1 (train) vs 5.76/5.82/5.53 (test US/India/France)** | Train-tuned thresholds may be too loose on test. Density-stress check (Step 8); lean strict if thresholds move |
| [D] 80.5% of S1 have matches in both S2 and S3 | **Use cross-source sibling evidence** [H]: a clean S2 match can anchor a noisy S3 one. (v3's "predict independently" did not follow) |
| [D] 60–76% of true pairs differ exactly on both name and address | Fuzzy retrieval + learned matcher are mandatory |
| [R] India name scripts: S2 Devanagari 13.4% + **other Indic 10.2%**; S3 7.5% + 5.7%; test = train; S1 100% Latin | Transliteration must be **all-Indic** (9 parallel Unicode blocks → one table + offset). A Devanagari-only version misses ~43% |
| [R] Low-name-overlap India pairs: **address-Jaccard median 0.65–0.76** | **Address-only retrieval view is the first India fix**; transliteration second; dense last |
| [R] Domain-style names (`X.COM`) are 3–4% | Space-stripped name similarity feature |
| [R] Orphan names rarely equal an S1 name (≈5% vs ≈25% for matched records) | Mined negatives are probably clean; audit the top 100 anyway |
| [D] S3 noisier than S2 on exact address (4.4% vs 13.3%); name-exact equal | Source (S2/S3) is a feature and the only segmentation candidate |
| [D] France = 15.0% of test S1; diacritics; "sarl"; addr-dup 21%; shift "moderate" is [H] (input marginals only) | ascii-fold everywhere; **no country feature, no country thresholds**; label-free checks + one LB probe |
| [D] Same-country invariant exact in train GT | Hard filter by exact country string |
| [D] IDs random; 0 train/test overlap | No ID features |
| [D] S1 fields never empty; S2/S3 ~3% empty addresses; literal "null" | Missingness-state features; "null" → "" |
| [D] Name-dup 27–37% (chains); 11–16% shared S2/S3 addresses | Name alone is never decisive; frequency/chain features; contradiction features |
| [D] Char-TF-IDF@10 on miniature: 99.5 / 96.6 | Shape only; **not used to set K or gates**. The full-universe rank table decides (Step 3) |

---

## 2. Final pipeline (one screen)

```text
 S1, S2, S3 (tsv → parquet on S3)
      │
 [1] NORMALIZATION (additive views; raw kept)
      │   lower_ascii (NFKD fold, punct→space, "null"→"") · legal-strip view · token-set
      │   space-stripped name · script flags · digits (house-no, postal, digit-runs)
      │   [EXP] all-Indic transliteration view
      ▼
 [2] CANDIDATE GENERATION (same-country hard filter first)
      │   char-3g TF-IDF (max_df-pruned), top-k PER SOURCE:
      │     name+addr (backbone)  ∪  address-only [EXP, strongly indicated]
      │     ∪ reverse S2/S3→S1 top-k [EXP]   ∪ name-only [EXP]
      │   plain union → per-source cap → candidate_pairs.tsv
      │   AUDIT: oracle macro-F0.5 per stratum (post-cap), |C|, RR, runtime
      ▼
 [3] PAIR SCORING
      │   features (no country, no target encoding) → LightGBM stage 1 (5-fold full-universe OOF)
      │   [EXP] stage 2: + record-competition & sibling features from stage-1 OOF
      │   [EXP, GPU] cross-encoder score on gray-zone pairs → fed as a GBDT feature
      ▼
 [4] ENTITY DECISION LAYER
      │   global T → T_first/T_rest → T[source]     (paired-bootstrap selection)
      │   [EXP] expected-F0.5 prefix selection on isotonic-calibrated p
      │   [EXP] hard arbitration (each S2/S3 → its argmax S1)
      ▼
 matching_results.tsv + candidate_pairs.tsv → validate_submission.py → submit
```

---

## 3. Step-by-step plan

### Step 0 — Repo, versioning, compute (Day 1, ~45 min)
- `git init`. Layout: `src/{normalization,blocking,features,models,decision,eval}/ · configs/ · output/ · phase0/ · subs/`.
- One `configs/pipeline.yaml` holds every number (paths, K per view/source, thresholds, seeds).
- **Per-submission snapshot** `subs/<n>_<date>/`: commit hash, config, both TSVs (or their S3 URIs), offline metrics, LB score. This meets the "version history" rule [C].
- Upload `data/dataset/` to S3. Start the SageMaker CPU instance.
- Copy the review scripts (`check*.py`) into `phase0/`.
- **Docstring every function as it's written** [C: commented code].

### Step 1 — Evaluation harness FIRST (Day 1, ~1.5h)
1. **Exact scorer:** per-entity P/R → F₀.₅ → macro, including the singleton rules. Unit-test ≥10 hand cases, including the [C] example (0.714).
2. **Paired-bootstrap Δ utility:** resample entities, report the 95% CI of Δ macro-F₀.₅ between two prediction sets on the same entities. **This is the only definition of "noise" used by every gate.**
3. **Oracle-F0.5 of a candidate set:** per entity, P=1 and R = fraction of true matches present.
4. Diagnostics by stratum: country × script{Latin, Devanagari, other-Indic} × k-bucket{0,1,2,3,4,5+} × source.
5. Submission writer + a wrapper around `data/utils/validate_submission.py`.

### Step 2 — Normalization (Day 1, ~1h)
Additive views, never destructive:
- `lower_ascii`
- `legal-strip` (suffix dictionary harvested from **unlabeled** token frequencies per dataset: llc, inc, corp, ltd, pvt, private, limited, llp, sarl, sas…; keep `had_suffix` and `suffix_conflict` flags)
- `token-set`
- `space-stripped`
- script flag per record
- digit extractions

All-Indic transliteration (ISO-15919-style table applied via per-block offset) is written now but used only after Step 3 shows where it helps.

### Step 3 — Full-universe rank table (Day 1, the decisive recall experiment)
1. **Timing run** on a 1% slice per view. Extrapolate before launching the full run. High-df n-grams dominate the cost, so prune them with `max_df`.
2. On the **full train universe** (all S1 × all S2/S3, per country; India first, then US):
   - Views: NA = name+addr, A = address-only, N = name-only (T = transliterated added later if needed).
   - Directions: forward S1→S2 and S1→S3 separately at k ≤ 50, plus reverse S2/S3→S1 at k ≤ 10 for NA.
3. Store **each true pair's rank in every (view, direction)**. Every view subset and K can then be evaluated offline without re-running retrieval.
4. Report per stratum:
   - oracle macro-F0.5 of the **capped union**
   - marginal oracle gain per view
   - |C| median / p90 / max
   - reduction ratio
   - wall-clock
5. **Selection rule:** add views greedily by oracle-F0.5 gain per unit |C|. Stop when the next gain's CI includes 0. K per source is the smallest K whose oracle F0.5 is within the CI of K_max.
6. The same code runs on test to produce `candidate_pairs.tsv`. These tables go straight into the methodology doc's blocking-audit section [C].

### Step 4 — Validation design (Day 1)
- **5-fold entity-level cross-fitting on the full train universe.** S1 entities are stratified by country × k-bucket; a pair inherits its S1's fold. Orphans enter only as negatives, wherever they're retrieved.
- This replaces v3's 70/15/15 slices and 3 seeds. Slices shrink the distractor universe and hide conflicts with other S1s.
- Unsupervised items (normalization dictionaries, IDF, frequency features, retrieval index) are computed on the full dataset's inputs: no label leakage. Anything learned from GT (e.g., target encodings, alignment-mined abbreviations) would be fold-internal. v4 uses none.
- Decision parameters are tuned on folds 1–4 OOF and **reported on fold 0**. Calibration (if used) is fitted on OOF only.
- **Order:** normalize → retrieve (full universe) → features → stage-1 CV (OOF) → stage-2 features from OOF → stage-2 CV → decision tuning → arbitration → final refit on all train → test inference with identical code.

### Step 5 — Training pairs (Day 1)
- All candidates of the sampled entities (all entities if memory allows on SageMaker). **No 3:1 negative subsampling**: it distorts the score distribution the thresholds are tuned on.
- Labels: GT pairs are positive; every other candidate is negative. R8 suggests the labels are clean. **Manually audit the top 100 highest-scored negatives** once, after stage 1.

### Step 6 — Features + stage-1 LightGBM (Day 1 → Sub #1)
- **Name:** Jaro-Winkler, normalized Damerau-Levenshtein, token Jaccard/containment, sorted-token equality, char-3g cosine, IDF-weighted token overlap, acronym-vs-expansion, **space-stripped similarity**, legal-suffix states, name-frequency (chain pressure).
- **Address:** the same battery, plus **address-only cosine**; house-no / postal state ∈ {match, conflict, one-missing, both-missing}; digit-run Jaccard; street-token overlap; missingness.
- **Retrieval:** per-view rank and score, number of views retrieving the pair, rank within entity per source.
- **Context (country-free):** source (S2/S3), script flags (both sides), candidate count of entity, name/address length ratio. **No country feature, no source×country target encoding**: France is unseen, and target encoding is a leakage vector.
- **Contradiction:** numeric conflict, suffix conflict, city-token conflict.
- **[EXP]** Transliterated-name similarity (fires on Indic-script pairs).
- **Compute:** vectorized `rapidfuzz.process.cpdist(..., workers=-1)` and sparse row-wise dot products. Time 1M pairs first.
- **Models:**
  - M0: best single similarity + threshold (cheap floor / fallback)
  - M1: LightGBM stage 1, 5-fold OOF
- The comparison metric is **end-to-end macro-F₀.₅ after the decision layer, by paired bootstrap**. Pair AUC is never the deciding metric.

### Step 7 — Stage 2: competition & sibling features (Day 2) [EXP]
From stage-1 OOF scores:
- **Record-side:** best score of any *other* S1 for this record, this S1's margin over it, and this S1's rank among the record's S1 candidates. This is soft arbitration.
- **Entity-side:** gap to the entity's best candidate in the same source, and rank within the source.
- **Sibling:** the candidate's max similarity to the entity's high-scoring candidates in the *other* source.

Then stage-2 LightGBM on the same folds.

### Step 8 — Decision layer (Day 2)
Evaluated on fold 0, with parameters tuned on folds 1–4. The simplest arm within the CI wins.

```text
D1  global:           include j iff p_j > T
D2  first/rest:       include top-1 iff p_1 > T_first; others iff p_j > T_rest     (2 params)
D3  per-source:       D2 with T_rest[S2], T_rest[S3]
D4  [EXP] expected-F0.5 prefix: sort by isotonic-calibrated p; choose prefix length
    (incl. 0) maximizing expected F0.5 under independence
A0/A1/A2 arbitration: none / hard (record → argmax S1 above T) / soft (= stage 2 only)
```

- Isotonic calibration is used **only with D4**. For D1–D3 it's a monotone re-parameterization of T and cannot change the result.
- **Density-stress check:** drop ~20% of train S1 entities from the query set (their partners become extra orphans), re-tune D2/D3, and compare the thresholds. If they move beyond the CI, ship the stricter setting (R4).
- Diagnostic only: predicted-k histogram on test vs train OOF; record-conflict rate on test vs OOF.

### Step 9 — France (Day 2)
- Same pipeline and the same thresholds (no country-specific parameter exists anywhere).
- **Label-free checks:** France vs US on (a) top-1 candidate-score distribution, (b) predicted-k histogram, (c) record-conflict rate. A large divergence flags a problem but doesn't pick a fix.
- **LB probe (1 submission):** identical to the current best except the France rows (e.g., France T_rest + δ). The LB Δ gives the direction for France. France is ~15% of public entities.

### Step 10 — Optional GPU arms (Day 2–3, time-boxed) [EXP]
- **Cross-encoder:**
  - Time 100k pairs first.
  - Fine-tune `xlm-roberta-base` (MIT) or a multilingual MiniLM (Apache-2.0; verify the model card) on OOF **gray-zone** pairs (stage-2 p near threshold). Serialize as "NAME … | ADDRESS …".
  - Feed the score **as a feature** into stage 2; never replace the GBDT.
  - Only pursue if gray-zone count × throughput fits within a few GPU hours.
- **Dense retrieval:** only on the Indic-script stratum (≈0.5M test S2/S3 + 0.81M India S1), and only if that stratum's oracle loss is still material after the address view and transliteration.

### Step 11 — Packaging (Day 3)
- Zip: `output/{matching_results.tsv, candidate_pairs.tsv}` + `code/business_entity_resolution/{src, README.md, requirements.txt}` + filled `Documentation_template.md`.
- Also produce the 1–2-page summary.
- README: exact reproduction steps, data → blocking → matching → output.
- Methodology: problem framing, rank-table / oracle-F0.5 audit, features, stage-1/2 results, decision-layer selection, ablation table (paired-bootstrap CIs), fair-play statement (no external data).
- Run `data/utils/validate_submission.py` before every upload [C].

### Submission schedule (15 total; public score also counts)

| Day | Submission |
|---|---|
| Day 1 | **Sub #1:** stage-1 LightGBM + D2. Validates the format and the LB↔offline correlation. |
| Day 2 | **Sub #2:** stage 2 + best decision arm. **Sub #3:** France-only probe (Sub #2 with only the France rows changed). **Sub #4:** best arm (arbitration / transliteration / density-strict) if one wins. Spend the remaining quota on low-risk variants rather than leave it unused. |
| Day 3 | **Freeze experiments by 12:00 IST.** Final refit on all train. **Sub #5:** best offline (+ France direction from Sub #3). If the organizers say "last submission counts", make the final one the best offline. Otherwise add a stricter-threshold hedge if the density check flagged it. |

- Never chase LB deltas smaller than the offline CI.
- At least one offline experiment per submission.
- Log offline and LB scores together in `subs/`.

---

## 4. Decision gates (pre-registered)

"CI" = 95% paired-bootstrap CI over fold-0 entities (full-universe OOF). The materiality bar **+0.002 macro-F0.5 is [J]**. Change it now if you want, not after seeing results.

| Gate | Question | Why this metric answers it | KEEP | DROP | INVESTIGATE |
|---|---|---|---|---|---|
| G1 view | Does view V (A, reverse, N, T, dense) pay for itself? | Marginal **oracle-F0.5** on its target stratum is the entity-level ceiling it adds | Gain CI > 0 and ≥ 0.002 on the stratum, runtime acceptable | CI includes 0 | Gain real but runtime > 2h → scope-reduced version |
| G1b K | Smallest safe K per source | Oracle-F0.5 vs K curve, post-cap | Smallest K within CI of K_max | — | Curve still rising at K_max → extend |
| G2 ML | Does M1 beat M0? | End-to-end macro-F0.5 | M1 − M0 CI > 0 (expected) | Otherwise ship M0 | — |
| G3 stage 2 | Do competition/sibling features help? | End-to-end, same decision arm | Δ CI > 0 and ≥ 0.002 | Otherwise | — |
| G4 decision | Is a richer rule better? | End-to-end on fold 0, params from folds 1–4 | D(n+1) beats D(n), CI > 0 | Keep the simpler rule | Gains differ strongly by country → transfer risk to France; keep the simpler rule |
| G5 arbitration | Does hard arbitration add over soft? | Full-universe OOF, where conflicts with *all* S1s are visible | A1 beats A2, CI > 0 | Otherwise | Test conflict rate ≫ OOF → re-evaluate |
| G6 France | Is the France policy directionally wrong? | The LB probe is the only France-labelled signal | LB Δ from the France-only change exceeds public noise → adopt that direction | Δ ≈ 0 → keep global | LB contradicts the label-free checks → trust the LB |
| G7 density | Do thresholds depend on density? | Re-tune under simulated extra orphans | Stable → ship as is | — | Moved beyond CI → ship the stricter setting |
| G8 GPU arms | Do the cross-encoder or dense arms add? | End-to-end with the score as a feature / oracle gain on the Indic stratum | Δ CI > 0 and ≥ 0.002 within the time box | Otherwise, or the time box is exceeded | — |
| G9 runtime | Can the full test pipeline re-run within one iteration slot? | Wall-clock on SageMaker | ≤ ~3h [J] | — | Longer → shrink K / drop the lowest-gain view (the audit is never dropped) |

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Train→test density shift (R4)**: thresholds too loose on test | Density-stress check (G7); lean strict; watch the predicted-k histogram on test |
| **Orphan-driven FPs** (~26% of records have no parent) | Record-competition features; full-universe validation; hard arbitration arm |
| Biased validation (small universe, hidden conflicts) | Full-universe 5-fold OOF (Step 4). *Mitigated by design* |
| India cross-script residue | Address-only view first; all-Indic transliteration; dense only via G1 |
| France miscalibration | No country parameters; label-free checks; LB probe (G6) |
| Unknown public/private selection rule | Ask via the Google Form Day 1; schedule assumes "last counts" until answered |
| Public score also counts for shortlisting | Probes are always near-best variants, never deliberately weak |
| SageMaker notebook restart / disk loss | Every stage writes to S3; idempotent stage scripts |
| Runtime blowups at 10M-record scale | 1% timing runs before every full job; `max_df` pruning; chunk per country |
| Overfitting the public LB | Offline-first; paired-bootstrap gates; ≤2 LB probes per pipeline generation |
| Format rejection | Validator before every upload [C] |
| Time slip | Pre-registered gates; experiment freeze Day 3 12:00 IST; docstrings written continuously |

---

## 6. Cut list (with reasons)

**Cut for logic or low information gain** (compute is not the reason, so SageMaker doesn't bring them back):

| Item | Reason |
|---|---|
| CatBoost check | Marginal expected gain in 3 days |
| BM25 / token view | Char n-grams cover the token cases; tokens fail on typos and `.com` forms |
| RRF / capping A/B | Per-source structure allows a plain union with small K; the model ranks |
| L0 exact keys, L3 postal/house-no as blockers | Subsumed by TF-IDF, or blocks too large to survive a cap. Kept as **features** |
| Per-rank threshold vector T_r | Not supported by the k histogram; D2 captures the metric-driven part |
| Country thresholds, country feature, source×country target encoding | Undefined for France; target encoding leaks labels |
| Isotonic applied to D1–D3 (v3 "F7") | Monotone, so it cannot change a threshold rule's result |
| 3-seed 70/15/15 splits | Replaced by full-universe OOF + paired bootstrap |
| France per-feature shift table as a gate | Input marginals can't predict matching loss |
| 3:1 negative subsampling | Distorts the score distribution used for thresholds |

**Demoted to gated experiments** (affordable on SageMaker, still must pass G8/G1): transformer cross-encoder (as a feature only), dense retrieval (Indic stratum only), GBDT+transformer ensemble (subsumed by "score as a feature").

## 7. Provenance appendix
- [C] problem statement PDF + guidelines PDF (outputs, metric, constraints, window, 5/day, both leaderboards, top-100, 1–2-page doc, commented code, version history, license cap) + `data/README.md`.
- [D] phase0_report.md §§1–6 (+ `phase0/stats/*.json`, not present locally).
- [R] phase0_report.md §8: full-data re-measurement 2026-09-25 (`check.py`, `check2.py`, `check3.py`).
- [L] strategy_v2_corrected.md references (hard negatives; blocking surveys; F-measure decision theory); feature-based GBDT as a strong ER baseline.
- [H]/[J] flagged inline; each [H] maps to a Step and a gate in §4.
- **Pending:** AWS "ML Challenge 2026 prep guide" blog (SageMaker setup and limits). The page is JS-rendered and couldn't be fetched; paste its text to fill in §0 Compute.
