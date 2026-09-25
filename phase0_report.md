# Phase 0 Report — Actual Dataset Measurements
## Amazon ML Challenge 2026 — Business Entity Resolution
**All numbers below are measured from the real challenge data** (downloaded 2026-09-24), replacing every assumption/hypothesis marked [H] in Strategy v1/v2 that they touch. Scripts in `phase0/`, raw stats in `phase0/stats/`, exemplars in `phase0/exemplars/`.

> **Review update (2026-09-25):** §8 holds the adversarial-review corrections and re-measurements on the full data, tagged **[R]**. **Where §8 conflicts with §§2–6, §8 wins.** Inline notes marked *[review]* point to it. Tags: [D] = Phase-0 measurement · [R] = re-measured on full train/test files · [H] = hypothesis.

**Environment constraint (sandbox only):** the Phase-0 sandbox had 1.9GB RAM / 2 cores, so all analysis was streaming + sqlite/external-sort. *[review]* This constraint **does not apply to the team machines**: the local workstation has 15.7GB RAM, 16 logical CPUs and no CUDA GPU [R], and production runs on **SageMaker** with the provided credits (instance types TBD). One thing does hold at any hardware size: full-pair scoring of 2.2M × 10.1M is impossible, so the cascade (blocking → scoring → decision) is still required.

---

## 1. Scale and splits

| File | Rows | Countries |
|---|---|---|
| train S1 | 2,206,821 | US 1,323,633 / India 883,188 |
| train S2 | 5,034,616 | US 3,016,817 / India 2,017,799 |
| train S3 | 5,285,603 | US 3,170,056 / India 2,115,547 |
| test S1 | 1,732,544 | US 663,106 / India 809,986 / **France 259,452 (15.0%)** |
| test S2 | 4,887,273 | US 1,871,330 / India 2,312,565 / France 703,378 |
| test S3 | 5,082,316 | US 1,945,701 / India 2,405,000 / France 731,615 |

- GT: one row per train S1 ✓; **0 dangling refs**; **0 cross-country GT pairs** (same-country filtering confirmed as a free, exact blocking filter); **0 train/test id overlap**.
- IDs are **random 9-digit numbers** (mean |Δnum| between true partners ≈ 333M ≈ uniform): **no positional/ID leakage or ordering signal**.

## 2. Match-count structure (the decision layer's prior)

| k matches | entities | share |
|---|---|---|
| 0 | 123,247 | **5.58%** |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.06% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7–11 | 87,424 | 3.96% |

Mean 3.46 matches/entity, max 11. **80.5% of S1 entities have matches in BOTH S2 and S3** (6.5% only-S2, 7.5% only-S3). Mean matches per source ≈ 1.67 (S2) / 1.79 (S3).

**S2/S3 exclusivity: EXACT — all 7,638,365 GT pairs have exactly 1 S1 parent (max 1, reuse 0.0%).** The arbitration diagnostic demanded by the adversarial review has now been run: the S2/S3 side *is* one-to-one in training GT. Arbitration is therefore *permissible*. *[review]* That test GT follows the same rule is a **strong inference, not verifiable**. Arbitration still enters only as a net-effect ablation (it only fires if the matcher proposes cross-entity conflicts).

*[review] Additional structure [R]:*
- **Orphans:** 26.6% of train S2 records (1,340,997) and 25.4% of train S3 records (1,340,857) have **no S1 parent**. About a quarter of the retrievable universe consists of records with no correct answer.
- **Per-source cap:** each S1 has **≤5 S2** and **≤6 S3** matches.
  - S2-count histogram: 0:287,745 · 1:789,108 · 2:652,779 · 3:333,957 · 4:119,078 · 5:24,154.
  - S3-count histogram: 0:266,276 · 1:716,417 · 2:668,375 · 3:372,443 · 4:145,116 · 5:35,378 · 6:2,816.
- **Train→test density shift:** S2+S3 records per S1 are **4.68 in train (identical for US and India)** versus **5.76 US / 5.82 India / 5.53 France in test** (from the §1 row counts). This is a measured fact. **Whether it comes from higher k or from more orphans is not established.** Consequence: the train k-prior and train-tuned thresholds may not transfer.

**Strategic reweighting:** singletons are only 5.6% of macro mass (v1 feared 25–40%). The dominant mass (k=2–6, ~85%) makes multi-match set decisions the bulk of the metric. *[review] Correction:* per error, **k∈{0,1} (11% of entities) is the highest-leverage band**. One FP on a k=0 entity, or a miss on a k=1 entity, costs the whole entity (1.0 → 0). For comparison, one FP on a k=4 entity with all 4 found costs 0.167, and one miss out of 4 costs 0.0625. Do not de-prioritize this band.

## 3. Noise structure (measured on GT pairs, per source-pair × country)

| metric | S2\|US | S2\|India | S3\|US | S3\|India |
|---|---|---|---|---|
| pairs | 2.213M | 1.481M | 2.365M | 1.579M |
| name exact (normalized) | 30.5% | 17.5% | 30.5% | 20.0% |
| **both name & addr differ exactly** | **60.2%** | **73.2%** | **65.1%** | **75.9%** |
| addr exact | 13.3% | 11.3% | 4.4% | 4.2% |
| name token-Jaccard < 0.25 | 8.3% | **28.7%** | 8.2% | **20.1%** |
| addr empty on one side | 4.9% | 3.8% | 4.6% | 4.0% |

- **The "maybe the data is clean" scenario is dead**: 60–76% of true pairs match exactly on *neither* normalized field. Fuzzy retrieval + learned matching is mandatory.
- **Cross-script India pairs are real**: S2-India has 13.4% Devanagari names while **train S1-India has 0%**. For ~20–29% of India true pairs the name shares almost no tokens with its partner (token-Jaccard < 0.25).
  - *[review] Correction [R]:* the original attribution of that residue to *Devanagari script mismatch* is **overclaimed**. Devanagari accounts for only **44% (S2) / 33% (S3)** of the low-overlap India pairs (S2: 192,076 of 436,216; S3: 111,142 of 336,315).
  - The rest is mostly **other Indic scripts**, plus domain-style names (`KAIROSSONS.COM`), garbled tokens (`Fayexylo`) and truncation (`HI`).
  - India name scripts (share of records):

    | | Latin | Devanagari | other Indic | breakdown of other Indic |
    |---|---|---|---|---|
    | train S2-India | 76.5% | 13.4% | **10.2%** | Telugu 1.95 · Kannada 1.84 · Tamil 1.67 · Gujarati 1.53 · Bengali 1.52 · Malayalam 0.93 · Odia 0.37 · Gurmukhi 0.33 |
    | train S3-India | 86.8% | 7.5% | **5.7%** | |
    | test S2/S3-India | same as train within 0.1pp | | | |
    | S1 (train and test, all countries) | 100% | 0 | 0 | |

  - **The address carries these pairs [R]:** for India pairs with name-Jaccard < 0.25, address token-Jaccard has a **median of 0.65–0.76**, and only 4–10% have address-Jaccard < 0.25. An **address-only retrieval view** is therefore the first fix. Transliteration must cover **all 9 Indic blocks**, not only Devanagari. The Unicode Indic blocks are laid out in parallel, so one table plus a per-block offset is enough.
- **Domain-style names [R]:** 3.4–4.4% of train S2/S3 names (2.8–3.6% test) contain `.com/.in/.net/…`. Token features fail on these; space-stripped char features handle them.
- **S3 is noisier than S2 on addresses** (addr-exact 4.4% vs 13.3% US). Source-pair is a first-class dimension. *[review]* This is measured on **exact equality only**: name-exact is identical (30.5% vs 30.5%), and much of the address gap is **component reordering** in S3, which token-set similarity tolerates. The gap under fuzzy metrics is not established.
- Missingness: S1 is perfectly clean (0% empty names/addresses); S2/S3 have ~2.3–3.7% empty addresses, 0% empty names.
- Duplication pressure: normalized-name duplicate rate in S1 is 27.3% (US) / 36.6% (India) — same-name chains everywhere; 11–16% of S2/S3 records share a normalized address (malls/plazas) → address agreement alone is far from sufficient.
- Exemplar noise model (from `phase0/exemplars/`): char typos ("colombier"→"wilblims"), truncation, token reordering, word substitution ("ap hospitality inc"→"ap inc service 80430"), ordinal corruption ("45th"→"45nd"), literal "null" strings, abbreviation variance (st/street), pure-Devanagari names.

## 4. France (measured from visible test inputs)

- France = **15.0% of test S1** (259K entities, plus 1.43M S2/S3 records) — a sixth of the macro average; cannot be ignored, cannot be specially *trained* for.
- Profile: names shorter (19.4–21.1 chars vs 22.5–27.4), diacritics in 15.7% of S1 / ~24% of S2/S3 names, legal suffix "sarl" observed inside name strings, addr-dup rate in S2/S3 is the *highest* of all countries (21%). No empty S1 addresses (same as US/IN). S1 France name-dup 25.9% — chain pressure comparable to US.
- **Shift verdict (headline): moderate.** France looks like "US-style addresses + diacritics + different suffix vocabulary," not a fundamentally different problem. That supports a language-agnostic char/numeric core; multilingual embeddings remain an experiment (E13), not a requirement.
- *[review]* This verdict is a **hypothesis based on input marginals only**. Shift in the noise process or in P(match | features) is **not established**, and no labels exist to measure it directly. Label-free checks (in plan v4, Step 9):
  - top-1 candidate-similarity curves, FR vs US
  - predicted-k histogram, FR vs train OOF
  - record-conflict rate by country
  - **one France-only LB probe**: two submissions identical except for the France rows

  Note: France S2+S3 density is 5.53 records per S1, versus 4.68 in train.

## 5. Retrieval feasibility probe (stratified miniature; hard-distractor bias — universe = matched neighborhoods only, so absolute numbers are optimistic; shape is informative)

| retriever | US | India |
|---|---|---|
| exact normalized-name key | 30.5% | 18.7% |
| char-TFIDF (3–4g, hashed) @10 | **99.5%** | 96.6% |
| @20 | 99.7% | 97.8% |
| @50 | 99.8% | 98.8% |
| union exact+TFIDF@10 | 99.5% | 96.8% |

- Lexical retrieval is confirmed as the workhorse (v2 hypothesis supported in *shape*), but **India leaves ~1.2–3% of true pairs unrecovered even @50** — the cross-script/weak-address residue. The full-scale ladder must add numeric/address keys, and India-specific recall likely needs Devanagari→Latin transliteration (pure algorithmic string processing — challenge-legal, no external data) and/or a multilingual dense pass (E13's trigger condition now has concrete evidence to test against). *[review] Superseded:* the fix order is now (1) address-only view, (2) **all-Indic** transliteration, (3) dense on the Indic stratum only if needed. See §3 and §8.
- On this miniature, exact keys add almost nothing over TFIDF@10 (union ≈ TFIDF alone). Their real value (speed, ultra-precision band) must be re-measured at full scale, where the universe is 100× larger.
- *[review] Caveats on this probe:*
  1. The universe is ~100× smaller than the real per-country index, so the absolute recall values **must not be used to set K or gates**.
  2. The view used (name vs name+addr), whether S2 and S3 were pooled, and hashing collisions at full scale were not recorded.
  3. In a pooled S2+S3 index, @10 cannot reach full recall for entities with k > 10.
  4. The India residue is only 1.2% @50 despite ~24% non-Latin S2-India names. That implies the probe used name+addr and **the address rescued the cross-script pairs**, which is consistent with §3 [R].
  5. Pair recall is not the objective. Report the **entity-level oracle macro-F0.5 ceiling**: per entity, P=1 and R = the fraction of its true matches present in its candidates. For reference: missing 1 of 4 matches gives 0.9375; missing the only match of a k=1 entity gives 0.
  6. The decisive measurement is the **full-universe rank table** in plan v4, Step 3.

---

## 6. Strategy deltas (what is now settled vs still open)

*[review] Each item is re-tagged. Items marked ✗ were not actually settled.*

**Settled by measurement ([D]/[R]):**
1. **[D] ✓** Same-country filter is exact in train → apply unconditionally in blocking. Holding in test is a strong inference.
2. **[D] ✓** S2/S3 one-to-one holds in train GT → arbitration permissible; still gated by a net-effect ablation evaluated on **full-universe OOF** (a validation slice undercounts conflicts with other S1s).
3. **✗ [H], not settled.** Singletons are 5.6% of entities. But "per-rank thresholds" do **not** follow from the k histogram, which is a marginal prior. The metric does justify testing a two-parameter rule, `T_first`/`T_rest`, against a global T. k∈{0,1} is the highest-leverage band per error (see §2).
4. **[D] ✓** Both-fields-differ dominates true pairs → no retrieval-only shortcut; ML matcher mandatory.
5. **[D] partial.** The source-pair asymmetry is real on exact equality (magnitude under fuzzy metrics not established). The segmentation to test is global → +source. **Country segmentation is dropped**: it is undefined for France.
6. **[R] revised.** The cross-script India residue exists, but it spans **all Indic scripts**, not only Devanagari (§3), and the **address carries it**. Priority order: (a) address-only retrieval view; (b) **all-Indic** transliteration (E15), first as matcher features and only then as a blocker; (c) dense retrieval (E13) only if (a)+(b) leave a material oracle-F0.5 loss on that stratum.
7. **✗ [H].** France "moderate shift" is inferred from input marginals only (§4). Default stays: same pipeline, no France-specific model, validated by label-free checks plus one LB probe. The per-feature shift table is no longer a gate.
8. **[D] ✓** IDs carry no signal → no ID features.
9. **[R] ✓** About 26% of S2/S3 records are orphans (§2). Mined negatives look clean: orphan names match a same-country S1 name exactly only 4.7% / 5.6% of the time, versus 24.9% / 25.5% for matched records. So orphans don't look like missing labels.
10. **[R] ✓** Per-source cap of ≤5 (S2) and ≤6 (S3) → top-K is set **per source**.

**Still open (ranked):**
- **Full-universe rank table** (plan v4 Step 3): recall and oracle macro-F0.5 per view, direction, K and stratum, measured post-cap.
- **Validation design:** full-universe 5-fold entity-level OOF (replaces 70/15/15 slices).
- **Train→test density shift** (4.68 vs ~5.8 records per S1): does it move the chosen thresholds? (density-stress check)
- Matcher: LightGBM stage 1 → stage 2 with record-competition/sibling features → optional GPU cross-encoder as a feature.
- Decision layer: global T → `T_first`/`T_rest` → T[source]; optional expected-F0.5 prefix + isotonic; arbitration arms (none / hard / soft).
- France: label-free checks + one France-only LB probe.

## 7. Artifacts persisted in the workspace

- `sample_pack/` (81MB): **full train GT**, train head+random rows (30k/60k per source), test head+random rows (20k/60k — includes France), and the stratified closed miniature (25k S1 entities across country×match-count strata with ALL their S2/S3 partner rows + GT).
- `phase0/`: all scripts (rerunnable locally on full data), `stats/*.json`, `exemplars/*.csv` (hard-case true pairs).
- `amazon_ml_challenge_2026_ER_strategy_report.md` (v1, superseded), `strategy_v2_corrected.md` (v2), this file.
- Raw 2.4GB data lives in a non-persisted cache: future turns work from `sample_pack/` or need the Drive links re-pasted.
- *[review]* Raw data is now at `data/dataset/{train,test}/` in the local workspace. `phase0/` and `sample_pack/` are **not** present locally. They need to be re-created or copied in when the repo is initialized.

---

## 8. Review corrections & re-measurement [R] (2026-09-25)

Measured with stdlib-only Python over the **full** train and test files (scripts `check.py`, `check2.py`, `check3.py`, currently in the session scratchpad; copy them to `phase0/` at `git init`).

| # | Measurement [R] | Consequence |
|---|---|---|
| R1 | Local workstation: 15.7GB RAM, 16 logical CPUs, no CUDA GPU. Production compute: SageMaker (credits provided; instance types TBD) | The 1.9GB framing is sandbox-only. GPU work (cross-encoder, dense) is affordable on SageMaker but still gated by evidence |
| R2 | Orphans: 26.6% of train S2 (1.341M), 25.4% of train S3 (1.341M) | Precision is driven partly by records with no correct answer. Validation must include them at realistic density |
| R3 | Per-S1 cap: ≤5 S2, ≤6 S3 matches (histograms in §2) | Set top-K per source; pooled @10 is structurally insufficient for some entities |
| R4 | Records per S1: train 4.68 (US and India identical); test 5.76 US / 5.82 India / 5.53 France | k-prior and thresholds may not transfer. Run a density-stress check; lean strict if thresholds move |
| R5 | India name scripts: S2 Latin 76.5 / Devanagari 13.4 / other Indic 10.2; S3 86.8 / 7.5 / 5.7; test = train ±0.1pp; S1 100% Latin | Devanagari-only transliteration misses ~43% of non-Latin names. No India script shift |
| R6 | Low-name-overlap (Jaccard < 0.25) India pairs: Devanagari 44% (S2), 33% (S3); the rest is other Indic, domain-style, garbled/truncated | The "Devanagari mismatch" attribution was overclaimed |
| R7 | Those pairs: address-Jaccard median 0.65–0.76; only 4–10% < 0.25 | Add an address-only retrieval view, before transliteration or dense |
| R8 | Exact-norm-name hit in a same-country S1: orphans 4.7% / 5.6% vs matched 24.9% / 25.5% | Evidence against missing labels; mined negatives are probably clean (still audit the top 100) |
| R9 | Domain-style names: 3.4–4.4% train, 2.8–3.6% test | Add a space-stripped name similarity feature |

**Claim-status summary (Phase 0 as originally written):**

| Status | Claims |
|---|---|
| Measured fact | Row counts/splits; 0 dangling/cross-country/overlap; ID randomness; k histogram (train); one-to-one (train); noise rates on exact equality; Devanagari share; dup rates; France input marginals |
| Strong inference | Same-country and one-to-one also hold in test; the cascade is required |
| Hypothesis | France shift "moderate"; S3 "much" noisier under fuzzy metrics; value of transliteration and dense; full-scale TF-IDF recall |
| Overclaimed | Devanagari as the cause of the India residue; "per-rank thresholds" as settled; de-prioritizing singletons; miniature recall used to set K and gates; "1.9GB" as a pipeline-wide constraint |
