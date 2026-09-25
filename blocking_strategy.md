# blocking_strategy.md — Candidate Generation, v5
## Amazon ML Challenge 2026 — Business Entity Resolution

**Supersedes** plan.md v4 Step 3 and the three-stage cascade sketched in the previous
turn. **Scope:** everything that produces `candidate_pairs.tsv`. The matcher and decision
layer (plan.md Steps 5–9) are unchanged.

**Provenance tags:** [D] Phase-0 measurement · [R] full-data re-measurement (phase0_report.md §8)
· [A] arithmetic derived from [D]/[R] · [S] measured on the **synthetic** simulator, mechanics
only · [L] literature · [J] judgement call.

> **Status of the numbers in this document.** The real 2.4GB dataset could not be
> downloaded in this environment (Google Drive is unreachable from the sandbox; only
> an allowlist including GitHub and PyPI resolves). So: every **[D]/[R]** number is
> quoted from `phase0_report.md`, every **[A]** number is arithmetic on those, and every
> **[S]** number comes from `phase0/simulate.py`, a generator calibrated to the Phase-0
> *marginals* with an invented noise model. **[S] numbers validate the mechanism and the
> code path, not the recall you will see on the leaderboard.** Step 0 below is the
> re-measurement on the real files, and it must be run before any of this is trusted.

---

## 1. Why the previous design was the wrong shape

The v4 plan and the follow-up "three-stage cascade" both retrieve top-K per **S1 entity**
per source, then union several views, then prune. Three problems, all visible in the
Phase-0 numbers:

1. **It optimises the wrong quantity.** `candidate_pairs.tsv` is grouped by S1, so the
   instinct is to control K per S1. But the thing being predicted is one-to-one on the
   *other* side: all 7,638,365 train GT pairs have exactly one S1 parent, reuse 0.0% [R].
   The object to estimate is a function `record → S1 ∪ {⊥}`, and its natural budget is
   candidates **per record**.
2. **It makes the union grow where it should shrink.** v4 adds the reverse direction
   (`S2/S3 → S1`) to the union, so the reverse pass *increases* |C|. The literature uses
   the second direction the other way round — as a *reciprocal* filter that keeps only
   edges both sides agree on. Reciprocal CNP/WNP "reduces CNP's recall slightly for much
   higher precision and, thus, often dominates CEP" [L, Papadakis survey].
3. **It leaves three measured hard constraints on the table.** ≤5 S2 and ≤6 S3 matches
   per S1 [R]; exactly ≤1 S1 per record [R]; ~26% of records have no parent at all [R].
   v4 uses all three as *features for the matcher*. They are cheaper and stronger as
   **capacity constraints inside blocking**.

The reframe: **blocking is a degree-constrained bipartite assignment, not a per-entity
search.** That is not a novel idea — Jaro (1989) used a maximum-weight one-to-one
assignment as the *blocking* step feeding a Fellegi–Sunter test, and it remains the
standard way to handle the maximum-one-to-one restriction [L].

---

## 2. The floor: how small can |C| per S1 honestly get? [A]

Everything below is arithmetic on the row counts in phase0_report.md §1.

| Quantity | Train | Test |
|---|---|---|
| S1 entities | 2,206,821 | 1,732,544 |
| S2+S3 records | 10,320,219 | 9,969,589 |
| **records per S1** | **4.677** | **5.754** (US 5.756 · India 5.824 · France 5.531) |
| GT pairs | 7,638,365 | unknown |
| matches per S1 (k̄) | 3.461 | unknown |
| matched-record share | 74.01% | unknown |

Because each record has at most one parent, a blocking policy that emits `a` candidates
per record produces **exactly `a × 5.754` candidates per test S1**. That single identity
sets the whole scale:

| policy | candidates / S1 (test) | |C| total | RR vs all-pairs |
|---|---|---|---|
| **perfect: 1 candidate per matched record, 0 per orphan** | **4.26** [A] | 7.4M | 0.99999957 |
| 1 candidate per record, no abstention | 5.75 | 9.97M | 0.99999942 |
| 2 per record | 11.51 | 19.9M | 0.99999885 |
| 3 per record | 17.26 | 29.9M | 0.99999827 |
| entity-side top-10 per source (v4 default) | 20.0 | 34.7M | 0.99999800 |
| entity-side top-50 per source (v4 `k_forward`) | 100.0 | 173.3M | 0.99999003 |

**The target band is 5–9 candidates per S1, and the floor is 4.26.** v4's stored K=50 is
23× above the floor. Anything above ~12 per S1 is leaving the ranking criterion on the
table for no recall we can demonstrate.

Two consequences worth writing on the wall:

* **Tune per record, report per S1.** Train has 4.68 records per S1 and test has 5.75, so
  an identical policy yields a **23% larger** per-S1 list on test purely from density
  [R4]. Any per-S1 target tuned on train is wrong on test by that factor.
* **Abstention is a first-class lever.** ~26% of records have no parent [R]. Every record
  the blocker correctly declines to propose anything for removes ~0.26 candidates per S1
  at zero recall cost. Nothing else in the pipeline buys size that cheaply. `candidate_pairs.tsv`
  explicitly permits an empty list.

---

## 3. The cascade

```text
S1, S2, S3 (per split)
   │
[0] PARTITION        exact country string.  6 test shards (3 countries x 2 sources),
   │                 each fully independent -> the scale-out story.
   │                 1.73e13 -> 6.72e12 comparisons before anything clever. [A]
   ▼
[1] INDEX            build the index over S1 only (the small, clean, deduplicated side:
   │                 1.73M rows vs 9.97M records).  char 3-gram TF-IDF, IDF fitted on the
   │                 whole country corpus (unsupervised).
   │                 PURGE n-grams whose posting list exceeds max_index_df.
   ▼
[2] PROBE            every S2/S3 record queries the S1 index, top-k_fwd, over 3 views:
   │                   NA = legal-stripped name + address   (backbone)
   │                   A  = address only                    (carries the Indic pairs [R7])
   │                   N  = name only                       (carries empty/garbled addresses)
   │                 + exact structural keys (near-free, ultra-precise)
   │                 + [gated] dense ANN on the Indic stratum only
   │                 QUERY SKETCH: only the sketch_terms rarest n-grams of each query are
   │                 scored -> work per query is bounded by sketch_terms x max_index_df.
   │
   │                 also: S1 -> record top-k_rev on view NA (1.73M queries, cheap) —
   │                 NOT unioned in; it only supplies r_ent for stage 3.
   ▼
[3] SELECT           four filters, in this order (src/ber/blocking/select.py):
   │                 3a score floor
   │                 3b adaptive record depth: 0 / 1 / a_max candidates per record,
   │                    decided from the record's own top-1 score and top1-top2 margin
   │                 3c reciprocal tiering: keep pairs both directions agree on; a
   │                    one-sided pair must clear a stricter score or have view agreement
   │                 3d capacity pruning: alternating top-a per record / top-b per
   │                    (S1, source), with b from the measured caps (<=5 S2, <=6 S3)
   │                 3e rescue: give back the single best edge of any record emptied by
   │                    3b-3d, then re-apply 3d so the per-S1 cap stays hard
   ▼
[4] EMIT             group by S1 -> candidate_pairs.tsv.  This is the exact set the
                     matcher runs inference over.  AUDIT: oracle macro-F0.5, PC, PQ,
                     |C| per S1 (mean/p90/max), |C| per record, RR, wall-clock.
```

### 3.1 Why the work in stage 1–2 is bounded (the actual fix for "it's too slow")

The cost of a sparse top-k is `Σ over query terms of posting-list length`. A char 3-gram
like `"ltd"`, `" pv"` or `"rue"` occurs in a large fraction of a country's records, so one
query can touch tens of millions of postings. **The number of queries was never the
problem; the tail of the df distribution was.** Two limits, both standard:

* **`max_index_df` — block purging.** Drop any n-gram whose posting list in the S1 index
  exceeds a cap. This is exactly *block purging* from the blocking survey, applied to
  n-gram blocks [L]. High-df grams also carry near-zero IDF weight, so the cosine barely
  moves. v4's `max_df: 0.10` is far too loose: at 663k US S1 rows that permits posting
  lists of 66,000.
* **`sketch_terms` — prefix filtering.** Keep only the N highest-weight (rarest) n-grams
  of each query. This is the prefix-filter principle behind All-Pairs/PPJoin: similar
  records must agree on a rare feature [L, Xiao et al. WWW'08]. Work per query is then
  ≤ `sketch_terms × max_index_df` postings, independent of how long the text is.

Measured on the simulator [S], purging is free or better — it removed 22% of postings and
18% of probe time while *slightly improving* oracle F0.5 (noise removal):

| `max_index_df` | postings kept | longest posting list | probe s | oracle F0.5 | PC | \|C\|/S1 |
|---|---|---|---|---|---|---|
| 0.02 | 16.9% | 64 | 8.3 | 0.97109 | 0.91496 | 9.59 |
| 0.05 | 33.1% | 158 | 9.1 | 0.99087 | 0.96962 | 9.21 |
| **0.20** | **77.7%** | **636** | **11.0** | **0.99759** | **0.99142** | **7.93** |
| 1.00 (off) | 100% | 1162 | 13.4 | 0.99748 | 0.99092 | 7.93 |

The fraction that is safe **will be different on the real data** — at 1/250 scale a 2% cap
means 53 documents, which is absurdly tight. On the real files set the cap in *absolute
posting length* (start at 20,000) and re-run this sweep; the code accepts either
(`max_index_df > 1` = absolute, `<= 1` = fraction of index rows).

### 3.2 Why capacity pruning is the biggest single lever

Stage 3d turns two measured facts into hard bounds:

* per record: ≤1 true parent [R] → cap at `a_max` (1–3)
* per (S1, source): ≤5 S2 and ≤6 S3 [R] → cap at `cap_s2`/`cap_s3`

With `cap_s2=6, cap_s3=7` (measured caps plus one slot of slack) **`|C|` per S1 is hard-bounded
at 13, no matter how pathological the chain-name collisions are.** Ablating it on the
simulator: the max jumps from 13 to 114 and the mean from 8.19 to 9.56 [S]. The tail is
what makes an average look bad, and chain names — 27% (US) / 37% (India) duplicate
normalised S1 names [D] — are exactly the mechanism that generates it.

The alternating implementation (top-a per record, then top-b per entity, repeat) is a
cheap deterministic stand-in for maximum-weight degree-constrained b-matching. On this
graph the edge scores are extremely skewed, so greedy and alternating agree almost
everywhere; if the audit shows otherwise, swap in greedy edge-sorted admission (it is
`O(E log E)` and 12M edges is nothing).

### 3.3 End-to-end mechanics, measured [S]

Simulator at 1/250 scale: 6,928 S1, 39,876 records, 5.756 records/S1, k̄ 3.48, 26–39%
orphans, caps ≤5/≤6, 36% duplicate S1 names. Full output in
`phase0/stats/blocking_demo_sim.txt`.

| config | oracle F0.5 | PC | PQ | **\|C\|/S1** | p90 | max | \|C\|/record | RR vs country |
|---|---|---|---|---|---|---|---|---|
| cascade `a_max=1` | 0.98515 | 0.9496 | 0.608 | **5.44** | 9 | 13 | 0.95 | 0.99965 |
| **cascade `a_max=2`** | **0.98612** | 0.9499 | 0.404 | **8.19** | 13 | 13 | 1.42 | 0.99947 |
| cascade `a_max=3` | 0.98242 | 0.9399 | 0.330 | 9.92 | 13 | 13 | 1.72 | 0.99936 |
| baseline entity-side top-5/source | 0.97679 | 0.9272 | 0.323 | 10.00 | 10 | 10 | 1.74 | 0.99936 |
| baseline entity-side top-10/source | 0.98273 | 0.9468 | 0.165 | 20.00 | 20 | 20 | 3.47 | 0.99871 |
| baseline entity-side top-20/source | 0.98752 | 0.9602 | 0.084 | 40.00 | 40 | 40 | 6.95 | 0.99742 |
| baseline entity-side top-50/source | 0.99256 | 0.9748 | 0.034 | 100.00 | 100 | 100 | 17.37 | 0.99356 |
| raw probe union (no selection) | 0.99864 | 0.9953 | 0.026 | 132.69 | 201 | 721 | 23.05 | 0.99145 |

Read the two rows that matter: **`a_max=2` reaches the oracle F0.5 of the top-20/source
baseline at 4.9× fewer candidates per S1, and beats top-10/source outright while being
2.4× smaller.** Pairs quality — the density of true pairs in the candidate set — improves
2.4× to 12×, which is also what makes the matcher's job easier and its training set less
degenerate.

Ablations at `a_max=2` [S], each removing one stage:

| removed | oracle F0.5 | \|C\|/S1 | max | reading |
|---|---|---|---|---|
| — (full cascade) | 0.98612 | 8.19 | 13 | |
| capacity cap | 0.98997 | 9.56 | **114** | +0.004 oracle for +17% size and a 9× worse tail |
| adaptive depth | 0.98516 | 9.03 | 13 | strictly worse: −0.001 oracle, +10% size |
| reciprocal tier | 0.98612 | 8.32 | 13 | free 1.5% size cut at this scale |
| rescue | 0.98612 | 8.16 | 13 | inert here; keep it as insurance on real data |

The policy is stable under both resolutions of the train→test density ambiguity [R4]:
regenerating with `--k-scale 1.23` (holding orphans at 26% and raising k̄ to 4.23 instead
of letting the surplus be orphans) moves `a_max=2` to 0.98394 oracle at 8.15 |C|/S1 [S] —
the ordering of every configuration is unchanged.

Two honest caveats on this table. **Adaptive depth barely fires** (4 of 39,774 records
abstained) because the simulator's score distribution is unrealistically clean; on real
data, where 60–76% of true pairs match exactly on neither field [D], both the abstain and
the confidence branches should do real work, and they need retuning against the actual
score histogram. And **PC ≈ 0.95 here is an artefact of the invented noise model**, not a
forecast.

---

## 4. Choosing the operating point on the real data

### Step 0 — re-measure (blocks everything else)
Run `phase0/blocking_demo.py`'s logic against `data/dataset/train/`, one country at a
time, India first (hardest). Produce, per country:
1. the `max_index_df` sweep of §3.1 in **absolute** posting lengths {5k, 20k, 50k, ∞};
2. the score histogram of top-1 and of the top1−top2 margin, split by
   `true partner present / absent` — this is what sets `conf_score`, `conf_margin` and
   `abstain_score`, and none of those defaults mean anything until it exists;
3. the raw probe union's oracle F0.5 ceiling per view, which bounds everything downstream.

### The selection rule
`ber.blocking.metrics.pick_operating_point` implements it: score every policy, take the
best oracle F0.5, and keep the policies whose **paired-bootstrap CI of the deficit against
the best contains 0**. Ship the one with the smallest `C_per_s1_mean` among those.

This is deliberately size-first: Amazon has said the smaller candidate set ranks higher,
but not how that trades against F0.5, so the rule buys size only where the recall cost is
**statistically indistinguishable from zero**. [J] — revisit if the organisers answer the
Google Form question.

### Sweep order (cheapest and highest-leverage first)
The probe dump is reusable: ranks and scores are stored once, so every stage-3 policy is
evaluated offline without re-retrieving. Only group 1 requires re-running the probe.

| # | Knob | Grid | Re-probe? | Expected effect |
|---|---|---|---|---|
| 1 | `max_index_df` | 5k / 20k / 50k / ∞ | yes | runtime, then recall |
| 1 | `sketch_terms` | 16 / 24 / 40 | yes | runtime |
| 1 | `k_fwd` per record | 5 / 10 / 20 | yes | ceiling only; stage 3 decides the rest |
| 1 | view set | NA / NA+A / NA+A+N / +keys | yes | Indic + empty-address recall |
| 2 | `a_max` | 1 / 2 / 3 | no | **the dominant size knob** |
| 2 | `conf_score`, `conf_margin` | from the §4 Step 0 histogram | no | size, ~free |
| 2 | `abstain_score` | target 10 / 20 / 26% abstention | no | size, ~free |
| 3 | `cap_s2`, `cap_s3` | (5,6) / (6,7) / (8,9) | no | tail control |
| 3 | `recip_r_rec`, `one_sided_score` | 1/2/3 × 0.35/0.45/0.55 | no | precision |
| 4 | `score_floor`, `rescue_floor` | from the same histogram | no | fine trim |

### Gates (replacing plan.md §4 G1/G1b)

| Gate | Question | Keep if | Drop if |
|---|---|---|---|
| **B1 view** | does view V pay for itself? | marginal oracle F0.5 on its target stratum ≥ 0.002, CI > 0, and the size cost is < 1 candidate/S1 | CI includes 0 |
| **B2 purge** | is the df cap safe? | oracle F0.5 CI vs `∞` contains 0 | otherwise loosen |
| **B3 size** | is this the operating point? | smallest \|C\|/S1 tied with the best (above) | — |
| **B4 tail** | is the tail controlled? | `C_per_s1_p99 ≤ cap_s2 + cap_s3` | otherwise tighten the caps |
| **B5 stratum** | does any stratum collapse? | no country × script × k-bucket cell loses > 0.01 oracle F0.5 against the ALL row | otherwise add a targeted view |
| **B6 France** | does the policy transfer? | France's \|C\|/S1, abstention rate and top-1 score distribution sit inside the US/India range | otherwise investigate before submitting |
| **B7 runtime** | can the full test pipeline re-run in one slot? | ≤ 3h on the SageMaker CPU instance | otherwise cut `k_fwd`, then the lowest-gain view |

B6 is the only France check available: there are no French labels, so the blocking policy
must contain **no country-conditioned parameter** and is validated only by the shape of its
outputs. The simulator agrees the shape is stable (France 7.75 |C|/S1 and 0.9931 oracle
against India 8.40 / 0.9842 [S]), but the simulator's France is the easy case by
construction.

---

## 5. The scalability argument (for the methodology document)

Amazon asks for blocking that would survive billions of records. The claim to make, with
the numbers:

1. **Nothing is ever compared all-to-all.** Test all-pairs = 1,732,544 × 9,969,589 =
   **1.73 × 10¹³** comparisons. The exact-country partition alone removes 61% of that
   (**6.72 × 10¹²**), and 0 of the 7,638,365 train GT pairs cross a country boundary [D],
   so the partition is exact, not heuristic.
2. **Work per query is a constant, not a function of corpus size.** `sketch_terms ×
   max_index_df` bounds the postings touched by any record (§3.1). Adding records grows the
   work linearly in the number of queries, never quadratically.
3. **The shards are independent.** (country × source) — 6 shards on test, and each can be
   split further by any exact key. Nothing is shared but the S1 index, which is the small
   side. This maps onto Spark/EMR or a partitioned map-reduce with no algorithmic change;
   we run it as a single-box job only because 10M records fits.
4. **The final candidate set is `|C| ≈ 1.2 × 10⁷`**, i.e. a reduction ratio of
   **0.9999993** against all-pairs and **0.9999982** against the country-partitioned space
   [A], with a hard per-entity bound of `cap_s2 + cap_s3` candidates.
5. **We do not use a managed entity-resolution service.** AWS Entity Resolution and Glue
   FindMatches would both fall under the prohibition on "commercial entity resolution APIs
   or services". AWS is used for compute only (SageMaker, S3); every algorithm is
   open-source or ours.

Report the audit table from §3.3 on the real data in the methodology doc; that table *is*
the blocking-quality evidence they say they will review.

---

## 6. Compliance note on the learned pruner — read before implementing it

The previous plan proposed a LightGBM that scores every candidate and prunes. The rules
say `candidate_pairs.tsv` must be "the *exact* set of records you feed into your matching
model for inference … the *last* one: whatever your model actually runs inference over".
A reviewer can reasonably read a trained pair classifier inside blocking as *the matcher,
run twice*, with `candidate_pairs.tsv` reporting only its survivors.

The v5 stance:

* **Default (ship this): stage 3 is label-free.** Every filter in `select.py` uses only
  retrieval evidence — cosine score, rank in each direction, number of agreeing views —
  plus two constants read off the training *structure* (the ≤5/≤6 caps, the ≤1-parent rule).
  No pair classifier, nothing fitted to match/non-match labels. This is unambiguously
  blocking.
* **Gated upgrade: supervised meta-blocking.** If B3 shows a material size win, a small
  model over *the same retrieval-evidence features only* (never the 60-feature battery,
  never a cross-encoder) can replace the hand-set thresholds. That is Generalized
  Supervised Meta-blocking [L] and it is a named blocking technique, so it is defensible —
  **provided** the methodology document names it as blocking, states its feature list, and
  shows the final matcher is strictly richer. Decide with §4's rule, and if the win is
  under ~1 candidate per S1, do not take the interpretive risk.

Either way, `matching_results.tsv ⊆ candidate_pairs.tsv` holds by construction, and
`data/utils/validate_submission.py` runs before every upload.

---

## 7. What is implemented

| Path | Contents | State |
|---|---|---|
| `src/ber/blocking/probe.py` | df-purged S1 index, query sketch, bounded-work top-k, pair-table fusion | new, exercised by the demo |
| `src/ber/blocking/select.py` | the stage-3 cascade: floor, adaptive depth, reciprocal tier, capacity b-matching, rescue | new, 20 unit tests |
| `src/ber/blocking/metrics.py` | oracle F0.5 / PC / PQ / \|C\| / RR scorecard, stratum table, `frontier`, `pick_operating_point` | new, unit-tested |
| `phase0/simulate.py` | calibrated synthetic split, `--k-scale` for the R4 density ambiguity | new |
| `phase0/blocking_demo.py` | end-to-end run + df sweep + ablations; output in `phase0/stats/` | new |
| `src/ber/blocking/retrieve.py`, `rank_table.py`, `audit.py` | v4 entity-side retrieval and audit | kept — `retrieve.py` is the fastest way to produce the §4 Step 0 dump; `audit.py` is superseded by `metrics.py` |
| `src/ber/normalize.py` | all-Indic transliteration, legal strip, no-space view | unchanged, already correct |

```bash
python -m pytest tests/test_select.py -q            # 20 tests
python -m phase0.simulate --scale 0.004             # marginals vs phase0_report
python -m phase0.blocking_demo --scale 0.004 --df-sweep
```

### Not yet built (ranked)
1. **Exact structural keys** as a probe (normalised name; space-stripped name; house-number
   + street token + postcode; sorted-token name; transliterated name). Near-free, very
   precise, and they mostly produce tier-0 pairs. Highest value per hour.
2. **The real-data driver**: `src/ber/blocking/run.py` that shards by (country, source),
   streams the probe, calls `select`, and writes `candidate_pairs.tsv` + the audit JSON.
3. **Dense ANN on the Indic stratum**, gated on B1 and on the A-view failing to close the
   gap. ~0.5M test records — small enough for one GPU hour.

---

## 8. Open questions

| # | Question | Why it matters | How to close it |
|---|---|---|---|
| 1 | Is test's extra density (5.75 vs 4.68) extra matches or extra orphans? [R4] | decides the |C|/S1 floor: 4.26 vs ~3.5 | not directly answerable; the policy is checked under both (`--k-scale`) |
| 2 | Does the ≤1-parent rule hold on test? | stage 3b/3d assume it | strong inference only (it is exact in train); the rescue pass limits the damage if it fails |
| 3 | How does Amazon weigh candidate size against F0.5? | decides how aggressive B3 should be | Google Form — ask alongside the "last or best submission" question |
| 4 | Is the learned pruner acceptable as blocking? | §6 | ask in the same form; default to the label-free policy until answered |

---

## 9. References

- Papadakis, Skoutas, Thanos, Palpanas, *Blocking and Filtering Techniques for Entity Resolution: A Survey*, ACM CSUR 2020 — block purging; Reciprocal WNP/CNP; meta-blocking. https://arxiv.org/pdf/1905.06167
- Xiao, Wang, Lin, Shang, *Efficient Similarity Joins for Near Duplicate Detection* (PPJoin), WWW 2008 — prefix, positional and length filtering. https://cgi.cse.unsw.edu.au/~lxue/WWW08.pdf
- Jaro, *Advances in Record-Linkage Methodology*, JASA 1989 — one-to-one assignment used as the blocking step before the Fellegi–Sunter test; see also Binette & Steorts, *(Almost) All of Entity Resolution*. https://arxiv.org/pdf/2008.04443
- Sadinle, *Bayesian Estimation of Bipartite Matchings for Record Linkage*, 2016 — why independent pairwise decisions violate the max-one-to-one constraint. https://arxiv.org/pdf/1601.06630
- Gagliardelli, Papadakis, Simonini, Bergamaschi, Palpanas, *Generalized Supervised Meta-blocking*, PVLDB 2022. https://arxiv.org/abs/2204.08801
- Brinkmann, Shraga, Bizer, *SC-Block*, 2023 — candidate-set size at fixed pairs completeness as the blocking benchmark. https://arxiv.org/pdf/2303.03132
- Shao, Wang, Lin, *Skyblocking*, 2018 — pairs completeness / pairs quality / reduction ratio as a multi-criteria frontier. https://arxiv.org/pdf/1805.12319
- BlockingPy (ANN blocking; RR reporting conventions), 2025/26. https://www.sciencedirect.com/science/article/pii/S2352711026000774
- AWS Entity Resolution FAQ — the managed service we deliberately do not use. https://aws.amazon.com/entity-resolution/faqs
