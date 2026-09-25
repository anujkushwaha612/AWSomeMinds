# RUNBOOK — candidate generation on the real data

Everything here runs on your workstation (15.7GB, 16 threads) or a SageMaker CPU
instance. Nothing needs a GPU. Strategy and rationale: [blocking_strategy.md](blocking_strategy.md).

> **Why you are running this and not the agent.** The agent sandbox reaches only
> `github.com`, `codeload.github.com`, `api.github.com` and PyPI. Google Drive, Dropbox,
> Hugging Face, S3 and even `objects.githubusercontent.com` (GitHub release assets and
> LFS) are blocked at the TLS layer — DNS resolves, the handshake is cut. No link will
> work. Step 5 below is the one channel that does.

## 0. Setup (once)

```bash
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
# data at data/dataset/{train,test}/, or:
aws s3 sync s3://awsomeminds-dataset-532749777349/dataset/ data/dataset/ --region ap-south-1
python -m pytest -q                      # must be green before you trust any number
```

`configs/pipeline.yaml` holds every knob. Nothing is hard-coded to `{US, India}` — the
driver reads the country list out of S1, so France flows through unchanged.

---

## 1. Timing run FIRST — do not skip this (~10 min)

Never launch a full shard blind. Probe 1% of India (the hardest country) and read the
extrapolation the driver prints.

```bash
python -m ber.blocking.run --split train --country India --sample-records 0.01
```

You get a line like `EXTRAPOLATION: a full run of these shards is about N minutes`.

| extrapolation | what to do |
|---|---|
| under ~45 min | go straight to step 2 |
| 45 min – 3 h | drop view `N` (`--views NA A`), or lower `max_index_df` to 5000 |
| over 3 h | `max_index_df: 5000`, `sketch_terms: 16`, `k_forward: 5`, re-time |

**Memory:** if you OOM, lower `idf_sample` (500k → 200k) and `query_chunk`
(20000 → 5000). Peak is one query chunk plus the S1 index, not the whole shard.

At `--sample-records < 1.0` the audit restricts ground truth to the records actually
probed, so `pairs_completeness`, `pairs_quality` and `C_per_record_mean` are **exact**.
`oracle_f05` and `C_per_s1_mean` are **not** — each entity only had a fraction of its
records probed. Re-measure those at 1.0.

---

## 2. Full train probe (the expensive step, once)

```bash
python -m ber.blocking.run --split train 2>&1 | tee logs/blocking_train.log
```

Writes `artifacts/blocking/train/pairs_<country>_S<src>.parquet` (the cache every
tuning command reuses), `candidate_pairs.tsv`, and `audit.json`.

**Send me `artifacts/blocking/train/audit.json`.** It is a few KB and contains the
scorecard, per-shard timings and the policy that produced them.

---

## 3. Measure the thresholds, then tune (minutes, no re-probing)

### 3a. The measurement that must come before any threshold
```bash
python -m ber.blocking.tune thresholds --country India
python -m ber.blocking.tune thresholds --country US
```

Four tables come out. The two that decide everything:

* **`abstain_score`** — for each threshold: what share of records get declined, what
  share of those are genuinely orphans (~26% of records have no parent), and what share
  of true pairs the threshold destroys. Pick the largest threshold whose
  `true_pairs_lost` is still under ~0.005.
* **`conf_score` × `conf_margin`** — where "top-1 is the true parent" is precise enough
  to stop looking at the runner-up. Pick a cell with `precision ≥ 0.99` and the largest
  `records_confident`; every confident record drops from `a_max` candidates to 1.

The defaults in `configs/pipeline.yaml` (0.20 / 0.80 / 0.15) are placeholders. They came
from a simulator, they will be wrong for the real score distribution, and they are the
second-biggest size lever after `a_max`.

Also check `parent_retrieved_at_all` in the summary — that is the hard recall ceiling of
the probe. If it is below ~0.97 on India, no amount of selection tuning will fix it;
the fix is a view (address-only, transliterated) or a bigger `k_forward`.

### 3b. Sweep the selection policy
```bash
python -m ber.blocking.tune grid --country India \
    --knobs a_max=1,2,3 abstain_score=0.0,<from 3a>,<higher> conf_margin=0.05,0.10,0.20
python -m ber.blocking.tune grid --country India --knobs cap_s2=5,6,8 cap_s3=6,7,9
python -m ber.blocking.tune grid --country India --knobs one_sided_score=0.35,0.45,0.55
```

Each finishes in seconds because it re-runs stage 3 over the cached pairs. The last
line prints the operating point: **the smallest candidate set whose oracle F0.5 deficit
against the best is either inside the bootstrap CI or under `bootstrap.materiality`
(0.002)**. Change that bar in the config if you want it stricter or looser — it is the
single number encoding "how much F0.5 is a candidate worth", which is exactly what we
still need Amazon to answer.

### 3c. Only if step 1 said you are runtime-bound
```bash
python -m ber.blocking.tune dfsweep --country India --caps 5000,20000,50000,0
```
Re-probes at each purge cap. Take the smallest cap whose oracle F0.5 is tied with `0`
(off). On the simulator, purging was free or slightly positive — it removes noise grams
as well as work.

**Send me `artifacts/tune/*.json`** (a few KB each).

---

## 4. Lock it in and produce the test candidates

Write the winning values into the `select:` block of `configs/pipeline.yaml`, then:

```bash
python -m ber.blocking.run --split train                 # confirm the final scorecard
python -m ber.blocking.run --split test --out output/candidate_pairs.tsv
python3 data/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/dataset/test
```

Sanity checks on the test run, all printed by the driver:

| check | expected | if it fails |
|---|---|---|
| `C_per_s1_mean` | 5–9 (floor is 4.26) | above 12, `a_max` or the caps are too loose |
| `C_per_s1_max` | exactly `cap_s2 + cap_s3` | higher means the capacity pass was bypassed |
| `projected_C_per_s1_from_records` ≈ `C_per_s1_mean` | yes | a mismatch means a country was skipped |
| France `cand_per_record` in `audit.json` | inside the US/India range | outside it, investigate before submitting (gate B6) |

Expect test `C_per_s1` to be ~23% above train's for the *same* policy — test has 5.75
records per S1 against train's 4.68. That is arithmetic, not a regression.

---

## 5. Getting real data to the agent (the only route that works)

```bash
python -m phase0.export_sample --denom 25 --out sample_pack
git add sample_pack && git commit -m "real data sample (1/25 hash bucket)" && git push
```

Hash-bucket sampling, not row sampling: an entity is kept when
`blake2b(entity_id) % 25 == 0`, and a record is kept when its parent was kept or — for
orphans — its own hash lands in the bucket. That keeps the slice a closed world with the
real density, orphan rate, k histogram, per-source caps and country mix. Verified on a
stand-in: density 4.619 vs 4.677, orphans 25.7% vs 26.0%, caps 5/6 exactly.

~30MB gzipped at `--denom 25`. Raise `--denom` if `manifest.json` reports over 90MB
(GitHub rejects single files over 100MB, and git LFS is unreachable from the sandbox).

The one thing it cannot preserve is distractor pressure, which drops 25×, so absolute
recall on the sample is optimistic. It is for tuning *shapes* — score distributions,
which views fire on which strata, error exemplars — not for setting final thresholds.

---

## 6. What to send back, in order of usefulness

1. `artifacts/blocking/train/audit.json` — scorecard + per-shard timings
2. `artifacts/tune/thresholds_India.json`, `thresholds_US.json` — the score distributions
3. `artifacts/tune/grid_*.json` — the size/recall frontier
4. `sample_pack/` committed to the repo — then I can iterate on real text directly
5. The tail of `logs/blocking_train.log` if anything crashed or was slower than the
   extrapolation predicted

## 7. Known rough edges

* `ber.blocking.retrieve` / `rank_table` / `audit` are the old entity-side v4 path. They
  still work; nothing in this runbook uses them.
* `--rev-mode exact` runs a real S1 → record pass instead of deriving `r_ent` from the
  forward table. On the stand-in it gave identical recall for 25% more time, so
  `derived` is the default; re-check once on real data with
  `--country India --sample-records 0.05 --rev-mode exact`.
* Exact structural keys (normalised name, house-number + street + postcode,
  transliterated name) are **not implemented yet**. They are the highest-value remaining
  addition: near-free, very precise, and they mostly land in the reciprocal tier.
