# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: normalization → same-country candidate generation → LightGBM pair scoring →
entity decision layer. Strategy and gates: [plan.md](plan.md). Candidate generation
(the part Amazon ranks separately): [blocking_strategy.md](blocking_strategy.md).
**Commands to run it on the real data: [RUNBOOK.md](RUNBOOK.md).**
Data findings: [phase0_report.md](phase0_report.md).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/SageMaker: .venv/bin/python
.venv/Scripts/python -m pip install -e .
```

Data goes in `data/dataset/{train,test}/` (not in git). On SageMaker, sync it from S3:

```bash
aws s3 sync s3://awsomeminds-dataset-532749777349/dataset/ data/dataset/ --region ap-south-1
```

All paths and numbers live in [configs/pipeline.yaml](configs/pipeline.yaml).
`artifacts_root` can be a local folder or an `s3://` URI; the code is the same either way.

## Reproduce (stages implemented so far)

```bash
python -m pytest                 # scorer + selection unit tests (incl. the 0.714 example)
python -m ber.folds              # artifacts/folds.parquet: 5 entity-level folds

# blocking cascade on the calibrated synthetic split (no real data needed)
python -m phase0.simulate --scale 0.004
python -m phase0.blocking_demo --scale 0.004 --df-sweep

# a synthetic dataset in exact challenge format, to smoke-test the real driver
python -m phase0.simulate --split train --scale 0.004 --as-dataset data/simset
python -m phase0.simulate --split test  --scale 0.004 --as-dataset data/simset
```

On the real data, follow [RUNBOOK.md](RUNBOOK.md):

```bash
python -m ber.blocking.run --split train --country India --sample-records 0.01  # time it
python -m ber.blocking.run --split train                                        # probe + cache
python -m ber.blocking.tune thresholds --country India                          # measure
python -m ber.blocking.tune grid --country India --knobs a_max=1,2,3            # sweep
python -m ber.blocking.run --split test --out output/candidate_pairs.tsv
```

## Layout

| Path | Contents |
|---|---|
| `src/ber/config.py` | config loading, local/S3 path helpers |
| `src/ber/io.py` | TSV readers (strings, no NA conversion), parquet cache, submission writer |
| `src/ber/eval/scorer.py` | exact per-entity F0.5 scorer, oracle ceiling, paired bootstrap, strata summary |
| `src/ber/blocking/probe.py` | df-purged S1 index, query sketch, bounded-work record→S1 top-k |
| `src/ber/blocking/select.py` | adaptive depth, reciprocal filter, capacity b-matching, rescue |
| `src/ber/blocking/metrics.py` | blocking scorecard: oracle F0.5, PC, PQ, \|C\|/S1, RR, operating point |
| `src/ber/blocking/run.py` | sharded production driver → `candidate_pairs.tsv` + `audit.json` |
| `src/ber/blocking/tune.py` | `thresholds` / `grid` / `dfsweep` — measurement-driven tuning |
| `phase0/export_sample.py` | 1/N hash-bucket slice of the real data, small enough for git |
| `src/ber/folds.py` | fold assignment stratified by country x k-bucket |
| `src/ber/submit.py` | write both TSVs, run the official validator, snapshot to `subs/<name>/` |
| `phase0/` | data-analysis scripts behind phase0_report.md §8 |
| `subs/` | one folder per leaderboard submission (commit, config, metrics) |

Every leaderboard upload goes through `ber.submit.make_submission`, which keeps the
version history the guidelines require.
