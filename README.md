# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: normalization → same-country candidate generation → LightGBM pair scoring →
entity decision layer. Strategy and gates: [plan.md](plan.md). Data findings:
[phase0_report.md](phase0_report.md).

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
python -m pytest                 # scorer unit tests (incl. the 0.714 worked example)
python -m ber.folds              # artifacts/folds.parquet: 5 entity-level folds
```

## Layout

| Path | Contents |
|---|---|
| `src/ber/config.py` | config loading, local/S3 path helpers |
| `src/ber/io.py` | TSV readers (strings, no NA conversion), parquet cache, submission writer |
| `src/ber/eval/scorer.py` | exact per-entity F0.5 scorer, oracle ceiling, paired bootstrap, strata summary |
| `src/ber/folds.py` | fold assignment stratified by country x k-bucket |
| `src/ber/submit.py` | write both TSVs, run the official validator, snapshot to `subs/<name>/` |
| `phase0/` | data-analysis scripts behind phase0_report.md §8 |
| `subs/` | one folder per leaderboard submission (commit, config, metrics) |

Every leaderboard upload goes through `ber.submit.make_submission`, which keeps the
version history the guidelines require.
