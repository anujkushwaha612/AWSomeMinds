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

## Reproduce: baseline, on a local machine (≥ 16 GB RAM recommended)

The pipeline is memory-lean (Arrow text store, streaming retrieval, chunked features; see
[experiments.md](experiments.md) E1). Every stage checkpoints under `artifacts/`; re-running a command
resumes where it stopped. Close memory-hungry apps first (browser tabs, IDEs): each stage logs
`[mem … free … GB]`.

Windows (Git Bash or PowerShell) from the repo root, data in `data/dataset/{train,test}/`:

```bash
.venv/Scripts/python -m pytest -q                              # 44 tests
.venv/Scripts/python -u -m ber.folds                           # ~1 min
.venv/Scripts/python -u -m ber.normalize                       # ~5 min (skipped if cached)
.venv/Scripts/python -u -m ber.baseline build --split train    # retrieval + features, India & US
.venv/Scripts/python -u -m ber.baseline build --split test     # France, India, US
.venv/Scripts/python -u -m ber.baseline train                  # 5 fold models, tuning, fold-0 report
.venv/Scripts/python -u -m ber.baseline predict --name sub01_baseline
```

(Linux/SageMaker: `.venv/bin/python`, see [SAGEMAKER.md](SAGEMAKER.md).) Outputs:
- `artifacts/baseline/metrics.json`: fold-0 macro F0.5, oracle ceiling, candidates per S1, per country / k
- `artifacts/baseline/log.txt`: full log with memory readings
- `output/matching_results.tsv` (upload this) and `output/candidate_pairs.tsv`
- `subs/sub01_baseline/`: snapshot for the version history

Knobs are in `configs/pipeline.yaml` → `baseline:`. To try a variant without overwriting, set
`BER_RUN=<name>` (outputs go to `artifacts/<name>/`).

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
