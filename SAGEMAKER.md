# Running the pipeline on SageMaker

Target: a SageMaker **notebook instance** in **ap-south-1 (Mumbai)**, the same region as the data bucket.

## 1. Instance

| Choice | vCPU / RAM | When |
|---|---|---|
| `ml.m5.4xlarge` | 16 / 64 GB | minimum that comfortably fits the full train stage (~31M candidate pairs) |
| `ml.m5.8xlarge` or `ml.c5.9xlarge` | 32–36 / 72–128 GB | faster: retrieval search and rapidfuzz features scale with cores |

No GPU is needed for the baseline. Give the notebook a **≥ 100 GB EBS volume** (data 2.4 GB + parquet
caches + candidates/features). Only `/home/ec2-user/SageMaker/` survives a stop/start, so work there.

The notebook's **execution role** needs `s3:GetObject` + `s3:ListBucket` on
`arn:aws:s3:::awsomeminds-dataset-532749777349` (and `/*`).

## 2. One-time setup (notebook terminal)

```bash
cd ~/SageMaker
git clone https://github.com/anujkushwaha612/AWSomeMinds.git   # later: cd AWSomeMinds && git pull
cd AWSomeMinds
aws s3 ls s3://awsomeminds-dataset-532749777349/dataset/ --region ap-south-1        # access check
aws s3 sync s3://awsomeminds-dataset-532749777349/dataset/ data/dataset/ --region ap-south-1

python3 --version        # needs >= 3.11 for the pinned numpy/scikit-learn
# if it is older:  conda create -y -n ber python=3.12 && source activate ber
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q      # expect: 40 passed
```

Why sync instead of reading `s3://` directly: one 2.4 GB copy is faster than streaming the TSVs on every
stage, and it survives kernel restarts. To read straight from S3 instead, set
`data_root: s3://awsomeminds-dataset-532749777349/dataset` in `configs/pipeline.yaml`.

## 3. Baseline run (Sub #1)

Run detached so closing the browser tab doesn't kill it; every stage checkpoints, so re-running the same
command after a crash resumes where it stopped.

```bash
cd ~/SageMaker/AWSomeMinds && source .venv/bin/activate
nohup sh -c '
  python -u -m ber.folds &&
  python -u -m ber.normalize &&
  python -u -m ber.baseline build --split train &&
  python -u -m ber.baseline build --split test &&
  python -u -m ber.baseline train &&
  python -u -m ber.baseline predict --name sub01_baseline
' > baseline_run.log 2>&1 &
tail -f baseline_run.log
```

| Stage | Output | Rough time on 16 vCPU (estimate, not yet measured at full scale) |
|---|---|---|
| folds, normalize | `artifacts/folds.parquet`, `artifacts/norm/*.parquet` | ~1 min, ~5 min |
| build train (India, US) | `artifacts/baseline/train/*.parquet` | retrieval ~25–40 min per country + features |
| build test (France, India, US) | `artifacts/baseline/test/*.parquet` | similar |
| train (5 fold models + decision tuning) | `artifacts/baseline/models/`, `metrics.json` | ~10–20 min |
| predict | `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `subs/sub01_baseline/` | a few min |

**The offline metric** is printed at the end of `train` and saved in `artifacts/baseline/metrics.json`:
fold-0 macro F0.5, the candidate-set oracle ceiling, pair recall, mean candidates per S1, and per-country /
per-k breakdowns.

## 4. Submit

Download `output/matching_results.tsv` (Jupyter file browser → right-click → Download) and upload it on
the portal from the one registered machine. After the leaderboard score appears, write it into
`subs/sub01_baseline/meta.json` (`leaderboard_score`) and commit the snapshot folder (TSVs are
git-ignored; the meta file is kept).

## 5. Knobs (configs/pipeline.yaml → `baseline:`)

| Key | Default | Effect |
|---|---|---|
| `k_rec` | 3 | candidates per S2/S3 record: smaller = smaller candidate set, lower recall ceiling |
| `max_df` | 0.01 | block-purging cap: smaller = faster search, lower recall (see experiments.md E0b) |
| `min_score` | 0.0 | cosine floor for a candidate: raises precision of the candidate set |
| `workers` | 8 | featurization processes: raise to 16 on ≥ 64 GB |
| `train_frac` | 0.3 | share of training entities per fold model: raise for accuracy, lower for speed |

Change a knob → set `BER_RUN=<new_name>` so outputs go to `artifacts/<new_name>/` and don't overwrite
the baseline, then re-run the stages.
