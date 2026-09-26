# GPU runbook — full pipeline on the 48 GB GPU machine

One script runs everything from a fresh clone: setup → TF-IDF baseline → bi-encoder (A) → union +
GBDT (B) → gap analysis (C) → submission. Strategy and hyperparameter evidence:
[strategy_v5.md](strategy_v5.md) and [experiments.md](experiments.md) E4.

## 1. Before starting

- Windows: **Smart App Control must be off**, otherwise PyTorch's DLLs are blocked
  (`An Application Control policy has blocked this file`).
- `nvidia-smi` works and shows the GPU. Note the **CUDA Version** in its header (driver).
- Python 3.11–3.13 on PATH (`python --version`).
- About 60 GB free disk (data 2.4 GB, caches, features, models).
- Close heavy apps: the CPU stages are sized for ~15.7 GB RAM.

## 2. Run (Windows PowerShell)

```powershell
git clone https://github.com/anujkushwaha612/AWSomeMinds.git
cd AWSomeMinds
# put the dataset here: data\dataset\train\*.tsv and data\dataset\test\*.tsv
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Cuda cu124
```

`-Cuda` picks the PyTorch wheel: `cu124` for a driver showing CUDA 12.4 or newer, `cu121` for 12.1,
`cu128` for very new GPUs (RTX 50xx / Blackwell). Linux or WSL2: `CUDA_TAG=cu124 bash run_all.sh`.

**If anything stops, run the same command again.** Every finished stage is skipped (markers in
`artifacts\run_all\`); encoder training resumes from its last checkpoint (every 2,000 steps) as long as
the config is unchanged. Console log: `artifacts\run_all\console_log.txt`.

## 3. Stages and what to look at

| # | Stage | Device | Check in the log |
|---|---|---|---|
| 1 | setup, tests | – | `torch ... cuda True <GPU name>`, tests pass |
| 2 | folds, normalize | CPU | – |
| 3 | baseline build train / test (TF-IDF candidates) | CPU, ~2–2.5 h | `pair recall` India ≈ 0.94, US ≈ 0.97 |
| 4 | baseline train | CPU | `fold-0 macro F0.5` ≈ 0.949 (the reference) |
| 5 | env_check | GPU | encoding and training speed, time estimates; stops if no CUDA |
| 6 | pairs | CPU | share of each negative kind (TF-IDF / same name / same first word / random) |
| 7 | train_encoder | GPU | loss falling; recall gate: **dense R@3 vs TF-IDF R@3**, especially `indic` |
| 8 | dense train / test | GPU | dense pairs per S1 |
| 9 | union train / test | CPU | **`pair recall: TF-IDF … \| union …` — union must be higher** |
| 10 | stage1, stage2 | CPU | `fold-0 macro F0.5`, prune curve, chosen decision rule |
| 11 | compare | CPU | **delta vs baseline with 95% CI** |
| 12 | stress | CPU | fold-0 F0.5 under test-like density (same rule vs re-tuned) |
| 13 | predict | CPU | validator `PASS`, candidates and matches per S1 |
| 14 | france | CPU | per-country prediction statistics |

If step 5 estimates encoder training above ~3 h, stop, set `v5.neural.n_pairs` in
`configs\pipeline.yaml` to the suggested value, and run again with `-Redo pairs`.

## 4. After it finishes

- **Upload** `output\matching_results.tsv` (snapshot with metadata: `subs\sub_v5\`).
- **Send back:** `artifacts\run_all\console_log.txt`, `artifacts\v5\stage1.json`, `stage2.json`,
  `compare.json`, `artifacts\neural\eval.json`, `artifacts\experiments\C1_stress_v5.json`, `C2_france.json`.

## 5. Out-of-memory knobs (configs\pipeline.yaml → v5)

| Symptom | Change, then re-run with `-Redo <stage>` |
|---|---|
| CUDA out of memory in training | `neural.batch_size: 128` |
| CUDA out of memory in dense search | `neural.tile_gb: 3.0`, `neural.encode_batch: 256` |
| RAM exhausted in stage1 | `max_train_rows: 12000000` |
| RAM exhausted in baseline/union build | `baseline.workers: 4`, `baseline.feature_chunk: 250000` |
