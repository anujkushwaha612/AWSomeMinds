# GPU runbook — strategy v5 (A + B + C) on the GPU machine

What runs where is in [strategy_v5.md](strategy_v5.md) §1. This page is the checklist for the machine
with the NVIDIA GPU (48 GB VRAM, ~15.7 GB RAM).

## 1. Get the inputs onto the GPU machine

The v5 pipeline builds on the baseline's TF-IDF candidates. Pick one:

- **Copy from the laptop (faster, ~7.5 GB):** the whole `artifacts\` folder (`folds.parquet`, `norm\`,
  `cache\`, `baseline\`) plus `data\dataset\`. Put both inside the repo folder on the GPU machine.
- **Rebuild there (~2.5 h CPU):** copy only `data\dataset\`, then run `.\run_baseline.ps1`.

## 2. One-time setup (PowerShell, repo folder)

```powershell
git pull                                   # or: git clone https://github.com/anujkushwaha612/AWSomeMinds.git
nvidia-smi                                 # note the "CUDA Version" in the header
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # cu121 if CUDA 12.1
.\.venv\Scripts\python -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name. On Windows with **Smart App Control** on, PyTorch's
DLLs may be blocked (`An Application Control policy has blocked this file`), as happened on the laptop.
Whether to turn Smart App Control off is the machine owner's decision; the alternative is WSL2 (Ubuntu),
which supports CUDA, where the same commands run with `.venv/bin/python`.

## 3. Measure before committing (first 10 minutes)

```powershell
.\.venv\Scripts\python -u -m ber.neural.env_check
```

It prints the encoding speed, training speed and time estimates for every GPU stage. If training on all
3.06M pairs would take more than ~3 h, it prints a suggested `v5.neural.n_pairs`. Set that in
`configs\pipeline.yaml` before continuing.

## 4. Run everything

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v5.ps1
```

| Stage | Where | Output |
|---|---|---|
| A0 env check | GPU | console |
| A1 training pairs | CPU, ~5 min | `artifacts\neural\train_pairs.parquet`, `eval_records.parquet` |
| A2–A3 fine-tune + recall gate | GPU (time from A0) | `artifacts\neural\biencoder\`, `eval.json` |
| A4 dense search train + test | GPU (time from A0) | `artifacts\dense\{train,test}\*` |
| A5 union features train + test | CPU, ~40–60 min [H] | `artifacts\union\{train,test}\*` |
| B stage 1, stage 2, compare | CPU, ~30–60 min [H] | `artifacts\v5\stage1.json`, `stage2.json`, `compare.json` |
| C1 density stress | CPU, ~10 min | `artifacts\experiments\C1_stress_v5.json` |
| B predict | CPU, ~10 min | `output\matching_results.tsv`, `subs\sub03_v5\` |
| C2 France diagnostics | CPU, ~5 min | `artifacts\experiments\C2_france.json` |

If a stage fails, re-run the same command. Finished stages are skipped (dense search and union per
country), training resumes from its last checkpoint (every 2,000 steps), and the GBDT stages rerun in
minutes. The full console log is in `artifacts\v5\run_v5_console.txt`.

## 5. Gates to read before submitting

1. `artifacts\neural\eval.json`: `gate_dense_R@3_ge_tfidf_R@3` should be `true`. Compare `indic` dense
   R@3 against TF-IDF R@3 as well.
2. Log line `[union train/...] pair recall: TF-IDF ... | union ...`: the union must be higher.
3. `artifacts\v5\compare.json`: stage 2 vs baseline delta with CI > 0.
4. `C1_stress_v5.json`: if "re-tuned" is clearly better than "same rule", tell the laptop side (the
   thresholds should then be tuned under stress for the submission).

## 6. Send back to the laptop

`artifacts\v5\log.txt`, `artifacts\v5\*.json`, `artifacts\neural\eval.json`, `artifacts\experiments\C*.json`,
and `output\matching_results.tsv` + `output\candidate_pairs.tsv` (or upload from the GPU machine).

## 7. If something runs out of memory

| Symptom | Setting (configs\pipeline.yaml → v5) |
|---|---|
| CUDA out of memory in training | `neural.batch_size: 128` |
| CUDA out of memory in search | `neural.tile_gb: 3.0`, `neural.encode_batch: 256` |
| RAM (MemoryError / killed) in stage 1 | `max_train_rows: 12000000` |
| Training too slow | `neural.n_pairs` as suggested by env_check, or `neural.model: intfloat/multilingual-e5-small` |
