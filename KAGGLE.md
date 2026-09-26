# Free-GPU plan: Kaggle T4s for the neural part, laptop for everything else

No team member has a GPU, so the work is split:

| Where | What | Time (estimate; the run measures and prints the real numbers) |
|---|---|---|
| Kaggle, account A | `train`: throughput probe, bi-encoder fine-tune (sized to fit the 12 h session), recall gate | up to ~11 h |
| Kaggle, accounts B and C, **in parallel** | `dense`: dense search, 2–3 split/country jobs per session (one per T4) | ~1–2 h |
| Laptop (CPU, 16 threads, 15.7 GB RAM) | union features → stage 1 → stage 2 (LightGBM + XGBoost) → submission | a few hours |

The cross-encoder (D) and the LLM judge are **off**: on a T4 they would not finish before the deadline.
The TF-IDF baseline stays the fallback submission (fold-0 macro F0.5 0.949).

**Why not 4 × `train`:** fine-tuning is one model trained step by step, so it cannot be split across
accounts. The dense search can: after training, its 5 jobs (train/US, train/India, test/US, test/India,
test/France) are independent. Every team member uses their **own** Kaggle account (Kaggle forbids one
person using several). Keep the dataset and notebooks **private**, shared only with teammates.

## 0. Once per account

kaggle.com → sign in → Settings → **phone verification** (needed for GPU and Internet).
Free quota: 30 GPU hours per week per account.

## 1. Laptop: build and upload the bundle (done once)

```powershell
.venv\Scripts\python kaggle\make_bundle.py        # -> artifacts\kaggle_bundle.zip (1.35 GB)
```

It holds only the columns the GPU stages read (normalized name/address, TF-IDF candidate row ids,
training triplets) plus `src/`, `configs/` and `kaggle/`. Account A: kaggle.com → Datasets → New Dataset
→ upload the zip (Kaggle unpacks it) → keep it **Private** → Settings → Sharing → add B and C.

## 2. Account A: `train`

New notebook → **Add Input** → the bundle dataset **and** the code dataset `ber-code` (the small
`artifacts\kaggle_code.zip` from `make_bundle.py --code-only`; update only this one when the code changes).
Right panel: **Accelerator: GPU T4 x2**, **Internet: on**. Cell 1 (the same in every notebook) combines
the newest code with the uploaded data:

```python
import glob, os, shutil
runs = glob.glob('/kaggle/input/**/code/kaggle/run_gpu.py', recursive=True)
data = next(p for p in runs if os.path.isdir(p.rsplit('/code/', 1)[0] + '/artifacts')).rsplit('/code/', 1)[0]
code = next(p for p in runs if not p.startswith(data + '/')).rsplit('/kaggle/run_gpu.py', 1)[0]
shutil.rmtree('/tmp/b', ignore_errors=True); os.makedirs('/tmp/b')
shutil.copytree(code, '/tmp/b/code'); os.symlink(data + '/artifacts', '/tmp/b/artifacts')
RUN = '/tmp/b/code/kaggle/run_gpu.py'
print('data:', data, '| code:', code)
!nvidia-smi -L
```

Cell 2:

```
!python {RUN} train
```

Run it with **Save Version → Save & Run All (Commit)**. It keeps running with the browser closed, and
the output is saved even on a timeout. First lines to check: the GPU names, then `plan: {...}`, which gives
the measured steps/s and how many training pairs fit the session. To finish sooner (fewer pairs), start
it as `!SESSION_H=8 python ...`.

Done when the log shows `saved fine-tuned encoder` and `GATE A-retrieval ...: PASS/FAIL`. Share the notebook
with B and C (Share → add collaborators).
**If the session ended early:** make a new version with its own output attached as an input
(Add Input → Notebook Output) and run the same cell. It resumes from the last checkpoint (every 2,000 steps).

## 3. Accounts B and C (in parallel): `dense`

Each person: new notebook → Add Input → the bundle dataset, `ber-code`, **and** A's notebook output (Notebook Output
tab). GPU T4 x2, Internet on. The job lists are balanced so each T4 gets roughly the same number of texts:

```
# account B
!python {RUN} dense train/US test/India
# account C
!python {RUN} dense train/India test/US test/France
```

(One person can also run `dense` with no job list: all 5 jobs on both GPUs, roughly twice as long.)
A job that fails can be re-run the same way; finished jobs are skipped.

## 4. Laptop: bring the results back and finish

From each notebook's **Output** tab, download its `results_*.zip` (A: `results_train.zip`; B:
`results_dense_train-US_test-India.zip`; C: `results_dense_train-India_test-US_test-France.zip`). Then:

```powershell
powershell -ExecutionPolicy Bypass -File .\kaggle\after_kaggle.ps1 -Results (Get-ChildItem $HOME\Downloads\results_*.zip).FullName
```

It checks that all 10 dense files exist and runs `union_train`, `union_test`, `stage1`, `stage2_noce`,
`predict_noce`, `compare`, `stress` and `france` (markers in `artifacts\free\`, so re-running resumes).

**Which file to upload:** `artifacts\v5\compare.json` → `stage2_noce_vs_baseline`. If `ci_low > 0`, upload
`subs\sub_v5_noce\matching_results.tsv`; otherwise keep the baseline submission.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `env_check failed at batch 256` | handled: it retries at batch 128 by itself |
| CUDA out of memory in `dense` | the tile is already 1 GB; run one job per session (`dense train/US`) |
| `no encoder found` in `dense` | A's notebook output is not attached, or A's training has not finished |
| Model download fails | Internet is off, or the account is not phone-verified |
| Kaggle disk / output limit (20 GB) | the runner deletes the 2 GB optimizer state after training and unlinks the inputs |
