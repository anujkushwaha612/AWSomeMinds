# GPU runbook — full pipeline on the GPU machine (sized for 32 GB VRAM)

One script runs everything from a fresh clone: setup → TF-IDF baseline → bi-encoder (A) → union +
GBDT (B) → cross-encoder (D) → gap analysis (C) → submissions. Strategy and hyperparameter evidence:
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
| 10 | stage1 | CPU | `fold-0 macro F0.5`, prune curve, **gain share** (are the `num_*` features used?) |
| 11 | stage2_noce, predict_noce | CPU | fold-0 F0.5 without cross-encoder; **`subs\sub_v5_noce` = safe submission** |
| 12 | ce_pairs | CPU | band share of pruned pairs and of true pairs; positive share of training pairs |
| 13 | ce_train | GPU | loss falling, acc rising; ETA |
| 14 | ce_score_train / ce_score_test | GPU | pairs/s and ETA; mean logit of positives far above negatives |
| 15 | stage2 | CPU | fold-0 F0.5 with `ce`; gain share of `ce` |
| 16 | compare | CPU | **delta vs baseline, and the `cross-encoder ablation` delta with 95% CI** |
| 17 | stress | CPU | fold-0 F0.5 under test-like density (same rule vs re-tuned) |
| 18 | predict | CPU | validator `PASS`; `subs\sub_v5` (with cross-encoder) |
| 19 | france | CPU | per-country prediction statistics |
| 20 | llm_check / llm_score_test / llm_score_train / llm_apply | local LLM | only if `v5.llm.enabled` (see §6); `[llm apply] ... -> KEEP` means upload `subs\sub_v5_llm` |

Stage 15 (and 11) also log `[ensemble] lgb / xgb / mean` tuning scores and the chosen option; LightGBM alone
is kept unless another option gains ≥ 0.0005 on the tuning folds.

If step 5 estimates encoder training above ~3 h, stop, set `v5.neural.n_pairs` in
`configs\pipeline.yaml` to the suggested value, and run again with `-Redo pairs`.

## 4. After it finishes

- **Which file to upload:** open `artifacts\v5\compare.json` → `ce_ablation_stage2_vs_stage2_noce`.
  If `ci_low > 0`, upload `subs\sub_v5\matching_results.tsv` (also in `output\`); otherwise upload
  `subs\sub_v5_noce\matching_results.tsv`.
- **Short on time?** `stage2_noce` / `predict_noce` finish before any cross-encoder stage, so a valid
  submission exists early. To skip the cross-encoder entirely set `v5.cross_encoder.enabled: false`.
- **Send back:** `artifacts\run_all\console_log.txt`, `artifacts\v5\stage1.json`, `stage2.json`, `stage2_noce.json`,
  `compare.json`, `artifacts\neural\eval.json`, `artifacts\experiments\C1_stress_v5.json`, `C2_france.json`.

## 5. Out-of-memory knobs (configs\pipeline.yaml → v5)

| Symptom | Change, then re-run with `-Redo <stage>` |
|---|---|
| CUDA out of memory in training | `neural.batch_size: 128` |
| CUDA out of memory in dense search | `neural.tile_gb: 2.0`, `neural.encode_batch: 256` |
| CUDA out of memory in the cross-encoder | `cross_encoder.batch_size: 64` (train) / `score_batch: 512` (score) |
| Cross-encoder scoring ETA too long | narrow `cross_encoder.band`, e.g. `[0.01, 0.99]`; delete `artifacts\v5\ce\`, then `-Redo ce_score_train` and `-Redo ce_score_test` |
| Union files built before v5.1 | stage1 warns that the `num_*` features are missing: `-Redo union_train`, `-Redo union_test` to add them |
| RAM exhausted in stage1 | `max_train_rows: 12000000` |
| RAM exhausted in baseline/union build | `baseline.workers: 4`, `baseline.feature_chunk: 250000` |

## 6. Optional: local LLM judge (stage 3)

Only MIT/Apache-2.0 models of ≤ 8B parameters are allowed, and entities must not be resolved through an
external service, so the model runs **on this machine** with Ollama (Ollama Cloud has no allowed model).

```powershell
# 1. install Ollama from https://ollama.com/download, then in a SEPARATE terminal:
$env:OLLAMA_NUM_PARALLEL = "16"; ollama serve
# 2. in the run terminal:
ollama pull qwen3:4b-instruct-2507-q4_K_M          # Qwen3-4B-Instruct-2507, Apache-2.0, 4.0B
# 3. configs\pipeline.yaml -> v5.llm.enabled: true   (stage2_tag: "noce" if the CE ablation was not KEEP)
# 4. run the same run_all command again: only the llm_* stages run
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Cuda cu124 -SkipSetup
```

`llm_check` prints three sample verdicts and the pairs/s → ETA for `max_pairs_test` + `max_pairs_train`
(defaults 40k + 20k). If the ETA is too long, lower those two numbers. Answers are cached in
`artifacts\v5\llm\`, so an interrupted run resumes. Result: `artifacts\v5\llm.json` (`verdict`) and
`subs\sub_v5_llm\` — upload it only if the verdict is `KEEP`.
