#!/usr/bin/env bash
# One command, fresh clone -> submission (Linux / WSL2 / cloud GPU). Same stages as run_all.ps1.
#
#   bash run_all.sh                    # data in data/dataset/{train,test}/
#   CUDA_TAG=cu121 bash run_all.sh     # PyTorch CUDA wheel tag matching `nvidia-smi` (default cu124)
#   REDO=stage2 bash run_all.sh        # re-run one stage
#   SKIP_SETUP=1 bash run_all.sh       # do not create the venv / install packages
# Smoke tests only: ALLOW_CPU=1 LIMIT=20000 TAG=smoke_run bash run_all.sh
# Every stage writes artifacts/<TAG>/<stage>.done; re-running skips finished stages.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
CUDA_TAG="${CUDA_TAG:-cu124}"; TAG="${TAG:-run_all}"; REDO="${REDO:-}"
MARK="artifacts/$TAG"; mkdir -p "$MARK"; LOG="$MARK/console_log.txt"
PY=.venv/bin/python

say() { echo -e "$1" | tee -a "$LOG"; }

stage() {
  local name="$1"; shift
  if [[ -f "$MARK/$name.done" && "$REDO" != "$name" ]]; then say "--- skip $name (done $(cat "$MARK/$name.done"))"; return; fi
  say "\n=== [$(date +%H:%M:%S)] $name :: python $*"
  if ! "$PY" "$@" 2>&1 | tee -a "$LOG"; then
    say "FAILED: $name. Fix the cause, then run the same command again to resume."; exit 1
  fi
  date '+%Y-%m-%d %H:%M:%S' > "$MARK/$name.done"
}

for f in train/train_source1.tsv train/train_source2.tsv train/train_source3.tsv train/train_ground_truth.tsv \
         test/test_source1.tsv test/test_source2.tsv test/test_source3.tsv; do
  [[ -f "data/dataset/$f" ]] || { say "missing data/dataset/$f"; exit 1; }
done

if [[ -z "${SKIP_SETUP:-}" ]]; then
  [[ -x "$PY" ]] || python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
  "$PY" -m pip install -q torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  "$PY" -m pip install -q -r requirements-gpu.txt
  "$PY" -m pip install -q -e .
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$LOG"

LIMIT_ARGS=(); [[ -n "${LIMIT:-}" ]] && LIMIT_ARGS=(--limit "$LIMIT")
CPU_ARGS=(); [[ -n "${ALLOW_CPU:-}" ]] && CPU_ARGS=(--allow-cpu)

stage tests                -m pytest -q
stage folds                -u -m ber.folds
stage normalize            -u -m ber.normalize
stage baseline_build_train -u -m ber.baseline build --split train "${LIMIT_ARGS[@]}"
stage baseline_build_test  -u -m ber.baseline build --split test "${LIMIT_ARGS[@]}"
stage baseline_train       -u -m ber.baseline train
stage env_check            -u -m ber.neural.env_check "${CPU_ARGS[@]}"
stage pairs                -u -m ber.neural.pairs
stage train_encoder        -u -m ber.neural.train_biencoder
stage dense_train          -u -m ber.neural.dense_retrieve --split train
stage dense_test           -u -m ber.neural.dense_retrieve --split test
stage union_train          -u -m ber.union --split train
stage union_test           -u -m ber.union --split test
stage stage1               -u -m ber.v5 stage1
stage stage2_noce          -u -m ber.v5 stage2 --tag noce --no-ce       # safe submission + CE ablation reference
stage predict_noce         -u -m ber.v5 predict --tag noce --name sub_v5_noce
stage ce_pairs             -u -m ber.neural.cross_encoder pairs
stage ce_train             -u -m ber.neural.cross_encoder train "${CPU_ARGS[@]}"
stage ce_score_train       -u -m ber.neural.cross_encoder score --split train "${CPU_ARGS[@]}"
stage ce_score_test        -u -m ber.neural.cross_encoder score --split test "${CPU_ARGS[@]}"
stage stage2               -u -m ber.v5 stage2
stage compare              -u -m ber.v5 compare
stage stress               -u -m ber.gap stress --run v5
stage predict              -u -m ber.v5 predict --name sub_v5
stage france               -u -m ber.gap france
say "\nDONE. Snapshots: subs/sub_v5_noce (no cross-encoder) and subs/sub_v5 (with; now in output/)."
say "Pick by artifacts/v5/compare.json -> ce_ablation_stage2_vs_stage2_noce (upload sub_v5 only if its CI > 0)."
