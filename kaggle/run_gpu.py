"""Kaggle side of the free-GPU plan (KAGGLE.md): bi-encoder fine-tune + dense search on free T4s.

Needs a Kaggle notebook with Accelerator "GPU T4 x2", Internet on, and the bundle dataset
(kaggle/make_bundle.py) attached. One cell:

    !python /kaggle/input/<bundle>/code/kaggle/run_gpu.py train
        env check, fine-tune (sized to fit the session), recall gate. Output: artifacts/neural/biencoder
    !python /kaggle/input/<bundle>/code/kaggle/run_gpu.py dense train/US test/India
        dense search for the listed split/country jobs, spread over the session's GPUs.
        Attach the output of the `train` notebook as a second input (it holds the encoder).
    !python /kaggle/input/<bundle>/code/kaggle/run_gpu.py all
        both in one session (one person, no hand-over).

Several team members can run `dense` in parallel on their own accounts, each with a different
job list (KAGGLE.md). Every run writes /kaggle/working/results_<mode>.zip: extract it into the
laptop's artifacts/ folder. Re-running after a timeout resumes (attach the previous output).
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

T0 = time.time()
BUNDLE = Path(__file__).resolve().parents[2]            # <bundle>/code/kaggle/run_gpu.py
CODE = BUNDLE / "code"
WORK = Path(os.environ.get("KAGGLE_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
ART = WORK / "artifacts"
LOG = WORK / "run_gpu_log.txt"
SESSION_H = float(os.environ.get("SESSION_H", "12"))    # Kaggle GPU session limit
SAFETY_H = 1.0                                           # checkpointing, packaging, slack
ALL_JOBS = ["train/US", "train/India", "test/US", "test/India", "test/France"]
VRAM_SHARE = 0.75       # env_check peak above this share of the card -> smaller batch (real batches are longer)
LAST_OOM = False        # set by run(): the last subprocess hit CUDA out of memory


def say(msg: str) -> None:
    """Print and append to the run log."""
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def hours() -> float:
    return (time.time() - T0) / 3600


def env(gpu: str | None = None) -> dict:
    """Environment of a pipeline subprocess (code from the bundle, config from WORK)."""
    e = dict(os.environ, PYTHONPATH=str(CODE / "src"), BER_CONFIG=str(WORK / "pipeline_kaggle.yaml"),
             PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONWARNINGS="ignore",
             TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True")
    if gpu is not None:
        e["CUDA_VISIBLE_DEVICES"] = gpu
    return e


def run(args: list[str], gpu: str | None = "0", log_path: Path | None = None) -> int:
    """Run ``python -u -m <args>``, stream its output, return the exit code (sets LAST_OOM)."""
    global LAST_OOM
    LAST_OOM = False
    say(f"\n=== [{hours():.2f} h] GPU {gpu}: python -m {' '.join(args)}")
    p = subprocess.Popen([sys.executable, "-u", "-m", *args], env=env(gpu), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    out = open(log_path or LOG, "a", encoding="utf-8")
    for line in p.stdout:
        print(line, end="", flush=True)
        out.write(line)
        LAST_OOM = LAST_OOM or "OutOfMemoryError" in line or "CUDA out of memory" in line
    out.close()
    return p.wait()


def write_config(**neural) -> None:
    """Bundle config with Kaggle paths, no cross-encoder / LLM, and ``neural`` overrides."""
    cfg = yaml.safe_load(open(CODE / "configs" / "pipeline.yaml", encoding="utf-8"))
    cfg["artifacts_root"] = str(ART)
    cfg["data_root"] = str(BUNDLE / "no_raw_data")      # the GPU stages never read the raw TSVs
    v = cfg["v5"]
    v["cross_encoder"]["enabled"] = False
    v["llm"]["enabled"] = False
    v["neural"]["tile_gb"] = 1.0                        # 16 GB T4: similarity tiles of 1 GB
    v["neural"].update(neural)
    yaml.safe_dump(cfg, open(WORK / "pipeline_kaggle.yaml", "w", encoding="utf-8"), sort_keys=False)


def link_inputs() -> None:
    """artifacts/ in WORK: inputs linked from the bundle (read-only), outputs written for real."""
    (ART / "neural").mkdir(parents=True, exist_ok=True)
    (ART / "dense").mkdir(parents=True, exist_ok=True)
    for name in ("norm", "baseline"):
        dst = ART / name
        if not dst.exists():
            dst.symlink_to(BUNDLE / "artifacts" / name, target_is_directory=True)
    for f in ("train_pairs.parquet", "eval_records.parquet"):
        if not (ART / "neural" / f).exists():
            shutil.copy2(BUNDLE / "artifacts" / "neural" / f, ART / "neural" / f)
    # outputs of earlier sessions (resume / hand-over): the most advanced encoder checkpoint
    best, best_step = None, -1
    for st in INPUT.glob("**/neural/biencoder/train_state.json"):
        s = json.load(open(st))["step"]
        if s > best_step:
            best, best_step = st.parent, s
    mine = ART / "neural" / "biencoder" / "train_state.json"
    if best is not None and (not mine.exists() or json.load(open(mine))["step"] < best_step):
        shutil.rmtree(ART / "neural" / "biencoder", ignore_errors=True)
        shutil.copytree(best, ART / "neural" / "biencoder")
        say(f"encoder checkpoint from {best} (step {best_step:,})")
        for f in ("kaggle_plan.json", "eval.json", "env_check.json"):
            if (best.parent / f).exists() and not (ART / "neural" / f).exists():
                shutil.copy2(best.parent / f, ART / "neural" / f)
    for f in INPUT.glob("**/dense/*/*"):                 # dense jobs finished by earlier sessions
        dst = ART / "dense" / f.parent.name / f.name
        if f.is_file() and not dst.exists() and BUNDLE not in f.parents:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)


def n_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))
    except FileNotFoundError:
        return 0


def ensure_packages() -> None:
    """Install only what the Kaggle image lacks (never touch its CUDA build of torch)."""
    need = []
    for mod, pkg in (("transformers", "transformers"), ("sentencepiece", "sentencepiece"),
                     ("safetensors", "safetensors"), ("psutil", "psutil"), ("pyarrow", "pyarrow")):
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])


def plan_training(dense_after: bool, batches=(256, 128, 64)) -> dict:
    """env_check at the largest batch that leaves VRAM headroom, then pick n_pairs to fit the session."""
    plan_path = ART / "neural" / "kaggle_plan.json"
    if plan_path.exists():                               # resume: same pairs, same fingerprint
        plan = json.load(open(plan_path))
        write_config(n_pairs=plan["n_pairs"], batch_size=plan["batch_size"])
        say(f"plan from the previous session: {plan}")
        return plan
    for i, batch in enumerate(batches):
        write_config(batch_size=batch)
        if run(["ber.neural.env_check"]) != 0:
            say(f"env_check failed at batch {batch} (see the log above; usually CUDA out of memory)")
            continue
        ec = json.load(open(ART / "neural" / "env_check.json"))
        limit = VRAM_SHARE * ec.get("vram_gb", 14.5)
        if ec["peak_vram_gb"] > limit and i + 1 < len(batches):
            say(f"batch {batch}: probe peak {ec['peak_vram_gb']:.1f} GB > {limit:.1f} GB; real batches are "
                f"longer (the first training run at 256 ran out of memory) -> trying batch {batches[i + 1]}")
            continue
        break
    else:
        raise SystemExit(f"env_check failed at batch {batches[-1]} too: is the accelerator set to GPU T4 x2?")
    ec = json.load(open(ART / "neural" / "env_check.json"))
    gate_h = (ec["n_texts_train_s1"] + 2 * ec["eval_records"]) / ec["enc_texts_per_s"] / 3600
    dense_h = 0.0
    if dense_after:                                      # train + test texts, split over the GPUs
        dense_h = 1.3 * ec["n_texts"] / ec["enc_texts_per_s"] / 3600 / max(1, min(2, n_gpus()))
    train_h = SESSION_H - SAFETY_H - hours() - gate_h - dense_h
    n_fit = int(max(0.0, train_h) * 3600 * ec["train_steps_per_s"] * ec["batch_size"])
    n_pairs = min(ec["n_pairs_available"], n_fit)
    if n_pairs < 300_000:
        say(f"WARNING: only {n_pairs:,} pairs fit this session; the encoder may not beat TF-IDF")
    plan = {"n_pairs": 0 if n_pairs >= ec["n_pairs_available"] else n_pairs,
            "batch_size": ec["batch_size"], "train_h": round(train_h, 2), "gate_h": round(gate_h, 2),
            "dense_h": round(dense_h, 2), "gpu": ec["gpu"], "steps_per_s": ec["train_steps_per_s"],
            "texts_per_s": ec["enc_texts_per_s"]}
    json.dump(plan, open(plan_path, "w"), indent=2)
    write_config(n_pairs=plan["n_pairs"], batch_size=plan["batch_size"])
    say(f"plan: {plan}  (n_pairs 0 = all {ec['n_pairs_available']:,})")
    return plan


def train_stage(dense_after: bool) -> None:
    plan = plan_training(dense_after)
    while run(["ber.neural.train_biencoder"]) != 0:
        ckpt = ART / "neural" / "biencoder"
        smaller = [b for b in (128, 64) if b < plan["batch_size"]]
        if not (LAST_OOM and smaller and not (ckpt / "train_state.json").exists()):
            raise SystemExit("train_biencoder failed: re-run the same command (it resumes from the checkpoint)")
        say(f"training ran out of GPU memory at batch {plan['batch_size']}: re-planning at {smaller[0]}")
        (ART / "neural" / "kaggle_plan.json").unlink()
        shutil.rmtree(ckpt, ignore_errors=True)
        plan = plan_training(dense_after, batches=tuple(smaller))
    opt = ART / "neural" / "biencoder" / "optimizer.pt"
    st = json.load(open(ART / "neural" / "biencoder" / "train_state.json"))
    if st["step"] >= st["total"] and opt.exists():
        opt.unlink()                                     # 2 GB, only needed to resume training


def dense_stage(jobs: list[str]) -> None:
    """Run split/country jobs, one worker process per GPU, each taking jobs in order."""
    state = ART / "neural" / "biencoder" / "train_state.json"
    if not state.exists():
        raise SystemExit("no encoder found: attach the output of the `train` notebook as an input")
    if not (WORK / "pipeline_kaggle.yaml").exists():
        write_config()
    for j in jobs:
        if j not in ALL_JOBS:
            raise SystemExit(f"unknown job {j!r}; choose from {ALL_JOBS}")
    k = max(1, min(n_gpus(), len(jobs)))
    queues = [jobs[i::k] for i in range(k)]
    procs = []
    for gpu, q in enumerate(queues):
        cmds = " && ".join(f"{sys.executable} -u -m ber.neural.dense_retrieve --split {j.split('/')[0]} "
                           f"--country {j.split('/')[1]}" for j in q)
        say(f"\n=== [{hours():.2f} h] GPU {gpu}: {q}")
        glog = open(WORK / f"dense_gpu{gpu}_log.txt", "a", encoding="utf-8")
        procs.append((gpu, q, subprocess.Popen(cmds, shell=True, env=env(str(gpu)), stdout=glog,
                                               stderr=subprocess.STDOUT)))
    while any(p.poll() is None for _, _, p in procs):   # progress from the per-GPU logs
        time.sleep(120)
        for gpu, _, _ in procs:
            lines = open(WORK / f"dense_gpu{gpu}_log.txt", encoding="utf-8", errors="replace").read().splitlines()
            if lines:
                say(f"[{hours():.2f} h] GPU {gpu}: {lines[-1][:160]}")
    bad = [(gpu, q) for gpu, q, p in procs if p.returncode != 0]
    if bad:
        raise SystemExit(f"dense jobs failed: {bad}; see dense_gpu*_log.txt, then re-run (finished jobs are skipped)")


def package(mode: str) -> None:
    """results_<mode>.zip: what the laptop needs (dense candidates, encoder report, logs)."""
    z = WORK / f"results_{mode}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted((ART / "dense").glob("*/*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                zf.write(f, f"dense/{f.parent.name}/{f.name}")
        for f in ("eval.json", "env_check.json", "kaggle_plan.json"):
            if (ART / "neural" / f).exists():
                zf.write(ART / "neural" / f, f"neural/{f}")
        for f in WORK.glob("*_log.txt"):
            zf.write(f, f"kaggle_logs/{f.name}")
    say(f"\nwrote {z} ({z.stat().st_size / 1e6:.0f} MB): download it and extract into the laptop's artifacts/")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "dense", "all"):
        raise SystemExit(__doc__)
    mode, jobs = sys.argv[1], sys.argv[2:] or ALL_JOBS
    WORK.mkdir(parents=True, exist_ok=True)
    say(f"bundle {BUNDLE} | mode {mode} | GPUs {n_gpus()}")
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working | tail -1; "
                   "free -g | head -2", shell=True)
    ensure_packages()
    link_inputs()
    try:
        if mode in ("train", "all"):
            train_stage(dense_after=(mode == "all"))
        if mode in ("dense", "all"):
            dense_stage(jobs)
    finally:
        package(mode if mode != "dense" else "dense_" + "_".join(j.replace("/", "-") for j in jobs))
        for name in ("norm", "baseline"):                # keep the saved output small
            if (ART / name).is_symlink():
                (ART / name).unlink()


if __name__ == "__main__":
    main()
