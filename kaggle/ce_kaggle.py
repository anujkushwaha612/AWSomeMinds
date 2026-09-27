"""E20 (Kaggle GPU): gray-zone cross-encoder, standalone (torch + transformers only; inputs from e20_ce_prep.py).

Reads ce_train.parquet / ce_score.parquet from any attached Kaggle dataset, fine-tunes a multilingual cross-encoder
(one logit, BCE) on the training pairs for one epoch (or until CE_TRAIN_MIN minutes), then scores every pair of
ce_score.parquet: fold 0 first (prints its AUC), then test, then folds 1-2. Results are written per part to
/kaggle/working/ce_scores_<part>.parquet (split, rid, ce) as soon as each part is done.

  !python ce_kaggle.py        # env: CE_MODEL, CE_TRAIN_MIN (30), CE_EPOCHS (1), CE_BS (128), CE_LR (3e-5)
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = os.environ.get("CE_MODEL", "intfloat/multilingual-e5-base")      # MIT, 278M parameters
MAX_LEN = int(os.environ.get("CE_MAX_LEN", 128))
BS = int(os.environ.get("CE_BS", 128))
LR = float(os.environ.get("CE_LR", 3e-5))
TRAIN_MIN = float(os.environ.get("CE_TRAIN_MIN", 30))
EPOCHS = int(os.environ.get("CE_EPOCHS", 1))
SCORE_BS = int(os.environ.get("CE_SCORE_BS", 1024))
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SEED = 2026


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(name)
    if not hits:
        raise SystemExit(f"{name} not found under /kaggle/input: attach the ce-data dataset")
    return hits[0]


def encode(tok, a, b):
    """Token ids per pair (no padding), in chunks so memory stays flat."""
    ids = []
    for s in range(0, len(a), 100_000):
        ids += tok(list(a[s:s + 100_000]), list(b[s:s + 100_000]), truncation="longest_first",
                   max_length=MAX_LEN)["input_ids"]
    return ids


def collate(ids, sel, pad, dev):
    L = max(len(ids[i]) for i in sel)
    x = np.full((len(sel), L), pad, np.int64)
    m = np.zeros((len(sel), L), np.int64)
    for r, i in enumerate(sel):
        x[r, :len(ids[i])] = ids[i]
        m[r, :len(ids[i])] = 1
    return torch.from_numpy(x).to(dev, non_blocking=True), torch.from_numpy(m).to(dev, non_blocking=True)


def main():
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # CPU: smoke tests only
    amp = dev.type == "cuda"
    n_gpu = torch.cuda.device_count()
    print(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpu)] or 'none (CPU)'} | model {MODEL} | batch {BS} | "
          f"lr {LR} | train cap {TRAIN_MIN} min", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    net = torch.nn.DataParallel(model) if n_gpu > 1 else model
    pad = tok.pad_token_id

    # ---------------------------------------------------------------- train
    tr = pd.read_parquet(find("ce_train.parquet"))
    if os.environ.get("CE_SMOKE"):
        tr = tr.sample(int(os.environ["CE_SMOKE"]), random_state=SEED)
    t0 = time.time()
    ids = encode(tok, tr["text_a"].to_numpy(), tr["text_b"].to_numpy())
    y = tr["label"].to_numpy().astype(np.float32)
    lens = np.fromiter((len(x) for x in ids), np.int32, count=len(ids))
    print(f"train pairs {len(ids):,} (positive share {y.mean():.3f}), tokens p50/p99/max "
          f"{np.percentile(lens, 50):.0f}/{np.percentile(lens, 99):.0f}/{lens.max()}, tokenized in "
          f"{time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(SEED)
    batches = []
    for _ in range(EPOCHS):
        perm = rng.permutation(len(ids))
        eb = []
        for s in range(0, len(perm), BS * 50):              # length buckets of 50 batches: little padding
            chunk = perm[s:s + BS * 50]
            chunk = chunk[np.argsort(lens[chunk], kind="stable")]
            eb += [chunk[i:i + BS] for i in range(0, len(chunk), BS) if len(chunk[i:i + BS]) == BS]
        rng.shuffle(eb)
        batches += eb
    total = len(batches)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    net.train()
    t0, losses, accs = time.time(), [], []
    for step, sel in enumerate(batches):
        x, m = collate(ids, sel, pad, dev)
        yy = torch.from_numpy(y[sel]).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            logit = net(input_ids=x, attention_mask=m).logits.squeeze(-1).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yy)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(float(loss))
        accs.append(float(((logit > 0).float() == yy).float().mean()))
        el = time.time() - t0
        if (step + 1) % 100 == 0 or step + 1 == total:
            rate = (step + 1) / el
            print(f"step {step + 1:,}/{total:,} loss {np.mean(losses[-100:]):.4f} acc {np.mean(accs[-100:]):.4f} "
                  f"{rate:.2f} steps/s ({rate * BS:.0f} pairs/s) ETA {(total - step - 1) / rate / 60:.1f} min",
                  flush=True)
        if el > TRAIN_MIN * 60:
            print(f"time cap reached after {step + 1:,}/{total:,} steps", flush=True)
            break
    del ids, tr
    model.save_pretrained(os.path.join(OUT, "ce_model"))
    tok.save_pretrained(os.path.join(OUT, "ce_model"))
    print(f"trained in {(time.time() - t0) / 60:.1f} min; saved {OUT}/ce_model", flush=True)

    # ---------------------------------------------------------------- score
    net.eval()
    sc = pd.read_parquet(find("ce_score.parquet"))
    if os.environ.get("CE_SMOKE"):
        sc = sc.groupby("part", group_keys=False).apply(lambda g: g.sample(min(len(g), int(os.environ["CE_SMOKE"])), random_state=SEED))
    order = [p for p in ("f0", "test", "f12") if (sc["part"] == p).any()]
    for part in order:
        d = sc[sc["part"] == part].reset_index(drop=True)
        path = os.path.join(OUT, f"ce_scores_{part}.parquet")
        if os.path.exists(path):
            print(f"{part}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        ids = encode(tok, d["text_a"].to_numpy(), d["text_b"].to_numpy())
        lens = np.fromiter((len(x) for x in ids), np.int32, count=len(ids))
        o = np.argsort(lens, kind="stable")
        ce = np.empty(len(ids), np.float32)
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            for n, s in enumerate(range(0, len(o), SCORE_BS)):
                sel = o[s:s + SCORE_BS]
                x, m = collate(ids, sel, pad, dev)
                ce[sel] = net(input_ids=x, attention_mask=m).logits.squeeze(-1).float().cpu().numpy()
                if n % 200 == 0:
                    done = s + len(sel)
                    rate = done / (time.time() - t0)
                    print(f"  {part}: {done:,}/{len(o):,} {rate:,.0f} pairs/s ETA {(len(o) - done) / rate / 60:.1f} min",
                          flush=True)
        pd.DataFrame({"split": d["split"], "rid": d["rid"], "ce": ce}).to_parquet(path, index=False)
        msg = f"{part}: {len(d):,} pairs scored in {(time.time() - t0) / 60:.1f} min -> {path}"
        if (d["label"] >= 0).all():
            from sklearn.metrics import roc_auc_score
            lab = d["label"].to_numpy()
            msg += f" | AUC {roc_auc_score(lab, ce):.4f} (positives {lab.mean():.3f})"
        print(msg, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
