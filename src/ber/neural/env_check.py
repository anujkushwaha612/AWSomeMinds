"""A0: environment check + throughput probe on the GPU box (run this first, ~5-10 min).

Prints torch / CUDA / GPU / VRAM / bf16 support, then measures on real data:
  * encoding throughput (texts/s) at the configured max_len and encode batch,
  * training throughput (steps/s) at the configured batch size,
and turns them into time estimates for every GPU stage, so the budget is measured,
not guessed.

Run:  python -m ber.neural.env_check
"""

import os
import time

import numpy as np
import pyarrow.parquet as pq

from ..config import artifact_path
from ..store import norm_path, split_countries
from .common import Encoder, amp_dtype, country_store, ncfg, store_texts, vdir

N_ENC = 20_000
N_STEPS = 30


def main() -> None:
    import torch

    print(f"torch {torch.__version__}  cuda available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU {p.name}  VRAM {p.total_memory / 2**30:.1f} GB  bf16 {torch.cuda.is_bf16_supported()}"
              f"  CUDA {torch.version.cuda}")
    else:
        print("WARNING: no CUDA GPU visible -- the GPU stages would run on CPU (far too slow).")
    c = ncfg()
    country = split_countries("train")[0]
    store = country_store("train", country, cols=["name_n", "addr_n"])
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(store.n(2), N_ENC, replace=False))
    texts = store_texts(store, 2, rows)

    enc = Encoder(c["model"])
    enc.encode(texts[:512])                                   # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    enc.encode(texts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    enc_rate = N_ENC / (time.time() - t)

    # training throughput: batch of (record, positive, negative) texts, in-batch softmax
    enc.model.train()
    opt = torch.optim.AdamW(enc.model.parameters(), lr=c["lr"])
    use_amp = enc.dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype() == torch.float16)
    B = c["batch_size"]
    t = None
    for step in range(N_STEPS + 3):
        idx = rng.choice(N_ENC, 3 * B, replace=True)
        batch = [texts[i] for i in idx]
        with torch.autocast(enc.dev.type, dtype=amp_dtype(), enabled=use_amp):
            q = enc.forward(batch[:B])
            d = enc.forward(batch[B:])
            logits = q @ d.T * c["scale"]
            loss = torch.nn.functional.cross_entropy(logits.float(), torch.arange(B, device=enc.dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step == 2:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_rate = N_STEPS / (time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")

    # sizes from the data itself: texts to encode = every S1/S2/S3 row of train and test
    n_texts = sum(pq.ParquetFile(norm_path(split, s)).metadata.num_rows
                  for split in ("train", "test") for s in (1, 2, 3))
    pairs_path = artifact_path(vdir("neural"), "train_pairs.parquet")
    n_pairs = c["n_pairs"] or (pq.ParquetFile(pairs_path).metadata.num_rows
                               if os.path.exists(pairs_path) else 3_060_000)   # E4 measurement until A1 ran
    train_h = n_pairs / B / step_rate / 3600 * c["epochs"]
    enc_h = n_texts / enc_rate / 3600
    print(f"\nencoding : {enc_rate:,.0f} texts/s  (max_len {c['max_len']}, batch {c['encode_batch']})")
    print(f"training : {step_rate:.2f} steps/s at batch {B}  (peak VRAM {peak:.1f} GB, "
          f"grad checkpointing {bool(c.get('grad_checkpointing', False))}, emb_store {c.get('emb_store', 'gpu')})")
    print(f"estimates: train encoder on {n_pairs:,} pairs ~ {train_h:.1f} h | "
          f"encode all {n_texts:,} texts (search) ~ {enc_h:.1f} h | recall gate ~ "
          f"{(2_300_000 + 40_000) / enc_rate / 3600:.2f} h")
    budget_h = float(c.get("train_budget_h", 3.0))
    if train_h > budget_h:
        suggest = int(budget_h * 3600 * step_rate * B / c["epochs"])
        print(f"SUGGESTION: training exceeds {budget_h:.0f} h -> set v5.neural.n_pairs to ~{suggest:,} "
              f"(or use intfloat/multilingual-e5-small)")


if __name__ == "__main__":
    main()
