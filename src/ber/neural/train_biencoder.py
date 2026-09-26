"""A2-A3: fine-tune the multilingual bi-encoder, then run the recall gate (GPU).

Loss: in-batch softmax (MultipleNegativesRanking) over [positives; hard negatives]
for each record query, scale ``v5.neural.scale``. Columns that are in fact the row's
parent (another record of the same entity, or a hard negative that is another row's
parent) and S1s with text identical to the parent's are masked out of the softmax. Batches are drawn from one country
at a time (``same_country_batches``): cross-country negatives are trivially easy.
Mixed precision (bf16 where supported), AdamW, linear warmup/decay, grad clip 1.0.
Checkpoints every 2,000 steps to artifacts/neural/biencoder (resumable).

Recall gate (``--eval-only`` runs just this): for the fold-0 sample, record-side
top-5 over the FULL S1 universe of the country; R@1/3/5 overall, Latin, Indic,
compared with TF-IDF top-3 on the same records. Written to artifacts/neural/eval.json.

Run:  python -m ber.neural.train_biencoder
      python -m ber.neural.train_biencoder --eval-only
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ..config import artifact_path, load_config
from ..store import split_countries
from .common import Encoder, amp_dtype, country_store, model_dir, ncfg, store_texts, tile_rows, vdir


def make_batches(pairs: pd.DataFrame, batch: int, same_country: bool, seed: int) -> list[np.ndarray]:
    """Row-index batches of the triplet table (optionally one country per batch), shuffled."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(pairs["country"].to_numpy() == c) for c in np.unique(pairs["country"])] \
        if same_country else [np.arange(len(pairs))]
    batches = []
    for g in groups:
        g = rng.permutation(g)
        batches += [g[i:i + batch] for i in range(0, len(g) - batch + 1, batch)]   # full batches only
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def texts_for_batch(stores: dict, pairs: pd.DataFrame, idx: np.ndarray):
    """(record, positive S1, negative S1) texts for one batch, per country and source.

    Works for mixed-country batches too (``same_country_batches: false``).
    """
    sub = pairs.iloc[idx]
    q, p, n = [None] * len(sub), [None] * len(sub), [None] * len(sub)
    country, src = sub["country"].to_numpy(), sub["src"].to_numpy()
    for code in np.unique(country):
        st = stores[int(code)]
        mc = np.flatnonzero(country == code)
        for j, t in zip(mc, store_texts(st, 1, sub["pos_s1"].to_numpy()[mc])):
            p[j] = t
        for j, t in zip(mc, store_texts(st, 1, sub["neg_s1"].to_numpy()[mc])):
            n[j] = t
        for s in (2, 3):
            m = mc[src[mc] == s]
            if len(m):
                for j, t in zip(m, store_texts(st, s, sub["doc_row"].to_numpy()[m])):
                    q[j] = t
    return q, p, n


def false_negative_mask(pairs: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """[B, 2B] True where column j is ANOTHER row's S1 that is in fact row i's parent.

    Columns = the batch's positives, then its hard negatives. Records of one entity (it has
    3.5 matches on average) can share a batch, and a hard negative can be another row's
    parent: those columns are true matches and must not be pushed away.
    """
    sub = pairs.iloc[idx]
    ent = (sub["country"].to_numpy(np.int64) << 32) | sub["pos_s1"].to_numpy(np.int64)
    neg = (sub["country"].to_numpy(np.int64) << 32) | sub["neg_s1"].to_numpy(np.int64)
    cols = np.concatenate([ent, neg])
    mask = ent[:, None] == cols[None, :]
    np.fill_diagonal(mask[:, :len(ent)], False)       # the row's own positive stays the target
    return mask


def duplicate_text_mask(pos_texts: list[str], neg_texts: list[str]) -> np.ndarray:
    """[B, 2B] True where column j's S1 text is identical to row i's positive text.

    S1 is deduplicated by entity, not by text: two S1 entities can share name and address
    exactly. Such a column cannot be told apart from the target, so it only adds a
    contradictory gradient; it is masked like a false negative. The row's own positive
    stays the target.
    """
    h = pd.util.hash_array(np.asarray(pos_texts + neg_texts, dtype=object))
    mask = h[:len(pos_texts), None] == h[None, :]
    np.fill_diagonal(mask[:, :len(pos_texts)], False)
    return mask


def run_fingerprint(pairs: pd.DataFrame, c: dict) -> str:
    """Identifies a training run; a checkpoint is resumed only if this matches."""
    import hashlib
    keys = ["model", "max_len", "batch_size", "lr", "epochs", "scale", "warmup", "same_country_batches"]
    raw = json.dumps({k: c[k] for k in keys}, sort_keys=True) + f"|{len(pairs)}|{int(pairs['doc_row'].sum())}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def train(max_steps: int | None = None) -> None:
    """Fine-tune and save the encoder (resumes only a checkpoint of the same run)."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from transformers import get_linear_schedule_with_warmup

    c, seed = ncfg(), load_config()["seed"]
    torch.manual_seed(seed)
    pairs = pd.read_parquet(artifact_path(vdir("neural"), "train_pairs.parquet"))
    names = split_countries("train")
    stores = {code: country_store("train", country, cols=["name_n", "addr_n"])
              for code, country in enumerate(names) if (pairs["country"] == code).any()}
    ckpt = model_dir()
    fp = run_fingerprint(pairs, c)
    state_path = os.path.join(ckpt, "train_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else None
    if state and state.get("fingerprint") != fp:
        print("checkpoint belongs to a different run (config or pairs changed): starting fresh", flush=True)
        shutil.rmtree(ckpt)
        state = None
    batches = []
    for ep in range(c["epochs"]):
        batches += make_batches(pairs, c["batch_size"], c["same_country_batches"], seed + ep)
    total = len(batches) if max_steps is None else min(max_steps, len(batches))
    if state and state["step"] >= total:
        print(f"encoder already trained ({state['step']:,}/{total:,} steps): skipping training", flush=True)
        return
    enc = Encoder(ckpt if state else c["model"])
    enc.model.train()
    opt = torch.optim.AdamW(enc.model.parameters(), lr=c["lr"], weight_decay=c.get("weight_decay", 0.01))
    sched = get_linear_schedule_with_warmup(opt, int(c["warmup"] * total), total)
    use_amp = enc.dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype() == torch.float16)
    start = 0
    if state:
        start = state["step"]
        s = torch.load(os.path.join(ckpt, "optimizer.pt"), map_location=enc.dev)
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        print(f"resuming from step {start:,}/{total:,}", flush=True)
    every_log, every_ckpt = c.get("log_every", 100), c.get("checkpoint_every", 2000)
    labels = None
    t0, losses = time.time(), []

    def prepare(step):
        """CPU work for one step, run on a helper thread while the GPU trains."""
        idx = batches[step]
        q, p, n = texts_for_batch(stores, pairs, idx)
        return enc.tokenize(q), enc.tokenize(p + n), false_negative_mask(pairs, idx) | duplicate_text_mask(p, n)

    with ThreadPoolExecutor(max_workers=1) as ex:
        nxt = ex.submit(prepare, start) if start < total else None
        for step in range(start, total):
            tq, td, fn = nxt.result()
            if step + 1 < total:
                nxt = ex.submit(prepare, step + 1)
            with torch.autocast(enc.dev.type, dtype=amp_dtype(), enabled=use_amp):
                eq = enc.embed(tq)                     # float32 embeddings
                ed = enc.embed(td)                     # positives then hard negatives
            logits = (eq @ ed.T) * c["scale"]          # float32 similarity / loss
            logits = logits.masked_fill(torch.as_tensor(fn, device=enc.dev), float("-inf"))
            if labels is None or len(labels) != len(eq):
                labels = torch.arange(len(eq), device=enc.dev)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.model.parameters(), c.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
            done = step + 1
            if done % every_log == 0 or done == total:
                rate = (done - start) / (time.time() - t0)
                print(f"step {done:,}/{total:,}  loss {np.mean(losses[-every_log:]):.4f}  {rate:.2f} steps/s  "
                      f"ETA {(total - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)
            if done % every_ckpt == 0 or done == total:
                os.makedirs(ckpt, exist_ok=True)
                enc.model.save_pretrained(ckpt)
                enc.tok.save_pretrained(ckpt)
                torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()},
                           os.path.join(ckpt, "optimizer.pt.tmp"))
                os.replace(os.path.join(ckpt, "optimizer.pt.tmp"), os.path.join(ckpt, "optimizer.pt"))
                json.dump({"step": done, "total": total, "fingerprint": fp}, open(state_path + ".tmp", "w"))
                os.replace(state_path + ".tmp", state_path)   # state last: marks the checkpoint complete
    print(f"saved fine-tuned encoder -> {ckpt}", flush=True)


def recall_gate(k: int | None = None) -> dict:
    """Record-side recall of the fold-0 sample against the full S1 universe, dense vs TF-IDF."""
    import torch

    c = ncfg()
    k = k or c.get("gate_k", 5)
    ev = pd.read_parquet(artifact_path(vdir("neural"), "eval_records.parquet"))
    enc = Encoder(model_dir() if os.path.isdir(model_dir()) else c["model"])
    report = {"encoder": enc.path}
    for code, country in enumerate(split_countries("train")):
        e = ev[ev["country"] == code].reset_index(drop=True)
        if e.empty:
            continue
        st = country_store("train", country, cols=["name_n", "addr_n"])
        E1 = enc.encode(store_texts(st, 1))
        ranks = np.full(len(e), -1, dtype=np.int32)
        for s in (2, 3):
            m = np.flatnonzero(e["src"].to_numpy() == s)
            if not len(m):
                continue
            Q = enc.encode(store_texts(st, s, e["doc_row"].to_numpy()[m]))
            step = tile_rows(E1.shape[0], c["tile_gb"])
            for a in range(0, len(m), step):
                top = torch.topk(Q[a:a + step] @ E1.T, k, dim=1).indices.cpu().numpy()
                par = e["parent_s1"].to_numpy()[m[a:a + step]]
                hit = top == par[:, None]
                ranks[m[a:a + step]] = np.where(hit.any(1), hit.argmax(1), -1)
        indic = e["script"].to_numpy() > 0
        tr = e["tfidf_rank"].to_numpy()
        row = {}
        for name, mask in (("all", np.ones(len(e), bool)), ("latin", ~indic), ("indic", indic)):
            if not mask.any():
                continue
            r = ranks[mask]
            row[name] = {f"dense_R@{j}": float(((r >= 0) & (r < j)).mean()) for j in sorted({1, 3, 5, k})}
            row[name].update({f"tfidf_R@{j}": float(((tr[mask] >= 0) & (tr[mask] < j)).mean())
                              for j in (1, 3)})
            row[name]["union_R@3+3"] = float((((r >= 0) & (r < 3)) | ((tr[mask] >= 0) & (tr[mask] < 3))).mean())
            row[name]["n"] = int(mask.sum())
        report[country] = row
        print(country, json.dumps(row, indent=1), flush=True)
        del E1
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gate = all(v["all"]["dense_R@3"] >= v["all"]["tfidf_R@3"] for k_, v in report.items() if k_ != "encoder")
    report["gate_dense_R@3_ge_tfidf_R@3"] = bool(gate)
    json.dump(report, open(artifact_path(vdir("neural"), "eval.json"), "w"), indent=2)
    print(f"GATE A-retrieval (dense R@3 >= TF-IDF R@3 in every country): {'PASS' if gate else 'FAIL'}")
    return report


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    args = ap.parse_args()
    if not args.eval_only:
        train(args.max_steps)
    recall_gate()


if __name__ == "__main__":
    main()
