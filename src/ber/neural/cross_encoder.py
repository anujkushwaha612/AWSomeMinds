"""D: cross-encoder re-scoring of the stage-1 gray zone -> stage-2 feature ``ce`` (GPU).

Why (experiments.md E2 / E4): the largest fold-0 losses of the baseline are true pairs that
were found but rejected with low p (0.021 F0.5) and false positives (0.015), 72% of which are
orphan records that merely *look* like an S1. A bi-encoder compares two independently built
vectors; a cross-encoder attends across both records token by token (a different house
number, one swapped word, a chain's other branch), which is what separates "similar" from
"same business". It only runs where stage 1 is unsure: p1 inside ``v5.cross_encoder.band``
(lower end never below ``v5.prune_tau``: pruned pairs never reach stage 2). Outside the band
``ce`` is NaN and stage 2 relies on p1 there.

Leakage rule (as for the bi-encoder, strategy_v5.md §2): a training pair needs its S1 in
``v5.encoder_folds`` AND its record owned by those folds (owner = the fold of the record's
true parent; orphan records get a deterministic hash fold). No label of a GBDT-fold entity
is trained on. Residual, documented: an orphan record owned by folds 3-4 can also be a
(negative) candidate of a fold 0-2 S1.

Model: ``AutoModelForSequenceClassification`` with one logit, initialised from the
fine-tuned bi-encoder (XLM-R architecture; intfloat/multilingual-e5-base is MIT-licensed),
input ``<S1 name | address> </s></s> <record name | address>``, BCE loss, bf16/fp16
autocast, AdamW with linear warmup/decay. Checkpoints are resumable (same fingerprint).

Stages:
  pairs            (CPU) training pairs from the union candidates and stage-1 p1
  train            (GPU) fine-tune
  score --split s  (GPU) logit for every union pair in the band -> artifacts/v5/ce_<split>.npy
                   (NaN elsewhere; aligned with the union row order; per-country resumable)

Run:  python -m ber.neural.cross_encoder pairs
      python -m ber.neural.cross_encoder train
      python -m ber.neural.cross_encoder score --split train
      python -m ber.neural.cross_encoder score --split test
"""

import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from ..config import artifact_path, ensure_parent, load_config
from ..store import split_countries
from .common import amp_dtype, country_store, device, length_order, model_dir, ncfg, v5cfg, vdir


# ------------------------------------------------------------------ config / paths
def ccfg() -> dict:
    """The ``v5.cross_encoder`` section (empty if absent: then the stage is disabled)."""
    return v5cfg().get("cross_encoder") or {}


def enabled() -> bool:
    return bool(ccfg().get("enabled", False))


def band() -> tuple[float, float]:
    """p1 interval that is scored; the lower end is at least ``v5.prune_tau``."""
    lo, hi = ccfg().get("band") or [None, 1.0]
    tau = float(v5cfg()["prune_tau"])
    return max(tau, float(lo) if lo is not None else tau), float(hi)


def ce_dir() -> str:
    return artifact_path(vdir("neural"), "cross_encoder")


def pairs_path() -> str:
    return artifact_path(vdir("neural"), "ce_train_pairs.parquet")


def init_path() -> str:
    """Initial weights: the fine-tuned bi-encoder (if complete), else a hub model name."""
    init = ccfg().get("init", "biencoder")
    if init != "biencoder":
        return init
    state = os.path.join(model_dir(), "train_state.json")
    if os.path.exists(state):
        st = json.load(open(state))
        if st["step"] >= st["total"]:
            return model_dir()
    print(f"WARNING: no complete fine-tuned bi-encoder in {model_dir()}; starting from "
          f"{ncfg()['model']}", flush=True)
    return ncfg()["model"]


def plain_texts(names: list[str], addrs: list[str]) -> list[str]:
    """One side of a cross-encoder input (no e5 prefix: the pair is encoded jointly)."""
    return [f"{n} | {a}" for n, a in zip(names, addrs)]


def pair_texts(store, s1_rows: np.ndarray, src: np.ndarray, doc_rows: np.ndarray):
    """(S1 texts, record texts) for aligned pair arrays of one country (mixed sources)."""
    a = plain_texts(store.strings(1, "name_n", s1_rows), store.strings(1, "addr_n", s1_rows))
    b = [None] * len(doc_rows)
    for s in (2, 3):
        m = np.flatnonzero(src == s)
        if len(m):
            for j, t in zip(m, plain_texts(store.strings(s, "name_n", doc_rows[m]),
                                           store.strings(s, "addr_n", doc_rows[m]))):
                b[j] = t
    return a, b


def require_gpu(allow_cpu: bool) -> None:
    import torch
    if not torch.cuda.is_available() and not allow_cpu:
        raise SystemExit("no CUDA GPU visible: the cross-encoder needs one (--allow-cpu for smoke tests)")


# ------------------------------------------------------------------ model wrapper
class CrossEncoder:
    """Tokenizer + sequence-classification transformer with a single logit."""

    def __init__(self, path: str, max_len: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.path = path
        self.max_len = max_len or ccfg().get("max_len", 128)
        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(self.dev)

    def tokenize(self, a: list[str], b: list[str]):
        """Joint encoding of the two sides; truncation trims the longer side first."""
        t = self.tok(a, b, padding=True, truncation="longest_first", max_length=self.max_len,
                     return_tensors="pt")
        if self.dev.type == "cuda":
            t = {k: v.pin_memory() for k, v in t.items()}
        return t

    def logits(self, t):
        """Float32 logits of a tokenized batch."""
        t = {k: v.to(self.dev, non_blocking=True) for k, v in t.items()}
        return self.model(**t).logits.squeeze(-1).float()

    def predict(self, a: list[str], b: list[str], batch: int) -> np.ndarray:
        """Logits in input order; length-sorted batches, next batch tokenized on a thread."""
        from concurrent.futures import ThreadPoolExecutor

        torch = self.torch
        self.model.eval()
        order = length_order([x + y for x, y in zip(a, b)])
        out = np.empty(len(a), dtype=np.float32)
        starts = list(range(0, len(a), batch))
        use_amp = self.dev.type == "cuda"

        def tok(i):
            sel = order[i:i + batch]
            return self.tokenize([a[j] for j in sel], [b[j] for j in sel])

        with ThreadPoolExecutor(max_workers=1) as ex, torch.inference_mode(), \
                torch.autocast(self.dev.type, dtype=amp_dtype(), enabled=use_amp):
            nxt = ex.submit(tok, starts[0]) if starts else None
            for n, i in enumerate(starts):
                t = nxt.result()
                if n + 1 < len(starts):
                    nxt = ex.submit(tok, starts[n + 1])
                out[order[i:i + batch]] = self.logits(t).cpu().numpy()
        return out


# ------------------------------------------------------------------ pairs (CPU)
def record_owner_fold(keys: pd.DataFrame, ents: pd.DataFrame, split: str = "train") -> np.ndarray:
    """Fold that owns each pair's record: its true parent's fold, or a hash fold for orphans."""
    from ..baseline import truth_parents
    from ..io import load_truth_pairs
    from ..v5 import entity_uniform

    cfg = load_config()
    n_folds = cfg["validation"]["n_folds"]
    truth = load_truth_pairs()
    own = np.empty(len(keys), dtype=np.int8)
    country = keys["country"].to_numpy()
    for code, name in enumerate(split_countries(split)):
        pm = np.flatnonzero(country == code)
        if not len(pm):
            continue
        parents = truth_parents(country_store(split, name, cols=["entity_id"]), truth)
        e_fold = ents.loc[(ents["country"] == code).to_numpy(), "fold"].to_numpy()
        src = keys["src"].to_numpy()[pm].astype(np.int64)
        doc = keys["doc_row"].to_numpy()[pm].astype(np.int64)
        par = np.full(len(pm), -1, dtype=np.int64)
        for s in (2, 3):
            m = src == s
            par[m] = parents[s][doc[m]]
        rkey = (np.int64(code) << 40) | (src << 32) | doc
        hashed = np.minimum((entity_uniform(rkey, cfg["seed"] + 11) * n_folds).astype(np.int8), n_folds - 1)
        own[pm] = np.where(par >= 0, e_fold[np.maximum(par, 0)], hashed)
    return own


def build_pairs() -> None:
    """Leakage-safe training pairs (natural label ratio inside the band)."""
    from .. import v5 as V

    if not enabled():
        print("cross-encoder disabled (v5.cross_encoder.enabled): nothing to do", flush=True)
        return
    c, v = ccfg(), v5cfg()
    keys, ents = V.read_keys("train")
    p1 = np.load(V.run_path("p1_train.npy"))
    if len(p1) != len(keys):
        raise ValueError(f"p1_train has {len(p1)} rows for {len(keys)} union pairs: rerun stage1")
    lo, hi = band()
    inb = (p1 >= lo) & (p1 <= hi)
    enc = np.array(v["encoder_folds"])
    s1_fold = V.pair_folds(keys, ents)
    own = record_owner_fold(keys, ents)
    sel = inb & np.isin(s1_fold, enc) & np.isin(own, enc)
    idx = np.flatnonzero(sel)
    cap = int(c.get("max_train_pairs") or 0)
    if cap and len(idx) > cap:
        idx = np.sort(np.random.default_rng(load_config()["seed"]).choice(idx, cap, replace=False))
    df = keys.iloc[idx][["country", "src", "doc_row", "s1_row", "label"]].reset_index(drop=True)
    kept = p1 >= v["prune_tau"]
    y = keys["label"].to_numpy()
    print(f"band p1 in [{lo}, {hi}]: {int(inb.sum()):,} of {int(kept.sum()):,} pruned pairs "
          f"({inb.sum() / max(kept.sum(), 1):.1%}); holds {int(y[inb].sum()):,} of "
          f"{int(y[kept].sum()):,} pruned true pairs", flush=True)
    print(f"training pairs: {len(df):,} (encoder folds {v['encoder_folds']}, cap {cap:,}), "
          f"positive share {df['label'].mean():.3f}", flush=True)
    ensure_parent(pairs_path())
    df.to_parquet(pairs_path(), index=False)


# ------------------------------------------------------------------ train (GPU)
def fingerprint(pairs: pd.DataFrame, c: dict, init: str) -> str:
    keys = ["max_len", "batch_size", "lr", "epochs", "warmup", "weight_decay"]
    raw = json.dumps({k: c.get(k) for k in keys}, sort_keys=True) + \
        f"|{init}|{len(pairs)}|{int(pairs['doc_row'].sum())}|{int(pairs['s1_row'].sum())}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def train(max_steps: int | None = None) -> None:
    """Fine-tune the cross-encoder (resumes only a checkpoint of the same run)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from transformers import get_linear_schedule_with_warmup

    if not enabled():
        print("cross-encoder disabled (v5.cross_encoder.enabled): nothing to do", flush=True)
        return
    c, seed = ccfg(), load_config()["seed"]
    torch.manual_seed(seed)
    pairs = pd.read_parquet(pairs_path())
    names = split_countries("train")
    stores = {code: country_store("train", names[code], cols=["name_n", "addr_n"])
              for code in np.unique(pairs["country"]).tolist()}
    ckpt, init = ce_dir(), init_path()
    fp = fingerprint(pairs, c, init)
    state_path = os.path.join(ckpt, "train_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else None
    if state and state.get("fingerprint") != fp:
        print("cross-encoder checkpoint belongs to a different run: starting fresh", flush=True)
        shutil.rmtree(ckpt)
        state = None
    rng = np.random.default_rng(seed)
    B = c.get("batch_size", 128)
    batches = []
    for _ in range(c.get("epochs", 1)):
        perm = rng.permutation(len(pairs))
        batches += [perm[i:i + B] for i in range(0, len(perm) - B + 1, B)]
    total = len(batches) if max_steps is None else min(max_steps, len(batches))
    if state and state["step"] >= total:
        print(f"cross-encoder already trained ({state['step']:,}/{total:,} steps)", flush=True)
        return
    model = CrossEncoder(ckpt if state else init)
    model.model.train()
    opt = torch.optim.AdamW(model.model.parameters(), lr=c.get("lr", 2e-5),
                            weight_decay=c.get("weight_decay", 0.01))
    sched = get_linear_schedule_with_warmup(opt, int(c.get("warmup", 0.05) * total), total)
    use_amp = model.dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype() == torch.float16)
    start = 0
    if state:
        start = state["step"]
        s = torch.load(os.path.join(ckpt, "optimizer.pt"), map_location=model.dev)
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        print(f"resuming from step {start:,}/{total:,}", flush=True)
    country = pairs["country"].to_numpy()
    cols = {k: pairs[k].to_numpy() for k in ("src", "doc_row", "s1_row", "label")}

    def prepare(step):
        """Texts + tokens + labels of one batch (runs on a helper thread)."""
        idx = batches[step]
        a, b = [None] * len(idx), [None] * len(idx)
        for code in np.unique(country[idx]):
            m = np.flatnonzero(country[idx] == code)
            r = idx[m]
            ta, tb = pair_texts(stores[int(code)], cols["s1_row"][r], cols["src"][r], cols["doc_row"][r])
            for j, x, yy in zip(m, ta, tb):
                a[j], b[j] = x, yy
        return model.tokenize(a, b), torch.as_tensor(cols["label"][idx], dtype=torch.float32)

    every_log, every_ckpt = c.get("log_every", 100), c.get("checkpoint_every", 2000)
    t0, losses, accs = time.time(), [], []
    with ThreadPoolExecutor(max_workers=1) as ex:
        nxt = ex.submit(prepare, start)
        for step in range(start, total):
            tok, y = nxt.result()
            if step + 1 < total:
                nxt = ex.submit(prepare, step + 1)
            y = y.to(model.dev, non_blocking=True)
            with torch.autocast(model.dev.type, dtype=amp_dtype(), enabled=use_amp):
                logit = model.logits(tok)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), c.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
            accs.append(float(((logit > 0).float() == y).float().mean()))
            done = step + 1
            if done % every_log == 0 or done == total:
                rate = (done - start) / (time.time() - t0)
                print(f"step {done:,}/{total:,}  loss {np.mean(losses[-every_log:]):.4f}  "
                      f"acc {np.mean(accs[-every_log:]):.4f}  {rate:.2f} steps/s  "
                      f"ETA {(total - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)
            if done % every_ckpt == 0 or done == total:
                os.makedirs(ckpt, exist_ok=True)
                model.model.save_pretrained(ckpt)
                model.tok.save_pretrained(ckpt)
                torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()},
                           os.path.join(ckpt, "optimizer.pt.tmp"))
                os.replace(os.path.join(ckpt, "optimizer.pt.tmp"), os.path.join(ckpt, "optimizer.pt"))
                json.dump({"step": done, "total": total, "fingerprint": fp}, open(state_path + ".tmp", "w"))
                os.replace(state_path + ".tmp", state_path)      # state last: checkpoint complete
    print(f"saved cross-encoder -> {ckpt}", flush=True)


# ------------------------------------------------------------------ score (GPU)
def score(split: str) -> None:
    """Logits for every union pair of ``split`` with p1 in the band (NaN elsewhere)."""
    from .. import v5 as V

    if not enabled():
        print("cross-encoder disabled (v5.cross_encoder.enabled): nothing to do", flush=True)
        return
    state_path = os.path.join(ce_dir(), "train_state.json")
    if not os.path.exists(state_path):
        raise SystemExit(f"no trained cross-encoder in {ce_dir()}: run the train stage first")
    st = json.load(open(state_path))
    if st["step"] < st["total"]:
        raise SystemExit(f"cross-encoder training is incomplete ({st['step']}/{st['total']} steps)")
    c = ccfg()
    keys, _ = V.read_keys(split)
    p1 = np.load(V.run_path(f"p1_{split}.npy"))
    if len(p1) != len(keys):
        raise ValueError(f"p1_{split} has {len(p1)} rows for {len(keys)} union pairs: rerun stage1")
    lo, hi = band()
    inb = (p1 >= lo) & (p1 <= hi)
    print(f"[ce {split}] scoring {int(inb.sum()):,} of {len(keys):,} union pairs "
          f"(p1 in [{lo}, {hi}])", flush=True)
    model = CrossEncoder(ce_dir())
    batch, chunk = c.get("score_batch", 512), c.get("score_chunk", 500_000)
    out = np.full(len(keys), np.nan, dtype=np.float32)
    country = keys["country"].to_numpy()
    s1, src, doc = (keys[k].to_numpy() for k in ("s1_row", "src", "doc_row"))
    t0, done = time.time(), 0
    for code, name in enumerate(split_countries(split)):
        rows = np.flatnonzero(inb & (country == code))
        part = V.run_path("ce", split, f"{name}.npy")
        if os.path.exists(part):
            vals = np.load(part)
            if len(vals) != len(rows):
                raise ValueError(f"{part} has {len(vals)} scores for {len(rows)} band rows: delete it")
            out[rows] = vals
            print(f"[ce {split}/{name}] exists, loaded", flush=True)
            continue
        store = country_store(split, name, cols=["name_n", "addr_n"])
        vals = np.empty(len(rows), dtype=np.float32)
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]
            ta, tb = pair_texts(store, s1[r], src[r], doc[r])
            vals[a:a + len(r)] = model.predict(ta, tb, batch)
            done += len(r)
            rate = done / (time.time() - t0)
            print(f"[ce {split}/{name}] {a + len(r):,}/{len(rows):,}  {rate:,.0f} pairs/s  "
                  f"ETA {(inb.sum() - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)
        ensure_parent(part)
        np.save(part + ".tmp.npy", vals)
        os.replace(part + ".tmp.npy", part)
        out[rows] = vals
        del store
    np.save(V.run_path(f"ce_{split}.npy"), out)
    fin = np.isfinite(out)
    msg = f"[ce {split}] done: {int(fin.sum()):,} scored"
    if split == "train" and fin.any():
        y = keys["label"].to_numpy()[fin].astype(bool)
        msg += f"; mean logit positives {out[fin][y].mean():.2f} / negatives {out[fin][~y].mean():.2f}"
    print(msg, flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs")
    t = sub.add_parser("train")
    t.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    t.add_argument("--allow-cpu", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--split", required=True, choices=["train", "test"])
    s.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    if args.cmd == "pairs":
        build_pairs()
    elif not enabled():
        print("cross-encoder disabled (v5.cross_encoder.enabled): nothing to do", flush=True)
    elif args.cmd == "train":
        require_gpu(args.allow_cpu)
        train(args.max_steps)
    else:
        require_gpu(args.allow_cpu)
        score(args.split)


if __name__ == "__main__":
    main()
