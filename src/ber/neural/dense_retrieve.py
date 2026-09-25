"""A4: streaming dense search with the fine-tuned encoder (GPU), per split and country.

For one country the encoder embeds S1, S2 and S3 once; then, per source S2 / S3, with only
the S1 matrix and that source's matrix on the GPU:
  * record-side: top-``k_rec`` S1 for every S2/S3 record          -> drank_rec
  * S1-side:     top-``k_s1`` records per source for every S1       -> drank_s1
  * cosine for every existing TF-IDF candidate, in the row order of the
    TF-IDF feature file (so the union step can attach it without a join)
Similarities are computed in tiles sized by ``tile_gb`` (tiled matmul + topk); brute force,
no ANN, so recall is exact.

Where embeddings live between encoding and search is ``v5.neural.emb_store``:
  gpu   all three matrices resident on the GPU (US train: 7.5M x 768 fp16 ~ 11.5 GB) -- 48 GB box
  disk  float16 memmaps under ``emb_dir`` (default artifacts/dense/emb/<split>/); peak VRAM is
        |S1| + |S_src| + tile (~9 GB on US train), peak RAM one tile -- 16-24 GB cards / Colab.
        Memmaps are reused if present, so an interrupted run resumes after encoding.

Outputs: artifacts/dense/<split>/<country>_dense.parquet
             (src, doc_row, s1_row, cos, drank_rec, drank_s1; -1 = not in that list)
         artifacts/dense/<split>/<country>_tfidf_cos.npy (float32, one per TF-IDF row)

Run:  python -m ber.neural.dense_retrieve --split train
      python -m ber.neural.dense_retrieve --split test
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import artifact_path, ensure_parent
from ..store import split_countries
from .common import (Encoder, country_store, emb_dir, emb_store_mode, ncfg, open_memmap,
                     store_texts, tile_rows, vdir)


def topk_tiles(Q, D, k: int, tile_gb: float, enc=None):
    """Top-``k`` columns of Q @ D.T per row of Q, tile by tile. Returns (idx, sim) on CPU.

    ``D`` must be on the device. ``Q`` may be a device tensor or a host array / memmap (then
    each tile is moved to the device on its own, via ``enc.to_device``).
    """
    import torch

    k = min(k, D.shape[0])
    step = tile_rows(D.shape[0], tile_gb)
    idx = np.empty((Q.shape[0], k), dtype=np.int32)
    sim = np.empty((Q.shape[0], k), dtype=np.float32)
    for a in range(0, Q.shape[0], step):
        q = Q[a:a + step]
        if not isinstance(q, torch.Tensor):
            q = enc.to_device(q)
        s, i = torch.topk(q @ D.T, k, dim=1)
        idx[a:a + step] = i.int().cpu().numpy()
        sim[a:a + step] = s.float().cpu().numpy()
    return idx, sim


def pair_cos(A, B, rows_a: np.ndarray, rows_b: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """Cosine of aligned pairs (A[rows_a[i]], B[rows_b[i]]); A, B device tensors, already normalized."""
    import torch

    out = np.empty(len(rows_a), dtype=np.float32)
    for a in range(0, len(rows_a), chunk):
        ia = torch.as_tensor(rows_a[a:a + chunk], device=A.device, dtype=torch.long)
        ib = torch.as_tensor(rows_b[a:a + chunk], device=B.device, dtype=torch.long)
        out[a:a + chunk] = (A[ia].float() * B[ib].float()).sum(1).cpu().numpy()
    return out


def embed_source(enc, st, split: str, country: str, s: int, log=print):
    """Embeddings of source ``s``: a device tensor (``gpu`` mode) or a memmap (``disk`` mode).

    In disk mode an existing, complete memmap is reused (resume after an interruption); a
    partially written one is recognised by its ``.done`` marker being absent and re-encoded.
    """
    n = st.n(s)
    if emb_store_mode() != "disk":
        return enc.encode(store_texts(st, s), log=log)
    path = os.path.join(emb_dir(split), f"{country}_S{s}.f16")
    done = path + ".done"
    if os.path.exists(path) and os.path.exists(done):
        log(f"    S{s}: reusing {path}")
        return open_memmap(path, n, enc.dim, mode="r")
    ensure_parent(path)
    mm = open_memmap(path, n, enc.dim, mode="w+")
    enc.encode(store_texts(st, s), out=mm, log=log)
    del mm
    open(done, "w").close()
    return open_memmap(path, n, enc.dim, mode="r")


def search_source(enc, E1, Es, src: int, k_rec: int, k_s1: int, tile_gb: float) -> pd.DataFrame:
    """Record-side and S1-side dense candidates of one source (E1, Es on the device)."""
    ri, rs = topk_tiles(Es, E1, k_rec, tile_gb, enc)
    rec = pd.DataFrame({"doc_row": np.repeat(np.arange(len(ri), dtype=np.int32), ri.shape[1]),
                        "s1_row": ri.ravel(), "cos": rs.ravel(),
                        "drank_rec": np.tile(np.arange(ri.shape[1], dtype=np.int8), len(ri))})
    si, ss = topk_tiles(E1, Es, k_s1, tile_gb, enc)
    s1s = pd.DataFrame({"s1_row": np.repeat(np.arange(len(si), dtype=np.int32), si.shape[1]),
                        "doc_row": si.ravel(), "cos_s1": ss.ravel(),
                        "drank_s1": np.tile(np.arange(si.shape[1], dtype=np.int8), len(si))})
    m = rec.merge(s1s, on=["doc_row", "s1_row"], how="outer")
    m["cos"] = m["cos"].fillna(m["cos_s1"]).astype(np.float32)
    m["drank_rec"] = m["drank_rec"].fillna(-1).astype(np.int8)
    m["drank_s1"] = m["drank_s1"].fillna(-1).astype(np.int8)
    m["src"] = np.int8(src)
    m.attrs["n_rec"], m.attrs["n_s1"] = len(rec), len(s1s)
    return m[["src", "doc_row", "s1_row", "cos", "drank_rec", "drank_s1"]]


def run_country(enc, split: str, country: str, log=print) -> None:
    """Dense candidates + TF-IDF-candidate cosines for one split/country."""
    import torch

    c = ncfg()
    out_dir = artifact_path(vdir("dense"), split)
    out_pq = os.path.join(out_dir, f"{country}_dense.parquet")
    out_np = os.path.join(out_dir, f"{country}_tfidf_cos.npy")
    if os.path.exists(out_pq) and os.path.exists(out_np):
        log(f"[dense {split}/{country}] exists, skipping")
        return
    t0 = time.time()
    st = country_store(split, country, cols=["name_n", "addr_n"])
    E = {s: embed_source(enc, st, split, country, s, log) for s in (1, 2, 3)}
    log(f"[dense {split}/{country}] encoded S1 {st.n(1):,} S2 {st.n(2):,} S3 {st.n(3):,} "
        f"in {time.time() - t0:.0f}s ({emb_store_mode()} mode)")

    tf = pq.read_table(artifact_path(vdir("tfidf"), split, f"{country}.parquet"),
                       columns=["src", "doc_row", "s1_row"]).to_pandas()
    cos = np.empty(len(tf), dtype=np.float32)
    parts = []
    E1 = enc.to_device(E[1])                                 # S1 stays resident for both sources
    for src in (2, 3):
        t = time.time()
        Es = enc.to_device(E[src])
        m = search_source(enc, E1, Es, src, c["k_rec"], c["k_s1"], c["tile_gb"])
        mm = (tf["src"] == src).to_numpy()
        cos[mm] = pair_cos(Es, E1, tf.loc[mm, "doc_row"].to_numpy(), tf.loc[mm, "s1_row"].to_numpy())
        parts.append(m)
        log(f"    S{src}: {m.attrs['n_rec']:,} record-side + {m.attrs['n_s1']:,} S1-side -> {len(m):,} pairs "
            f"in {time.time() - t:.0f}s")
        del Es
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dense = pd.concat(parts, ignore_index=True).astype({"doc_row": np.int32, "s1_row": np.int32})

    ensure_parent(out_pq)
    dense.to_parquet(out_pq + ".tmp", index=False)
    np.save(out_np + ".tmp.npy", cos)
    os.replace(out_pq + ".tmp", out_pq)
    os.replace(out_np + ".tmp.npy", out_np)
    del E, E1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"[dense {split}/{country}] done in {time.time() - t0:.0f}s: {len(dense):,} dense pairs "
        f"({len(dense) / max(st.n(1), 1):.2f}/S1), {len(tf):,} TF-IDF cosines")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--country", default=None)
    args = ap.parse_args()
    enc = Encoder()
    print(f"encoder: {enc.path}", flush=True)
    for country in ([args.country] if args.country else split_countries(args.split)):
        run_country(enc, args.split, country, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
