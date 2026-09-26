"""A4: streaming dense search with the fine-tuned encoder (GPU), per split and country.

For one country, all embeddings stay on the GPU (largest: US train 7.5M x 768 fp16
~ 11.5 GB); nothing is written except results:
  * record-side: top-``k_rec`` S1 for every S2/S3 record          -> drank_rec
  * S1-side:     top-``k_s1`` records per source for every S1       -> drank_s1
  * cosine for every existing TF-IDF candidate, in the row order of the
    TF-IDF feature file (so the union step can attach it without a join)
Similarities are computed in tiles sized by ``tile_gb`` (tiled matmul + topk).

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
from .common import Encoder, country_store, ncfg, store_texts, tile_rows, v5cfg, vdir


def topk_tiles(Q, D, k: int, tile_gb: float):
    """Top-``k`` columns of Q @ D.T per row of Q, tile by tile. Returns (idx, sim) on CPU."""
    import torch

    k = min(k, D.shape[0])
    step = tile_rows(D.shape[0], tile_gb)
    idx = np.empty((Q.shape[0], k), dtype=np.int32)
    sim = np.empty((Q.shape[0], k), dtype=np.float32)
    for a in range(0, Q.shape[0], step):
        s, i = torch.topk(Q[a:a + step] @ D.T, k, dim=1)
        idx[a:a + step] = i.int().cpu().numpy()
        sim[a:a + step] = s.float().cpu().numpy()
    return idx, sim


def pair_cos(A, B, rows_a: np.ndarray, rows_b: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """Cosine of aligned pairs (A[rows_a[i]], B[rows_b[i]]) (embeddings already normalized)."""
    import torch

    out = np.empty(len(rows_a), dtype=np.float32)
    for a in range(0, len(rows_a), chunk):
        ia = torch.as_tensor(rows_a[a:a + chunk], device=A.device, dtype=torch.long)
        ib = torch.as_tensor(rows_b[a:a + chunk], device=B.device, dtype=torch.long)
        out[a:a + chunk] = (A[ia].float() * B[ib].float()).sum(1).cpu().numpy()
    return out


def run_country(enc, split: str, country: str, log=print) -> None:
    """Dense candidates + TF-IDF-candidate cosines for one split/country."""
    import torch

    c, v = ncfg(), v5cfg()
    out_dir = artifact_path(vdir("dense"), split)
    out_pq = os.path.join(out_dir, f"{country}_dense.parquet")
    out_np = os.path.join(out_dir, f"{country}_tfidf_cos.npy")
    if os.path.exists(out_pq) and os.path.exists(out_np):
        log(f"[dense {split}/{country}] exists, skipping")
        return
    t0 = time.time()
    st = country_store(split, country, cols=["name_n", "addr_n"])
    E = {s: enc.encode(store_texts(st, s)) for s in (1, 2, 3)}
    log(f"[dense {split}/{country}] encoded S1 {st.n(1):,} S2 {st.n(2):,} S3 {st.n(3):,} "
        f"in {time.time() - t0:.0f}s")
    parts = []
    for src in (2, 3):
        t = time.time()
        ri, rs = topk_tiles(E[src], E[1], c["k_rec"], c["tile_gb"])
        rec = pd.DataFrame({"doc_row": np.repeat(np.arange(len(ri), dtype=np.int32), ri.shape[1]),
                            "s1_row": ri.ravel(), "cos": rs.ravel(),
                            "drank_rec": np.tile(np.arange(ri.shape[1], dtype=np.int8), len(ri))})
        si, ss = topk_tiles(E[1], E[src], c["k_s1"], c["tile_gb"])
        s1s = pd.DataFrame({"s1_row": np.repeat(np.arange(len(si), dtype=np.int32), si.shape[1]),
                            "doc_row": si.ravel(), "cos_s1": ss.ravel(),
                            "drank_s1": np.tile(np.arange(si.shape[1], dtype=np.int8), len(si))})
        m = rec.merge(s1s, on=["doc_row", "s1_row"], how="outer")
        m["cos"] = m["cos"].fillna(m["cos_s1"]).astype(np.float32)
        m["drank_rec"] = m["drank_rec"].fillna(-1).astype(np.int8)
        m["drank_s1"] = m["drank_s1"].fillna(-1).astype(np.int8)
        m["src"] = np.int8(src)
        parts.append(m[["src", "doc_row", "s1_row", "cos", "drank_rec", "drank_s1"]])
        log(f"    S{src}: {len(rec):,} record-side + {len(s1s):,} S1-side -> {len(m):,} pairs "
            f"in {time.time() - t:.0f}s")
    dense = pd.concat(parts, ignore_index=True).astype({"doc_row": np.int32, "s1_row": np.int32})

    # cosine for every TF-IDF candidate, in file order
    tf = pq.read_table(artifact_path(vdir("tfidf"), split, f"{country}.parquet"),
                       columns=["src", "doc_row", "s1_row"]).to_pandas()
    for s, n in ((1, "s1_row"), (2, "doc_row"), (3, "doc_row")):     # row universes must agree
        rows = tf[n] if s == 1 else tf.loc[tf["src"] == s, n]
        if len(rows) and rows.max() >= st.n(s):
            raise ValueError(f"{split}/{country}: TF-IDF rows index beyond the store (S{s}); "
                             f"the TF-IDF run and v5.limit disagree")
    cos = np.empty(len(tf), dtype=np.float32)
    for src in (2, 3):
        mm = (tf["src"] == src).to_numpy()
        cos[mm] = pair_cos(E[src], E[1], tf.loc[mm, "doc_row"].to_numpy(), tf.loc[mm, "s1_row"].to_numpy())
    ensure_parent(out_pq)
    dense.to_parquet(out_pq + ".tmp", index=False)
    np.save(out_np + ".tmp.npy", cos)
    os.replace(out_pq + ".tmp", out_pq)
    os.replace(out_np + ".tmp.npy", out_np)
    del E
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"[dense {split}/{country}] done in {time.time() - t0:.0f}s: {len(dense):,} dense pairs "
        f"({len(dense) / max(st.n(1), 1):.2f}/S1), {len(tf):,} TF-IDF cosines")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--country", default=None)
    ap.add_argument("--allow-base-model", action="store_true",
                    help="search with the pretrained model if no fine-tuned one exists (not for real runs)")
    args = ap.parse_args()
    import json
    from .common import model_dir
    state_path = os.path.join(model_dir(), "train_state.json")
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        if st["step"] < st["total"]:
            raise SystemExit(f"encoder training is incomplete ({st['step']}/{st['total']} steps): "
                             f"finish train_biencoder first")
        enc = Encoder(model_dir())
    elif args.allow_base_model:
        enc = Encoder(ncfg()["model"])
    else:
        raise SystemExit("no fine-tuned encoder in " + model_dir() + ": run train_biencoder first")
    print(f"encoder: {enc.path}", flush=True)
    for country in ([args.country] if args.country else split_countries(args.split)):
        run_country(enc, args.split, country, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
