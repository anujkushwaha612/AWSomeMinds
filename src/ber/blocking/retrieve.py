"""Same-country char TF-IDF top-k retrieval (plan.md Step 3).

For one split and one country, and for each text view, this fits a char n-gram
TF-IDF on that country's S1+S2+S3 records (unsupervised; test is fitted on test
inputs) and stores:

  fwd_s{2,3}  [n_S1, k_forward]   top-k S2 (or S3) rows for every S1 row
  rev_s{2,3}  [n_S2|S3, k_reverse] top-k S1 rows for every S2 (or S3) row

as ``artifacts/retrieval/<split>/<country>/<view>.npz`` (row indices into the
country's id arrays in ``ids.npz``, -1 = no hit; float16 cosine scores). The
audit then evaluates any view / direction / K combination offline from these.

Run:  python -m ber.blocking.retrieve --split train --country India
      python -m ber.blocking.retrieve --split train --country India --sample-frac 0.01
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ..config import artifact_path, ensure_parent, load_config
from ..normalize import load_normalized

VIEWS = {
    "NA": lambda d: d["name_n"] + " " + d["addr_n"],
    "A": lambda d: d["addr_n"],
    "N": lambda d: d["name_n"],
    "T": lambda d: d["name_tr"] + " " + d["addr_tr"],
    "NS": lambda d: d["name_nospace"] + " " + d["addr_n"],
}


def country_tables(split: str, country: str) -> dict:
    """Normalized S1/S2/S3 rows of one country, keyed by source number."""
    out = {}
    for s in (1, 2, 3):
        df = load_normalized(split, s)
        out[s] = df[df["country"] == country].reset_index(drop=True)
    return out


def vectorize(texts: dict, cfg: dict) -> dict:
    """Fit one TF-IDF on the union of ``texts`` values; return a CSR per key."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(cfg["ngram"]),
                          min_df=cfg["min_df"], max_df=cfg["max_df"],
                          sublinear_tf=True, dtype=np.float32)
    keys = list(texts)
    mat = vec.fit_transform(pd.concat([texts[k] for k in keys], ignore_index=True))
    out, start = {}, 0
    for k in keys:
        n = len(texts[k])
        out[k] = mat[start:start + n]
        start += n
    return out


def topk(queries: sp.csr_matrix, index: sp.csr_matrix, k: int, chunk: int,
         n_threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` cosine neighbours in ``index`` for every row of ``queries``.

    Rows are L2-normalized TF-IDF, so the dot product is the cosine. Returns
    ``(idx int32 [n, k], score float16 [n, k])`` sorted by score, -1 padded.
    """
    index_t = index.T.tocsr()
    n = queries.shape[0]
    idx = np.full((n, k), -1, dtype=np.int32)
    score = np.zeros((n, k), dtype=np.float16)
    for start in range(0, n, chunk):
        res = sp_matmul_topn(queries[start:start + chunk], index_t, top_n=k,
                             threshold=0.0, n_threads=n_threads, sort=True)
        counts = np.diff(res.indptr)
        rows = np.repeat(np.arange(res.shape[0]), counts)
        cols = np.arange(res.nnz) - np.repeat(res.indptr[:-1], counts)
        idx[start + rows, cols] = res.indices
        score[start + rows, cols] = res.data
    return idx, score


def run(split: str, country: str, views: list[str], sample_frac: float = 1.0,
        seed: int = 0) -> dict:
    """Retrieve for every view; save results; return per-view timings."""
    cfg = load_config()["blocking"]
    n_threads = cfg["n_threads"] or os.cpu_count()
    tables = country_tables(split, country)
    tag = "" if sample_frac >= 1 else f"_sample{sample_frac:g}"
    out_dir = artifact_path("retrieval", split + tag, country)

    q_rows = {}
    rng = np.random.default_rng(seed)
    for s, df in tables.items():
        n = len(df)
        q_rows[s] = (np.arange(n) if sample_frac >= 1
                     else np.sort(rng.choice(n, max(1, int(n * sample_frac)), replace=False)))
    ids_path = os.path.join(out_dir, "ids.npz")
    ensure_parent(ids_path)
    np.savez(ids_path, **{f"s{s}": tables[s]["entity_id"].to_numpy(dtype=str) for s in tables},
             **{f"q{s}": q_rows[s] for s in tables})

    timings = {}
    for view in views:
        t0 = time.time()
        mats = vectorize({s: VIEWS[view](tables[s]) for s in tables}, cfg)
        t_fit = time.time() - t0
        res, t_q = {}, {}
        for s in (2, 3):
            t = time.time()
            res[f"fwd_s{s}_idx"], res[f"fwd_s{s}_score"] = topk(
                mats[1][q_rows[1]], mats[s], cfg["k_forward"], cfg["query_chunk"], n_threads)
            t_q[f"fwd_s{s}"] = time.time() - t
            t = time.time()
            res[f"rev_s{s}_idx"], res[f"rev_s{s}_score"] = topk(
                mats[s][q_rows[s]], mats[1], cfg["k_reverse"], cfg["query_chunk"], n_threads)
            t_q[f"rev_s{s}"] = time.time() - t
        np.savez(os.path.join(out_dir, f"{view}.npz"), **res)
        timings[view] = {"fit_s": round(t_fit, 1), **{k: round(v, 1) for k, v in t_q.items()},
                         "vocab": int(mats[1].shape[1]),
                         "nnz_per_row": round(mats[2].nnz / max(1, mats[2].shape[0]), 1)}
        # extrapolated full-run query time for sampled runs
        if sample_frac < 1:
            timings[view]["query_s_full_est"] = round(sum(t_q.values()) / sample_frac, 0)
        print(view, json.dumps(timings[view]), flush=True)

    meta = {"split": split, "country": country, "sample_frac": sample_frac,
            "sizes": {f"s{s}": len(tables[s]) for s in tables}, "config": cfg,
            "timings": timings}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return timings


def main() -> None:
    """CLI: retrieve for one split and one (or every) country."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--country", default=None, help="default: every country in S1")
    ap.add_argument("--views", nargs="+", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="query a random fraction of rows (timing runs); index stays full")
    args = ap.parse_args()
    views = args.views or load_config()["blocking"]["views"]
    countries = ([args.country] if args.country
                 else sorted(load_normalized(args.split, 1)["country"].unique()))
    for c in countries:
        print(f"== {args.split} / {c} / views {views}", flush=True)
        run(args.split, c, views, args.sample_frac)


if __name__ == "__main__":
    main()
