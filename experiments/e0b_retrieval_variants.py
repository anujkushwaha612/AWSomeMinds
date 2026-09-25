"""E0b: which text representation makes record-side retrieval fast AND accurate?

Train India. Index = all 883k S1 records. Queries = 3,000 sampled S2 + 3,000
sampled S3 records with a known S1 parent (seeded). For each representation and
n-gram cap it reports ms/query, the full-run estimate for all S2+S3 records, and
recall@1 / recall@5 (overall, S2, S3, Latin vs Indic-script partner names).

Representations
  char_wb n-grams n=3,4,5 and word 1-grams / 1+2-grams (hashed, 2^22 features),
  sublinear TF x IDF (IDF over S1+S2+S3), optional df cap, L2-normalized;
  address-only view for the best sparse variants and the NA ∪ A union;
  SVD(256) of char-4g TF-IDF + FAISS HNSW (approximate nearest neighbours).

Run:  python experiments/e0b_retrieval_variants.py  (results -> artifacts/experiments/E0b.json)
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from ber.blocking.retrieve import VIEWS, country_tables
from ber.config import artifact_path, ensure_parent
from ber.io import load_truth_pairs

N_FEATURES = 2 ** 22
N_THREADS = os.cpu_count()      # search threads (native, memory-light)
N_WORKERS = 8                   # featurization processes: each holds a chunk's temporaries
CAPS = [None, 0.02, 0.01, 0.005]
T0 = time.time()


def log(*a):
    """Timestamped, flushed progress line."""
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def make_vectorizer(kind: str, n: int) -> HashingVectorizer:
    """Stateless hashed n-gram counter for ``kind`` in {"char", "word"}."""
    if kind == "char":
        return HashingVectorizer(analyzer="char_wb", ngram_range=(n, n), n_features=N_FEATURES,
                                 alternate_sign=False, norm=None, dtype=np.float32)
    return HashingVectorizer(analyzer="word", ngram_range=(1, n), token_pattern=r"\S+",
                             n_features=N_FEATURES, alternate_sign=False, norm=None,
                             dtype=np.float32)


def _transform(args):
    """Worker: raw hashed counts for one chunk of texts."""
    kind, n, texts = args
    return make_vectorizer(kind, n).transform(texts).tocsr()


def _df(args):
    """Worker: document frequency of every hashed feature in one chunk."""
    m = _transform(args)
    return np.bincount(m.indices, minlength=N_FEATURES).astype(np.int32)


def featurize(kind, n, texts_by_src, query_texts, pool, chunk=50_000):
    """Return (S1 counts, query counts, df over S1+S2+S3, N docs)."""
    jobs = []
    for s in (1, 2, 3):
        t = texts_by_src[s].tolist()
        jobs += [(kind, n, t[i:i + chunk]) for i in range(0, len(t), chunk)]
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for part in pool.map(_df, jobs):
        df += part
    t1 = texts_by_src[1].tolist()
    s1 = sp.vstack(list(pool.map(_transform, [(kind, n, t1[i:i + chunk])
                                              for i in range(0, len(t1), chunk)]))).tocsr()
    q = _transform((kind, n, query_texts))
    n_docs = sum(len(texts_by_src[s]) for s in (1, 2, 3))
    return s1, q, df, n_docs


def weight(counts, df, n_docs, cap):
    """Sublinear TF x IDF, drop features with df > cap * n_docs, L2 normalize."""
    m = counts.copy()
    m.data = 1.0 + np.log(m.data)
    idf = (np.log((n_docs + 1) / (df + 1)) + 1.0).astype(np.float32)
    if cap is not None:
        idf[df > cap * n_docs] = 0.0
    m.data *= idf[m.indices]
    m.eliminate_zeros()
    return normalize(m, norm="l2", axis=1, copy=False).astype(np.float32)


def search_sparse(q, s1, k=5):
    """Top-k S1 per query row; returns (lists [n_q, k] with -1 padding, seconds)."""
    s1t = s1.T.tocsr()
    t = time.time()
    res = sp_matmul_topn(q, s1t, top_n=k, threshold=0.0, n_threads=N_THREADS, sort=True)
    secs = time.time() - t
    out = np.full((q.shape[0], k), -1, dtype=np.int64)
    for i in range(q.shape[0]):
        cols = res.indices[res.indptr[i]:res.indptr[i + 1]]
        out[i, :len(cols)] = cols
    return out, secs


def recall(lists, parent, groups):
    """Recall@1/@5 overall and per group mask."""
    hit1 = lists[:, 0] == parent
    hit5 = (lists == parent[:, None]).any(axis=1)
    out = {"r1": float(hit1.mean()), "r5": float(hit5.mean())}
    for name, mask in groups.items():
        out[f"r5_{name}"] = float(hit5[mask].mean())
    return out, hit5


def main():
    keep = ["entity_id", "name_n", "addr_n", "script"]
    tb = {s: df[keep].copy() for s, df in country_tables("train", "India").items()}
    n_queries_full = len(tb[2]) + len(tb[3])
    log("loaded", {s: len(d) for s, d in tb.items()})

    truth = load_truth_pairs()
    s1_index = pd.Index(tb[1]["entity_id"])
    samples = []
    for src in (2, 3):
        t = truth[truth["rid"].str.startswith(f"S{src}-") & truth["s1"].isin(s1_index)]
        smp = t.sample(3000, random_state=src)
        rows = pd.Index(tb[src]["entity_id"]).get_indexer(smp["rid"])
        samples.append(pd.DataFrame({"src": src, "row": rows,
                                     "parent": s1_index.get_indexer(smp["s1"]),
                                     "script": tb[src]["script"].to_numpy()[rows]}))
    smp = pd.concat(samples, ignore_index=True)
    parent = smp["parent"].to_numpy()
    groups = {"S2": (smp["src"] == 2).to_numpy(), "S3": (smp["src"] == 3).to_numpy(),
              "latin": (smp["script"] == 0).to_numpy(), "indic": (smp["script"] > 0).to_numpy()}
    log(f"queries: {len(smp)} (Indic-script partners {groups['indic'].mean():.1%})")

    def query_texts(view):
        return pd.concat([VIEWS[view](tb[src]).iloc[smp.loc[smp["src"] == src, "row"]]
                          for src in (2, 3)], ignore_index=True).tolist()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*",
                    default=["NA:word:1", "NA:word:2", "A:char:4", "A:word:2", "NA:char:4"],
                    help="VIEW:KIND:N entries, e.g. NA:char:4")
    ap.add_argument("--svd", action="store_true", help="run SVD(256)+HNSW on NA char-4g first")
    ap.add_argument("--tag", default="run2")
    args = ap.parse_args()
    variants = [(v, k, int(n)) for v, k, n in (s.split(":") for s in args.variants)]

    results, hits = [], {}
    import gc
    gc.collect()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        if args.svd:
            results += run_svd(tb, query_texts, parent, groups, n_queries_full, pool)
        for view, kind, n in variants:
            t = time.time()
            texts = {s: VIEWS[view](tb[s]) for s in (1, 2, 3)}
            s1c, qc, df, n_docs = featurize(kind, n, texts, query_texts(view), pool)
            t_feat = time.time() - t
            log(f"{view} {kind}{n}: featurized in {t_feat:.0f}s, "
                f"distinct features {int((df > 0).sum()):,}, nnz/row S1 {s1c.nnz / s1c.shape[0]:.1f}")
            for cap in CAPS:
                s1w, qw = weight(s1c, df, n_docs, cap), weight(qc, df, n_docs, cap)
                lists, secs = search_sparse(qw, s1w)
                rec, hit5 = recall(lists, parent, groups)
                ms = secs / len(smp) * 1e3
                row = {"view": view, "rep": f"{kind}{n}", "cap": cap, "ms_per_query": round(ms, 3),
                       "full_run_h": round(ms / 1e3 * n_queries_full / 3600, 2),
                       "featurize_s": round(t_feat), **{k: round(v, 4) for k, v in rec.items()}}
                results.append(row)
                hits[(view, f"{kind}{n}", cap)] = hit5
                log(f"  cap {str(cap):>5}: {ms:6.2f} ms/q  full {row['full_run_h']:6.2f} h  "
                    f"r@1 {rec['r1']:.3f}  r@5 {rec['r5']:.3f}  S2 {rec['r5_S2']:.3f}  "
                    f"S3 {rec['r5_S3']:.3f}  latin {rec['r5_latin']:.3f}  indic {rec['r5_indic']:.3f}")
            del s1c, qc

    # NA ∪ A unions (top-5 each) for matching representations and caps
    for rep in ("char4", "word2"):
        for cap in CAPS:
            a, b = hits.get(("NA", rep, cap)), hits.get(("A", rep, cap))
            if a is not None and b is not None:
                u = a | b
                results.append({"view": "NA∪A", "rep": rep, "cap": cap, "r5": round(float(u.mean()), 4),
                                "r5_indic": round(float(u[groups["indic"]].mean()), 4),
                                "r5_latin": round(float(u[groups["latin"]].mean()), 4)})
                log(f"NA∪A {rep} cap {cap}: r@5 {u.mean():.3f}  latin {u[groups['latin']].mean():.3f}"
                    f"  indic {u[groups['indic']].mean():.3f}")

    out = artifact_path("experiments", f"E0b_{args.tag}.json")
    ensure_parent(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {out}")


def run_svd(tb, query_texts, parent, groups, n_queries_full, pool):
    """SVD(256) of char-4g NA TF-IDF + FAISS HNSW; returns result rows."""
    import faiss
    from sklearn.decomposition import TruncatedSVD

    t = time.time()
    texts = {s: VIEWS["NA"](tb[s]) for s in (1, 2, 3)}
    s1c, qc, df, n_docs = featurize("char", 4, texts, query_texts("NA"), pool)
    used = np.flatnonzero(df > 0)                 # compact 2^22 hashed columns -> used ones
    s1w = weight(s1c, df, n_docs, None)[:, used]
    qw = weight(qc, df, n_docs, None)[:, used]
    del s1c, qc
    log(f"SVD input: featurized+compacted in {time.time() - t:.0f}s, {len(used):,} columns")
    results = []
    t = time.time()
    rng = np.random.default_rng(0)
    fit_rows = rng.choice(s1w.shape[0], 200_000, replace=False)
    svd = TruncatedSVD(n_components=256, algorithm="randomized", n_iter=4, random_state=0)
    svd.fit(s1w[fit_rows])
    e1 = normalize(svd.transform(s1w)).astype(np.float32)
    eq = normalize(svd.transform(qw)).astype(np.float32)
    t_svd = time.time() - t
    exact = faiss.IndexFlatIP(e1.shape[1]); exact.add(e1)
    _, lists_exact = exact.search(eq, 5)
    rec_exact, _ = recall(lists_exact, parent, groups)
    t = time.time()
    hnsw = faiss.IndexHNSWFlat(e1.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.add(e1)
    t_build = time.time() - t
    for ef in (64, 128, 256):
        hnsw.hnsw.efSearch = ef
        t = time.time(); _, lists = hnsw.search(eq, 5); secs = time.time() - t
        rec, _ = recall(lists, parent, groups)
        ms = secs / len(eq) * 1e3
        row = {"view": "NA", "rep": "char4-svd256-hnsw", "ef_search": ef, "ms_per_query": round(ms, 3),
               "full_run_h": round(ms / 1e3 * n_queries_full / 3600, 2), "svd_s": round(t_svd),
               "hnsw_build_s": round(t_build), **{k: round(v, 4) for k, v in rec.items()},
               "exact_r5": round(rec_exact["r5"], 4)}
        results.append(row)
        log(f"SVD256+HNSW ef={ef}: {ms:.3f} ms/q full {row['full_run_h']} h  r@1 {rec['r1']:.3f} "
            f"r@5 {rec['r5']:.3f} (exact SVD r@5 {rec_exact['r5']:.3f})  latin {rec['r5_latin']:.3f} "
            f"indic {rec['r5_indic']:.3f}  [svd {t_svd:.0f}s, build {t_build:.0f}s]")
    return results


if __name__ == "__main__":
    main()
