"""Record-side word 1+2-gram TF-IDF retrieval (plan.md Step 3, retriever R1).

Chosen by experiment E0b (experiments.md): word unigrams + bigrams of the
normalized name + address, sublinear TF x IDF, with "block purging" (features
whose document frequency exceeds ``max_df`` of the partition are dropped),
then exact top-k cosine search from every S2/S3 record to the S1 records of
the same country. On train India this gave recall@5 0.949 at a 1% cap in
~0.4 ms/query, versus 0.938 in ~28 ms/query for uncapped char 3-grams.

Memory-lean, two passes over S2/S3 (streaming):
  pass 1  hash every chunk only to accumulate document frequencies, then drop it
          (S1 counts are kept: they become the index)
  pass 2  re-hash each S2/S3 chunk, weight it, search it against the S1 index,
          keep only compact candidate arrays (int32/int8/float32, no text)
Resident: the S1 index + at most ``2 * workers`` hashed chunks in flight.
Hashing is Python-level (GIL-bound), so it runs in a bounded process pool;
the sparse search runs on native threads (no memory duplication).
"""

import os
import time
from collections import deque

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

N_FEATURES = 2 ** 22


def _vectorizer() -> HashingVectorizer:
    """Stateless word 1+2-gram counter (tokens are whitespace-separated)."""
    return HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
                             n_features=N_FEATURES, alternate_sign=False, norm=None,
                             dtype=np.float32)


def hash_counts(texts: list[str]) -> sp.csr_matrix:
    """Raw hashed n-gram counts for a list of texts (top-level so pools can pickle it)."""
    m = _vectorizer().transform(texts).tocsr()
    m.indices = m.indices.astype(np.int32, copy=False)
    return m


def iter_counts(store, s: int, pool, chunk: int, window: int):
    """Yield ``(start, counts)`` for consecutive chunks of source ``s``, in order.

    At most ``window`` chunks are submitted but not yet consumed, which bounds
    the memory held by in-flight texts and results.
    """
    n = store.n(s)
    starts = iter(range(0, n, chunk))
    q = deque()

    def submit(start):
        texts = store.na_text(s, start, min(start + chunk, n))
        fut = pool.submit(hash_counts, texts) if pool is not None else None
        q.append((start, fut if fut is not None else hash_counts(texts)))

    for start in starts:
        submit(start)
        if len(q) >= window:
            break
    while q:
        start, res = q.popleft()
        yield start, (res.result() if hasattr(res, "result") else res)
        nxt = next(starts, None)
        if nxt is not None:
            submit(nxt)


def tfidf(counts: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """Sublinear TF x IDF (zero-IDF columns removed), L2-normalized rows (in place)."""
    counts.data = (1.0 + np.log(counts.data)) * idf[counts.indices]
    counts.eliminate_zeros()
    return normalize(counts, norm="l2", axis=1, copy=False).astype(np.float32, copy=False)


def retrieve_country(store, k: int, max_df: float, min_score: float, pool,
                     chunk: int = 50_000, window: int = 16, log=print) -> pd.DataFrame:
    """Top-``k`` S1 rows for every S2/S3 row of one country (see module docstring).

    Returns compact candidates: ``src int8, doc_row int32, s1_row int32,
    score float32, rank int8`` (rank 0 = best S1 for that record).
    """
    t0 = time.time()
    df = np.zeros(N_FEATURES, dtype=np.int64)
    s1_parts = []
    for s in (1, 2, 3):
        for _, counts in iter_counts(store, s, pool, chunk, window):
            df += np.bincount(counts.indices, minlength=N_FEATURES)
            if s == 1:
                s1_parts.append(counts)
    n_docs = store.n(1) + store.n(2) + store.n(3)
    idf = (np.log((n_docs + 1) / (df + 1)) + 1.0).astype(np.float32)
    purged = df > max_df * n_docs
    idf[purged] = 0.0                                   # block purging
    idf[df == 0] = 0.0
    s1 = tfidf(sp.vstack(s1_parts).tocsr(), idf)
    del s1_parts
    s1t = s1.T.tocsr()
    del s1
    log(f"    pass 1 (df) over {n_docs:,} records in {time.time() - t0:.0f}s; "
        f"purged {int(purged.sum()):,} n-grams; S1 index nnz {s1t.nnz:,}")

    n_threads = os.cpu_count() or 1
    cols = {"src": [], "doc_row": [], "s1_row": [], "score": [], "rank": []}
    for src in (2, 3):
        t = time.time()
        for start, counts in iter_counts(store, src, pool, chunk, window):
            res = sp_matmul_topn(tfidf(counts, idf), s1t, top_n=k, threshold=min_score,
                                 n_threads=n_threads, sort=True)
            per_row = np.diff(res.indptr)
            cols["src"].append(np.full(res.nnz, src, dtype=np.int8))
            cols["doc_row"].append((np.repeat(np.arange(res.shape[0], dtype=np.int32), per_row)
                                    + np.int32(start)))
            cols["s1_row"].append(res.indices.astype(np.int32))
            cols["score"].append(res.data.astype(np.float32))
            cols["rank"].append((np.arange(res.nnz) - np.repeat(res.indptr[:-1], per_row))
                                .astype(np.int8))
        log(f"    pass 2 S{src}: {store.n(src):,} queries in {time.time() - t:.0f}s")
    return pd.DataFrame({c: np.concatenate(v) if v else np.array([], dtype=np.int32)
                         for c, v in cols.items()})
