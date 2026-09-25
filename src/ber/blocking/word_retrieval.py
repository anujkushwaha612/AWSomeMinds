"""Record-side word 1+2-gram TF-IDF retrieval (plan.md Step 3, retriever R1).

Chosen by experiment E0b (experiments.md): word unigrams + bigrams of the
normalized name + address, sublinear TF x IDF, with "block purging" (features
whose document frequency exceeds ``max_df`` of the partition are dropped),
then exact top-k cosine search from every S2/S3 record to the S1 records of
the same country. On train India this gave recall@5 0.949 at a 1% cap in
~0.4 ms/query, versus 0.938 in ~28 ms/query for uncapped char 3-grams.

Memory: features are hashed (2^22 columns, no vocabulary to hold), texts are
vectorized in parallel chunks, and only one country partition is in memory.
"""

import os
import time

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
    return _vectorizer().transform(texts).tocsr()


def count_matrix(texts: pd.Series, pool, chunk: int = 50_000) -> sp.csr_matrix:
    """Hashed counts for ``texts``, vectorized in parallel chunks when ``pool`` is given."""
    items = texts.tolist()
    parts = [items[i:i + chunk] for i in range(0, len(items), chunk)] or [[]]
    mats = list(pool.map(hash_counts, parts)) if pool is not None else [hash_counts(p) for p in parts]
    return sp.vstack(mats).tocsr()


def tfidf(counts: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """Sublinear TF x IDF (zero-IDF columns removed), L2-normalized rows."""
    m = counts.copy()
    m.data = (1.0 + np.log(m.data)) * idf[m.indices]
    m.eliminate_zeros()
    return normalize(m, norm="l2", axis=1, copy=False).astype(np.float32)


def retrieve_country(texts: dict, k: int, max_df: float, min_score: float, pool,
                     chunk: int = 50_000, log=print) -> pd.DataFrame:
    """Top-``k`` S1 rows for every S2/S3 row of one country.

    ``texts`` maps source number (1, 2, 3) to a Series of view texts (row order =
    the country's table order). Returns long-form candidates with columns
    ``src, doc_row, s1_row, score, rank`` (rank 0 = best S1 for that record).
    """
    t0 = time.time()
    counts = {s: count_matrix(texts[s], pool, chunk) for s in (1, 2, 3)}
    n_docs = sum(m.shape[0] for m in counts.values())
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for m in counts.values():
        df += np.bincount(m.indices, minlength=N_FEATURES)
    idf = (np.log((n_docs + 1) / (df + 1)) + 1.0).astype(np.float32)
    idf[df > max_df * n_docs] = 0.0                     # block purging
    idf[df == 0] = 0.0
    weights = {s: tfidf(counts[s], idf) for s in (1, 2, 3)}
    del counts
    s1t = weights[1].T.tocsr()
    log(f"    vectorized {n_docs:,} records in {time.time() - t0:.0f}s "
        f"(purged {int((df > max_df * n_docs).sum()):,} frequent n-grams)")

    n_threads = os.cpu_count() or 1
    parts = []
    for src in (2, 3):
        t = time.time()
        w = weights[src]
        for start in range(0, w.shape[0], chunk):
            res = sp_matmul_topn(w[start:start + chunk], s1t, top_n=k, threshold=min_score,
                                 n_threads=n_threads, sort=True)
            counts_per_row = np.diff(res.indptr)
            rows = np.repeat(np.arange(res.shape[0], dtype=np.int32), counts_per_row)
            rank = (np.arange(res.nnz) - np.repeat(res.indptr[:-1], counts_per_row)).astype(np.int8)
            parts.append(pd.DataFrame({
                "src": np.int8(src), "doc_row": rows + start,
                "s1_row": res.indices.astype(np.int32),
                "score": res.data.astype(np.float32), "rank": rank}))
        log(f"    S{src}: {w.shape[0]:,} queries in {time.time() - t:.0f}s")
    return pd.concat(parts, ignore_index=True)
