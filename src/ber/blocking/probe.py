"""Bounded-work record -> S1 retrieval (stage 2 of the blocking cascade).

Two design decisions distinguish this from a plain ``TfidfVectorizer`` +
top-k product, and both come straight from the Phase-0 measurements:

**The query side is the S2/S3 record, not the S1 entity.** Train ground truth
has every one of its 7,638,365 pairs owned by exactly one S1 (0.0% reuse), so
the object being predicted is a function ``record -> S1 or none``. Indexing S1
makes the index the small, clean, deduplicated side (1.73M test rows against
9.97M records) and makes the natural unit of candidate-set size "candidates per
record", which is the quantity that transfers between train (4.68 records per
S1) and test (5.75).

**Work per query is bounded, not data-dependent.** The cost of a sparse top-k is
``sum over query terms of the posting-list length``. A char 3-gram such as
``"ltd"`` or ``" pv"`` occurs in a large fraction of a country's records, so a
single query can touch tens of millions of postings - this, not the number of
queries, is what makes naive TF-IDF retrieval at this scale slow. Two limits fix
it, and both are standard:

* ``max_index_df`` purges any n-gram whose posting list in the S1 index is
  longer than a cap. This is *block purging* from the blocking literature,
  applied to n-gram blocks. High-df grams also carry almost no IDF weight, so
  the cosine barely moves.
* ``sketch_terms`` keeps only the N highest-weight (rarest) n-grams of each
  query. This is the *prefix filter* idea: similar records must agree on a rare
  feature. Work per query is then at most ``sketch_terms * max_index_df``
  postings regardless of how long the record's text is.

Together they turn an unbounded scan into a fixed per-query budget, which is
what makes the approach shard-and-scale-out-able: every (country, source) shard
is independent, so the same code runs on one box or on N workers.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

try:                                        # optional, much faster when present
    from sparse_dot_topn import sp_matmul_topn
except ImportError:                         # pragma: no cover - fallback path
    sp_matmul_topn = None


# ---------------------------------------------------------------- sparse utils


def row_sketch(mat: sp.csr_matrix, n_terms: int, chunk: int = 200_000) -> sp.csr_matrix:
    """Keep only the ``n_terms`` largest entries of every row (prefix filter).

    Rows are re-L2-normalised afterwards so the dot product stays a cosine over
    the retained terms. Done in row chunks with a single ``lexsort`` each, so it
    stays vectorised at 10M-row scale.
    """
    if n_terms <= 0:
        return mat
    out = []
    for start in range(0, mat.shape[0], chunk):
        block = mat[start:start + chunk].tocsr().copy()
        counts = np.diff(block.indptr)
        rows = np.repeat(np.arange(block.shape[0]), counts)
        # sort by (row asc, weight desc); a row's entries are then contiguous and
        # start at exactly indptr[row], so the within-row rank is a subtraction
        order = np.lexsort((-block.data, rows))
        rank = np.empty(block.nnz, dtype=np.int64)
        rank[order] = np.arange(block.nnz) - block.indptr[rows[order]]
        block.data = np.where(rank < n_terms, block.data, 0.0)
        block.eliminate_zeros()
        out.append(block)
    res = sp.vstack(out).tocsr() if len(out) > 1 else out[0]
    return _l2_normalize(res)


def _l2_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
    """L2-normalise the rows of a CSR matrix in place (zero rows stay zero)."""
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    mat.data /= np.repeat(norms, np.diff(mat.indptr))
    return mat


def prune_columns(mats: dict, index_key, max_df: int) -> dict:
    """Drop every n-gram with more than ``max_df`` postings in the index matrix.

    ``mats`` maps a name to a CSR with a shared vocabulary; the column mask is
    computed on ``mats[index_key]`` (the S1 side) and applied to all of them.
    Returns re-normalised copies.
    """
    idx = mats[index_key]
    df = np.diff(sp.csc_matrix(idx).indptr)
    keep = np.flatnonzero(df <= max_df)
    return {k: _l2_normalize(v[:, keep].tocsr()) for k, v in mats.items()}


def topk(queries: sp.csr_matrix, index: sp.csr_matrix, k: int, chunk: int = 20_000,
         n_threads: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` cosine neighbours in ``index`` for every row of ``queries``.

    Uses ``sparse_dot_topn`` when installed, otherwise a chunked dense fallback
    (fine for development-size data, too memory-hungry for a full country).
    Returns ``(idx int32 [n, k], score float32 [n, k])``, -1 / 0 padded.
    """
    n = queries.shape[0]
    idx = np.full((n, k), -1, dtype=np.int32)
    score = np.zeros((n, k), dtype=np.float32)
    if n == 0:
        return idx, score

    if sp_matmul_topn is not None:
        index_t = index.T.tocsr()
        for s in range(0, n, chunk):
            res = sp_matmul_topn(queries[s:s + chunk], index_t, top_n=k,
                                 threshold=0.0, n_threads=n_threads or None, sort=True)
            counts = np.diff(res.indptr)
            rows = np.repeat(np.arange(res.shape[0]), counts)
            cols = np.arange(res.nnz) - np.repeat(res.indptr[:-1], counts)
            idx[s + rows, cols] = res.indices
            score[s + rows, cols] = res.data
        return idx, score

    index_t = index.T.tocsc()
    for s in range(0, n, chunk):
        dense = (queries[s:s + chunk] @ index_t).toarray()
        kk = min(k, dense.shape[1])
        part = np.argpartition(-dense, kk - 1, axis=1)[:, :kk]
        vals = np.take_along_axis(dense, part, axis=1)
        order = np.argsort(-vals, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        part = np.where(vals > 0, part, -1)
        idx[s:s + dense.shape[0], :kk] = part
        score[s:s + dense.shape[0], :kk] = vals
    return idx, score


# ---------------------------------------------------------------- probe index


class ProbeIndex:
    """A df-pruned char-n-gram index over one country's S1 rows, for one view.

    ``corpus`` (all of S1 + S2 + S3 for the country) only supplies the IDF
    statistics; it contains no labels, so fitting it on the test inputs is
    legitimate and is what keeps test-time IDF correct.
    """

    def __init__(self, ngram=(3, 3), min_df=2, max_index_df=0.02,
                 sketch_terms=24, analyzer="char_wb"):
        """``max_index_df`` is an absolute posting-list cap when > 1, or a
        fraction of the index rows when in (0, 1]; 0 disables purging."""
        self.params = dict(ngram=ngram, min_df=min_df, max_index_df=max_index_df,
                           sketch_terms=sketch_terms, analyzer=analyzer)
        self.vec = TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(ngram),
                                   min_df=min_df, sublinear_tf=True, dtype=np.float32)
        self.index = None
        self.stats = {}

    def fit(self, s1_texts, corpus_texts) -> "ProbeIndex":
        """Fit IDF on ``corpus_texts`` and build the pruned S1 index."""
        self.vec.fit(corpus_texts)
        mat = self.vec.transform(s1_texts)
        df = np.diff(sp.csc_matrix(mat).indptr)
        cap = self.params["max_index_df"]
        cap = mat.shape[0] if not cap else (cap if cap > 1 else max(1, int(cap * mat.shape[0])))
        self._keep_cols = np.flatnonzero(df <= cap)
        self.index = _l2_normalize(mat[:, self._keep_cols].tocsr())
        self.stats = {
            "vocab_full": int(mat.shape[1]),
            "vocab_kept": int(len(self._keep_cols)),
            "postings_before": int(mat.nnz),
            "postings_after": int(self.index.nnz),
            "max_posting_list": int(df[self._keep_cols].max()) if len(self._keep_cols) else 0,
            "df_cap": int(cap),
        }
        return self

    def transform(self, texts) -> sp.csr_matrix:
        """Vectorise query texts with the same pruning and the query sketch."""
        mat = self.vec.transform(texts)[:, self._keep_cols].tocsr()
        return row_sketch(_l2_normalize(mat), self.params["sketch_terms"])

    def query(self, texts, k: int, chunk: int = 20_000):
        """Top-``k`` S1 rows for each text in ``texts``."""
        return topk(self.transform(texts), self.index, k, chunk)

    def query_from_index(self, k: int, target: sp.csr_matrix, chunk: int = 20_000):
        """Reverse direction: top-``k`` rows of ``target`` for every S1 row."""
        return topk(self.index, target, k, chunk)


# ---------------------------------------------------------------- pair table


def pair_table(s1_ids, rec_ids, src: int, probes: dict, rev: dict | None = None,
               weights: dict | None = None) -> pd.DataFrame:
    """Fuse several probe results into the long pair table ``select`` consumes.

    ``probes`` maps a view name to ``(idx, score)`` from :meth:`ProbeIndex.query`
    (record -> S1). ``rev`` optionally maps a view name to ``(idx, score)`` from
    the S1 -> record direction, and supplies the ``r_ent`` column that the
    reciprocal filter needs.

    The fused score is the weighted maximum over views rather than a sum: the
    views are deliberately redundant (name+address, address-only, name-only), so
    a strong hit on any one of them is the evidence, while agreement across views
    is carried separately in ``n_views`` and used by the one-sided rule.
    """
    weights = weights or {}
    frames = []
    for view, (idx, score) in probes.items():
        k = idx.shape[1]
        rows = np.repeat(np.arange(idx.shape[0]), k)
        flat_idx, flat_score = idx.ravel(), score.ravel()
        ok = flat_idx >= 0
        frames.append(pd.DataFrame({
            "rec_row": rows[ok].astype(np.int32),
            "s1_row": flat_idx[ok].astype(np.int32),
            "score": flat_score[ok] * weights.get(view, 1.0),
            "r_rec": (np.tile(np.arange(k), idx.shape[0])[ok]).astype(np.int16),
        }))
    if not frames:
        return pd.DataFrame(columns=["s1", "rid", "src", "score", "r_rec", "r_ent", "n_views"])

    long = pd.concat(frames, ignore_index=True)
    agg = long.groupby(["rec_row", "s1_row"], sort=False).agg(
        score=("score", "max"), r_rec=("r_rec", "min"), n_views=("score", "size")
    ).reset_index()

    agg["r_ent"] = np.int16(-1)
    if rev:
        rev_frames = []
        for idx, score in rev.values():
            k = idx.shape[1]
            rows = np.repeat(np.arange(idx.shape[0]), k)
            flat = idx.ravel()
            ok = flat >= 0
            rev_frames.append(pd.DataFrame({
                "s1_row": rows[ok].astype(np.int32),
                "rec_row": flat[ok].astype(np.int32),
                "r_ent": np.tile(np.arange(k), idx.shape[0])[ok].astype(np.int16),
            }))
        rev_long = pd.concat(rev_frames, ignore_index=True).groupby(
            ["s1_row", "rec_row"], sort=False)["r_ent"].min().reset_index()
        agg = agg.drop(columns="r_ent").merge(rev_long, on=["s1_row", "rec_row"], how="left")
        agg["r_ent"] = agg["r_ent"].fillna(-1).astype(np.int16)

    return pd.DataFrame({
        "s1": np.asarray(s1_ids)[agg["s1_row"].to_numpy()],
        "rid": np.asarray(rec_ids)[agg["rec_row"].to_numpy()],
        "src": np.int8(src),
        "score": agg["score"].to_numpy(dtype=np.float32),
        "r_rec": agg["r_rec"].to_numpy(dtype=np.int16),
        "r_ent": agg["r_ent"].to_numpy(dtype=np.int16),
        "n_views": agg["n_views"].to_numpy(dtype=np.int8),
    })
