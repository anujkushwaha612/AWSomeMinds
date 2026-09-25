"""Memory-lean pipeline pieces: chunking must not change results (plan.md Step 3/6/8)."""

import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.baseline import Decider, write_lists
from ber.blocking.word_retrieval import iter_counts, hash_counts
from ber.features import string_block, write_features
from ber.normalize import normalize_frame


class FakeStore:
    """Minimal stand-in for ber.store.CountryStore built from in-memory frames."""

    def __init__(self, frames):
        self.tables = {s: pa.Table.from_pandas(normalize_frame(f), preserve_index=False)
                       for s, f in frames.items()}

    def n(self, s):
        return self.tables[s].num_rows

    def column(self, s, name):
        return self.tables[s].column(name).combine_chunks()

    def strings(self, s, name, rows=None, start=0, stop=None):
        col = self.column(s, name)
        if rows is not None:
            return col.take(pa.array(rows, type=pa.int64())).to_pylist()
        return col.slice(start, (stop or len(col)) - start).to_pylist()

    def na_text(self, s, start, stop):
        import pyarrow.compute as pc
        return pc.binary_join_element_wise(self.column(s, "name_n").slice(start, stop - start),
                                           self.column(s, "addr_n").slice(start, stop - start),
                                           " ").to_pylist()

    def numpy(self, s, name):
        return self.column(s, name).to_numpy(zero_copy_only=False)


def frame(prefix, names, addrs):
    return pd.DataFrame({"entity_id": [f"{prefix}-{i}" for i in range(len(names))],
                         "business_name": names, "business_address": addrs,
                         "country": "India"})


STORE = FakeStore({
    1: frame("S1", ["Alpha Traders Pvt Ltd", "Beta Foods", "Gamma LLP"],
             ["12 MG Road Pune", "4 Park St Kolkata", "9 Ring Rd Delhi"]),
    2: frame("S2", ["ALPHA TRADERS", "बीटा फूड्स", "Delta"], ["12 MG ROAD PUNE", "4 PARK ST", ""]),
    3: frame("S3", ["Gamma", "Alpha Trader"], ["9 RING ROAD DELHI", "MG Road 12 Pune"]),
})


def test_iter_counts_chunking_is_lossless():
    whole = hash_counts(STORE.na_text(2, 0, STORE.n(2))).toarray()
    parts = np.vstack([c.toarray() for _, c in iter_counts(STORE, 2, None, chunk=1, window=2)])
    np.testing.assert_array_equal(whole, parts)


def test_write_features_chunk_invariant(tmp_path):
    cand = pd.DataFrame({"src": np.array([2, 2, 2, 3, 3], dtype=np.int8),
                         "doc_row": np.array([0, 1, 2, 0, 1], dtype=np.int32),
                         "s1_row": np.array([0, 1, 0, 2, 0], dtype=np.int32),
                         "score": np.float32(0.5), "rank": np.int8(0)})
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_features(cand, STORE, str(a), chunk=100, log=None)
    write_features(cand, STORE, str(b), chunk=2, log=None)
    ta, tb = pq.read_table(a).to_pandas(), pq.read_table(b).to_pandas()
    pd.testing.assert_frame_equal(ta, tb)
    assert ta["doc_row"].tolist() == cand["doc_row"].tolist()      # row order preserved
    direct = string_block(STORE, 2, np.array([0]), np.array([0]))
    assert ta.loc[0, "name_tset"] == direct["name_tset"][0]


def test_write_lists_every_s1_and_subset():
    fm, fc = io.StringIO(), io.StringIO()
    s1_ids = np.array(["S1-a", "S1-b", "S1-c"], dtype=object)
    rid = np.array(["S2-1", "S3-1", "S2-2"], dtype=object)
    rows = np.array([2, 0, 2])
    keep = np.array([True, False, False])
    nm, nc = write_lists(fm, fc, s1_ids, rid, rows, keep)
    assert (nm, nc) == (1, 3)
    assert fm.getvalue().splitlines() == ["S1-a\t", "S1-b\t", "S1-c\tS2-1"]
    assert fc.getvalue().splitlines() == ["S1-a\tS3-1", "S1-b\t", "S1-c\tS2-1,S2-2"]


def test_decider_arbitration_and_first_rest():
    # record (src2, doc0) is claimed by S1 rows 0 and 1; row 1 has higher p
    keys = pd.DataFrame({"country": np.int8(0), "src": np.array([2, 2, 2, 3], dtype=np.int8),
                         "doc_row": np.array([0, 0, 1, 0], dtype=np.int32),
                         "s1_row": np.array([0, 1, 1, 1], dtype=np.int32)})
    p = np.array([0.9, 0.95, 0.6, 0.4], dtype=np.float32)
    d = Decider(keys, p)
    assert d.rec_best.tolist() == [False, True, True, True]
    # S1 row 1: best 0.95 (first), then 0.6, 0.4
    assert d.keep(0.5, 0.5).tolist() == [False, True, True, False]
    assert d.keep(0.99, 0.1).tolist() == [False, False, False, False]   # rest need the first
    assert d.keep(0.5, 0.5, arbitrate=False).tolist() == [True, True, True, False]
