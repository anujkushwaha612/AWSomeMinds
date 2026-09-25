"""Retrieval top-k, rank lookup and audit parsing (plan.md Step 3)."""

import numpy as np
import scipy.sparse as sp

from ber.blocking.audit import parse
from ber.blocking.rank_table import NOT_FOUND, UNKNOWN, find_rank
from ber.blocking.retrieve import topk


def test_topk_matches_bruteforce():
    q =sp.random(50, 300, density=0.05, format="csr", dtype=np.float32, random_state=1)
    d = sp.random(400, 300, density=0.05, format="csr", dtype=np.float32, random_state=2)
    idx, score = topk(q, d, k=5, chunk=16, n_threads=2)
    dense = (q @ d.T).toarray()
    for i in range(q.shape[0]):
        nz = np.flatnonzero(dense[i] > 0)
        want = nz[np.argsort(-dense[i, nz], kind="stable")][:5]
        got = idx[i][idx[i] >= 0]
        assert len(got) == len(want)
        np.testing.assert_allclose(np.sort(dense[i, got]), np.sort(dense[i, want]), rtol=1e-3)
        assert (idx[i][score[i] == 0] == -1).all()      # padding only where no hit


def test_find_rank():
    lists = np.array([[4, 7, 9], [1, -1, -1]])
    row_pos = np.array([0, -1, 1])          # table row 1 was not queried
    rows = np.array([0, 0, 2, 1])
    target = np.array([9, 5, 1, 3])
    assert find_rank(lists, rows, target, row_pos).tolist() == [2, NOT_FOUND, 0, UNKNOWN]


def test_parse():
    assert parse("NA:f20, A:r3") == [("NA", "fwd", 20), ("A", "rev", 3)]
