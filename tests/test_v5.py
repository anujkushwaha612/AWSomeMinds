"""Strategy v5 pieces that run on CPU: union merge, stage-2 features, expected-F rule, tiling."""

import numpy as np
import pandas as pd
import pytest

from ber.neural.common import length_order, make_texts, tile_rows
from ber.neural.train_biencoder import make_batches
from ber.union import ABSENT_RANK, dense_group_features, merge_candidates, pair_key
from ber.v5 import expected_f_keep, group_stats, stage2_features


def test_pair_key_roundtrip_and_bounds():
    k = pair_key(np.array([2, 3]), np.array([5, (1 << 22) - 1]), np.array([7, (1 << 21) - 1]))
    assert (k >> 43).tolist() == [2, 3]
    assert ((k >> 21) & ((1 << 22) - 1)).tolist() == [5, (1 << 22) - 1]
    assert (k & ((1 << 21) - 1)).tolist() == [7, (1 << 21) - 1]
    with pytest.raises(ValueError):
        pair_key(np.array([2]), np.array([0]), np.array([1 << 21]))


def test_merge_candidates_union_and_defaults():
    tf = pd.DataFrame({"src": [2, 2], "doc_row": [0, 1], "s1_row": [10, 11],
                       "score": [0.9, 0.4], "rank": [0, 0]})
    tf_cos = np.array([0.8, 0.3], dtype=np.float32)
    dense = pd.DataFrame({"src": [2, 3], "doc_row": [0, 4], "s1_row": [10, 12],
                          "cos": [0.81, 0.7], "drank_rec": [0, -1], "drank_s1": [-1, 2]})
    m = merge_candidates(tf, tf_cos, dense)
    assert len(m) == 3                                      # (2,0,10) shared, (2,1,11) tfidf, (3,4,12) dense
    row = m[(m.src == 2) & (m.doc_row == 0)].iloc[0]
    assert row.in_tfidf == 1 and row.in_dense == 1 and row.n_retrievers == 2
    assert row.cos == pytest.approx(0.8)                    # TF-IDF row keeps its own cosine
    d = m[m.src == 3].iloc[0]
    assert d.in_tfidf == 0 and d.score == 0 and d["rank"] == ABSENT_RANK
    assert d.drank_rec == ABSENT_RANK and d.drank_s1 == 2 and d.cos == pytest.approx(0.7)
    assert m["src"].is_monotonic_increasing                 # write_features requires src order
    g = dense_group_features(m.copy())
    assert g.loc[g.doc_row == 0, "dgap_rec"].iloc[0] == 0


def test_group_stats():
    key = np.array([1, 1, 1, 2])
    p = np.array([0.2, 0.9, 0.5, 0.4])
    s = group_stats(key, p, extra=(p > 0.3).astype(float))
    assert s["rank"].tolist() == [2, 0, 1, 0]
    assert s["max"].tolist() == [0.9, 0.9, 0.9, 0.4]
    assert s["second"].tolist() == [0.5, 0.5, 0.5, 0.0]
    assert s["n"].tolist() == [3, 3, 3, 1]
    assert s["extra_sum"].tolist() == [2, 2, 2, 1]


def test_stage2_features_margins_and_counts():
    # record (src2, doc0) has two S1 candidates; S1 row 1 also claims doc1 (src2) and doc0 (src3)
    keys = pd.DataFrame({"country": [0, 0, 0, 0], "src": [2, 2, 2, 3], "doc_row": [0, 0, 1, 0],
                         "s1_row": [0, 1, 1, 1]})
    p = np.array([0.3, 0.8, 0.6, 0.7])
    f = stage2_features(keys, p)
    np.testing.assert_allclose(f["p_minus_rec_other"], [-0.5, 0.5, 0.6, 0.7], atol=1e-6)
    assert f["s1src_n05"].tolist() == [0, 2, 2, 1]
    assert f["s1_n05"].tolist() == [0, 3, 3, 3]
    np.testing.assert_allclose(f["s1_pmax_other_src"], [0.0, 0.7, 0.7, 0.8], atol=1e-6)
    assert f["s1_n_best"].tolist() == [0, 3, 3, 3]          # best S1 of all three records


def test_expected_f_keep():
    keys = pd.DataFrame({"country": [0] * 5, "s1_row": [0, 0, 0, 1, 2]})
    rec_best = np.ones(5, dtype=bool)
    q = np.array([0.99, 0.95, 0.05, 0.02, 0.9])
    keep = expected_f_keep(keys, q, rec_best)
    # entity 0: keep the two confident candidates, not the 0.05 one; entity 1: empty is better
    assert keep.tolist() == [True, True, False, False, True]


def test_tile_rows_and_text_helpers():
    assert tile_rows(1_000_000, 2.0) == 1000
    assert tile_rows(10, 2.0) == 16384 and tile_rows(10**12, 1.0) == 64
    assert make_texts(["a b"], [""]) == ["query: a b | "]
    assert length_order(["ccc", "a", "bb"]).tolist() == [1, 2, 0]


def test_make_batches_same_country():
    pairs = pd.DataFrame({"country": [0] * 5 + [1] * 5})
    b = make_batches(pairs, 2, True, 0)
    assert len(b) == 4                                      # full batches only (2 + 2)
    for idx in b:
        assert len(set(pairs["country"].to_numpy()[idx])) == 1


def test_sibling_in_group_only_allowed_and_never_self():
    from ber.neural.pairs import sibling_in_group
    keys = pd.Series(["subway", "subway", "subway", "cafe", "", "cafe"])
    allowed = np.array([True, True, False, True, True, False])
    out = sibling_in_group(keys, allowed)
    assert out[0] == 1 and out[1] == 0            # cyclic within allowed rows of the group
    assert out[2] == -1 and out[5] == -1          # not allowed -> no sibling
    assert out[3] == -1 and out[4] == -1          # singleton group / empty key


def test_false_negative_mask():
    from ber.neural.train_biencoder import false_negative_mask
    # rows 0 and 1 share parent 7 (two records of one entity); row 2's hard negative is 7
    pairs = pd.DataFrame({"country": [0, 0, 0], "pos_s1": [7, 7, 9], "neg_s1": [5, 6, 7]})
    m = false_negative_mask(pairs, np.arange(3))
    assert m.shape == (3, 6)
    assert m[0, 1] and m[1, 0]                    # the other row's positive is my parent
    assert not m[0, 0] and not m[1, 1]            # own positive stays the target
    assert m[0, 5] and m[1, 5]                    # row 2's hard negative is my parent
    assert not m[2].any()


def test_entity_uniform_is_per_entity_and_deterministic():
    from ber.v5 import entity_uniform
    k = np.array([5, 5, 9, 12345678901], dtype=np.int64)
    u = entity_uniform(k, 7)
    assert u[0] == u[1] and 0 <= u.min() and u.max() < 1
    assert np.array_equal(u, entity_uniform(k, 7)) and not np.array_equal(u, entity_uniform(k, 8))


def test_number_features_conflict_vs_missing():
    from ber.features import _number_features
    a_d = ["12 400 411001", "12 411001", "", "7"]
    b_d = ["12 500 411001", "12", "", "7"]
    a_n = ["studio 54", "alpha", "7 eleven", "x"]
    b_n = ["studio 55", "alpha", "eleven", "x"]
    f = _number_features(a_d, b_d, a_n, b_n)
    assert f["num_conflict"].tolist() == [1, 0, 0, 0]        # suite 400 vs 500 is a conflict
    assert f["num_conflict_len"].tolist() == [3, 0, 0, 0]
    assert f["num_only_s1"].tolist() == [1, 1, 0, 0]         # a missing PIN is one-sided, not a conflict
    assert f["num_only_rec"].tolist() == [1, 0, 0, 0]
    assert f["name_num_conflict"].tolist() == [1, 0, 0, 0]
    assert f["name_num_diff"].tolist() == [2, 0, 1, 0]


def test_duplicate_text_mask():
    from ber.neural.train_biencoder import duplicate_text_mask
    pos = ["a | x", "b | y"]
    neg = ["a | x", "c | z"]                                 # row 0's hard negative has its parent's text
    m = duplicate_text_mask(pos, neg)
    assert m.shape == (2, 4)
    assert m[0, 2] and not m[0, 0] and not m[1].any()


def test_ce_features_margin_and_nan():
    from ber.v5 import ce_features
    keys = pd.DataFrame({"country": [0, 0, 0, 0], "src": [2, 2, 2, 3], "doc_row": [0, 0, 1, 0],
                         "s1_row": [0, 1, 1, 1]})
    ce = np.array([2.0, -1.0, 3.0, np.nan])
    f = ce_features(keys, ce)
    np.testing.assert_allclose(f["ce_minus_rec_other"].to_numpy()[:2], [3.0, -3.0])
    assert np.isnan(f["ce_minus_rec_other"].iloc[2])          # only scored candidate of its record
    assert np.isnan(f["ce"].iloc[3]) and np.isnan(f["ce_minus_rec_other"].iloc[3])


def test_ce_band_never_below_prune_tau():
    from ber.neural.cross_encoder import band
    from ber.config import load_config
    lo, hi = band()
    assert lo >= load_config()["v5"]["prune_tau"] and hi <= 1.0


def test_decision_distance_uses_first_and_rest_thresholds():
    from ber.llm_judge import decision_distance
    keys = pd.DataFrame({"country": np.int8(0), "src": np.array([2, 2, 2, 3], dtype=np.int8),
                         "doc_row": np.array([0, 0, 1, 0], dtype=np.int32),
                         "s1_row": np.array([0, 1, 1, 1], dtype=np.int32)})
    p = np.array([0.9, 0.95, 0.6, 0.4], dtype=np.float32)
    d = decision_distance(keys, p, {"threshold": {"t_first": 0.7, "t_rest": 0.5}})
    assert np.isinf(d[0])                                   # lost arbitration: never predicted
    np.testing.assert_allclose(d[1:], [0.25, 0.1, 0.1], atol=1e-6)


def test_llm_model_allowlist(monkeypatch):
    import ber.llm_judge as L
    monkeypatch.setattr(L, "lcfg", lambda: {"model": "qwen3:4b-instruct-2507-q4_K_M",
                                            "allowed_models": {"qwen3:4b-instruct-2507": "4.0B Apache-2.0"}})
    assert "Apache" in L.model_card()
    monkeypatch.setattr(L, "lcfg", lambda: {"model": "gemma4:31b", "allowed_models": {"qwen3:4b": "x"}})
    with pytest.raises(SystemExit):
        L.model_card()


def test_xgb_wrapper_and_combine():
    import xgboost as xgb
    from ber.v5 import XGBModel, combine_members
    rng = np.random.default_rng(0)
    X = rng.random((400, 3)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    d = xgb.DMatrix(X[:300], y[:300])
    b = xgb.train({"objective": "binary:logistic", "max_depth": 2, "device": "cpu"}, d, 50,
                  evals=[(xgb.DMatrix(X[300:], y[300:]), "es")], early_stopping_rounds=5, verbose_eval=False)
    p = XGBModel(b).predict(X, num_threads=1)
    assert p.shape == (400,) and ((p > 0.5) == y).mean() > 0.9
    m = {"lgb": np.array([0.2, 0.8], dtype=np.float32), "xgb": np.array([0.4, 0.6], dtype=np.float32)}
    np.testing.assert_allclose(combine_members(m, "mean"), [0.3, 0.7])
    assert combine_members(m, "xgb") is m["xgb"]
