"""Hand-computed checks of the competition scorer (plan.md Step 1)."""

import numpy as np
import pandas as pd
import pytest

from ber.eval.scorer import (macro_f05, oracle_scores, paired_bootstrap,
                             per_entity_scores, summarize)


def pairs(rows):
    """Build a long-form pairs frame from ``[(s1, rid), ...]``."""
    return pd.DataFrame(rows, columns=["s1", "rid"], dtype=str)


def f05(p, r):
    return 1.25 * p * r / (0.25 * p + r)


def score(truth_rows, pred_rows, entities):
    return per_entity_scores(pairs(truth_rows), pairs(pred_rows), entities)["f05"]


def test_problem_statement_example():
    # predict [47, 193, 812], truth [47, 812] -> P=2/3, R=1 -> 0.714
    s = score([("e", "S2-47"), ("e", "S3-812")],
              [("e", "S2-47"), ("e", "S2-193"), ("e", "S3-812")], ["e"])
    assert s["e"] == pytest.approx(f05(2 / 3, 1.0))
    assert round(s["e"], 3) == 0.714


@pytest.mark.parametrize("truth, pred, expected", [
    ([], [], 1.0),                                        # singleton, empty -> 1
    ([], [("e", "S2-1")], 0.0),                           # singleton, any pred -> 0
    ([("e", "S2-1")], [], 0.0),                           # non-singleton, empty -> 0
    ([("e", "S2-1")], [("e", "S2-1")], 1.0),              # exact
    ([("e", "S2-1")], [("e", "S2-2")], 0.0),              # no TP
    ([("e", f"S2-{i}") for i in range(4)],                # 3 of 4, no FP -> 0.9375
     [("e", f"S2-{i}") for i in range(3)], 0.9375),
    ([("e", f"S2-{i}") for i in range(4)],                # 1 of 4 -> 0.625
     [("e", "S2-0")], 0.625),
    ([("e", f"S2-{i}") for i in range(4)],                # 4 of 4 + 1 FP -> P=.8
     [("e", f"S2-{i}") for i in range(5)], f05(0.8, 1.0)),
    ([("e", "S2-1"), ("e", "S3-1")],                      # 2 TP + 1 FP -> 0.714
     [("e", "S2-1"), ("e", "S3-1"), ("e", "S3-9")], f05(2 / 3, 1.0)),
    ([("e", "S2-1"), ("e", "S3-1")],                      # 1 TP + 1 FP, R=.5
     [("e", "S2-1"), ("e", "S3-9")], f05(0.5, 0.5)),
])
def test_hand_cases(truth, pred, expected):
    assert score(truth, pred, ["e"])["e"] == pytest.approx(expected)


def test_f05_from_counts_matches_scorer():
    from ber.eval.scorer import f05_from_counts
    # k, m, tp for: singleton empty, singleton FP, miss, exact, 3/4, 4/4+1FP, 2TP+1FP
    k = [0, 0, 1, 1, 4, 4, 2]
    m = [0, 1, 0, 1, 3, 5, 3]
    tp = [0, 0, 0, 1, 3, 4, 2]
    want = [1.0, 0.0, 0.0, 1.0, 0.9375, f05(0.8, 1.0), f05(2 / 3, 1.0)]
    np.testing.assert_allclose(f05_from_counts(k, m, tp), want)


def test_duplicate_predictions_count_once():
    s = score([("e", "S2-1")], [("e", "S2-1"), ("e", "S2-1")], ["e"])
    assert s["e"] == 1.0


def test_macro_includes_entities_without_rows_and_ignores_outsiders():
    truth = pairs([("a", "S2-1")])
    pred = pairs([("a", "S2-1"), ("z", "S2-5")])   # z is not in the eval set
    # a = 1.0, b is a correctly-empty singleton = 1.0, c has a wrong... none -> 1.0
    assert macro_f05(truth, pred, ["a", "b", "c"]) == 1.0
    # a predicted empty now -> 0; mean over 3 entities
    assert macro_f05(truth, pairs([]), ["a", "b", "c"]) == pytest.approx(2 / 3)


def test_oracle_is_perfect_matcher_on_candidates():
    truth = pairs([("a", "S2-1"), ("a", "S2-2"), ("b", "S3-1")])
    cands = pairs([("a", "S2-1"), ("a", "S2-9"), ("c", "S2-7")])
    s = oracle_scores(truth, cands, ["a", "b", "c"])["f05"]
    assert s["a"] == pytest.approx(f05(1.0, 0.5))   # 1 of 2 reachable
    assert s["b"] == 0.0                            # true match not retrieved
    assert s["c"] == 1.0                            # singleton: oracle predicts empty


def test_paired_bootstrap():
    a = pd.Series(np.zeros(2000), index=[f"e{i}" for i in range(2000)])
    same = paired_bootstrap(a, a, n_resamples=200)
    assert same["delta"] == 0 and not same["significant"]
    better = paired_bootstrap(a, a + 0.01, n_resamples=200)
    assert better["delta"] == pytest.approx(0.01) and better["significant"]
    # index alignment: shuffled order must give the same result
    shuffled = (a + 0.01).sample(frac=1.0, random_state=1)
    assert paired_bootstrap(a, shuffled, n_resamples=50)["delta"] == pytest.approx(0.01)


def test_summarize_strata():
    truth = pairs([("a", "S2-1")])
    s = per_entity_scores(truth, pairs([]), ["a", "b"])
    strata = pd.DataFrame({"country": ["US", "India"]}, index=pd.Index(["a", "b"], name="s1"))
    out = summarize(s, strata).set_index("stratum")
    assert out.loc["ALL", "f05"] == 0.5
    assert out.loc["country=US", "f05"] == 0.0
    assert out.loc["k_bucket=0", "f05"] == 1.0
