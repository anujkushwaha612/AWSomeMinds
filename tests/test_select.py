"""Unit tests for the candidate-selection stage (ber.blocking.select)."""

import numpy as np
import pandas as pd
import pytest

from ber.blocking.metrics import report, size_stats, universe_sizes
from ber.blocking.select import (SelectPolicy, capacity_prune, group_rank,
                                 reciprocal_tier, record_depth, select)


def make_pairs(rows):
    """Build a pair table from ``(s1, rid, src, score, r_rec, r_ent, n_views)`` tuples."""
    return pd.DataFrame(rows, columns=["s1", "rid", "src", "score", "r_rec",
                                       "r_ent", "n_views"])


# ---------------------------------------------------------------- group_rank


def test_group_rank_orders_by_descending_score_within_group():
    codes = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    assert list(group_rank(codes, score)) == [2, 0, 1, 1, 0]


def test_group_rank_is_deterministic_on_ties():
    codes = np.zeros(4, dtype=int)
    score = np.array([0.5, 0.5, 0.5, 0.5])
    assert list(group_rank(codes, score)) == [0, 1, 2, 3]


def test_group_rank_handles_a_single_group_and_empty_input():
    assert list(group_rank(np.array([3, 3]), np.array([1.0, 2.0]))) == [1, 0]
    assert len(group_rank(np.array([], dtype=int), np.array([]))) == 0


# ---------------------------------------------------------------- record depth


def test_confident_record_keeps_only_its_top_candidate():
    pairs = make_pairs([("A", "S2-1", 2, 0.95, 0, 0, 2),
                        ("B", "S2-1", 2, 0.30, 1, -1, 1),
                        ("C", "S2-1", 2, 0.25, 2, -1, 1)])
    p = SelectPolicy(conf_score=0.8, conf_margin=0.15, a_max=3)
    keep, diag = record_depth(pairs, p, np.ones(3, dtype=bool))
    assert list(keep) == [True, False, False]
    assert diag.loc[0, "depth"] == 1


def test_ambiguous_record_keeps_a_max_candidates():
    pairs = make_pairs([("A", "S2-1", 2, 0.61, 0, 0, 1),
                        ("B", "S2-1", 2, 0.60, 1, 0, 1),
                        ("C", "S2-1", 2, 0.59, 2, -1, 1)])
    keep, diag = record_depth(pairs, SelectPolicy(a_max=2), np.ones(3, dtype=bool))
    assert list(keep) == [True, True, False]
    assert diag.loc[0, "depth"] == 2


def test_hopeless_record_abstains_entirely():
    pairs = make_pairs([("A", "S2-1", 2, 0.12, 0, -1, 1),
                        ("B", "S2-1", 2, 0.11, 1, -1, 1)])
    keep, diag = record_depth(pairs, SelectPolicy(abstain_score=0.2, score_floor=0.0),
                              np.ones(2, dtype=bool))
    assert not keep.any()
    assert diag.loc[0, "depth"] == 0


# ---------------------------------------------------------------- reciprocal


def test_reciprocal_tiering_labels_all_three_tiers():
    pairs = make_pairs([
        ("A", "S2-1", 2, 0.50, 0, 1, 1),     # both directions -> tier 0
        ("B", "S2-2", 2, 0.90, 5, -1, 1),    # one-sided but strong -> tier 1
        ("C", "S2-3", 2, 0.30, 0, 99, 1),    # rank too deep on the S1 side -> tier 2
    ])
    p = SelectPolicy(recip_r_rec=2, recip_r_ent=10, one_sided_score=0.45, one_sided_views=2)
    assert list(reciprocal_tier(pairs, p)) == [0, 1, 2]


def test_view_agreement_can_save_a_one_sided_pair():
    pairs = make_pairs([("A", "S2-1", 2, 0.20, 4, -1, 3)])
    p = SelectPolicy(one_sided_score=0.45, one_sided_views=2)
    assert reciprocal_tier(pairs, p)[0] == 1


# ---------------------------------------------------------------- capacity


def test_capacity_prune_enforces_the_per_entity_source_cap():
    rows = [("A", f"S2-{i}", 2, 1.0 - i / 100, 0, 0, 1) for i in range(12)]
    keep = capacity_prune(make_pairs(rows), SelectPolicy(cap_s2=5, a_max=3),
                          np.ones(12, dtype=bool))
    assert keep.sum() == 5
    assert list(keep[:5]) == [True] * 5          # the five highest scores survive


def test_capacity_prune_keeps_s2_and_s3_budgets_separate():
    rows = ([("A", f"S2-{i}", 2, 0.9, 0, 0, 1) for i in range(8)]
            + [("A", f"S3-{i}", 3, 0.9, 0, 0, 1) for i in range(8)])
    keep = capacity_prune(make_pairs(rows), SelectPolicy(cap_s2=5, cap_s3=6),
                          np.ones(16, dtype=bool))
    out = make_pairs(rows)[keep]
    assert (out["src"] == 2).sum() == 5
    assert (out["src"] == 3).sum() == 6


def test_capacity_prune_enforces_the_per_record_cap():
    rows = [(f"S1-{i}", "S2-1", 2, 1.0 - i / 100, i, 0, 1) for i in range(6)]
    keep = capacity_prune(make_pairs(rows), SelectPolicy(a_max=2), np.ones(6, dtype=bool))
    assert keep.sum() == 2


# ---------------------------------------------------------------- end to end


def test_select_output_respects_every_cap_and_is_a_subset_of_the_input():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(300):
        for s in rng.choice(60, size=8, replace=False):
            rows.append((f"S1-{s}", f"S2-{r}", 2, float(rng.random()),
                         int(rng.integers(0, 8)), int(rng.integers(-1, 12)), 1))
    pairs = make_pairs(rows)
    out = select(pairs, SelectPolicy(a_max=2, cap_s2=5, cap_s3=6))

    assert len(out) <= len(pairs)
    assert out.groupby("rid").size().max() <= 2
    assert out.groupby(["s1", "src"]).size().max() <= 5
    merged = out.merge(pairs, on=["s1", "rid"], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_rescue_gives_an_emptied_record_its_best_edge_back():
    # a single weak, non-reciprocal pair: tiering drops it, the rescue restores it
    pairs = make_pairs([("A", "S2-1", 2, 0.30, 5, -1, 1)])
    p = SelectPolicy(one_sided_score=0.45, one_sided_views=2, rescue_floor=0.25,
                     abstain_score=0.0)
    out = select(pairs, p)
    assert len(out) == 1 and out.loc[0, "tier"] == 2

    out_no_rescue = select(pairs, p.replace(rescue_floor=0.9))
    assert len(out_no_rescue) == 0


def test_select_handles_an_empty_pair_table():
    empty = make_pairs([]).astype({"score": float, "r_rec": int, "r_ent": int,
                                   "n_views": int, "src": int})
    out = select(empty)
    assert len(out) == 0 and "tier" in out.columns


def test_smaller_a_max_never_produces_a_larger_candidate_set():
    rng = np.random.default_rng(1)
    rows = [(f"S1-{int(s)}", f"S2-{r}", 2, float(rng.random()), int(i), 0, 1)
            for r in range(200)
            for i, s in enumerate(rng.choice(40, size=5, replace=False))]
    pairs = make_pairs(rows)
    sizes = [len(select(pairs, SelectPolicy(a_max=a))) for a in (1, 2, 3)]
    assert sizes == sorted(sizes)


# ---------------------------------------------------------------- metrics


def test_report_recovers_known_recall_size_and_reduction_ratio():
    truth = pd.DataFrame({"s1": ["A", "A", "B"], "rid": ["S2-1", "S3-1", "S2-2"]})
    cand = pd.DataFrame({"s1": ["A", "A", "B", "C"],
                         "rid": ["S2-1", "S3-9", "S2-2", "S2-7"]})
    entities = ["A", "B", "C"]
    r = report(cand, truth, entities, {"cross_all_pairs": 100, "cross_country": 50,
                                       "n_records": 10})
    assert r["pairs_completeness"] == pytest.approx(2 / 3)
    assert r["pairs_quality"] == pytest.approx(2 / 4)
    assert r["C_per_s1_mean"] == pytest.approx(4 / 3)
    assert r["RR_vs_country"] == pytest.approx(1 - 4 / 50)
    # A: 1 of its 2 true matches is reachable, so the perfect matcher gets
    # P = 1, R = 0.5 -> F0.5 = 1.25 * 0.5 / (0.25 + 0.5) = 0.8333.
    # B: both reachable -> 1.0. C: k = 0, so predicting empty scores 1.0 even
    # though blocking handed it a candidate.
    assert r["oracle_f05"] == pytest.approx((1.25 * 0.5 / 0.75 + 1.0 + 1.0) / 3)


def test_size_stats_counts_entities_with_no_candidates():
    cand = pd.DataFrame({"s1": ["A", "A"], "rid": ["S2-1", "S2-2"]})
    s = size_stats(cand, ["A", "B", "C", "D"], n_records=8)
    assert s["share_s1_empty"] == pytest.approx(0.75)
    assert s["C_per_record_mean"] == pytest.approx(0.25)


def test_universe_sizes_uses_the_country_partition():
    s1 = pd.DataFrame({"country": ["US", "US", "India"]})
    rec = pd.DataFrame({"country": ["US", "India", "India", "France"]})
    u = universe_sizes(s1, rec)
    assert u["cross_all_pairs"] == 12
    assert u["cross_country"] == 2 * 1 + 1 * 2      # US 2x1, India 1x2, France unmatched


def test_policy_loads_from_the_pipeline_config():
    from ber.blocking.select import SelectPolicy as P
    pol = P.from_config()
    assert pol.cap_s2 >= 5 and pol.cap_s3 >= 6      # never below the measured caps
    assert 1 <= pol.a_max <= 5
    assert P.from_config({"a_max": 1, "unknown_key": 3}).a_max == 1


# ---------------------------------------------------------------- run helpers


def test_derived_r_ent_ranks_records_competing_for_the_same_entity():
    from ber.blocking.run import derive_r_ent
    pairs = pd.DataFrame({
        "s1_row": [0, 0, 0, 1],
        "src": [2, 2, 2, 2],
        "score": [0.3, 0.9, 0.6, 0.2],
    })
    assert list(derive_r_ent(pairs)) == [2, 0, 1, 0]


def test_derived_r_ent_separates_the_two_sources():
    from ber.blocking.run import derive_r_ent
    pairs = pd.DataFrame({"s1_row": [0, 0], "src": [2, 3], "score": [0.3, 0.9]})
    assert list(derive_r_ent(pairs)) == [0, 0]


def test_parse_knobs_keeps_integer_knobs_integral():
    from ber.blocking.tune import parse_knobs
    knobs = parse_knobs(["a_max=1,2,3", "score_floor=0.1,0.2"])
    assert knobs["a_max"] == [1, 2, 3]
    assert all(isinstance(v, int) for v in knobs["a_max"])
    assert knobs["score_floor"] == [0.1, 0.2]


def test_parse_knobs_rejects_an_unknown_knob():
    from ber.blocking.tune import parse_knobs
    with pytest.raises(SystemExit):
        parse_knobs(["not_a_knob=1"])


def test_record_view_locates_the_true_parent_rank():
    from ber.blocking.tune import record_view
    pairs = make_pairs([("A", "S2-1", 2, 0.9, 0, 0, 1),
                        ("B", "S2-1", 2, 0.5, 1, -1, 1),
                        ("C", "S2-2", 2, 0.4, 0, -1, 1)])
    truth = pd.DataFrame({"s1": ["B"], "rid": ["S2-1"]})
    rv = record_view(pairs, truth).set_index("rid")
    assert rv.loc["S2-1", "parent_rank"] == 1          # parent is the runner-up
    assert not rv.loc["S2-1", "parent_is_top1"]
    assert rv.loc["S2-1", "has_parent"]
    assert not rv.loc["S2-2", "has_parent"]            # orphan record
    assert rv.loc["S2-2", "parent_rank"] == -1


def test_abstain_table_trades_declined_records_against_lost_pairs():
    from ber.blocking.tune import abstain_table
    rv = pd.DataFrame({
        "top1": [0.9, 0.1, 0.1, 0.8],
        "has_parent": [True, True, False, False],
        "parent_rank": [0, 0, -1, -1],
    })
    t = abstain_table(rv, grid=[0.0, 0.5]).set_index("abstain_score")
    assert t.loc[0.0, "records_declined"] == 0.0
    assert t.loc[0.5, "records_declined"] == pytest.approx(0.5)
    assert t.loc[0.5, "true_pairs_lost"] == pytest.approx(0.5)   # 1 of 2 retrievable
    assert t.loc[0.5, "orphans_caught"] == pytest.approx(0.5)    # 1 of 2 orphans


def test_operating_point_treats_sub_materiality_deficits_as_ties():
    from ber.blocking.metrics import pick_operating_point
    rng = np.random.default_rng(3)
    rows = [(f"S1-{int(s)}", f"S2-{r}", 2, float(rng.random()), int(i), 0, 1)
            for r in range(400)
            for i, s in enumerate(rng.choice(80, size=4, replace=False))]
    pairs = make_pairs(rows)
    truth = pairs.groupby("rid").head(1)[["s1", "rid"]]
    entities = sorted(pairs["s1"].unique())
    pols = {"tight": SelectPolicy(a_max=1), "loose": SelectPolicy(a_max=3)}
    op = pick_operating_point(pairs, truth, entities, pols, n_resamples=100,
                              materiality=1.0)
    assert op["tied_with_best"].all()                  # everything ties at a huge bar
    assert op.iloc[0]["C_per_s1_mean"] <= op.iloc[-1]["C_per_s1_mean"]
