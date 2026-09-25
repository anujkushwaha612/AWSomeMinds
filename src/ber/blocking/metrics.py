"""Blocking scorecard: the numbers the methodology document has to report.

Amazon judges ``candidate_pairs.tsv`` on how small it is per Source-1 entity at
a given recall, so every configuration is reported on one line with all four
axes at once:

* **oracle macro-F0.5** - the leaderboard ceiling of the candidate set. Per
  entity, a perfect matcher scores P = 1 and R = (true matches retrieved) / k;
  k = 0 entities score 1.0. This, not pair recall, is what the competition
  metric can ever reach given these candidates.
* **pairs completeness (PC)** - classic blocking recall, ``|C n GT| / |GT|``.
* **pairs quality (PQ)** - ``|C n GT| / |C|``, the density of true pairs.
* **size** - ``|C|`` per S1 entity (mean / median / p90 / max, plus the share of
  entities with an empty candidate list) and ``|C|`` per S2/S3 record, which is
  the quantity that actually transfers between train and test.
* **reduction ratio (RR)** - ``1 - |C| / |universe|`` against both the naive
  cross product and the country-partitioned one.

Why report per-record as well as per-S1: train has 4.68 S2/S3 records per S1
entity and test has 5.75, so an identical blocking policy mechanically produces
a 23% larger per-S1 candidate list on test. Tune on per-record, report per-S1.
"""

import numpy as np
import pandas as pd

from ..eval.scorer import k_bucket, oracle_scores, paired_bootstrap


def universe_sizes(s1: pd.DataFrame, records: pd.DataFrame) -> dict:
    """Comparison-space sizes from the raw tables (columns ``country``).

    Returns the naive cross product, the country-partitioned one, and the row
    counts, i.e. the denominators of every reduction ratio in the report.
    """
    a = s1["country"].value_counts()
    b = records["country"].value_counts()
    common = a.index.intersection(b.index)
    return {
        "n_s1": int(len(s1)),
        "n_records": int(len(records)),
        "cross_all_pairs": int(len(s1)) * int(len(records)),
        "cross_country": int((a[common].astype("int64") * b[common].astype("int64")).sum()),
    }


def size_stats(candidates: pd.DataFrame, entities, n_records: int | None = None) -> dict:
    """``|C|`` statistics per S1 entity and per S2/S3 record."""
    entities = pd.Index(pd.unique(pd.Series(list(entities), dtype=str)), name="s1")
    cand = candidates[["s1", "rid"]].drop_duplicates()
    per_entity = cand.groupby("s1").size().reindex(entities, fill_value=0).to_numpy()
    out = {
        "n_candidates": int(len(cand)),
        "C_per_s1_mean": float(per_entity.mean()),
        "C_per_s1_median": float(np.median(per_entity)),
        "C_per_s1_p90": float(np.percentile(per_entity, 90)),
        "C_per_s1_p99": float(np.percentile(per_entity, 99)),
        "C_per_s1_max": int(per_entity.max()) if len(per_entity) else 0,
        "share_s1_empty": float((per_entity == 0).mean()),
    }
    if n_records:
        n_touched = cand["rid"].nunique()
        out["C_per_record_mean"] = len(cand) / n_records
        out["share_records_used"] = n_touched / n_records
    return out


def report(candidates: pd.DataFrame, truth: pd.DataFrame, entities,
           universe: dict | None = None, name: str = "") -> dict:
    """One-line scorecard for a candidate set (see the module docstring)."""
    entities = pd.Index(pd.unique(pd.Series(list(entities), dtype=str)), name="s1")
    cand = candidates[["s1", "rid"]].drop_duplicates()
    cand = cand[cand["s1"].isin(entities)]
    gt = truth[truth["s1"].isin(entities)][["s1", "rid"]].drop_duplicates()

    hit = len(cand.merge(gt, on=["s1", "rid"], how="inner"))
    scores = oracle_scores(gt, cand, entities)

    row = {"config": name,
           "oracle_f05": float(scores["f05"].mean()),
           "pairs_completeness": hit / len(gt) if len(gt) else float("nan"),
           "pairs_quality": hit / len(cand) if len(cand) else float("nan")}
    row.update(size_stats(cand, entities, universe.get("n_records") if universe else None))
    if universe:
        row["RR_vs_all_pairs"] = 1 - row["n_candidates"] / universe["cross_all_pairs"]
        row["RR_vs_country"] = 1 - row["n_candidates"] / universe["cross_country"]
    return row


def by_stratum(candidates: pd.DataFrame, truth: pd.DataFrame, entities,
               strata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Oracle F0.5 and |C| per stratum (``strata`` is indexed by S1 id).

    Always includes the k-bucket breakdown, because the k in {0, 1} band carries
    the most F0.5 per error: one false positive on a k = 0 entity costs the full
    1.0, whereas one extra candidate on a k = 4 entity costs 0.167.
    """
    entities = pd.Index(pd.unique(pd.Series(list(entities), dtype=str)), name="s1")
    cand = candidates[["s1", "rid"]].drop_duplicates()
    gt = truth[truth["s1"].isin(entities)][["s1", "rid"]].drop_duplicates()
    scores = oracle_scores(gt, cand, entities)
    scores["C"] = cand.groupby("s1").size().reindex(entities, fill_value=0)
    scores["k_bucket"] = k_bucket(scores["k"])
    if strata is not None:
        scores = scores.join(strata, how="left")

    cols = [c for c in ("country", "k_bucket") if c in scores.columns]
    rows = [{"stratum": "ALL", "n": len(scores), "oracle_f05": scores["f05"].mean(),
             "C_per_s1": scores["C"].mean()}]
    for col in cols:
        for key, grp in scores.groupby(col, sort=True):
            rows.append({"stratum": f"{col}={key}", "n": len(grp),
                         "oracle_f05": grp["f05"].mean(), "C_per_s1": grp["C"].mean()})
    return pd.DataFrame(rows)


def frontier(pairs: pd.DataFrame, truth: pd.DataFrame, entities, base_policy,
             sweep: dict, universe: dict | None = None,
             select_fn=None) -> pd.DataFrame:
    """Sweep one policy knob and return the size / recall trade-off curve.

    ``sweep`` is ``{"a_max": [1, 2, 3]}``-style; one row per value, sorted by
    candidate-set size. This is the table that picks the operating point: take
    the smallest ``C_per_s1_mean`` whose ``oracle_f05`` is statistically tied
    with the best (see :func:`pick_operating_point`).
    """
    from .select import select as default_select

    select_fn = select_fn or default_select
    (knob, values), = sweep.items()
    rows = []
    for v in values:
        cand = select_fn(pairs, base_policy.replace(**{knob: v}))
        rows.append({knob: v, **report(cand, truth, entities, universe,
                                       name=f"{knob}={v}")})
    return pd.DataFrame(rows).sort_values("C_per_s1_mean").reset_index(drop=True)


def pick_operating_point(pairs: pd.DataFrame, truth: pd.DataFrame, entities,
                         policies: dict, n_resamples: int = 500,
                         select_fn=None) -> pd.DataFrame:
    """Rank candidate policies by size, keeping only those tied with the best.

    For every policy this reports the oracle F0.5, the candidate-set size and a
    paired-bootstrap CI of its oracle F0.5 *deficit* against the best policy in
    the set. ``tied_with_best`` marks the policies whose deficit CI contains 0;
    the operating point is the smallest such candidate set.
    """
    from .select import select as default_select

    select_fn = select_fn or default_select
    entities = pd.Index(pd.unique(pd.Series(list(entities), dtype=str)), name="s1")
    gt = truth[truth["s1"].isin(entities)][["s1", "rid"]].drop_duplicates()

    per_entity, rows = {}, []
    for name, pol in policies.items():
        cand = select_fn(pairs, pol)
        per_entity[name] = oracle_scores(gt, cand[["s1", "rid"]], entities)["f05"]
        rows.append({"config": name, **size_stats(cand, entities),
                     "oracle_f05": float(per_entity[name].mean())})
    df = pd.DataFrame(rows)
    best = df.loc[df["oracle_f05"].idxmax(), "config"]

    stats = [paired_bootstrap(per_entity[best], per_entity[c], n_resamples)
             for c in df["config"]]
    df["delta_vs_best"] = [s["delta"] for s in stats]
    df["ci_low"] = [s["ci_low"] for s in stats]
    df["ci_high"] = [s["ci_high"] for s in stats]
    df["tied_with_best"] = [s["ci_low"] <= 0 <= s["ci_high"] for s in stats]
    return df.sort_values("C_per_s1_mean").reset_index(drop=True)
