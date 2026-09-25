"""Exact replica of the competition metric: per-S1-entity F0.5, macro-averaged.

Rules (problem statement):
  * F0.5 = 1.25 P R / (0.25 P + R), computed per S1 entity, averaged over ALL
    entities in the evaluation set.
  * An entity with no true matches scores 1.0 for an empty prediction and 0.0
    for any non-empty prediction.
  * An entity with true matches scores 0.0 for an empty prediction (and for any
    prediction with no true positives).

Pairs are long-form DataFrames with columns ``s1`` (S1 id) and ``rid`` (S2/S3 id).
"""

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5


def _counts(pairs: pd.DataFrame, entities: pd.Index) -> pd.Series:
    """Number of distinct ``rid`` per entity, 0 for entities without pairs."""
    return pairs.groupby("s1").size().reindex(entities, fill_value=0).astype(np.int64)


def per_entity_scores(truth: pd.DataFrame, pred: pd.DataFrame, entities) -> pd.DataFrame:
    """Score every entity in ``entities``.

    Returns a DataFrame indexed by entity with columns ``k`` (true matches),
    ``m`` (predicted), ``tp`` and ``f05``. Predictions and truth for entities
    outside ``entities`` are ignored; duplicate predicted pairs count once.
    """
    entities = pd.Index(pd.unique(pd.Series(list(entities), dtype=str)), name="s1")
    truth = truth.loc[truth["s1"].isin(entities), ["s1", "rid"]].drop_duplicates()
    pred = pred.loc[pred["s1"].isin(entities), ["s1", "rid"]].drop_duplicates()

    k = _counts(truth, entities)
    m = _counts(pred, entities)
    tp = _counts(pred.merge(truth, on=["s1", "rid"], how="inner"), entities)

    kv, mv, tv = k.to_numpy(), m.to_numpy(), tp.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(mv > 0, tv / np.maximum(mv, 1), 0.0)
        r = np.where(kv > 0, tv / np.maximum(kv, 1), 0.0)
        f = np.where(tv > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    f = np.where(kv == 0, (mv == 0).astype(float), f)
    return pd.DataFrame({"k": kv, "m": mv, "tp": tv, "f05": f}, index=entities)


def macro_f05(truth: pd.DataFrame, pred: pd.DataFrame, entities) -> float:
    """Macro-averaged F0.5 over ``entities`` (the leaderboard number)."""
    return float(per_entity_scores(truth, pred, entities)["f05"].mean())


def oracle_scores(truth: pd.DataFrame, candidates: pd.DataFrame, entities) -> pd.DataFrame:
    """Per-entity F0.5 ceiling of a candidate set under a perfect matcher.

    A perfect matcher keeps exactly the true pairs present among the candidates,
    so P = 1 and R = (true pairs in candidates) / k. Entities with k = 0 score
    1.0 (the perfect matcher predicts empty). This is the entity-level recall
    ceiling that blocking decisions are judged on (plan.md Step 3).
    """
    reachable = candidates[["s1", "rid"]].merge(truth[["s1", "rid"]], on=["s1", "rid"])
    return per_entity_scores(truth, reachable, entities)


def paired_bootstrap(scores_a, scores_b, n_resamples: int = 1000, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """Paired bootstrap CI of mean(scores_b - scores_a) over the same entities.

    ``scores_a``/``scores_b`` are per-entity F0.5 aligned on the same entities
    (Series are aligned by index; arrays must already be aligned). Returns the
    observed delta, the (1 - alpha) percentile CI, and whether the CI excludes 0.
    This is the single definition of "noise" used by every gate in plan.md §4.
    """
    if isinstance(scores_a, pd.Series) and isinstance(scores_b, pd.Series):
        scores_a, scores_b = scores_a.align(scores_b, join="inner")
    d = np.asarray(scores_b, dtype=float) - np.asarray(scores_a, dtype=float)
    n = d.size
    if n == 0:
        raise ValueError("no entities to compare")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "n_entities": int(n),
    }


def k_bucket(k, top: int = 6) -> np.ndarray:
    """Map true match counts to buckets 0..top, where ``top`` means ``>= top``."""
    return np.minimum(np.asarray(k), top)


def summarize(scores: pd.DataFrame, strata: pd.DataFrame | None = None,
              by=("country", "k_bucket")) -> pd.DataFrame:
    """Macro-F0.5 overall and per stratum.

    ``scores`` comes from :func:`per_entity_scores` or :func:`oracle_scores`.
    ``strata`` is indexed by entity and holds the ``by`` columns (a missing
    ``k_bucket`` column is derived from ``scores['k']``). Returns one row per
    stratum plus an ``ALL`` row, with entity count and share of entities.
    """
    df = scores.copy()
    if strata is not None:
        df = df.join(strata, how="left")
    if "k_bucket" in by and "k_bucket" not in df.columns:
        df["k_bucket"] = k_bucket(df["k"])
    rows = [{"stratum": "ALL", "n": len(df), "share": 1.0, "f05": df["f05"].mean()}]
    for col in by:
        if col not in df.columns:
            continue
        for key, grp in df.groupby(col, sort=True):
            rows.append({"stratum": f"{col}={key}", "n": len(grp),
                         "share": len(grp) / len(df), "f05": grp["f05"].mean()})
    return pd.DataFrame(rows)
