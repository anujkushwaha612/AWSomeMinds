"""Candidate selection: turn a wide probe result into a small candidate set.

This is stage 3 of the blocking cascade described in ``blocking_strategy.md``.
It consumes a long *pair table* produced by the probe stage and applies four
label-free (or retrieval-evidence-only) filters, in this order:

1. **score floor** - drop pairs whose fused retrieval score is hopeless.
2. **adaptive record depth** - each S2/S3 record keeps between 0 and ``a_max``
   S1 candidates, chosen from its own top score and its top-1/top-2 margin.
   Confident records keep 1, hopeless records keep 0 (they are probably
   orphans: 26.0% of train records have no S1 parent), ambiguous records keep
   ``a_max``.
3. **reciprocal tiering** - a pair seen in both directions (the record ranks the
   S1 highly *and* the S1 ranks the record highly) is kept; a one-sided pair is
   kept only if its score clears a stricter bar. This is Reciprocal CNP from
   the meta-blocking literature, which trades a little recall for a large
   precision gain.
4. **capacity pruning** - train ground truth caps an S1 at <=5 S2 and <=6 S3
   matches, and caps a record at exactly 1 S1 parent. Those caps are applied as
   b-matching capacities on the candidate graph, by alternating "top-a per
   record" and "top-b per (S1, source)" passes until the assignment is stable.

A rescue pass then re-admits the single best edge of any record that steps 2-4
emptied, provided it clears ``rescue_floor``; this bounds the recall damage of
the aggressive steps.

The pair table is a DataFrame with these columns (extra columns are carried
through untouched):

=========== ==========================================================
``s1``      S1 entity id (any hashable dtype)
``rid``     S2/S3 record id
``src``     2 or 3
``score``   fused retrieval score, higher is better
``r_rec``   rank of this S1 in the record's probe list (0-based, -1 = absent)
``r_ent``   rank of this record in the S1's probe list (0-based, -1 = absent)
``n_views`` number of probes that produced this pair (>=1)
=========== ==========================================================

Every function is pure and vectorised; nothing here reads ground truth, so the
same code runs unchanged on test.
"""

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- group helpers


def group_rank(codes: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of each row inside its ``codes`` group, by descending ``score``.

    Ties break on the original row order, so the result is deterministic.
    """
    if len(codes) == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.lexsort((np.arange(len(codes)), -score, codes))
    c = codes[order]
    pos = np.arange(len(c))
    first = np.empty(len(c), dtype=bool)
    first[0] = True
    np.not_equal(c[1:], c[:-1], out=first[1:])
    start = np.maximum.accumulate(np.where(first, pos, -1))
    rank = np.empty(len(c), dtype=np.int32)
    rank[order] = pos - start
    return rank


def group_nth_score(codes: np.ndarray, score: np.ndarray, n_groups: int,
                    rank: np.ndarray, nth: int, fill: float = -np.inf) -> np.ndarray:
    """Score of the ``nth``-ranked row of every group, ``fill`` where absent.

    ``rank`` is the output of :func:`group_rank` for the same inputs, so the
    caller pays for the sort only once.
    """
    out = np.full(n_groups, fill, dtype=np.float64)
    sel = rank == nth
    out[codes[sel]] = score[sel]
    return out


# ---------------------------------------------------------------- policy object


@dataclass
class SelectPolicy:
    """Every knob of the selection stage. Swept by ``ber.blocking.frontier``.

    The defaults are the starting point argued for in ``blocking_strategy.md``;
    they are *not* tuned numbers. Tune ``a_max``, ``conf_*`` and ``abstain_score``
    first: they move |C| the most.
    """

    # 1. floor
    score_floor: float = 0.10

    # 2. adaptive record depth
    a_max: int = 3                 # max S1 candidates per S2/S3 record
    conf_score: float = 0.80       # top-1 this strong ...
    conf_margin: float = 0.15      # ... and this far ahead -> keep only top-1
    abstain_score: float = 0.20    # top-1 weaker than this -> keep nothing

    # 3. reciprocal tiering
    recip_r_rec: int = 2           # "record ranks the S1 in its top-r_rec"
    recip_r_ent: int = 10          # "S1 ranks the record in its top-r_ent"
    one_sided_score: float = 0.45  # a one-sided pair needs at least this score
    one_sided_views: int = 2       # ... or agreement from this many probes

    # 4. capacity pruning (from train GT: <=5 S2 and <=6 S3 per S1)
    cap_s2: int = 6
    cap_s3: int = 7
    cap_rounds: int = 3

    # 5. rescue
    rescue_floor: float = 0.25

    def replace(self, **kw) -> "SelectPolicy":
        """Copy of the policy with ``kw`` overridden (for sweeps)."""
        return SelectPolicy(**{**asdict(self), **kw})

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "SelectPolicy":
        """Build the policy from the ``select:`` block of ``configs/pipeline.yaml``.

        Unknown keys are ignored so the config can carry documentation entries
        the code does not consume yet.
        """
        if cfg is None:
            from ..config import load_config
            cfg = load_config().get("select", {})
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (cfg or {}).items() if k in fields})


# ---------------------------------------------------------------- stages


def apply_floor(pairs: pd.DataFrame, p: SelectPolicy) -> np.ndarray:
    """Boolean keep-mask for stage 1 (absolute score floor)."""
    return pairs["score"].to_numpy() >= p.score_floor


def record_depth(pairs: pd.DataFrame, p: SelectPolicy,
                 keep: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Stage 2: per-record adaptive depth.

    Returns the updated keep-mask and a per-record diagnostic frame with the
    chosen depth, so the audit can report how often the blocker was confident,
    ambiguous or abstained.
    """
    rec_codes, rec_uniq = pd.factorize(pairs["rid"], sort=False)
    score = np.where(keep, pairs["score"].to_numpy(dtype=np.float64), -np.inf)
    rank = group_rank(rec_codes, score)
    n = len(rec_uniq)
    top1 = group_nth_score(rec_codes, score, n, rank, 0)
    top2 = group_nth_score(rec_codes, score, n, rank, 1, fill=0.0)

    margin = top1 - np.maximum(top2, 0.0)
    depth = np.full(n, p.a_max, dtype=np.int32)
    depth[(top1 >= p.conf_score) & (margin >= p.conf_margin)] = 1
    depth[top1 < p.abstain_score] = 0

    keep = keep & (rank < depth[rec_codes])
    diag = pd.DataFrame({"rid": rec_uniq, "top1": top1, "margin": margin, "depth": depth})
    return keep, diag


def reciprocal_tier(pairs: pd.DataFrame, p: SelectPolicy) -> np.ndarray:
    """Tier of every pair: 0 = reciprocal, 1 = one-sided but strong, 2 = weak.

    Tier 2 pairs are dropped by :func:`select` (subject to the rescue pass).
    A pair with unknown reverse rank (``r_ent`` = -1, e.g. the S1-side probe was
    not run) is treated as one-sided rather than reciprocal.
    """
    r_rec = pairs["r_rec"].to_numpy()
    r_ent = pairs["r_ent"].to_numpy()
    score = pairs["score"].to_numpy(dtype=np.float64)
    views = pairs["n_views"].to_numpy()

    recip = (r_rec >= 0) & (r_rec < p.recip_r_rec) & (r_ent >= 0) & (r_ent < p.recip_r_ent)
    strong = (score >= p.one_sided_score) | (views >= p.one_sided_views)
    return np.where(recip, 0, np.where(strong, 1, 2)).astype(np.int8)


def capacity_prune(pairs: pd.DataFrame, p: SelectPolicy, keep: np.ndarray) -> np.ndarray:
    """Stage 4: alternating b-matching against the measured degree caps.

    Each round keeps the top ``a_max`` surviving edges per record and then the
    top ``cap_s{2,3}`` per (S1, source). The two passes interact - dropping an
    edge on one side frees capacity on the other - so the rounds are repeated
    until the mask stops changing (``cap_rounds`` is the cap on iterations).

    This is the cheap deterministic stand-in for a maximum-weight degree-
    constrained bipartite matching; on this graph the edges are extremely
    skewed in score, so the greedy assignment and the alternating one agree
    almost everywhere.
    """
    rec_codes = pd.factorize(pairs["rid"], sort=False)[0]
    ent_codes = pd.factorize(
        pd.Series(pairs["s1"].astype(str) + "\x00" + pairs["src"].astype(str)), sort=False)[0]
    raw = pairs["score"].to_numpy(dtype=np.float64)
    cap_by_src = np.where(pairs["src"].to_numpy() == 2, p.cap_s2, p.cap_s3)

    for _ in range(max(1, p.cap_rounds)):
        before = keep.copy()
        score = np.where(keep, raw, -np.inf)
        keep &= group_rank(rec_codes, score) < p.a_max
        score = np.where(keep, raw, -np.inf)
        keep &= group_rank(ent_codes, score) < cap_by_src
        if np.array_equal(keep, before):
            break
    return keep


def rescue(pairs: pd.DataFrame, p: SelectPolicy, keep: np.ndarray,
           eligible: np.ndarray) -> np.ndarray:
    """Re-admit the best ``eligible`` edge of every record left with none.

    ``eligible`` is the mask of pairs that survived the score floor; the rescue
    never resurrects a pair below ``rescue_floor``. This bounds how much recall
    the aggressive stages can cost on records whose evidence is merely weak
    rather than absent.
    """
    rec_codes = pd.factorize(pairs["rid"], sort=False)[0]
    n_rec = int(rec_codes.max()) + 1 if len(rec_codes) else 0
    has = np.zeros(n_rec, dtype=bool)
    np.logical_or.at(has, rec_codes, keep)

    cand = eligible & ~has[rec_codes] & (pairs["score"].to_numpy() >= p.rescue_floor)
    score = np.where(cand, pairs["score"].to_numpy(dtype=np.float64), -np.inf)
    return keep | (cand & (group_rank(rec_codes, score) == 0))


# ---------------------------------------------------------------- driver


def select(pairs: pd.DataFrame, policy: SelectPolicy | None = None,
           return_diagnostics: bool = False):
    """Run the whole selection cascade and return the surviving pairs.

    The output carries a ``tier`` column (0 reciprocal / 1 one-sided / 2 rescued)
    so downstream features and the blocking audit can use it.
    """
    p = policy or SelectPolicy()
    if pairs.empty:
        out = pairs.assign(tier=np.array([], dtype=np.int8))
        return (out, {}) if return_diagnostics else out

    pairs = pairs.reset_index(drop=True)
    eligible = apply_floor(pairs, p)
    keep, depth_diag = record_depth(pairs, p, eligible)

    tier = reciprocal_tier(pairs, p)
    keep &= tier < 2

    keep = capacity_prune(pairs, p, keep)
    before_rescue = keep.copy()
    keep = rescue(pairs, p, keep, eligible)

    # the rescue ignores entity capacity, so re-apply it once with rescued edges
    # ranked behind regular ones. |C| per (S1, source) is then hard-bounded by
    # cap_s{2,3}, which is what keeps C_per_s1_max at cap_s2 + cap_s3.
    if keep.sum() > before_rescue.sum():
        ent_codes = pd.factorize(
            pd.Series(pairs["s1"].astype(str) + "\x00" + pairs["src"].astype(str)),
            sort=False)[0]
        cap_by_src = np.where(pairs["src"].to_numpy() == 2, p.cap_s2, p.cap_s3)
        prio = np.where(keep, pairs["score"].to_numpy(dtype=np.float64)
                        + 10.0 * before_rescue, -np.inf)
        keep &= group_rank(ent_codes, prio) < cap_by_src

    out = pairs.loc[keep].copy()
    out["tier"] = np.where(before_rescue[keep], tier[keep], 2).astype(np.int8)
    out = out.reset_index(drop=True)
    if not return_diagnostics:
        return out

    diag = {
        "n_in": int(len(pairs)),
        "n_after_floor": int(eligible.sum()),
        "n_out": int(len(out)),
        "records_in": int(pairs["rid"].nunique()),
        "records_confident": int((depth_diag["depth"] == 1).sum()),
        "records_abstained": int((depth_diag["depth"] == 0).sum()),
        "records_ambiguous": int((depth_diag["depth"] > 1).sum()),
        "n_rescued": int(len(out) - before_rescue.sum()),
        "tier_counts": out["tier"].value_counts().sort_index().to_dict(),
    }
    return out, diag
