"""Hyperparameter tuning for candidate generation, driven by measurement.

The probe is expensive and the selection cascade is cheap, so
``ber.blocking.run`` caches the raw pair table per shard and everything here
re-runs **stage 3 only**. One probe pass, then hundreds of policies per minute.

Three subcommands, in the order they should be run:

``thresholds``
    The measurement that has to happen before any threshold means anything.
    For every record it recovers where its true S1 parent sits in the probe
    list, then reports the top-1 score distribution split by *has a parent* vs
    *orphan* (26% of records). From that it reads off:
      * ``abstain_score`` - how many records can be declined, and what share of
        true pairs each threshold destroys;
      * ``conf_score`` / ``conf_margin`` - the region where top-1 is the true
        parent often enough to stop looking at the runner-up.
    The defaults in ``configs/pipeline.yaml`` are placeholders until this runs.

``grid``
    Cartesian sweep of selection knobs over the cached pairs, each scored on
    oracle macro-F0.5, pairs completeness, and candidates per S1 *and* per
    record. Ends with the operating point: the smallest candidate set whose
    oracle F0.5 is statistically tied with the best (paired bootstrap).

``dfsweep``
    The one sweep that needs re-probing: ``max_index_df`` and ``sketch_terms``
    trade retrieval runtime against recall.

Run:
    python -m ber.blocking.tune thresholds --country India
    python -m ber.blocking.tune grid --country India --knobs a_max=1,2,3 abstain_score=0.0,0.2,0.3
    python -m ber.blocking.tune dfsweep --country India --caps 5000,20000,50000,0
"""

import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from ..config import artifact_path, ensure_parent, load_config
from ..io import load_source, load_truth_pairs
from . import metrics as bm
from .select import SelectPolicy, group_rank, select

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


# ---------------------------------------------------------------- loading


def load_pairs(split: str, country: str | None = None) -> pd.DataFrame:
    """Load the cached probe tables written by ``ber.blocking.run``."""
    pat = artifact_path("blocking", split, f"pairs_{country or '*'}_S*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(
            f"no cached pair tables at {pat}. Run `python -m ber.blocking.run "
            f"--split {split}` first (it caches them as a side effect).")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def entity_index(split: str, country: str | None = None) -> pd.Series:
    """S1 ids of the split (optionally one country), in file order."""
    s1 = load_source(split, 1)
    if country:
        s1 = s1[s1["country"] == country]
    return s1["entity_id"]


# ---------------------------------------------------------------- thresholds


def record_view(pairs: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """One row per probed record: its top-1 evidence and where its parent sits.

    Columns: ``rid``, ``top1``, ``top2``, ``margin``, ``has_parent`` (the record
    appears in ground truth at all), ``parent_rank`` (rank of the true parent in
    the record's probe list, -1 if the probe missed it), ``parent_is_top1``.
    """
    p = pairs.reset_index(drop=True)
    codes, rids = pd.factorize(p["rid"], sort=False)
    score = p["score"].to_numpy(dtype=np.float64)
    rank = group_rank(codes, score)
    n = len(rids)

    top1 = np.full(n, -np.inf)
    top1[codes[rank == 0]] = score[rank == 0]
    top2 = np.zeros(n)
    top2[codes[rank == 1]] = score[rank == 1]

    parent = truth.set_index("rid")["s1"]
    parent_of = pd.Series(rids).map(parent)
    is_parent = (p["s1"].to_numpy() == parent_of.to_numpy()[codes])
    prank = np.full(n, -1, dtype=np.int32)
    prank[codes[is_parent]] = rank[is_parent]

    return pd.DataFrame({
        "rid": rids, "top1": top1, "top2": top2, "margin": top1 - top2,
        "has_parent": parent_of.notna().to_numpy(),
        "parent_rank": prank, "parent_is_top1": prank == 0,
    })


def abstain_table(rv: pd.DataFrame, grid=None) -> pd.DataFrame:
    """What each ``abstain_score`` buys and costs.

    ``records_declined`` is the size win (each declined record removes its whole
    candidate list); ``true_pairs_lost`` is the recall bill - the share of
    retrievable true pairs that sit in a record the threshold silences.
    """
    grid = grid if grid is not None else np.round(np.arange(0.0, 0.75, 0.05), 2)
    retrievable = (rv["parent_rank"] >= 0).sum()
    rows = []
    for t in grid:
        silenced = rv["top1"] < t
        rows.append({
            "abstain_score": float(t),
            "records_declined": round(float(silenced.mean()), 4),
            "declined_that_are_orphans": round(
                float((silenced & ~rv["has_parent"]).sum() / max(1, silenced.sum())), 4),
            "orphans_caught": round(
                float((silenced & ~rv["has_parent"]).sum()
                      / max(1, (~rv["has_parent"]).sum())), 4),
            "true_pairs_lost": round(
                float((silenced & (rv["parent_rank"] >= 0)).sum() / max(1, retrievable)), 5),
        })
    return pd.DataFrame(rows)


def confidence_table(rv: pd.DataFrame, scores=None, margins=None) -> pd.DataFrame:
    """Where "trust top-1 alone" is safe.

    ``precision`` is P(top-1 is the true parent | the record is called
    confident and has a parent). ``records_confident`` is the share of records
    that would drop from ``a_max`` candidates to 1.
    """
    scores = scores if scores is not None else [0.5, 0.6, 0.7, 0.8, 0.9]
    margins = margins if margins is not None else [0.0, 0.05, 0.10, 0.20]
    rows = []
    for s, m in itertools.product(scores, margins):
        conf = (rv["top1"] >= s) & (rv["margin"] >= m)
        withp = conf & rv["has_parent"]
        rows.append({
            "conf_score": s, "conf_margin": m,
            "records_confident": round(float(conf.mean()), 4),
            "precision": round(float(rv.loc[withp, "parent_is_top1"].mean()), 4)
            if withp.any() else float("nan"),
            "true_pairs_lost": round(float(
                (conf & (rv["parent_rank"] > 0)).sum()
                / max(1, (rv["parent_rank"] >= 0).sum())), 5),
        })
    return pd.DataFrame(rows)


def cmd_thresholds(args) -> dict:
    """Run the pre-tuning measurement and print the three tables."""
    pairs = load_pairs("train", args.country)
    truth = load_truth_pairs()
    rv = record_view(pairs, truth)

    probed = len(rv)
    reach = (rv["parent_rank"] >= 0).sum()
    haveparent = rv["has_parent"].sum()
    summary = {
        "records_probed": int(probed),
        "records_with_a_parent": int(haveparent),
        "orphan_share": round(1 - haveparent / probed, 4),
        "parent_retrieved_at_all": round(reach / max(1, haveparent), 4),
        "parent_is_top1": round(rv["parent_is_top1"].sum() / max(1, haveparent), 4),
        "probe_ceiling_pair_recall": round(reach / max(1, haveparent), 4),
    }
    print("\n== record-level probe summary")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    dist = pd.DataFrame({
        "quantile": q,
        "top1 | has parent": rv.loc[rv["has_parent"], "top1"].quantile(q).to_numpy().round(4),
        "top1 | orphan": rv.loc[~rv["has_parent"], "top1"].quantile(q).to_numpy().round(4),
        "margin | has parent": rv.loc[rv["has_parent"], "margin"].quantile(q).to_numpy().round(4),
        "margin | orphan": rv.loc[~rv["has_parent"], "margin"].quantile(q).to_numpy().round(4),
    })
    print("\n== top-1 score distribution (this is what sets every threshold)")
    print(dist.to_string(index=False))

    ab = abstain_table(rv)
    print("\n== abstain_score: size win vs recall bill")
    print(ab.to_string(index=False))

    cf = confidence_table(rv)
    print("\n== conf_score x conf_margin: where top-1 can be trusted alone")
    print(cf.to_string(index=False))

    print("\n== parent rank histogram (0 = top-1; -1 = probe missed it entirely)")
    print(rv.loc[rv["has_parent"], "parent_rank"].value_counts().sort_index()
          .rename("records").to_frame().T.to_string())

    return {"summary": summary, "distribution": dist.to_dict("records"),
            "abstain": ab.to_dict("records"), "confidence": cf.to_dict("records")}


# ---------------------------------------------------------------- grid


def parse_knobs(specs) -> dict:
    """``["a_max=1,2,3", "cap_s2=5,6"]`` -> ``{"a_max": [1,2,3], "cap_s2": [5,6]}``."""
    fields = SelectPolicy.__dataclass_fields__
    out = {}
    for spec in specs:
        key, raw = spec.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown knob {key!r}; valid: {sorted(fields)}")
        # dataclass annotations are real types here (no `from __future__`)
        cast = int if fields[key].type in (int, "int") else float
        out[key] = [cast(v) for v in raw.split(",")]
    return out


def cmd_grid(args) -> dict:
    """Sweep selection knobs over the cached pairs and pick the operating point."""
    pairs = load_pairs("train", args.country)
    truth = load_truth_pairs()
    entities = entity_index("train", args.country)
    records = pd.concat([load_source("train", s) for s in (2, 3)])
    s1 = load_source("train", 1)
    if args.country:
        s1 = s1[s1["country"] == args.country]
        records = records[records["country"] == args.country]
    universe = bm.universe_sizes(s1, records)

    base = SelectPolicy.from_config()
    knobs = parse_knobs(args.knobs)
    policies, rows = {}, []
    for combo in itertools.product(*knobs.values()):
        kw = dict(zip(knobs, combo))
        name = " ".join(f"{k}={v}" for k, v in kw.items())
        pol = base.replace(**kw)
        policies[name] = pol
        t0 = time.time()
        cand = select(pairs, pol)
        rows.append({**kw, **bm.report(cand, truth, entities, universe, name=name),
                     "seconds": round(time.time() - t0, 1)})

    cols = ["config", "oracle_f05", "pairs_completeness", "pairs_quality",
            "C_per_s1_mean", "C_per_s1_p99", "C_per_s1_max", "share_s1_empty",
            "C_per_record_mean", "RR_vs_country", "seconds"]
    table = pd.DataFrame(rows)[cols].sort_values("C_per_s1_mean")
    print("\n== grid (sorted by candidate-set size)")
    print(table.round(5).to_string(index=False))

    print("\n== operating point: smallest candidate set tied with the best oracle F0.5")
    op = bm.pick_operating_point(pairs, truth, entities, policies,
                                 n_resamples=args.bootstrap,
                                 materiality=args.materiality)
    keep = ["config", "oracle_f05", "C_per_s1_mean", "delta_vs_best",
            "ci_low", "ci_high", "tied_with_best"]
    print(op[keep].round(5).to_string(index=False))
    winner = op[op["tied_with_best"]].iloc[0]
    print(f"\n  -> choose: {winner['config']}  "
          f"({winner['C_per_s1_mean']:.2f} candidates/S1, "
          f"oracle F0.5 {winner['oracle_f05']:.5f})")
    return {"grid": table.to_dict("records"), "operating_point": op.to_dict("records"),
            "winner": str(winner["config"])}


# ---------------------------------------------------------------- df sweep


def cmd_dfsweep(args) -> dict:
    """Re-probe at several ``max_index_df`` values and compare cost to recall."""
    from .run import run

    truth = load_truth_pairs()
    entities = entity_index("train", args.country)
    rows = []
    for cap in [float(c) for c in args.caps.split(",")]:
        # load_config is lru_cached and returns the live dict, so mutating it
        # here is what re-parameterises the probe inside run()
        load_config()["blocking"]["max_index_df"] = cap if cap else 0
        t0 = time.time()
        cand, rep, _ = run("train", [args.country] if args.country else None,
                           save_pairs=False)
        rows.append({"max_index_df": cap, "probe_seconds": round(time.time() - t0, 1),
                     **{k: v for k, v in bm.report(cand, truth, entities).items()
                        if k in ("oracle_f05", "pairs_completeness", "C_per_s1_mean")}})
        print(f"  cap={cap}: {rows[-1]}", flush=True)
    table = pd.DataFrame(rows)
    print("\n== max_index_df sweep")
    print(table.round(5).to_string(index=False))
    return {"dfsweep": table.to_dict("records")}


# ---------------------------------------------------------------- CLI


def main() -> None:
    """CLI dispatcher; every subcommand also writes its tables to artifacts/tune/."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("thresholds", help="measure the score distributions first")
    t.add_argument("--country", default=None)

    g = sub.add_parser("grid", help="sweep selection knobs over cached pairs")
    g.add_argument("--country", default=None)
    g.add_argument("--knobs", nargs="+", default=["a_max=1,2,3"])
    g.add_argument("--bootstrap", type=int, default=400)
    g.add_argument("--materiality", type=float, default=None,
                   help="oracle-F0.5 deficit treated as noise "
                        "(default: bootstrap.materiality from the config)")

    d = sub.add_parser("dfsweep", help="re-probe at several n-gram purge caps")
    d.add_argument("--country", default=None)
    d.add_argument("--caps", default="5000,20000,50000,0")

    args = ap.parse_args()
    result = {"thresholds": cmd_thresholds, "grid": cmd_grid,
              "dfsweep": cmd_dfsweep}[args.cmd](args)

    path = artifact_path("tune", f"{args.cmd}_{args.country or 'all'}.json")
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nwrote {path}  <- paste this file back for the next round")


if __name__ == "__main__":
    main()
