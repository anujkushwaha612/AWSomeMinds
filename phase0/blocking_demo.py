"""End-to-end run of the blocking cascade on the synthetic split.

This is the regression harness for ``blocking_strategy.md``: it runs the real
pipeline code (``ber.normalize`` -> ``ber.blocking.probe`` -> ``ber.blocking.select``
-> ``ber.blocking.metrics``) over ``phase0.simulate`` data and prints the same
scorecard the methodology document has to report on the real files.

It answers two questions that do not need the real data to be meaningful:

1. does the cascade execute correctly end to end, and
2. does the size/recall mechanism behave as the strategy claims - i.e. does the
   record-side cascade sit far below the entity-side top-K baseline on
   candidates per S1 at comparable oracle F0.5?

It does **not** tell you what recall to expect on the challenge data. The noise
model is invented; only the structural marginals are calibrated.

Run:  python -m phase0.blocking_demo --scale 0.004
"""

import argparse
import time

import numpy as np
import pandas as pd

from ber.blocking import metrics as bm
from ber.blocking.probe import ProbeIndex, pair_table
from ber.blocking.select import SelectPolicy, select
from ber.normalize import normalize_frame
from phase0.simulate import generate, profile

VIEWS = {
    "NA": lambda d: (d["name_legal"] + " " + d["addr_n"]).str.strip(),
    "A": lambda d: d["addr_n"],
    "N": lambda d: d["name_legal"],
}


def build_pairs(data: dict, k_fwd: int = 10, k_rev: int = 50, views=("NA", "A", "N"),
                max_index_df: float = 0.02, sketch_terms: int = 24) -> tuple:
    """Normalise, index S1 per country and probe every S2/S3 record against it.

    Returns ``(pairs, rev_lists, timings)`` where ``pairs`` is the long pair
    table for :func:`ber.blocking.select.select` and ``rev_lists`` holds the
    S1 -> record top-``k_rev`` lists used to build the entity-side baseline.
    """
    norm = {s: normalize_frame(data[f"s{s}"] if s > 1 else data["s1"]) for s in (1, 2, 3)}
    out, rev_out, timings = [], [], {}

    for country in sorted(norm[1]["country"].unique()):
        tab = {s: norm[s][norm[s]["country"] == country].reset_index(drop=True)
               for s in (1, 2, 3)}
        corpus = pd.concat([VIEWS["NA"](tab[s]) for s in (1, 2, 3)], ignore_index=True)

        indices = {}
        for v in views:
            t0 = time.time()
            indices[v] = ProbeIndex(max_index_df=max_index_df,
                                    sketch_terms=sketch_terms).fit(VIEWS[v](tab[1]), corpus)
            timings.setdefault(f"fit:{country}:{v}", 0.0)
            timings[f"fit:{country}:{v}"] += time.time() - t0

        for src in (2, 3):
            probes, rev = {}, {}
            for v in views:
                t0 = time.time()
                probes[v] = indices[v].query(VIEWS[v](tab[src]), k_fwd)
                timings[f"fwd:{country}:S{src}:{v}"] = time.time() - t0
            # reverse direction only on the backbone view: it supplies r_ent for
            # the reciprocal filter, and the entity-side baseline for comparison
            t0 = time.time()
            rev["NA"] = indices["NA"].query_from_index(
                k_rev, indices["NA"].transform(VIEWS["NA"](tab[src])))
            timings[f"rev:{country}:S{src}:NA"] = time.time() - t0

            out.append(pair_table(tab[1]["entity_id"].to_numpy(),
                                  tab[src]["entity_id"].to_numpy(), src, probes, rev))
            idx, score = rev["NA"]
            rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
            ok = idx.ravel() >= 0
            rev_out.append(pd.DataFrame({
                "s1": tab[1]["entity_id"].to_numpy()[rows[ok]],
                "rid": tab[src]["entity_id"].to_numpy()[idx.ravel()[ok]],
                "rank": np.tile(np.arange(idx.shape[1]), idx.shape[0])[ok],
                "score": score.ravel()[ok]}))

        for v, ix in indices.items():
            timings[f"index:{country}:{v}"] = ix.stats

    return (pd.concat(out, ignore_index=True), pd.concat(rev_out, ignore_index=True), timings)


def df_cap_sweep(data: dict, caps=(0.02, 0.05, 0.20, 1.0), policy=None) -> pd.DataFrame:
    """Cost/benefit of n-gram purging: posting-list cap vs work vs oracle F0.5.

    ``max_index_df`` is the single knob that decides whether retrieval at 10M
    scale is hours or minutes, because query cost is the sum of the posting-list
    lengths of the query's n-grams. This table is how it gets chosen on the real
    data: pick the smallest cap whose oracle F0.5 is still tied with cap = 1.0.
    """
    entities, truth = data["s1"]["entity_id"], data["truth"]
    universe = bm.universe_sizes(data["s1"], pd.concat([data["s2"], data["s3"]]))
    rows = []
    for cap in caps:
        t0 = time.time()
        pairs, _, timings = build_pairs(data, views=("NA", "A"), max_index_df=cap)
        stats = [v for k, v in timings.items() if k.startswith("index:") and k.endswith(":NA")]
        rows.append({
            "max_index_df": cap,
            "postings_kept_pct": round(100 * sum(s["postings_after"] for s in stats)
                                       / sum(s["postings_before"] for s in stats), 1),
            "longest_posting_list": max(s["max_posting_list"] for s in stats),
            "probe_seconds": round(time.time() - t0, 1),
            **{k: v for k, v in bm.report(select(pairs, policy or SelectPolicy()), truth,
                                          entities, universe).items()
               if k in ("oracle_f05", "pairs_completeness", "C_per_s1_mean")},
        })
    return pd.DataFrame(rows)


def main() -> None:
    """CLI: generate, probe, sweep the selection policy and print the scorecard."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", type=float, default=0.004)
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-index-df", type=float, default=0.20)
    ap.add_argument("--df-sweep", action="store_true")
    args = ap.parse_args()

    pd.set_option("display.width", 200, "display.max_columns", 40)
    t0 = time.time()
    data = generate(args.scale, args.seed, args.k_scale)
    print(profile(data).to_string(index=False))

    if args.df_sweep:
        print("\n== n-gram purging sweep (view NA+A)")
        print(df_cap_sweep(data).round(5).to_string(index=False))

    pairs, rev, timings = build_pairs(data, max_index_df=args.max_index_df)
    print(f"\nprobe: {len(pairs):,} raw pairs in {time.time() - t0:.0f}s")
    for k, v in timings.items():
        if k.startswith("index:"):
            print(f"  {k}: {v}")

    entities = data["s1"]["entity_id"]
    truth = data["truth"]
    universe = bm.universe_sizes(data["s1"], pd.concat([data["s2"], data["s3"]]))
    cols = ["config", "oracle_f05", "pairs_completeness", "pairs_quality",
            "C_per_s1_mean", "C_per_s1_p90", "C_per_s1_max", "share_s1_empty",
            "C_per_record_mean", "RR_vs_country"]

    rows = [bm.report(rev[rev["rank"] < K], truth, entities, universe,
                      name=f"baseline entity-side top-{K}/source") for K in (5, 10, 20, 50)]
    rows.append(bm.report(pairs, truth, entities, universe, name="raw probe union"))

    base = SelectPolicy()
    for label, pol in {
        "cascade a_max=3": base,
        "cascade a_max=2": base.replace(a_max=2),
        "cascade a_max=1": base.replace(a_max=1),
        "  ablation: no capacity cap": base.replace(a_max=2, cap_s2=999, cap_s3=999),
        "  ablation: no adaptive depth": base.replace(a_max=2, conf_score=2.0),
        "  ablation: no reciprocal tier": base.replace(a_max=2, one_sided_score=0.0),
        "  ablation: no rescue": base.replace(a_max=2, rescue_floor=2.0),
    }.items():
        rows.append(bm.report(select(pairs, pol), truth, entities, universe, name=label))

    table = pd.DataFrame(rows)[cols].sort_values("C_per_s1_mean")
    print("\n== blocking scorecard (SYNTHETIC DATA - mechanics only, not real recall)")
    print(table.round(5).to_string(index=False))

    cand, diag = select(pairs, base.replace(a_max=2), return_diagnostics=True)
    print("\n== selection diagnostics (cascade a_max=2)")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    strata = data["s1"].set_index("entity_id")[["country"]]
    print("\n== per stratum (cascade a_max=2)")
    print(bm.by_stratum(cand, truth, entities, strata).round(4).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
