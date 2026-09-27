"""E14: offline leaderboard simulator — fold-0 macro F0.5 at the TEST orphan rate.

Hypothesis: the offline -> LB gap comes from test's higher share of orphan records (~40% vs ~29%, E7a), so
false positives on orphan records cost more on test than on train. The simulator scores fold-0 entities with
every orphan-record false positive weighted by w = test / train orphan odds (per country and source):
    F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP_parented + w * FP_orphan)
If it reproduces the three leaderboard scores (0.938 / 0.957 / 0.980), it is an offline proxy for the LB.
Run:  python experiments/e14_lb_simulator.py baseline baseline_gen      (baseline-style runs)
      BER_CONFIG=artifacts/free/pipeline.yaml python experiments/e14_lb_simulator.py v5:noce   (v5 stage 2)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from ber.baseline import Decider, truth_parents
from ber.config import artifact_path
from ber.io import load_truth_pairs
from ber.store import CountryStore, split_countries

E7 = json.load(open(artifact_path("experiments", "E7_test_orphans.json")))
TRAIN_ORPHAN = {("India", 2): 0.308, ("India", 3): 0.303, ("US", 2): 0.291, ("US", 3): 0.276}


def orphan_weight(country: str, src: int) -> float:
    a_t = E7[f"{country}/S{src}"]["test_unmatched_share_by_score"]
    a_tr = TRAIN_ORPHAN[(country, src)]
    return (a_t / (1 - a_t)) / (a_tr / (1 - a_tr))


def orphan_flags(keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per pair: record is an orphan (no S1 parent anywhere), and its orphan weight."""
    truth = load_truth_pairs()
    orphan = np.zeros(len(keys), bool)
    w = np.ones(len(keys))
    for code, country in enumerate(split_countries("train")):
        st = CountryStore("train", country, cols=["entity_id"])
        par = truth_parents(st, truth)
        for src in (2, 3):
            m = ((keys["country"] == code) & (keys["src"] == src)).to_numpy()
            orphan[m] = par[src][keys.loc[m, "doc_row"].to_numpy()] < 0
            w[m] = orphan_weight(country, src)
        del st
    return orphan, w


def sim_scores(keys, ents, keep, orphan, w, rep_mask) -> dict:
    """Plain and test-orphan-rate macro F0.5 over the fold-0 entities."""
    e = ents[rep_mask]
    ekey = pd.Index((e["country"].to_numpy(np.int64) << 32) | e["s1_row"].to_numpy(np.int64))
    pk = (keys["country"].to_numpy(np.int64) << 32) | keys["s1_row"].to_numpy(np.int64)
    idx = ekey.get_indexer(pk)
    m = (idx >= 0) & keep
    lab = keys["label"].to_numpy().astype(bool)
    n = len(e)
    tp = np.bincount(idx[m & lab], minlength=n).astype(float)
    fpo = np.bincount(idx[m & ~lab & orphan], minlength=n).astype(float)
    fpw = np.bincount(idx[m & ~lab & orphan], weights=w[m & ~lab & orphan], minlength=n)
    fpn = np.bincount(idx[m & ~lab & ~orphan], minlength=n).astype(float)
    k = e["k"].to_numpy().astype(float)
    fn = k - tp

    def f(fp):
        den = 1.25 * tp + 0.25 * fn + fp
        return np.where(den > 0, 1.25 * tp / np.where(den > 0, den, 1), 1.0)

    return {"plain": float(f(fpn + fpo).mean()), "test_orphan_rate": float(f(fpn + fpw).mean()),
            "fp_orphan_share": float(fpo.sum() / max(fpo.sum() + fpn.sum(), 1))}


def baseline_run(run: str) -> dict:
    os.environ["BER_RUN"] = os.environ["BER_FEATURES"] = run
    from ber import baseline as B
    keys, ents = B.read_keys("train")
    d = json.load(open(artifact_path(run, "metrics.json")))["decision"]
    keep = Decider(keys, np.load(artifact_path(run, "oof.npy"))).keep(d["t_first"], d["t_rest"], d["arbitrate"])
    orphan, w = orphan_flags(keys)
    return sim_scores(keys, ents, keep, orphan, w, (ents["fold"] == 0).to_numpy())


def v5_run(tag: str, p_file: str | None = None, rules_file: str | None = None) -> dict:
    from ber import v5 as V
    keys, ents = V.read_keys("train")
    kk = keys[np.load(V.run_path(f"kept_train_{tag}.npy"))].reset_index(drop=True)
    p = np.load(V.run_path(p_file or f"p2_train_{tag}.npy"))
    rules = json.load(open(V.run_path(rules_file or f"stage2_{tag}.json")))["rules"]
    keep = V.apply_rule(kk, p, rules)
    orphan, w = orphan_flags(kk)
    return sim_scores(kk, ents, keep, orphan, w, (ents["fold"] == 0).to_numpy())


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        r = v5_run(arg.split(":", 1)[1]) if arg.startswith("v5:") else baseline_run(arg)
        print(f"{arg:16} plain fold-0 {r['plain']:.5f} | at test orphan rate {r['test_orphan_rate']:.5f} "
              f"| orphan share of false positives {r['fp_orphan_share']:.3f}", flush=True)
