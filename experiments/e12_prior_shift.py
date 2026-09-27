"""E12: prior-shift (label-shift) correction of stage-2 p for test's higher orphan rate, and a probe submission.

E7a: ~40% of test records have no S1 parent vs ~29% on train. A pair is a match only if its record has a parent
(h) and this S1 is it; only the first factor shifts. Per record h = min(1, sum of its candidates' p2);
p' = p * r / (h r + 1 - h), with r = test / train odds of "record has a parent" per country and source.
France (no train): orphan share from the invariant 1 - 3.46 / (records per S1).
Reports the fold-0 cost of the correction under the TRAIN prior (where it is not needed) for several strengths,
then writes test submissions for the requested strengths.
Run:  BER_CONFIG=artifacts/free/pipeline.yaml python experiments/e12_prior_shift.py 1.0 0.5
      (arguments: strength multipliers on log r; 1.0 = the measured shift, 0.5 = half of it)
"""
import json
import sys

import numpy as np
import pandas as pd

from ber import v5 as V
from ber.baseline import EntityScorer
from ber.config import artifact_path
from ber.store import split_countries

TAG = "noce"
TRAIN_ORPHAN = {("India", 2): 0.308, ("India", 3): 0.303, ("US", 2): 0.291, ("US", 3): 0.276}
E7 = json.load(open(artifact_path("experiments", "E7_test_orphans.json")))


def test_orphan_share(country: str, src: int) -> float:
    if country == "France":
        return 1 - 3.46 / 5.53
    return E7[f"{country}/S{src}"]["test_unmatched_share_by_score"]


def train_orphan_share(country: str, src: int) -> float:
    return TRAIN_ORPHAN.get((country, src), float(np.mean(list(TRAIN_ORPHAN.values()))))


def log_r(countries: list[str], keys: pd.DataFrame) -> np.ndarray:
    """Per pair: log of (test odds / train odds) of 'record has a parent', by country and source."""
    out = np.zeros(len(keys))
    for code, c in enumerate(countries):
        for src in (2, 3):
            m = ((keys["country"] == code) & (keys["src"] == src)).to_numpy()
            a_t, a_tr = test_orphan_share(c, src), train_orphan_share(c, src)
            out[m] = np.log(((1 - a_t) / a_t) / ((1 - a_tr) / a_tr))
    return out


def shift(keys: pd.DataFrame, p: np.ndarray, lr: np.ndarray, strength: float) -> np.ndarray:
    """p' = p * r / (h r + 1 - h), h = min(1, record's sum of p), r = exp(strength * log r)."""
    r = np.exp(strength * lr)
    rec = (keys["country"].to_numpy(np.int64) << 40) | (keys["src"].to_numpy(np.int64) << 32) | keys["doc_row"].to_numpy(np.int64)
    h = np.minimum(1.0, pd.Series(p).groupby(rec).transform("sum").to_numpy())
    return (p * r / (h * r + 1 - h)).astype(np.float32)


strengths = [float(x) for x in sys.argv[1:]] or [1.0]
res = json.load(open(V.run_path(f"stage2_{TAG}.json")))
rules = res["rules"]
keys, ents = V.read_keys("train")
kk = keys[np.load(V.run_path(f"kept_train_{TAG}.npy"))].reset_index(drop=True)
p2 = np.load(V.run_path(f"p2_train_{TAG}.npy"))
rep = (ents["fold"] == 0).to_numpy()
sc = EntityScorer(ents, kk, rep)
lr_train = log_r(split_countries("train"), kk)
print("fold-0 macro F0.5 under the TRAIN prior (cost of shifting where no shift is needed):")
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    k = V.apply_rule(kk, shift(kk, p2, lr_train, s) if s else p2, rules)
    print(f"  strength {s:4.2f}: {sc.per_entity(k).mean():.5f}  | predicted per S1 {k[sc.pairs].sum() / rep.sum():.3f}")

tkeys, _ = V.read_keys("test")
tk = tkeys[np.load(V.run_path(f"kept_test_{TAG}.npy"))].reset_index(drop=True)
p2t = np.load(V.run_path(f"p2_test_{TAG}.npy"))
lr_test = log_r(split_countries("test"), tk)
print("test r by country/source:", {f"{c}/S{s}": round(float(np.exp(log_r([c], pd.DataFrame({"country": [0], "src": [s]}))[0])), 3)
                                     for c in split_countries("test") for s in (2, 3)})
from ber.submit import finalize_submission
base_keep = V.apply_rule(tk, p2t, rules)
for s in strengths:
    keep = V.apply_rule(tk, shift(tk, p2t, lr_test, s), rules)
    name = f"sub_v5_prior{int(round(s * 100)):03d}"
    print(f"[{name}] test matches {int(keep.sum()):,} vs {int(base_keep.sum()):,} unshifted "
          f"({keep.sum() / base_keep.sum() - 1:+.2%}); removed {int((base_keep & ~keep).sum()):,}, added {int((keep & ~base_keep).sum()):,}")
    mpath, cpath, n_m, n_c = V.write_outputs(tk, keep, name)
    finalize_submission(name, mpath, cpath, offline_metrics=res["report"], n_match_pairs=n_m, n_candidate_pairs=n_c,
                        notes=f"E12 probe: sub_v5_noce with prior-shift correction, strength {s} (r from E7a orphan shares)")
