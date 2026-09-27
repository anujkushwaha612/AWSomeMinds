"""E13: stage 3 = the stage-2 competition features recomputed from stage-2 p (iterative collective refinement).

Same pruned pairs, base features and folds as stage2_<tag>; STAGE2 features (record best / 2nd-best, S1 counts,
cross-source) now from the sharper out-of-fold p2 instead of p1. Cross-fitted LightGBM, decision tuned on the
tuning folds, fold-0 report (per_entity_stage3.npy), paired bootstrap vs stage 2, and a test submission.
Run:  BER_CONFIG=artifacts/free/pipeline.yaml python experiments/e13_stage3.py
"""
import gc
import json
import sys

import numpy as np

from ber import v5 as V
from ber.baseline import entity_key
from ber.eval.scorer import paired_bootstrap
from ber.submit import finalize_submission

TAG = sys.argv[1] if len(sys.argv) > 1 else "noce"
res2 = json.load(open(V.run_path(f"stage2_{TAG}.json")))
base = res2["base_features"]
v = V.vcfg()
keys, ents = V.read_keys("train")
kept = np.load(V.run_path(f"kept_train_{TAG}.npy"))
p2 = np.load(V.run_path(f"p2_train_{TAG}.npy"))
kk = keys[kept].reset_index(drop=True)
del keys
gc.collect()
y = kk["label"].to_numpy()
fold = V.pair_folds(kk, ents)
in_gbdt = np.isin(fold, v["gbdt_folds"])
ent_k = entity_key(kk["country"], kk["s1_row"])
names = base + V.STAGE2


def matrix(split, keys_kept, kept_mask, p):
    return np.hstack([V.read_matrix(split, kept_mask, base), V.stage2_features(keys_kept, p).to_numpy(np.float32)])


holder = [matrix("train", kk, kept, p2)[in_gbdt]]
gc.collect()
models = V.cross_fit(holder.pop(), y[in_gbdt], fold[in_gbdt], ent_k[in_gbdt], names, "stage3")
gc.collect()
p3 = V.predict_cross(models, matrix("train", kk, kept, p2), fold)
rules = V.tune_rules(kk, ents, p3, y, v["report_fold"])
keep = V.apply_rule(kk, p3, rules)
rep = V.report("stage3", kk, ents, keep)
np.save(V.run_path("p3_train.npy"), p3)
a, b = np.load(V.run_path(f"per_entity_stage2_{TAG}.npy")), np.load(V.run_path("per_entity_stage3.npy"))
bs = paired_bootstrap(a, b)
print(f"stage3 vs stage2_{TAG}: {b.mean():.5f} vs {a.mean():.5f}  delta {bs['delta']:+.5f}  CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]", flush=True)
json.dump({"base_features": base, "features": names, "rules": rules, "report": rep, "vs_stage2": bs},
          open(V.run_path("stage3.json"), "w"), indent=2, default=float)

# test
tkeys, _ = V.read_keys("test")
tkept = np.load(V.run_path(f"kept_test_{TAG}.npy"))
tk = tkeys[tkept].reset_index(drop=True)
del tkeys
p2t = np.load(V.run_path(f"p2_test_{TAG}.npy"))
p3t = V.predict_cross(models, matrix("test", tk, tkept, p2t), None)
np.save(V.run_path("p3_test.npy"), p3t)
keept = V.apply_rule(tk, p3t, rules)
mpath, cpath, n_m, n_c = V.write_outputs(tk, keept, "stage3")
finalize_submission("sub_v5_stage3", mpath, cpath, offline_metrics=rep, n_match_pairs=n_m, n_candidate_pairs=n_c,
                    notes=f"E13 stage 3: stage-2 competition features recomputed from stage-2 p ({TAG})")
