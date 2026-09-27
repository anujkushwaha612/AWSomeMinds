"""E17: deleted-S1 stress for stage 2 (experiments.md E17).

Test S1s were subsampled: the records of a deleted S1 become orphans, and when the deleted S1 had a near-duplicate
sibling (same name, nearby house number) its records latch onto the surviving sibling. Stage 2 leans on record-side
competition (p_minus_rec_other ~0.9 of gain), which in train is supplied by the sibling S1 itself; on test it is gone.

Simulation on train: delete a fraction of S1 entities (all folds) -> drop their pairs, recompute the stage-2
competition features on what is left (stage 1 is ~95% gap_s1, an S1-side feature that deletion leaves unchanged, so
p1 is kept). Consensus features of surviving S1s are unchanged (whole S1 groups are removed). Deleted entities get
fold -1, so they are neither trained on nor scored.

  python experiments/e17_stress.py eval --tags noce consr [--frac 0.19]
  python experiments/e17_stress.py train --tag stress [--cons consr] [--frac 0.19]
  python experiments/e17_stress.py predict --tag stress --name <sub>
"""
import argparse
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e16_consensus as E16  # noqa: E402
from ber import v5 as V  # noqa: E402
from ber.baseline import EntityScorer, entity_key  # noqa: E402
from ber.consensus import CONS_FEATURES  # noqa: E402
from ber.eval.scorer import paired_bootstrap  # noqa: E402

log = V.log
SALT = 1717


def universe(frac: float):
    keys, ents = V.read_keys("train")
    p1 = np.load(V.run_path("p1_train.npy"))
    kept = p1 >= V.vcfg()["prune_tau"]
    ents = ents.copy()
    removed = V.entity_uniform(entity_key(ents["country"], ents["s1_row"]), SALT) < frac
    orig_fold = ents["fold"].to_numpy().copy()
    ents.loc[removed, "fold"] = -1
    rem_pair = V.entity_uniform(entity_key(keys["country"], keys["s1_row"]), SALT) < frac
    return keys, ents, p1, kept, rem_pair, removed, orig_fold


def matrix(keys, p1, rows, base, cons_cols, cons_rows):
    kk = keys[rows].reset_index(drop=True)
    parts = [V.read_matrix("train", rows, base), V.stage2_features(kk, p1[rows]).to_numpy(np.float32)]
    if cons_cols:
        cons = np.load(V.run_path("cons_train.npy"), mmap_mode="r")
        parts.append(np.asarray(cons[np.flatnonzero(cons_rows)][:, cons_cols], dtype=np.float32))
    return kk, np.hstack(parts)


def cons_cols_of(names):
    return [CONS_FEATURES.index(f) for f in names if f in CONS_FEATURES]


def score(kk, ents, keep):
    m = (ents["fold"] == V.vcfg()["report_fold"]).to_numpy()
    sc = EntityScorer(ents, kk, m)
    f = sc.per_entity(keep)
    cn = ents.loc[m, "country_name"].to_numpy()
    return f, {c: round(float(f[cn == c].mean()), 5) for c in np.unique(cn)}


def evaluate(tags, frac):
    keys, ents, p1, kept, rem_pair, removed, orig_fold = universe(frac)
    rows_s = kept & ~rem_pair
    surv0 = ~removed[orig_fold == V.vcfg()["report_fold"]]     # fold-0 survivors, in per_entity_*.npy order
    out = {}
    for tag in tags:
        res = json.load(open(V.run_path(f"stage2{V.sfx(tag)}.json")))
        base = res.get("base_features")
        cc = cons_cols_of(res["features"]) if res.get("stage2_rel") else []
        kk, X = matrix(keys, p1, rows_s, base, cc, rem_pair[kept] == False)  # noqa: E712
        fold = V.pair_folds(kk, ents)
        p2 = V.predict_cross(V.load_models("stage2" + V.sfx(tag)), X, fold)
        del X
        gc.collect()
        keep = V.apply_rule(kk, p2, res["rules"])
        f, per = score(kk, ents, keep)
        f_un = np.load(V.run_path(f"per_entity_stage2{V.sfx(tag)}.npy"))[surv0]
        out[tag] = f
        c = kk["country"].to_numpy()
        log.info(f"[e17] {tag}: fold-0 survivors unstressed {f_un.mean():.5f} -> deleted {frac:.0%} {f.mean():.5f} "
                 f"({f.mean() - f_un.mean():+.5f}) {per}; pred/S1 {keep.sum() / len(ents[ents.fold >= 0]):.3f}")
        del kk, p2, keep
        gc.collect()
    tags = list(out)
    for a in range(len(tags)):
        for b in range(a + 1, len(tags)):
            bs = paired_bootstrap(out[tags[a]], out[tags[b]])
            log.info(f"[e17] stressed {tags[b]} vs {tags[a]}: {bs['delta']:+.5f} CI [{bs['ci_low']:+.5f}, "
                     f"{bs['ci_high']:+.5f}]")


def train(tag, cons_tag, frac):
    v = V.vcfg()
    keys, ents, p1, kept, rem_pair, removed, orig_fold = universe(frac)
    rows_s = kept & ~rem_pair
    base = V.stage1_features()
    feats = list(E16.SUBSETS[cons_tag]) if cons_tag else []
    cc = cons_cols_of(feats)
    names = base + V.STAGE2 + feats
    name = "stage2" + V.sfx(tag)
    kk, X = matrix(keys, p1, rows_s, base, cc, rem_pair[kept] == False)  # noqa: E712
    y = kk["label"].to_numpy()
    fold = V.pair_folds(kk, ents)
    in_gbdt = np.isin(fold, v["gbdt_folds"])
    ent_k = entity_key(kk["country"], kk["s1_row"])
    log.info(f"[{name}] deleted {frac:.0%} of S1s: {len(kk):,} pairs, matrix {X.shape}, consensus {feats} {V.mem_str()}")
    holder = [X[in_gbdt]]
    del X
    gc.collect()
    models = V.cross_fit(holder.pop(), y[in_gbdt], fold[in_gbdt], ent_k[in_gbdt], names, name)
    gc.collect()
    kk, X = matrix(keys, p1, rows_s, base, cc, rem_pair[kept] == False)  # noqa: E712
    p2 = V.predict_cross(models, X, fold)
    del X
    gc.collect()
    p2, rules, ens = V.choose_ensemble(kk, ents, {"lgb": p2}, y, v["report_fold"])
    keep = V.apply_rule(kk, p2, rules)
    rep = V.report(name, kk, ents, keep)                     # stressed fold-0 survivors
    res = {"prune_tau": v["prune_tau"], "tag": tag, "use_ce": False, "stage2_rel": bool(feats), "base_features": base,
           "features": names, "ensemble": ens, "rules": rules, "report": rep, "stress_frac": frac,
           "cons_subset": cons_tag}
    json.dump(res, open(V.run_path(f"stage2{V.sfx(tag)}.json"), "w"), indent=2, default=float)
    log.info(f"[{name}] rule {rules['chosen']}; stressed fold-0 {rep['macro_f05']:.5f}")
    # unstressed fold 0 with the same model (the plain universe), for the record
    ents_u = ents.copy()
    ents_u["fold"] = orig_fold
    kk, X = matrix(keys, p1, kept, base, cc, np.ones(int(kept.sum()), bool))
    p2u = V.predict_cross(models, X, V.pair_folds(kk, ents_u))
    del X
    f, per = score(kk, ents_u, V.apply_rule(kk, p2u, rules))
    np.save(V.run_path(f"per_entity_{name}_unstressed.npy"), f)
    log.info(f"[{name}] unstressed fold-0 {f.mean():.5f} {per}")


def predict(tag, name):
    res = json.load(open(V.run_path(f"stage2{V.sfx(tag)}.json")))
    if res.get("stage2_rel"):
        E16.SUBSETS[tag] = [f for f in res["features"] if f in CONS_FEATURES]
        E16.install(tag)
    V.predict(name, tag)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--tags", nargs="+", required=True)
    e.add_argument("--frac", type=float, default=0.19)
    t = sub.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--cons", default="")
    t.add_argument("--frac", type=float, default=0.19)
    p = sub.add_parser("predict")
    p.add_argument("--tag", required=True)
    p.add_argument("--name", required=True)
    a = ap.parse_args()
    V.setup_logging()
    if a.cmd == "eval":
        evaluate(a.tags, a.frac)
    elif a.cmd == "train":
        train(a.tag, a.cons, a.frac)
    else:
        predict(a.tag, a.name)


if __name__ == "__main__":
    main()
