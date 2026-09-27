"""E20 (CPU): combine consr stage-2 p2 with the gray-zone cross-encoder logit (kaggle/ce_kaggle.py outputs).

Inside the band (pairs e20_ce_prep.py scored) p = a small LightGBM on logit(p2) and the CE logit (v1), optionally with
rival margins (v2: CE / p2 margin over the record's other S1s, CE margin over the S1's other records of that source);
cross-fitted on folds 1-2, the variant chosen on those tuning folds; outside the band p = p2. Decision rule re-tuned on folds 1-2 (ber.v5.tune_rules), fold 0 reported and compared with consr by
paired bootstrap. Test: the same combiner and rule -> subs/<name> (+ a France-only variant: India / US = consr).

  python experiments/e20_ce_combine.py --scores <dir with ce_scores_*.parquet> --name sub_v5_ce
"""
import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from ber import v5 as V
from ber.eval.scorer import paired_bootstrap
from ber.submit import finalize_submission

TAG = "consr"


def load_ce(scores_dir, split, n):
    ce = np.full(n, np.nan, np.float32)
    parts = {}
    for f in glob.glob(os.path.join(scores_dir, "**", "ce_scores_*.parquet"), recursive=True):
        d = pd.read_parquet(f)
        d = d[d["split"] == split]
        if len(d):
            ce[d["rid"].to_numpy()] = d["ce"].to_numpy()
            parts[os.path.basename(f)] = len(d)
    return ce, parts


def logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def rival_margin(group, v, valid):
    """v minus the best v of the OTHER valid rows of the same group (NaN when there is no other valid row)."""
    g = group[valid]
    x = v[valid].astype(np.float64)
    o = np.lexsort((-x, g))
    gs, xs = g[o], x[o]
    first = np.ones(len(o), bool)
    first[1:] = gs[1:] != gs[:-1]
    starts = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    best = xs[starts][gid]
    size = np.diff(np.append(starts, len(o)))[gid]
    second = np.where(size > 1, xs[np.minimum(starts + 1, len(o) - 1)][gid], np.nan)
    m = np.where(np.arange(len(o)) == starts[gid], xs - second, xs - best)
    out = np.full(len(v), np.nan)
    tmp = np.empty(len(o))
    tmp[o] = m
    out[np.flatnonzero(valid)] = tmp
    return out


# v1: the pair alone; v2: + margins over the record's rival S1s and the S1's rival records (sibling competition)
VARIANTS = {"v1": ["z", "ce"], "v2": ["z", "ce", "ce_rec_margin", "z_rec_margin", "ce_s1src_margin"]}


def feature_table(kk, p2, ce):
    c, src = kk["country"].to_numpy(np.int64), kk["src"].to_numpy(np.int64)
    rec = (c << 40) | (src << 32) | kk["doc_row"].to_numpy(np.int64)
    s1src = (c << 40) | (src << 32) | kk["s1_row"].to_numpy(np.int64)
    z = logit(p2)
    band = np.isfinite(ce)
    return {"z": z, "ce": ce, "ce_rec_margin": rival_margin(rec, ce, band),
            "z_rec_margin": rival_margin(rec, z, np.ones(len(z), bool)),
            "ce_s1src_margin": rival_margin(s1src, ce, band)}


def feats(T, rows, names):
    return np.column_stack([T[n][rows] for n in names]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--name", default="sub_v5_ce")
    ap.add_argument("--no-test", action="store_true", help="fold-0 evaluation only (dry runs)")
    ap.add_argument("--tag", default="ce", help="output names: stage2_<tag>.json, per_entity_stage2_<tag>.npy ...")
    a = ap.parse_args()
    V.setup_logging()
    keys, ents = V.read_keys("train")
    kk = keys[np.load(V.run_path(f"kept_train_{TAG}.npy"))].reset_index(drop=True)
    del keys
    p2 = np.load(V.run_path(f"p2_train_{TAG}.npy")).astype(np.float64)
    y = kk["label"].to_numpy()
    fold = V.pair_folds(kk, ents)
    ce, parts = load_ce(a.scores, "train", len(kk))
    V.log.info(f"[e20] train CE scores {parts}")
    band = np.isfinite(ce)
    if (band & np.isin(fold, [1, 2])).sum() < 50_000:
        raise SystemExit("folds 1-2 are not scored (ce_scores_f12.parquet missing): cannot fit the combiner")
    T = feature_table(kk, p2, ce)
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 500,
              "verbosity": -1, "seed": 2026}
    from sklearn.metrics import roc_auc_score
    f0 = band & (fold == 0)
    best = None
    for var, names in VARIANTS.items():
        mono = [1 if n in ("z", "ce") else 0 for n in names]
        pr = {**params, "monotone_constraints": mono}
        p = p2.copy()
        for f in (1, 2):                     # cross-fit on folds 1-2: their p is out-of-fold for rule tuning
            trn, prd = band & (fold == 3 - f), band & (fold == f)
            m = lgb.train(pr, lgb.Dataset(feats(T, trn, names), y[trn]), 300)
            p[prd] = m.predict(feats(T, prd, names))
        fit = band & np.isin(fold, [1, 2])
        comb = lgb.train(pr, lgb.Dataset(feats(T, fit, names), y[fit]), 300)
        p[f0] = comb.predict(feats(T, f0, names))
        p = p.astype(np.float32)
        V.log.info(f"[e20] {var}: fold-0 band pairs {f0.sum():,} AUC p2 {roc_auc_score(y[f0], p2[f0]):.4f} | ce "
                   f"{roc_auc_score(y[f0], ce[f0]):.4f} | combined {roc_auc_score(y[f0], p[f0]):.4f}")
        rules = V.tune_rules(kk, ents, p, y, V.vcfg()["report_fold"])
        tune = rules[rules["chosen"]]["tune_f05"]
        V.log.info(f"[e20] {var}: tune (folds 1-2, out-of-fold) {tune:.5f} | report fold-0 "
                   f"{rules[rules['chosen']]['report_f05']:.5f}")
        if best is None or tune > best[0] + 0.0001:          # the simpler variant wins near-ties
            best = (tune, var, names, comb, rules, p)
    _, var, names, comb, rules, p = best
    comb.save_model(V.run_path("models", f"e20_{a.tag}_combiner.txt"))
    keep = V.apply_rule(kk, p, rules)
    rep = V.report(f"stage2_{a.tag}", kk, ents, keep)
    base = np.load(V.run_path(f"per_entity_stage2_{TAG}.npy"))
    new = np.load(V.run_path(f"per_entity_stage2_{a.tag}.npy"))
    bs = paired_bootstrap(base, new)
    V.log.info(f"[e20] chosen {var}: fold-0 consr {base.mean():.5f} -> ce {new.mean():.5f}: delta {bs['delta']:+.5f} CI "
               f"[{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}] | rule {rules['chosen']} | {rep['per_country']}")
    res = {"variant": var, "features": names, "rules": rules, "report": rep, "bootstrap_vs_consr": bs,
           "band_pairs_train": int(band.sum())}
    json.dump(res, open(V.run_path(f"stage2_{a.tag}.json"), "w"), indent=2, default=float)
    if a.no_test:
        return

    # ---------------------------------------------------------------- test
    keys, _ = V.read_keys("test")
    kept = np.load(V.run_path(f"kept_test_{TAG}.npy"))
    kt = keys[kept].reset_index(drop=True)
    del keys
    p2t = np.load(V.run_path(f"p2_test_{TAG}.npy")).astype(np.float64)
    cet, parts = load_ce(a.scores, "test", len(kt))
    bt = np.isfinite(cet)
    V.log.info(f"[e20] test CE scores {parts}: {bt.sum():,} pairs")
    Tt = feature_table(kt, p2t, cet)
    pt = p2t.copy()
    pt[bt] = comb.predict(feats(Tt, bt, names))
    keep_t = V.apply_rule(kt, pt.astype(np.float32), rules)
    base_t = np.load(V.run_path(f"keep_test_{TAG}.npy"))
    np.save(V.run_path(f"keep_test_{a.tag}.npy"), keep_t)
    for code, c in enumerate(V.split_countries("test")):
        m = (kt["country"] == code).to_numpy()
        n1 = V.country_store("test", c, cols=["entity_id"]).n(1)
        V.log.info(f"[e20] test {c}: matches/S1 consr {base_t[m].sum() / n1:.4f} -> ce {keep_t[m].sum() / n1:.4f} "
                   f"(added {(keep_t & ~base_t)[m].sum() / n1:.4f}, removed {(~keep_t & base_t)[m].sum() / n1:.4f})")
    mpath, cpath, n_m, n_c = V.write_outputs(kt, keep_t, f"predict_{a.tag}")
    finalize_submission(a.name, mpath, cpath, offline_metrics=rep, n_match_pairs=n_m, n_candidate_pairs=n_c,
                        notes="consr stage 2 + gray-zone multilingual cross-encoder (E20), combiner on folds 1-2")
    fr = (kt["country"] == V.split_countries("test").index("France")).to_numpy()
    mpath, cpath, n_m, n_c = V.write_outputs(kt, np.where(fr, keep_t, base_t), f"predict_{a.tag}_fr_only")
    finalize_submission(a.name + "_fronly", mpath, cpath, offline_metrics=json.load(open(V.run_path("stage2_consr.json")))["report"],
                        n_match_pairs=n_m, n_candidate_pairs=n_c,
                        notes="India / US = sub_v5_consr decisions; France = consr + cross-encoder (E20)")


if __name__ == "__main__":
    main()
