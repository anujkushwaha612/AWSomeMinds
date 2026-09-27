"""E18: leave-one-country-out (LOCO) stage 2 — an offline proxy for the unseen-country (France) gap.

Every France decision so far was judged by 3-decimal LB probes. Here stage 2 is trained on ONE country and scored on
the OTHER (US -> India, India -> US), next to the in-country model of the same size, for several feature subsets.
The gap in-country minus cross-country is the country-shift loss; a subset that shrinks it should transfer to France.

Protocol (per train country A): fit on A fold 2 (early stopping on 10% of its entities), isotonic calibration +
expected-F rule on A fold 1, report on fold 0 of A (in-country) and of B (cross-country). p1 comes from the 0.980 run's
stage 1 (trained on both countries: a leak that makes cross-country look better than it is, as do the E6 noise
vocabulary and E9 LR table, both mined on both countries). "self" = the model's own expected F0.5 on the report fold
(France: self 0.973 vs LB-implied 0.936).

  python experiments/e18_loco.py [--variants full agnostic ...]
"""
import argparse
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e16_consensus as E16  # noqa: E402
from ber import v5 as V  # noqa: E402
from ber.baseline import Decider, EntityScorer, entity_key  # noqa: E402
from ber.consensus import CONS_FEATURES  # noqa: E402

log = V.log

EO = ["eo_llr", "eo_decoy_rep", "eo_ocr_rep", "eo_legal_ins", "eo_appended", "eo_dba", "eo_house_rel", "eo_num_rel"]
VOCAB = ["nm_ins", "nm_del", "nm_sub", "nm_shared", "nm_content_cov", "nm_phonetic", "nm_typo", "legal_ratio"]
COUNTRY = ["translit_ratio", "rec_indic", "s1_name_freq", "rec_name_freq"]
DENSE = ["in_tfidf", "cos", "drank_rec", "drank_s1", "in_dense", "dgap_rec", "drank_rec_all", "dgap_s1",
         "drank_s1_all", "n_retrievers"]
VARIANTS = {
    "full": [],
    "no_eo": EO,
    "agnostic": EO + VOCAB + COUNTRY,
    "no_dense": DENSE,
}


def expected_f_self(keys, q, rec_best, ent_mask_keys):
    """Per-entity model-expected F0.5 of the expected-F decision (the same prefix score it maximizes)."""
    idx = np.flatnonzero(rec_best)
    ent = entity_key(keys["country"].to_numpy()[idx], keys["s1_row"].to_numpy()[idx])
    qq = np.clip(q[idx].astype(np.float64), 1e-6, 1 - 1e-6)
    o = np.lexsort((-qq, ent))
    e, qs = ent[o], qq[o]
    start = np.ones(len(o), bool)
    start[1:] = e[1:] != e[:-1]
    starts = np.flatnonzero(start)
    grp = np.cumsum(start) - 1
    pos = np.arange(len(o)) - starts[grp]
    cs = np.cumsum(qs)
    cs = cs - np.concatenate([[0.0], cs[starts[1:] - 1]])[grp]
    ek = np.add.reduceat(qs, starts)[grp]
    score = 1.25 * cs / (0.25 * ek + pos + 1)
    s0 = np.exp(np.add.reduceat(np.log1p(-qs), starts))
    best = np.maximum(np.maximum.reduceat(score, starts), s0)
    m = dict(zip(e[starts], best))
    return np.array([m.get(k, 1.0) for k in ent_mask_keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    V.setup_logging()
    t0 = time.time()
    keys, ents = V.read_keys("train")
    p1 = np.load(V.run_path("p1_train.npy"))
    kept = p1 >= V.vcfg()["prune_tau"]
    kk = keys[kept].reset_index(drop=True)
    y = kk["label"].to_numpy()
    base = V.stage1_features()
    cons = np.load(V.run_path("cons_train.npy"), mmap_mode="r")
    cc = [CONS_FEATURES.index(f) for f in E16.ROBUST]
    names = base + V.STAGE2 + E16.ROBUST
    X = np.hstack([V.read_matrix("train", kept, base), V.stage2_features(kk, p1[kept]).to_numpy(np.float32),
                   np.asarray(cons[:, cc], dtype=np.float32)])
    del keys, p1, cons
    gc.collect()
    fold = V.pair_folds(kk, ents)
    cty = kk["country"].to_numpy()
    cnames = V.split_countries("train")
    ekey_all = entity_key(ents["country"], ents["s1_row"])
    ent_pair = entity_key(kk["country"], kk["s1_row"])
    log.info(f"[e18] matrix {X.shape} {V.mem_str()} {time.time() - t0:.0f}s")
    params = V.lgb_params("stage2")
    results = {}
    for var in a.variants:
        drop = set(VARIANTS[var])
        cols = [i for i, n in enumerate(names) if n not in drop]
        fn = [names[i] for i in cols]
        for ca, A in enumerate(cnames):
            t = time.time()
            tr = (cty == ca) & (fold == 2)
            es = tr & (V.entity_uniform(ent_pair, 99) < 0.1)
            dtr = lgb.Dataset(X[np.flatnonzero(tr & ~es)][:, cols], y[tr & ~es], feature_name=fn, params=params)
            dva = lgb.Dataset(X[np.flatnonzero(es)][:, cols], y[es], reference=dtr)
            bst = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            del dtr, dva
            p = np.empty(len(X), np.float32)
            for s in range(0, len(X), 2_000_000):
                p[s:s + 2_000_000] = bst.predict(X[s:s + 2_000_000][:, cols], num_threads=os.cpu_count())
            dec = Decider(kk, p)
            cal = dec.rec_best & (cty == ca) & (fold == 1)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[cal], y[cal])
            q = iso.predict(p).astype(np.float32)
            keep = V.expected_f_keep(kk, q, dec.rec_best)
            for cb, B in enumerate(cnames):
                em = ((ents["country"] == cb) & (ents["fold"] == 0)).to_numpy()
                f = EntityScorer(ents, kk, em).per_entity(keep)
                selfF = expected_f_self(kk, q, dec.rec_best, ekey_all[em])
                pm = (cty == cb) & (fold == 0)
                gray = ((p > 0.05) & (p < 0.95) & dec.rec_best & pm).sum() / em.sum()
                results[(var, A, B)] = dict(F=float(f.mean()), self=float(selfF.mean()), gray=float(gray),
                                            pred=float(keep[pm].sum() / em.sum()), trees=bst.best_iteration)
                np.save(V.run_path(f"per_entity_e18_{var}_{A}_to_{B}.npy"), f)
                log.info(f"[e18] {var:9s} train {A:5s} -> {B:5s}: F {f.mean():.5f} self {selfF.mean():.5f} "
                         f"(optimism {selfF.mean() - f.mean():+.5f}) gray/S1 {gray:.3f} pred/S1 "
                         f"{keep[pm].sum() / em.sum():.3f} trees {bst.best_iteration} {time.time() - t:.0f}s")
            del bst, p, q, keep, dec
            gc.collect()
        for B in cnames:
            A = [c for c in cnames if c != B][0]
            ins, crs = results[(var, B, B)], results[(var, A, B)]
            log.info(f"[e18] {var:9s} on {B}: in-country {ins['F']:.5f} cross {crs['F']:.5f} "
                     f"gap {ins['F'] - crs['F']:+.5f}")
    json.dump({"|".join(k): v for k, v in results.items()}, open(V.run_path("e18_loco.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
