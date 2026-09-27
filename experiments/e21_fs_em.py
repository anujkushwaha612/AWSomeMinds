"""E21: unsupervised Fellegi-Sunter / EM for the unseen country (France); India / US keep the supervised decisions.

Fellegi-Sunter (Splink-style): each pair's comparison vector = discrete agreement levels of language-agnostic fields
(core-name similarity, content-word swaps / insertions, address similarity, house-number relation, dense cosine,
candidate rank). EM estimates, on ONE country's candidate pairs and without labels, P(level | match) = m and
P(level | non-match) = u per field plus the match prior; posterior log-odds = log prior odds + sum of log(m/u).
EM starts from the supervised posterior (isotonic q of the stage-2 p). Variants:
  fs   the fields above only
  hyb  + the supervised score (and the cross-encoder logit where scored) as two more fields, so EM re-weights them
       for the target country instead of trusting their India/US calibration
Decision: record arbitration + expected-F prefix rule on the posterior (as for the supervised model).

  python experiments/e21_fs_em.py validate    # label-free EM on a country the supervised model did NOT train on:
                                              #   stage 2 fit on US only -> EM on India (and India -> US); labels only score
  python experiments/e21_fs_em.py france      # EM on test France -> subs/sub_v5_ce_frfs, subs/sub_v5_ce_frhyb
                                              #   (India / US decisions = sub_v5_ce, LB 0.982)
"""
import argparse
import gc
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber import v5 as V  # noqa: E402
from ber.baseline import Decider, EntityScorer  # noqa: E402

log = V.log
BASE_COLS = ["name_tset", "nm_concat_sim", "nm_sub", "nm_ins", "addr_tset", "rec_addr_empty", "eo_house_rel", "cos"]
CE_DIR = os.path.join("artifacts", "ce_kaggle", "out")


def levels(F, p=None, ce=None):
    """Integer agreement levels per field (columns), from the union features F (dict of arrays)."""
    cols = {
        "name": np.digitize(F["name_tset"], [65, 80, 90, 99.5]),
        "core": np.digitize(F["nm_concat_sim"], [0.5, 0.75, 0.9, 0.999]),
        "sub": np.minimum(F["nm_sub"], 2).astype(int),
        "ins": np.minimum(F["nm_ins"], 2).astype(int),
        "addr": np.where(F["rec_addr_empty"] > 0, 5, np.digitize(F["addr_tset"], [50, 75, 90, 99.5])),
        "house": F["eo_house_rel"].astype(int),
        "cos": np.digitize(F["cos"], [0.8, 0.85, 0.9, 0.95]),
        "rec_rank": np.minimum(F["rec_prank"], 2).astype(int),
        "s1_rank": np.minimum(F["s1src_prank"], 3).astype(int),
    }
    if p is not None:
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        cols["sup"] = np.digitize(z, [-5, -3, -2, -1, 0, 1, 2, 3, 5])
    if ce is not None:
        cols["ce"] = np.where(np.isfinite(ce), 1 + np.digitize(np.nan_to_num(ce), [-4, -2, -1, 0, 1, 2, 4]), 0)
    return pd.DataFrame(cols)


def fs_em(L: pd.DataFrame, r0: np.ndarray, iters: int = 60, alpha: float = 1.0):
    """EM for the two-class Fellegi-Sunter model over categorical fields; returns posterior and (m, u, prior)."""
    X = [L[c].to_numpy() for c in L.columns]
    K = [int(x.max()) + 1 for x in X]
    r = np.clip(r0.astype(np.float64), 1e-4, 1 - 1e-4)
    for it in range(iters):
        lam = r.mean()
        lo = np.full(len(r), np.log(lam / (1 - lam)))
        mu = []
        for x, k in zip(X, K):
            m = np.bincount(x, weights=r, minlength=k) + alpha
            u = np.bincount(x, weights=1 - r, minlength=k) + alpha
            m, u = m / m.sum(), u / u.sum()
            mu.append((m, u))
            lo += np.log(m / u)[x]
        new = 1 / (1 + np.exp(-np.clip(lo, -30, 30)))
        delta = np.abs(new - r).mean()
        r = new
        if delta < 1e-5:
            break
    log.info(f"[e21] EM {it + 1} iterations, match prior {r.mean():.4f}, last mean |dr| {delta:.2e}")
    return r, dict(zip(L.columns, mu)), float(r.mean())


def weights_table(mu):
    return {f: [round(float(np.log2(m[i] / u[i])), 2) for i in range(len(m))] for f, (m, u) in mu.items()}


def read_fields(split, kept, kk, p1):
    X = V.read_matrix(split, kept, BASE_COLS)
    F = {c: X[:, i] for i, c in enumerate(BASE_COLS)}
    S = V.stage2_features(kk, p1[kept])
    F["rec_prank"], F["s1src_prank"] = S["rec_prank"].to_numpy(), S["s1src_prank"].to_numpy()
    return F


def sub(F, m):
    return {k: v[m] for k, v in F.items()}


def decide(kk, q, mask):
    """Expected-F decisions on the rows of ``mask`` (other rows never predicted)."""
    qq = np.where(mask, q, 0.0).astype(np.float32)
    return V.expected_f_keep(kk, qq, Decider(kk, qq).rec_best) & mask


def iso(rules, p):
    return np.interp(p, rules["iso_x"], rules["iso_y"])


def load_ce(split, n):
    ce = np.full(n, np.nan, np.float32)
    for f in glob.glob(os.path.join(CE_DIR, "ce_scores_*.parquet")):
        d = pd.read_parquet(f)
        d = d[d["split"] == split]
        ce[d["rid"].to_numpy()] = d["ce"].to_numpy()
    return ce


# ------------------------------------------------------------------ validate
def validate():
    """Can label-free EM adapt a transferred model to an unseen country? (US -> India, India -> US)."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    import e16_consensus as E16
    from ber.consensus import CONS_FEATURES
    keys, ents = V.read_keys("train")
    p1 = np.load(V.run_path("p1_train.npy"))
    kept = np.load(V.run_path("kept_train_consr.npy"))
    kk = keys[kept].reset_index(drop=True)
    del keys
    y = kk["label"].to_numpy()
    fold = V.pair_folds(kk, ents)
    cty = kk["country"].to_numpy()
    F = read_fields("train", kept, kk, p1)
    base = V.stage1_features()
    cons = np.load(V.run_path("cons_train.npy"), mmap_mode="r")
    cc = [CONS_FEATURES.index(f) for f in E16.ROBUST]
    Xs = np.hstack([V.read_matrix("train", kept, base), V.stage2_features(kk, p1[kept]).to_numpy(np.float32),
                    np.asarray(cons[:, cc], dtype=np.float32)])
    names = base + V.STAGE2 + E16.ROBUST
    params = V.lgb_params("stage2")
    cn = V.split_countries("train")
    out = {}
    for ca, A in enumerate(cn):
        cb = 1 - ca
        B = cn[cb]
        tr = (cty == ca) & (fold == 2)
        es = tr & (V.entity_uniform((kk["country"].to_numpy().astype(np.int64) << 32) | kk["s1_row"].to_numpy(), 99) < 0.1)
        dtr = lgb.Dataset(Xs[tr & ~es], y[tr & ~es], feature_name=names, params=params)
        bst = lgb.train(params, dtr, 3000, valid_sets=[lgb.Dataset(Xs[es], y[es], reference=dtr)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        mb = cty == cb
        p = np.zeros(len(kk))
        p[mb] = bst.predict(Xs[mb], num_threads=os.cpu_count())
        pa = bst.predict(Xs[(cty == ca) & (fold == 1)], num_threads=os.cpu_count())
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pa, y[(cty == ca) & (fold == 1)])
        q = np.zeros(len(kk))
        q[mb] = cal.predict(p[mb])
        em = ((ents["country"] == cb) & (ents["fold"] == 0)).to_numpy()
        sc = EntityScorer(ents, kk, em)
        f_tr = sc.per_entity(decide(kk, q, mb))
        res = {"transferred": float(f_tr.mean())}
        Fb = sub(F, mb)
        for var in ("fs", "hyb"):
            L = levels(Fb, p=p[mb] if var == "hyb" else None)
            post, mu, lam = fs_em(L, q[mb])
            qq = np.zeros(len(kk))
            qq[mb] = post
            f = sc.per_entity(decide(kk, qq, mb))
            from ber.eval.scorer import paired_bootstrap
            bs = paired_bootstrap(f_tr, f)
            res[var] = float(f.mean())
            res[var + "_delta_ci"] = [bs["delta"], bs["ci_low"], bs["ci_high"]]
            log.info(f"[e21] validate {A} -> {B} (unseen): transferred {f_tr.mean():.5f} | EM-{var} {f.mean():.5f} "
                     f"delta {bs['delta']:+.5f} CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}] | EM prior {lam:.3f}")
        out[f"{A}->{B}"] = res
        del bst
        gc.collect()
    json.dump(out, open(V.run_path("e21_validate.json"), "w"), indent=2)


# ------------------------------------------------------------------ france
def france():
    from ber.submit import finalize_submission
    keys, _ = V.read_keys("test")
    p1 = np.load(V.run_path("p1_test.npy"))
    kept = np.load(V.run_path("kept_test_consr.npy"))
    kk = keys[kept].reset_index(drop=True)
    del keys
    p2 = np.load(V.run_path("p2_test_consr.npy")).astype(np.float64)
    ce = load_ce("test", len(kk))
    rules = json.load(open(V.run_path("stage2_consr.json")))["rules"]
    fr = (kk["country"] == V.split_countries("test").index("France")).to_numpy()
    keep_ce = np.load(V.run_path("keep_test_ce.npy"))
    if len(keep_ce) != len(kk):
        raise RuntimeError("keep_test_ce.npy does not match the consr pruned rows")
    F = sub(read_fields("test", kept, kk, p1), fr)
    n1 = V.country_store("test", "France", cols=["entity_id"]).n(1)
    q0 = iso(rules, p2[fr])
    for var, name in (("fs", "sub_v5_ce_frfs"), ("hyb", "sub_v5_ce_frhyb")):
        L = levels(F, p=p2[fr] if var == "hyb" else None, ce=ce[fr] if var == "hyb" else None)
        post, mu, lam = fs_em(L, q0)
        log.info(f"[e21] France EM-{var}: match weights log2(m/u) per level {json.dumps(weights_table(mu))}")
        q = np.zeros(len(kk))
        q[fr] = post
        kf = decide(kk, q, fr)
        mix = np.where(fr, kf, keep_ce)
        log.info(f"[e21] France EM-{var}: matches/S1 {kf.sum() / n1:.4f} (sub_v5_ce {keep_ce[fr].sum() / n1:.4f}); "
                 f"agree with sub_v5_ce on {((kf == keep_ce) | ~fr)[fr].mean():.4f} of France pairs; added "
                 f"{(kf & ~keep_ce)[fr].sum() / n1:.4f}, removed {(~kf & keep_ce)[fr].sum() / n1:.4f} per S1")
        mpath, cpath, n_m, n_c = V.write_outputs(kk, mix, f"e21_{var}")
        finalize_submission(name, mpath, cpath, offline_metrics=json.load(open(V.run_path("stage2_ce.json")))["report"],
                            n_match_pairs=n_m, n_candidate_pairs=n_c,
                            notes=f"India / US = sub_v5_ce decisions; France = unsupervised Fellegi-Sunter EM ({var}, E21)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "france"])
    a = ap.parse_args()
    V.setup_logging()
    validate() if a.cmd == "validate" else france()


if __name__ == "__main__":
    main()
