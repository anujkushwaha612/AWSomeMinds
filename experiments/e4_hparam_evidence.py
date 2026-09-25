"""E4: measurements that set the hyperparameters of strategy v5 (A bi-encoder, B GBDT, C gap).

Everything is read from existing artifacts (baseline candidates, OOF predictions, stores);
nothing is trained. Output: printed tables + artifacts/experiments/E4_evidence.json.

Run:  python experiments/e4_hparam_evidence.py
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.baseline import Decider, EntityScorer, entity_key, read_keys, run_path, truth_parents
from ber.config import artifact_path, ensure_parent
from ber.eval.scorer import f05_from_counts
from ber.features import FEATURES
from ber.io import load_truth_pairs
from ber.store import CountryStore, split_countries

OUT = {}


def section(title):
    print(f"\n==== {title}", flush=True)


def main():
    os.environ.setdefault("BER_RUN", "baseline")
    keys, ents = read_keys("train")
    names = split_countries("train")
    rank = np.concatenate([pq.read_table(run_path("train", f"{c}.parquet"), columns=["rank"])
                           ["rank"].to_numpy() for c in names])
    keys["rank"] = rank
    oof = np.load(run_path("oof.npy"))
    d = json.load(open(run_path("metrics.json")))["decision"]
    lab = keys["label"].to_numpy().astype(bool)
    fold_of = dict(zip(entity_key(ents["country"], ents["s1_row"]), ents["fold"]))
    pair_fold = pd.Series(entity_key(keys["country"], keys["s1_row"])).map(fold_of).to_numpy()

    # ---------------------------------------------------------------- 1. TF-IDF rank of found parents
    section("1. TF-IDF record-side rank of the true parent (all train pairs)")
    k_total = ents.groupby("country")["k"].sum()
    rows = {}
    for code, c in enumerate(names):
        m = (keys["country"] == code).to_numpy() & lab
        r = keys.loc[m, "rank"].to_numpy()
        tot = int(k_total[code])
        rows[c] = {f"recall@{k}": float((r < k).sum() / tot) for k in (1, 2, 3)}
        rows[c]["share_of_found_at_rank0"] = float((r == 0).mean())
    print(pd.DataFrame(rows).round(4).to_string())
    OUT["tfidf_rank"] = rows

    # ---------------------------------------------------------------- 2. error anatomy (fold 0)
    section("2. Anatomy of fold-0 errors (baseline, T_first=T_rest=%.2f, arbitration)" % d["t_first"])
    dec = Decider(keys, oof)
    keep = dec.keep(d["t_first"], d["t_rest"], d["arbitrate"])
    f0 = pair_fold == 0
    pos = f0 & lab
    n_pos = int(pos.sum())
    tp = pos & keep
    lost_arb = pos & ~keep & ~dec.rec_best             # another S1 had higher p for this record
    low_p = pos & ~keep & dec.rec_best                 # won arbitration but p below threshold
    fp = f0 & keep & ~lab
    # is the FP record an orphan, or does it have a true parent somewhere?
    rec = (keys["country"].to_numpy(np.int64) << 40) | (keys["src"].to_numpy(np.int64) << 32) \
        | keys["doc_row"].to_numpy(np.int64)
    rec_has_pos_cand = pd.Series(lab).groupby(rec).transform("max").to_numpy().astype(bool)
    truth = load_truth_pairs()
    has_parent = np.zeros(len(keys), dtype=bool)
    for code, c in enumerate(names):
        st = CountryStore("train", c, cols=["entity_id"])
        par = truth_parents(st, truth)
        m = (keys["country"] == code).to_numpy()
        for src in (2, 3):
            ms = m & (keys["src"] == src).to_numpy()
            has_parent[ms] = par[src][keys.loc[ms, "doc_row"].to_numpy()] >= 0
        del st
    anatomy = {
        "retrieved_true_pairs": n_pos,
        "kept (TP)": float(tp.sum() / n_pos),
        "rejected: lost arbitration to another S1": float(lost_arb.sum() / n_pos),
        "rejected: won arbitration but p < T": float(low_p.sum() / n_pos),
        "false_positives": int(fp.sum()),
        "FP record is an orphan (no parent anywhere)": float((fp & ~has_parent).sum() / max(fp.sum(), 1)),
        "FP record's true parent is among its candidates": float((fp & rec_has_pos_cand).sum() / max(fp.sum(), 1)),
        "FP record's true parent exists but not retrieved": float((fp & has_parent & ~rec_has_pos_cand).sum()
                                                                  / max(fp.sum(), 1)),
    }
    for k_, v in anatomy.items():
        print(f"  {k_:50s} {v:.4f}" if isinstance(v, float) else f"  {k_:50s} {v:,}")
    OUT["error_anatomy"] = anatomy

    # ---------------------------------------------------------------- 3. pruning threshold
    section("3. Candidate pruning by stage-1 p (fold 0): oracle F0.5 vs candidates per S1")
    rep_mask = (ents["fold"] == 0).to_numpy()
    sc = EntityScorer(ents, keys, rep_mask)
    n_rep = int(rep_mask.sum())
    in_rep = np.zeros(len(keys), dtype=bool)
    in_rep[sc.pairs] = True
    prune = []
    for tau in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1):
        kept = oof >= tau
        orc = sc.per_entity(lab & kept).mean()
        prune.append({"tau": tau, "oracle_f05": float(orc),
                      "cand_per_s1": float((kept & in_rep).sum() / n_rep),
                      "pos_lost": float(((lab & ~kept) & in_rep).sum() / max((lab & in_rep).sum(), 1))})
    # record-side top-1 by p (+ tiny floor)
    best_only = dec.rec_best & (oof >= 1e-3)
    prune.append({"tau": "rec_best_p & p>=1e-3", "oracle_f05": float(sc.per_entity(lab & best_only).mean()),
                  "cand_per_s1": float((best_only & in_rep).sum() / n_rep),
                  "pos_lost": float(((lab & ~best_only) & in_rep).sum() / max((lab & in_rep).sum(), 1))})
    print(pd.DataFrame(prune).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    OUT["pruning"] = prune

    # p distribution of positives vs negatives
    q = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    OUT["p_quantiles"] = {"positives": np.quantile(oof[lab], q).tolist(),
                          "negatives_q50_q90_q99": np.quantile(oof[~lab], [0.5, 0.9, 0.99]).tolist()}
    print("  positive p quantiles", dict(zip(q, np.round(OUT["p_quantiles"]["positives"], 4))))
    print("  negative p q50/q90/q99", np.round(OUT["p_quantiles"]["negatives_q50_q90_q99"], 4))

    # ---------------------------------------------------------------- 4. density
    section("4. Density (S2+S3 records per S1): train vs test")
    dens = {}
    for split in ("train", "test"):
        for c in split_countries(split):
            st = CountryStore(split, c, cols=["entity_id"])
            dens[f"{split}/{c}"] = (st.n(2) + st.n(3)) / st.n(1)
            del st
    for k_, v in dens.items():
        print(f"  {k_:14s} {v:.3f}")
    stress = {c: 1 - dens[f"train/{c}"] / dens[f"test/{c}"] for c in names}
    print("  S1 share to remove so train density matches test:", {c: round(v, 3) for c, v in stress.items()})
    OUT["density"] = dens
    OUT["stress_fraction"] = stress

    # ---------------------------------------------------------------- 5. encoder training data
    section("5. Encoder training data: positives whose S1 is in folds 3-4")
    enc = {}
    for code, c in enumerate(names):
        st = CountryStore("train", c, cols=["entity_id", "name_n", "addr_n", "script"])
        par = truth_parents(st, truth)
        e = ents[ents["country"] == code]
        fold = e["fold"].to_numpy()
        names1 = pd.Series(st.strings(1, "name_n"))
        dup = names1.map(names1.value_counts()).to_numpy() > 1
        kc = keys[keys["country"] == code]
        for src in (2, 3):
            p = par[src]
            has = p >= 0
            enc_pos = has & np.isin(fold[np.maximum(p, 0)], [3, 4])
            docs = np.flatnonzero(enc_pos)
            ks = kc[(kc["src"] == src)]
            neg = ks[ks["label"] == 0]
            has_neg = np.zeros(st.n(src), dtype=bool)
            has_neg[neg["doc_row"].to_numpy()] = True
            posr = ks[ks["label"] == 1][["doc_row", "rank"]].set_index("doc_row")["rank"]
            neg_best = neg.groupby("doc_row")["rank"].min()
            above = neg_best.reindex(docs).to_numpy() < posr.reindex(docs).fillna(99).to_numpy()
            addr = np.array(st.strings(src, "addr_n", docs))
            scr = st.numpy(src, "script")[docs]
            name_len = np.fromiter((len(x) for x in st.strings(src, "name_n", docs)), int)
            addr_len = np.fromiter((len(x) for x in addr), int)
            tl = name_len + addr_len + 3
            enc[f"{c}/S{src}"] = {
                "positives_in_folds_3_4": int(len(docs)),
                "share_indic": float((scr > 0).mean()),
                "share_empty_addr": float((addr == "").mean()),
                "share_parent_name_shared": float(dup[p[docs]].mean()),
                "share_with_tfidf_hard_negative": float(has_neg[docs].mean()),
                "share_hard_negative_ranked_above_parent": float(np.nanmean(above)),
                "text_chars_p50_p95_p99": np.quantile(tl, [0.5, 0.95, 0.99]).tolist(),
            }
        s1_tl = np.fromiter((len(a) + len(b) + 3 for a, b in zip(st.strings(1, "name_n"), st.strings(1, "addr_n"))), int)
        enc[f"{c}/S1"] = {"text_chars_p50_p95_p99": np.quantile(s1_tl, [0.5, 0.95, 0.99]).tolist()}
        del st
    print(pd.DataFrame(enc).T.to_string())
    OUT["encoder_data"] = enc

    # ---------------------------------------------------------------- 6. GBDT capacity + importance
    section("6. Baseline LightGBM: trees used and gain importance")
    mdir = run_path("models")
    trees, imp = [], np.zeros(len(FEATURES))
    for f in sorted(os.listdir(mdir)):
        b = lgb.Booster(model_file=os.path.join(mdir, f))
        trees.append(b.num_trees())
        imp += b.feature_importance("gain")
    imp = imp / imp.sum()
    order = np.argsort(-imp)
    print("  trees per fold model:", trees)
    print("  top features by gain:", [(FEATURES[i], round(float(imp[i]), 3)) for i in order[:12]])
    print("  near-zero features:", [FEATURES[i] for i in order if imp[i] < 0.002])
    OUT["gbdt"] = {"trees": trees, "gain": {FEATURES[i]: float(imp[i]) for i in order}}

    out = artifact_path("experiments", "E4_evidence.json")
    ensure_parent(out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(OUT, fh, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
