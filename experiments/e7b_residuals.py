"""E7b: what does the best model (v5-lite stage 2, fold-0 0.9717) still get wrong?

Loss decomposition on fold 0 (false positives / true pairs rejected / true pairs pruned by stage 1 /
never retrieved), per k bucket, then a profile of 1,500 false positives, 1,500 rejected true pairs and
1,500 accepted true pairs with the E6 pattern flags, plus texts for reading.
Output: artifacts/experiments/E7b_residuals.json and E7b_examples.parquet
Run:  BER_CONFIG=artifacts/run_v5lite/pipeline.yaml python experiments/e7b_residuals.py
"""
import json
import os

import numpy as np
import pandas as pd

from ber import v5 as V
from ber.baseline import EntityScorer, truth_parents
from ber.config import artifact_path
from ber.eval.scorer import k_bucket
from ber.features import _gen_number_features, _name_edit_features, store_chain_features
from ber.io import load_truth_pairs
from ber.store import CountryStore, split_countries

N_EX = 1500
keys, ents = V.read_keys("train")
kept = np.load(V.run_path("kept_train.npy"))
p2 = np.load(V.run_path("p2_train.npy"))
rules = json.load(open(V.run_path("stage2.json")))["rules"]
kk = keys[kept].reset_index(drop=True)
keep = V.apply_rule(kk, p2, rules)
rep = (ents["fold"] == 0).to_numpy()
lab_k = kk["label"].to_numpy().astype(bool)
sc = EntityScorer(ents, kk, rep)
f_act, f_nofp, f_kept = sc.per_entity(keep), sc.per_entity(keep & lab_k), sc.per_entity(lab_k)
f_orc = EntityScorer(ents, keys, rep).per_entity(keys["label"].to_numpy().astype(bool))
loss = {"actual": float(f_act.mean()), "false_positives": float(f_nofp.mean() - f_act.mean()),
        "rejected_true": float(f_kept.mean() - f_nofp.mean()), "pruned_true": float(f_orc.mean() - f_kept.mean()),
        "never_retrieved": float(1 - f_orc.mean())}
print("fold-0 loss decomposition:", {k: round(v, 5) for k, v in loss.items()})
re_ = ents[rep].reset_index(drop=True)
kb = k_bucket(re_["k"])
tab = pd.DataFrame({"k": kb, "act": f_act, "nofp": f_nofp, "kept": f_kept, "orc": f_orc})
g = tab.groupby("k").agg(share=("act", "size"), act=("act", "mean"), nofp=("nofp", "mean"),
                          kept=("kept", "mean"), orc=("orc", "mean"))
g["share"] /= len(tab)
g["FP_x_share"] = (g.nofp - g.act) * g.share
g["REJ_x_share"] = (g.kept - g.nofp) * g.share
g["MISS_x_share"] = (1 - g.orc) * g.share
print(g.round(4).to_string())

# ------------------------------------------------------------------ profile + examples
truth = load_truth_pairs()
f0 = ents.loc[rep, ["country", "s1_row"]]
rng = np.random.default_rng(0)
rows = []
for code, country in enumerate(split_countries("train")):
    st = CountryStore("train", country, cols=["entity_id", "name_n", "addr_n", "name_tr", "addr_digits"])
    par = truth_parents(st, truth)
    m = (kk["country"] == code).to_numpy()
    inf0 = np.zeros(st.n(1), bool)
    inf0[f0.loc[f0.country == code, "s1_row"].to_numpy()] = True
    sr = kk["s1_row"].to_numpy()
    m[m] = inf0[sr[m]]
    for kind, mask in (("FP", m & keep & ~lab_k), ("REJ", m & ~keep & lab_k), ("TP", m & keep & lab_k)):
        idx = np.flatnonzero(mask)
        idx = rng.choice(idx, min(N_EX // 2, len(idx)), replace=False)
        sub = kk.iloc[idx]
        for src in (2, 3):
            ss = sub[sub.src == src]
            if ss.empty:
                continue
            s1, doc = ss.s1_row.to_numpy(), ss.doc_row.to_numpy()
            g1 = _gen_number_features(st.strings(1, "addr_digits", s1), st.strings(src, "addr_digits", doc))
            g1.update(_name_edit_features(st.strings(1, "name_tr", s1), st.strings(src, "name_tr", doc)))
            g1.update(store_chain_features(st, st.strings(1, "name_n", s1), st.strings(src, "name_n", doc)))
            tp = par[src][doc]
            d = pd.DataFrame({"kind": kind, "country": country, "src": src, "p2": p2[idx][sub.src.to_numpy() == src],
                              "s1": [a + " | " + b for a, b in zip(st.strings(1, "name_n", s1), st.strings(1, "addr_n", s1))],
                              "rec": [a + " | " + b for a, b in zip(st.strings(src, "name_n", doc), st.strings(src, "addr_n", doc))],
                              "true_parent": [(st.strings(1, "name_n", [t])[0] + " | " + st.strings(1, "addr_n", [t])[0]) if t >= 0 else "<orphan>" for t in tp],
                              "rec_addr_empty": [x.strip() in ("", "null") for x in st.strings(src, "addr_n", doc)],
                              **{k: v for k, v in g1.items()}})
            rows.append(d)
    del st
ex = pd.concat(rows, ignore_index=True)
flags = {
    "rec_addr_empty": ex.rec_addr_empty,
    "orphan_record": ex.true_parent == "<orphan>",
    "house_conflict (zhouse -1)": ex.zhouse_state == -1,
    "number 1-edit off": ex.znum_min_edit == 1,
    "number conflict (edit>=2)": ex.znum_min_edit >= 2,
    "content word swapped (nm_sub>0)": ex.nm_sub > 0,
    "content word inserted only": (ex.nm_ins > 0) & (ex.nm_del == 0),
    "no shared name token (trade name)": ex.nm_shared == 0,
    "record name shared by >1 S1 (chain)": ex.rec_name_freq > 1,
    "S1 name shared by >1 S1": ex.s1_name_freq > 1,
}
prof = pd.DataFrame({k: v.groupby(ex.kind).mean() for k, v in flags.items()}).T.round(3)
print("\npattern shares by kind (FP = false positive, REJ = rejected true pair, TP = accepted true pair):")
print(prof.to_string())
os.makedirs(artifact_path("experiments"), exist_ok=True)
ex.to_parquet(artifact_path("experiments", "E7b_examples.parquet"), index=False)
json.dump({"loss": loss, "by_k": g.reset_index().to_dict("records"), "profile": prof.to_dict()},
          open(artifact_path("experiments", "E7b_residuals.json"), "w"), indent=2, default=float)
