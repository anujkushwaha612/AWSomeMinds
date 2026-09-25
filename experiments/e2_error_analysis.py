"""E2: where does the baseline lose macro F0.5? (fold 0, full universe, train)

Decomposes the loss into false positives, found-but-rejected true pairs, and true
pairs the retrieval never produced; profiles each group and prints examples.
Run:  python experiments/e2_error_analysis.py
"""
import json
import numpy as np, pandas as pd
from ber.baseline import Decider, EntityScorer, read_keys, run_path, truth_parents
from ber.eval.scorer import f05_from_counts, k_bucket
from ber.io import load_truth_pairs
from ber.store import CountryStore, split_countries

keys, ents = read_keys("train")
oof = np.load(run_path("oof.npy"))
d = json.load(open(run_path("metrics.json")))["decision"]
keep = Decider(keys, oof).keep(d["t_first"], d["t_rest"], d["arbitrate"])
rep = (ents["fold"] == 0).to_numpy()
sc = EntityScorer(ents, keys, rep)
lab = keys["label"].to_numpy().astype(bool)
f_act = sc.per_entity(keep)
f_noFP = sc.per_entity(keep & lab)                      # same TPs, false positives removed
f_orc = sc.per_entity(lab)                              # every retrieved true pair, no FP
print(f"fold-0 macro F0.5 actual {f_act.mean():.4f} | without FPs {f_noFP.mean():.4f} | "
      f"oracle (all retrieved TPs) {f_orc.mean():.4f}")
print(f"loss: false positives {f_noFP.mean()-f_act.mean():.4f} | found-but-rejected "
      f"{f_orc.mean()-f_noFP.mean():.4f} | never retrieved {1-f_orc.mean():.4f}")
re = ents[rep].reset_index(drop=True); kb = k_bucket(re["k"])
tab = pd.DataFrame({"k": kb, "share": 1.0, "actual": f_act, "noFP": f_noFP, "oracle": f_orc})
g = tab.groupby("k").agg(share=("share","sum"), actual=("actual","mean"), noFP=("noFP","mean"), oracle=("oracle","mean"))
g["share"] /= len(tab); g["loss_FP_x_share"] = (g.noFP-g.actual)*g.share
g["loss_rej_x_share"] = (g.oracle-g.noFP)*g.share; g["loss_miss_x_share"] = (1-g.oracle)*g.share
print(g.round(4).to_string())

# retrieval misses: true pairs of fold-0 entities not among candidates
truth = load_truth_pairs(); rng = np.random.default_rng(0)
for code, country in enumerate(split_countries("train")):
    st = CountryStore("train", country)
    par = truth_parents(st, truth)
    e = ents[(ents.country == code)]; f0 = e.loc[e.fold == 0, "s1_row"].to_numpy()
    inf0 = np.zeros(st.n(1), bool); inf0[f0] = True
    kc = keys[keys.country == code]
    for src in (2, 3):
        p = par[src]; docs = np.flatnonzero((p >= 0) & inf0[np.maximum(p, 0)])
        kk = kc[(kc.src == src) & kc.label.astype(bool)]
        found = np.zeros(st.n(src), bool); found[kk.doc_row.to_numpy()] = True
        miss = docs[~found[docs]]
        scr = st.numpy(src, "script")[miss]; addr = np.array(st.strings(src, "addr_n", miss))
        # is the S1 name shared by other S1s (chain)?
        names1 = pd.Series(st.strings(1, "name_n")); dup = names1.map(names1.value_counts()).to_numpy()
        print(f"\n{country} S{src}: {len(miss):,} missed of {len(docs):,} fold-0 true pairs "
              f"({len(miss)/len(docs):.2%}); Indic-script {np.mean(scr>0):.1%}; empty addr "
              f"{np.mean(addr==''):.1%}; parent name shared by >1 S1: {np.mean(dup[p[miss]]>1):.1%} "
              f"(all pairs: {np.mean(dup[p[docs]]>1):.1%})")
        for i in rng.choice(len(miss), 6, replace=False):
            r = miss[i]; s = p[r]
            print(f"   S1: {st.strings(1,'name_n',[s])[0]} | {st.strings(1,'addr_n',[s])[0]}\n"
                  f"   S{src}: {st.strings(src,'name_n',[r])[0]} | {st.strings(src,'addr_n',[r])[0]}")
