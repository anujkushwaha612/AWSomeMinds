"""E10: can the true parent of an ambiguous record be picked from its same-name S1 candidates by set-level counts?

Ambiguous = the record's candidate list holds >= 2 S1s whose name_n equals the record's name_n (chains).
For each candidate S1: how many OTHER records point to it with stage-1 p1 > 0.5 (overall / same source / other
source). If the generator gives each entity a few records, the true parent may be the chain member that is
"missing" one. Reports how often the true parent has the min / max count vs chance, for empty-address records
and records with an address. Uses v5-lite stage-1 OOF p (artifacts/v5_lite/p1_train.npy); all folds (analysis).
Run:  BER_CONFIG=artifacts/run_v5lite/pipeline.yaml python experiments/e10_chain_resolution.py
"""
import numpy as np
import pandas as pd

from ber import v5 as V
from ber.neural.common import country_store
from ber.store import split_countries

keys, ents = V.read_keys("train")
p1 = np.load(V.run_path("p1_train.npy"))
keys["p1"] = p1
res = []
for code, country in enumerate(split_countries("train")):
    st = country_store("train", country, cols=["name_n", "addr_n"])
    k = keys[keys.country == code].copy()
    k["hi"] = (k.p1 > 0.5).astype(np.int32)
    # other-record counts per S1 (total and per source), excluding the pair itself
    tot = k.groupby("s1_row").hi.transform("sum") - k.hi
    per_src = k.groupby(["s1_row", "src"]).hi.transform("sum") - k.hi
    k["n_other"], k["n_same_src"], k["n_other_src"] = tot, per_src, tot - per_src
    s1n = np.array(st.strings(1, "name_n"), dtype=object)
    for src in (2, 3):
        rn = np.array(st.strings(src, "name_n"), dtype=object)
        ra = np.array(st.strings(src, "addr_n"), dtype=object)
        kk = k[k.src == src]
        same = s1n[kk.s1_row.to_numpy()] == rn[kk.doc_row.to_numpy()]
        kk = kk[same]
        g = kk.groupby("doc_row")
        kk = kk[g.s1_row.transform("size") >= 2]                      # >= 2 same-name S1 candidates
        kk = kk[kk.groupby("doc_row").label.transform("max") == 1]    # the true parent is among them
        kk["empty_addr"] = [ra[d].replace("null", "").strip() == "" for d in kk.doc_row.to_numpy()]
        for col in ("n_other", "n_same_src", "n_other_src", "p1"):
            gg = kk.groupby("doc_row")[col]
            kk[f"{col}_is_min"] = kk[col] == gg.transform("min")
            kk[f"{col}_is_max"] = kk[col] == gg.transform("max")
        kk["m"] = kk.groupby("doc_row").s1_row.transform("size")
        par = kk[kk.label == 1]
        for empty in (True, False):
            q = par[par.empty_addr == empty]
            if len(q) == 0:
                continue
            row = {"country": country, "src": src, "empty_addr": empty, "records": len(q), "mean_chain_size": q.m.mean(),
                   "chance_1_over_m": (1 / q.m).mean()}
            for col in ("n_other", "n_same_src", "n_other_src", "p1"):
                row[f"parent_is_min_{col}"] = q[f"{col}_is_min"].mean()
                row[f"parent_is_max_{col}"] = q[f"{col}_is_max"].mean()
            res.append(row)
    del st
out = pd.DataFrame(res)
pd.set_option("display.width", 250)
print(out.round(3).T.to_string())
