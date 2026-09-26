"""E6: baseline fold-0 false positives / found-but-rejected pairs with texts -> <out>/errors.parquet (experiments.md E6)."""
import json, sys
import numpy as np, pandas as pd
from ber.baseline import Decider, read_keys, run_path, truth_parents
from ber.io import load_truth_pairs
from ber.store import CountryStore, split_countries
OUT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/experiments"
keys, ents = read_keys("train")
oof = np.load(run_path("oof.npy"))
d = json.load(open(run_path("metrics.json")))["decision"]
keep = Decider(keys, oof).keep(d["t_first"], d["t_rest"], d["arbitrate"])
f0 = ents.loc[ents.fold == 0, ["country", "s1_row"]]
keys = keys.assign(p=oof, keep=keep)
truth = load_truth_pairs()
rows = []
for code, country in enumerate(split_countries("train")):
    st = CountryStore("train", country, cols=["entity_id", "name_n", "addr_n"])
    par = truth_parents(st, truth)
    kc = keys[keys.country == code]
    inf0 = np.zeros(st.n(1), bool); inf0[f0.loc[f0.country == code, "s1_row"].to_numpy()] = True
    kc = kc[inf0[kc.s1_row.to_numpy()]]
    lab = kc.label.to_numpy().astype(bool)
    fp = kc[kc.keep.to_numpy() & ~lab]
    rej = kc[~kc.keep.to_numpy() & lab]
    for kind, df in (("FP", fp), ("REJ", rej)):
        df = df.sample(min(len(df), 400), random_state=0)
        for r in df.itertuples():
            src, doc, s1 = int(r.src), int(r.doc_row), int(r.s1_row)
            tp = int(par[src][doc])
            # best-p candidate for this record + whether the true parent was a candidate for it
            mine = kc[(kc.src == src) & (kc.doc_row == doc)] if kind == "FP" else None
            rows.append({"kind": kind, "country": country, "src": src, "p": float(r.p),
                         "s1": st.strings(1, "name_n", [s1])[0] + " | " + st.strings(1, "addr_n", [s1])[0],
                         "rec": st.strings(src, "name_n", [doc])[0] + " | " + st.strings(src, "addr_n", [doc])[0],
                         "true_parent": (st.strings(1, "name_n", [tp])[0] + " | " + st.strings(1, "addr_n", [tp])[0]) if tp >= 0 else "<orphan>",
                         "parent_in_cands": bool(tp >= 0 and mine is not None and (mine.s1_row == tp).any())})
    print(country, "FP", len(fp), "REJ", len(rej), flush=True)
    del st
pd.DataFrame(rows).to_parquet(OUT + "/errors.parquet", index=False)
