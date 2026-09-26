"""E7c ablation: do the decoy features (ber.features.DECOY_FEATURES) help on top of GEN + CHAIN?

(Copy of e6_ablation.py with two arms.)

Same LightGBM (baseline parameters) on the baseline candidates, trained twice: baseline FEATURES vs
FEATURES + GEN_FEATURES. Entities (pooled over countries): 150k from folds 2-4 train, 50k from fold 1
tune the threshold (global T, arbitration), 100k from fold 0 report; paired bootstrap on fold 0.
Output: artifacts/experiments/E6_ablation.json
Run:  python experiments/e6_ablation.py
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.baseline import Decider, EntityScorer
from ber.config import artifact_path, load_config
from ber.eval.scorer import paired_bootstrap
from ber.features import (ADDR_FEATURES, CHAIN_FEATURES, DECOY_FEATURES, _decoy_name_features, FEATURES, GEN_FEATURES, KEY_COLS,
                          _addr_edit_features, _gen_number_features, _name_edit_features, chain_features)
from ber.store import CountryStore, split_countries

N_TRAIN, N_TUNE, N_EVAL = 150_000, 50_000, 100_000
rng = np.random.default_rng(2026)
t0 = time.time()

ents = []
for code, country in enumerate(split_countries("train")):
    e = pd.read_parquet(artifact_path("baseline", "train", f"{country}_entities.parquet"))
    ents.append(e.assign(country=np.int8(code), country_name=country))
ents = pd.concat(ents, ignore_index=True)
role = np.full(len(ents), "", dtype=object)
for r, folds, n in (("train", [2, 3, 4], N_TRAIN), ("tune", [1], N_TUNE), ("eval", [0], N_EVAL)):
    idx = np.flatnonzero(ents["fold"].isin(folds).to_numpy())
    role[rng.choice(idx, n, replace=False)] = r
ents["role"] = role
ents = ents[ents.role != ""].reset_index(drop=True)

parts = []
for code, country in enumerate(split_countries("train")):
    want = pa.array(ents.loc[ents.country == code, "s1_row"].to_numpy(np.int32))
    pf = pq.ParquetFile(artifact_path("baseline", "train", f"{country}.parquet"))
    rows = [rb.filter(pc.is_in(rb.column("s1_row"), value_set=want))
            for rb in pf.iter_batches(batch_size=1_000_000, columns=KEY_COLS + ["label"] + FEATURES)]
    df = pa.Table.from_batches(rows).to_pandas()
    st = CountryStore("train", country, cols=["name_tr", "addr_digits", "name_n", "addr_tr"])
    all_s1_names = st.strings(1, "name_n")
    for src in (2, 3):
        m = (df["src"] == src).to_numpy()
        s1, doc = df.loc[m, "s1_row"].to_numpy(), df.loc[m, "doc_row"].to_numpy()
        g = _gen_number_features(st.strings(1, "addr_digits", s1), st.strings(src, "addr_digits", doc))
        g.update(_name_edit_features(st.strings(1, "name_tr", s1), st.strings(src, "name_tr", doc)))
        g.update(_decoy_name_features(st.strings(1, "name_tr", s1), st.strings(src, "name_tr", doc)))
        g.update(chain_features(all_s1_names, st.strings(1, "name_n", s1), st.strings(src, "name_n", doc)))
        for k, v in g.items():
            if k not in df:
                df[k] = np.zeros(len(df), dtype=v.dtype)
            df.loc[m, k] = v
    del st
    df["country"] = np.int8(code)
    parts.append(df)
    print(f"{country}: {len(df):,} candidate pairs, features in {time.time() - t0:.0f}s", flush=True)
keys = pd.concat(parts, ignore_index=True)
del parts
ekey = ents.set_index(["country", "s1_row"])["role"]
keys["role"] = ekey.reindex(pd.MultiIndex.from_arrays([keys["country"], keys["s1_row"]])).to_numpy()

c = load_config()["baseline"]["lgb"]
params = {"objective": "binary", "verbosity": -1, "num_threads": os.cpu_count(), "seed": 2026,
          **{k: v for k, v in c.items() if k not in ("num_rounds", "early_stopping")}}
tr = (keys["role"] == "train").to_numpy()
es_ents = set(rng.choice(ents.loc[ents.role == "train", "s1_row"].to_numpy() * 4 +
                         ents.loc[ents.role == "train", "country"].to_numpy(), N_TRAIN // 10, replace=False))
es = tr & np.isin(keys["s1_row"].to_numpy() * 4 + keys["country"].to_numpy(), list(es_ents))
y = keys["label"].to_numpy()
grid = np.round(np.arange(0.10, 0.951, 0.05), 2)
tune_mask = (ents["role"] == "tune").to_numpy()
eval_mask = (ents["role"] == "eval").to_numpy()
sc_tune, sc_eval = EntityScorer(ents, keys, tune_mask), EntityScorer(ents, keys, eval_mask)
out, per = {}, {}
ARMS = (("genchain", FEATURES + GEN_FEATURES + CHAIN_FEATURES),
        ("genchain+decoy", FEATURES + GEN_FEATURES + CHAIN_FEATURES + DECOY_FEATURES))
for arm, names in ARMS:
    X = keys[names].to_numpy(np.float32)
    dtr = lgb.Dataset(X[tr & ~es], y[tr & ~es], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[es], y[es], reference=dtr)
    b = lgb.train(params, dtr, c["num_rounds"], valid_sets=[dva],
                  callbacks=[lgb.early_stopping(c["early_stopping"], verbose=False)])
    p = b.predict(X, num_threads=os.cpu_count())
    dec = Decider(keys, p)
    f_tune = {t: sc_tune.per_entity(dec.keep(t, t, True)).mean() for t in grid}
    T = max(f_tune, key=f_tune.get)
    per[arm] = sc_eval.per_entity(dec.keep(T, T, True))
    gain = dict(zip(names, b.feature_importance("gain")))
    tot = sum(gain.values())
    out[arm] = {"fold0_macro_f05": float(per[arm].mean()), "T": float(T), "trees": b.best_iteration,
                "new_gain_share": {k: round(gain[k] / tot, 4) for k in GEN_FEATURES + CHAIN_FEATURES + DECOY_FEATURES
                                   if k in gain}}
    print(f"[{arm}] fold-0 macro F0.5 {per[arm].mean():.5f} (T {T}, {b.best_iteration} trees) "
          f"{time.time() - t0:.0f}s", flush=True)
bs = out["decoy_vs_genchain"] = paired_bootstrap(per["genchain"], per["genchain+decoy"])
out["sizes"] = {"pairs": int(len(keys)), "train_entities": N_TRAIN, "tune": N_TUNE, "eval": N_EVAL}
print(f"decoy vs genchain: delta {bs['delta']:+.5f}  CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]")
print("gain shares:", {k: v for k, v in out["genchain+decoy"]["new_gain_share"].items()})
os.makedirs(artifact_path("experiments"), exist_ok=True)
json.dump(out, open(artifact_path("experiments", "E7c_decoy_ablation.json"), "w"), indent=2)
