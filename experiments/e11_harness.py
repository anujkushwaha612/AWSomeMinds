"""E11: A/B harness on the real union feature code (``string_block(extra=True)`` + worker pool), larger samples.

Baseline TF-IDF candidates (retrieval features from artifacts/baseline) for sampled entities: TRAIN from fold 2
only (the E9 LR table was learned on folds 3-4), TUNE fold 1, REPORT fold 0; every string feature is recomputed
with the exact function the v5 union stage uses. Arms are feature lists; same LightGBM (baseline params, up to
1500 trees), global threshold tuned on TUNE with arbitration, paired bootstrap on REPORT against the first arm.
Output: artifacts/experiments/E11_<name>.json
Run:  python experiments/e11_harness.py <name>        (arms defined in ARMS below)
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.baseline import Decider, EntityScorer
from ber.config import artifact_path, load_config
from ber.edit_ops import EDITOP_FEATURES, EDITOP_LOGIT
from ber.eval.scorer import paired_bootstrap
from ber.features import (CHAIN_FEATURES, DECOY_FEATURES, FEATURES, GEN_FEATURES, KEY_COLS, RECFREQ_FEATURES,
                          STRUCT_FEATURES, string_block)
from ber.neural.common import country_store
from ber.store import split_countries

N_TRAIN, N_TUNE, N_EVAL = 200_000, 50_000, 150_000
RETRIEVAL = ["score", "rank", "gap_rec", "n_cand_rec", "rank_s1", "gap_s1", "n_cand_s1", "n_cand_s1_src"]
CURRENT = FEATURES + STRUCT_FEATURES + GEN_FEATURES + CHAIN_FEATURES + DECOY_FEATURES + EDITOP_FEATURES
ARMS = {"recfreq": [("current", CURRENT), ("current+recfreq", CURRENT + RECFREQ_FEATURES)],
        "logit": [("current", CURRENT), ("current+logit", CURRENT + EDITOP_LOGIT)]}


def main(name: str) -> None:
    t0 = time.time()
    rng = np.random.default_rng(2026)
    ents = []
    for code, country in enumerate(split_countries("train")):
        e = pd.read_parquet(artifact_path("baseline", "train", f"{country}_entities.parquet"))
        ents.append(e.assign(country=np.int8(code), country_name=country))
    ents = pd.concat(ents, ignore_index=True)
    role = np.full(len(ents), "", dtype=object)
    for r, folds, n in (("train", [2], N_TRAIN), ("tune", [1], N_TUNE), ("eval", [0], N_EVAL)):
        idx = np.flatnonzero(ents["fold"].isin(folds).to_numpy())
        role[rng.choice(idx, n, replace=False)] = r
    ents["role"] = role
    ents = ents[ents.role != ""].reset_index(drop=True)

    parts = []
    with ProcessPoolExecutor(max_workers=max(1, min(12, (os.cpu_count() or 4) - 2))) as pool:
        for code, country in enumerate(split_countries("train")):
            want = pa.array(ents.loc[ents.country == code, "s1_row"].to_numpy(np.int32))
            pf = pq.ParquetFile(artifact_path("baseline", "train", f"{country}.parquet"))
            rows = [rb.filter(pc.is_in(rb.column("s1_row"), value_set=want))
                    for rb in pf.iter_batches(batch_size=1_000_000, columns=KEY_COLS + ["label"] + RETRIEVAL)]
            df = pa.Table.from_batches(rows).to_pandas()
            st = country_store("train", country)
            blocks = []
            for src in (2, 3):
                sub = df[df.src == src].reset_index(drop=True)
                f = string_block(st, src, sub["s1_row"].to_numpy(), sub["doc_row"].to_numpy(), extra=True, pool=pool)
                blocks.append(pd.concat([sub, pd.DataFrame(f)], axis=1))
            del st
            df = pd.concat(blocks, ignore_index=True)
            df["country"] = np.int8(code)
            parts.append(df)
            print(f"{country}: {len(df):,} pairs, features done at {time.time() - t0:.0f}s", flush=True)
    keys = pd.concat(parts, ignore_index=True)
    del parts
    ekey = ents.set_index(["country", "s1_row"])["role"]
    keys["role"] = ekey.reindex(pd.MultiIndex.from_arrays([keys["country"], keys["s1_row"]])).to_numpy()

    c = load_config()["baseline"]["lgb"]
    params = {"objective": "binary", "verbosity": -1, "num_threads": os.cpu_count(), "seed": 2026,
              **{k: v for k, v in c.items() if k not in ("num_rounds", "early_stopping")}}
    tr = (keys["role"] == "train").to_numpy()
    tr_ent = keys["s1_row"].to_numpy() * 4 + keys["country"].to_numpy()
    es_ents = rng.choice(np.unique(tr_ent[tr]), N_TRAIN // 10, replace=False)
    es = tr & np.isin(tr_ent, es_ents)
    y = keys["label"].to_numpy()
    grid = np.round(np.arange(0.10, 0.951, 0.05), 2)
    sc_tune = EntityScorer(ents, keys, (ents["role"] == "tune").to_numpy())
    sc_eval = EntityScorer(ents, keys, (ents["role"] == "eval").to_numpy())
    out, per = {"sizes": {"pairs": int(len(keys)), "train": N_TRAIN, "tune": N_TUNE, "eval": N_EVAL}}, {}
    for arm, names in ARMS[name]:
        X = keys[names].to_numpy(np.float32)
        dtr = lgb.Dataset(X[tr & ~es], y[tr & ~es], feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(X[es], y[es], reference=dtr)
        b = lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(c["early_stopping"], verbose=False)])
        p = b.predict(X, num_threads=os.cpu_count())
        dec = Decider(keys, p)
        f_tune = {t: sc_tune.per_entity(dec.keep(t, t, True)).mean() for t in grid}
        T = max(f_tune, key=f_tune.get)
        per[arm] = sc_eval.per_entity(dec.keep(T, T, True))
        gain = dict(zip(names, b.feature_importance("gain")))
        tot = sum(gain.values())
        out[arm] = {"fold0_macro_f05": float(per[arm].mean()), "T": float(T), "trees": b.best_iteration,
                    "top_gain": dict(sorted(((k, round(v / tot, 4)) for k, v in gain.items()), key=lambda x: -x[1])[:15])}
        print(f"[{arm}] fold-0 macro F0.5 {per[arm].mean():.5f} (T {T}, {b.best_iteration} trees) {time.time() - t0:.0f}s", flush=True)
    first = ARMS[name][0][0]
    for arm, _ in ARMS[name][1:]:
        bs = out[f"{arm}_vs_{first}"] = paired_bootstrap(per[first], per[arm])
        print(f"{arm} vs {first}: delta {bs['delta']:+.5f}  CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]")
    os.makedirs(artifact_path("experiments"), exist_ok=True)
    json.dump(out, open(artifact_path("experiments", f"E11_{name}.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "recfreq")
