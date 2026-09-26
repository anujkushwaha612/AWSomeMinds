"""E9: likelihood-ratio scan of the generator's edit operations: decoy orphans vs true pairs.

Confusable zone only (baseline_gen OOF p >= P_MIN): true pairs (label 1) vs orphan records (no S1 parent
anywhere) paired with their best-p S1 candidate ("decoys"). Each pair is broken into atomic descriptors:
  name (name_tr tokens): per S1 token its closest record token -> char edit ops (Levenshtein editops),
      phonetic-equal / OCR-swap / transposition / doubled-letter / appended-char classes; unmatched tokens
      as word insertions / deletions (filler vs content);
  address: zero-stripped number relation of the house number and of the whole number set (equal, digit
      dropped / added / replaced, transposed, small shift, unrelated), record address empty, extra numbers.
For every descriptor d: LR = P(d | decoy) / P(d | true) (add-one smoothed), with support counts.
Output: artifacts/experiments/E9_edit_ops.json and the feature table configs/E9_lr_table.json (folds 3-4 only)
Run:  python experiments/e9_edit_ops.py
"""
import collections
import json
import math
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from ber.baseline import truth_parents
from ber.config import artifact_path
from ber.edit_ops import LR_TABLE, LR_TABLE_PATH, addr_desc, name_desc
from ber.io import load_truth_pairs
from ber.store import CountryStore, split_countries

RUN, P_MIN, N_PER = "baseline_gen", 0.3, 40_000
TABLE_FOLDS = (3, 4)   # the LR table is learned on these entities only (encoder folds: never in the final GBDT, never fold 0)
CLIP = 4.0
rng = np.random.default_rng(0)


def collect(country: str, keys: pd.DataFrame, p: np.ndarray, truth, fold_ok: np.ndarray) -> list:
    st = CountryStore("train", country, cols=["entity_id", "name_tr", "addr_n", "addr_digits"])
    par = truth_parents(st, truth)
    lab = keys["label"].to_numpy().astype(bool)
    orphan = np.zeros(len(keys), bool)
    for src in (2, 3):
        m = (keys["src"] == src).to_numpy()
        orphan[m] = par[src][keys.loc[m, "doc_row"].to_numpy()] < 0
    # best-p candidate per record
    order = np.lexsort((-p, keys["doc_row"].to_numpy(), keys["src"].to_numpy()))
    rec = keys["src"].to_numpy()[order].astype(np.int64) << 32 | keys["doc_row"].to_numpy()[order]
    first = np.ones(len(order), bool)
    first[1:] = rec[1:] != rec[:-1]
    best = np.zeros(len(keys), bool)
    best[order[first]] = True
    zone = (p >= P_MIN) & fold_ok
    groups = {"true": np.flatnonzero(lab & zone), "decoy": np.flatnonzero(orphan & best & zone)}
    out = []
    for g, idx in groups.items():
        idx = rng.choice(idx, min(N_PER // 2, len(idx)), replace=False)
        sub = keys.iloc[idx]
        for src in (2, 3):
            ss = sub[sub.src == src]
            s1, doc = ss.s1_row.to_numpy(), ss.doc_row.to_numpy()
            an, bn = st.strings(1, "name_tr", s1), st.strings(src, "name_tr", doc)
            ad, bd = st.strings(1, "addr_digits", s1), st.strings(src, "addr_digits", doc)
            ba = st.strings(src, "addr_n", doc)
            for i in range(len(ss)):
                out.append((g, country, name_desc(an[i], bn[i]) | addr_desc(ad[i], bd[i], ba[i])[0]))
    print(f"{country}: true {len(groups['true']):,} | decoy {len(groups['decoy']):,} in the p >= {P_MIN} zone", flush=True)
    del st
    return out


truth = load_truth_pairs()
oof = np.load(artifact_path(RUN, "oof.npy"))
rows, off = [], 0
for country in split_countries("train"):
    k = pq.read_table(artifact_path(RUN, "train", f"{country}.parquet"), columns=["src", "doc_row", "s1_row", "label"]).to_pandas()
    ef = pd.read_parquet(artifact_path(RUN, "train", f"{country}_entities.parquet"))["fold"].to_numpy()
    rows += collect(country, k, oof[off:off + len(k)], truth, np.isin(ef[k["s1_row"].to_numpy()], TABLE_FOLDS))
    off += len(k)
cnt = {"true": collections.Counter(), "decoy": collections.Counter()}
n = collections.Counter()
for g, _, d in rows:
    cnt[g].update(d)
    n[g] += 1
res = []
for dsc in set(cnt["true"]) | set(cnt["decoy"]):
    ct, cd = cnt["true"][dsc], cnt["decoy"][dsc]
    if ct + cd < 60:
        continue
    pt, pd_ = (ct + 1) / (n["true"] + 2), (cd + 1) / (n["decoy"] + 2)
    res.append({"desc": dsc, "p_true": pt, "p_decoy": pd_, "log_lr": math.log(pd_ / pt), "n_true": ct, "n_decoy": cd})
res.sort(key=lambda r: -r["log_lr"])
print(f"\npairs: true {n['true']:,} | decoy {n['decoy']:,}\n")
print("DECOY fingerprints (LR = P(d|decoy)/P(d|true)):")
for r in res[:30]:
    print(f"  LR {math.exp(r['log_lr']):7.2f}  decoy {r['p_decoy']:.3f}  true {r['p_true']:.4f}  {r['desc']}")
print("\nTRUE-NOISE fingerprints (LR << 1):")
for r in res[-30:][::-1]:
    print(f"  LR {math.exp(r['log_lr']):7.3f}  decoy {r['p_decoy']:.4f}  true {r['p_true']:.3f}  {r['desc']}")
os.makedirs(artifact_path("experiments"), exist_ok=True)
json.dump({"p_min": P_MIN, "folds": TABLE_FOLDS, "n": dict(n), "descriptors": res}, open(artifact_path("experiments", "E9_edit_ops.json"), "w"), indent=1)
json.dump({"folds": TABLE_FOLDS, "p_min": P_MIN, "clip": CLIP,
           "log_lr": {r["desc"]: float(np.clip(r["log_lr"], -CLIP, CLIP)) for r in res}},
          open(LR_TABLE_PATH, "w"), indent=1)                      # versioned: configs/E9_lr_table.json
print(f"wrote {LR_TABLE}: {len(res)} descriptors (folds {TABLE_FOLDS})")
