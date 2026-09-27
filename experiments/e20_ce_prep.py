"""E20 (CPU): texts for a gray-zone cross-encoder on Kaggle (kaggle/ce_kaggle.py), from the consr stage 2.

Only pairs where the consr stage 2 is unsure (band on p2, default 0.02 < p2 < 0.98) are trained on / scored: every
other decision stays with consr. Leakage rule as in ber.neural.cross_encoder: training pairs need their S1 in folds 3-4
AND their record owned by folds 3-4 (true parent's fold, hash fold for orphans); folds 0-2 are only scored
(fold 1-2 fits the local combiner, fold 0 is the report fold).

  python experiments/e20_ce_prep.py [--lo 0.02 --hi 0.98 --n-train 320000]
    -> artifacts/ce_kaggle/ce_train.parquet (text_a, text_b, label)
       artifacts/ce_kaggle/ce_score.parquet (split, rid, part, text_a, text_b, label)   rid = row in kept_<split>_consr
"""
import argparse
import os

import numpy as np
import pandas as pd

from ber import v5 as V
from ber.config import artifact_path
from ber.neural.common import country_store
from ber.neural.cross_encoder import pair_texts, record_owner_fold

TAG = "consr"


def texts(split, kk, rows):
    a, b = np.empty(len(rows), object), np.empty(len(rows), object)
    for code, cname in enumerate(V.split_countries(split)):
        m = np.flatnonzero(kk["country"].to_numpy()[rows] == code)
        if not len(m):
            continue
        r = rows[m]
        st = country_store(split, cname, cols=["name_n", "addr_n"])
        ta, tb = pair_texts(st, kk["s1_row"].to_numpy()[r], kk["src"].to_numpy()[r], kk["doc_row"].to_numpy()[r])
        a[m], b[m] = ta, tb
        del st
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--n-train", type=int, default=320_000)
    ap.add_argument("--train-lo", type=float, default=None, help="training band (default = scoring band)")
    ap.add_argument("--train-hi", type=float, default=None)
    ap.add_argument("--out", default="ce_kaggle", help="folder under artifacts/")
    ap.add_argument("--train-only", action="store_true", help="write ce_train.parquet only (scoring set unchanged)")
    a = ap.parse_args()
    V.setup_logging()
    out = artifact_path(a.out)
    tlo = a.lo if a.train_lo is None else a.train_lo
    thi = a.hi if a.train_hi is None else a.train_hi
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(2026)
    parts = []
    for split in ("train", "test"):
        keys, ents = V.read_keys(split)
        kk = keys[np.load(V.run_path(f"kept_{split}_{TAG}.npy"))].reset_index(drop=True)
        del keys
        p2 = np.load(V.run_path(f"p2_{split}_{TAG}.npy"))
        band = (p2 > a.lo) & (p2 < a.hi)
        if split == "train":
            fold = V.pair_folds(kk, ents)
            own = record_owner_fold(kk, ents)
            tband = (p2 > tlo) & (p2 < thi)
            tr = np.flatnonzero(tband & np.isin(fold, [3, 4]) & np.isin(own, [3, 4]))
            if len(tr) > a.n_train:
                tr = np.sort(rng.choice(tr, a.n_train, replace=False))
            ta, tb = texts(split, kk, tr)
            y = kk["label"].to_numpy()[tr].astype(np.int8)
            pd.DataFrame({"text_a": ta, "text_b": tb, "label": y}).to_parquet(
                os.path.join(out, "ce_train.parquet"), index=False, compression="zstd")
            V.log.info(f"[e20] CE training pairs {len(tr):,} (folds 3-4, band {a.lo}-{a.hi}), positive share "
                       f"{y.mean():.3f}")
            if a.train_only:
                return
            groups = {"f0": band & (fold == 0), "f12": band & np.isin(fold, [1, 2])}
            label = kk["label"].to_numpy().astype(np.int8)
        else:
            groups = {"test": band}
            label = np.full(len(kk), -1, np.int8)
        for part, m in groups.items():
            rows = np.flatnonzero(m)
            ta, tb = texts(split, kk, rows)
            parts.append(pd.DataFrame({"split": split, "rid": rows.astype(np.int64), "part": part,
                                       "country": kk["country"].to_numpy()[rows].astype(np.int8),
                                       "text_a": ta, "text_b": tb, "label": label[rows]}))
            V.log.info(f"[e20] score {split}/{part}: {len(rows):,} pairs")
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(os.path.join(out, "ce_score.parquet"), index=False, compression="zstd")
    V.log.info(f"[e20] ce_score.parquet {len(df):,} pairs; files in {out}: "
               + ", ".join(f"{f} {os.path.getsize(os.path.join(out, f)) / 1e6:.0f} MB" for f in os.listdir(out)))
    print(df.groupby(["split", "part"]).size())
    print(df.sample(5, random_state=1)[["part", "text_a", "text_b", "label"]].to_string())


if __name__ == "__main__":
    main()
