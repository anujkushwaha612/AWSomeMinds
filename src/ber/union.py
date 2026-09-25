"""A5: union of TF-IDF and dense candidates, with features for both retrievers (CPU).

Inputs per split / country:
  artifacts/<v5.tfidf_run>/<split>/<country>.parquet     TF-IDF candidates (+ label on train)
  artifacts/dense/<split>/<country>_dense.parquet         dense candidates (record- + S1-side)
  artifacts/dense/<split>/<country>_tfidf_cos.npy         cosine of every TF-IDF candidate
Output: artifacts/union/<split>/<country>.parquet with KEY_COLS, label (train) and
``FEATURES_V5`` = the baseline features (TF-IDF score/rank = 0/9 when absent) + dense
features; entity files are copied unchanged (k and folds do not depend on candidates).

Run:  python -m ber.union --split train
      python -m ber.union --split test
"""

import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import artifact_path, ensure_parent
from .features import FEATURES, KEY_COLS, retrieval_features, write_features
from .memory import mem_str
from .neural.common import country_store, vdir
from .store import split_countries

DENSE_FEATURES = ["in_tfidf", "cos", "drank_rec", "drank_s1", "in_dense", "dgap_rec",
                  "drank_rec_all", "dgap_s1", "drank_s1_all", "n_retrievers"]
FEATURES_V5 = FEATURES + DENSE_FEATURES
ABSENT_RANK = 9


def pair_key(src, doc_row, s1_row) -> np.ndarray:
    """Unique int64 per (src, doc_row, s1_row): src << 43 | doc_row << 21 | s1_row."""
    s1 = np.asarray(s1_row, dtype=np.int64)
    doc = np.asarray(doc_row, dtype=np.int64)
    if len(s1) and (s1.max() >= 1 << 21 or doc.max() >= 1 << 22):
        raise ValueError("row index exceeds the pair-key bit budget")
    return (np.asarray(src, dtype=np.int64) << 43) | (doc << 21) | s1


def merge_candidates(tf: pd.DataFrame, tf_cos: np.ndarray, dense: pd.DataFrame) -> pd.DataFrame:
    """Outer-join TF-IDF and dense candidates on (src, doc_row, s1_row), sorted by that key.

    ``tf`` has KEY_COLS + score + rank (row order = ``tf_cos``); ``dense`` has KEY_COLS +
    cos + drank_rec + drank_s1 (-1 = not in that list).
    """
    a = pd.DataFrame({"key": pair_key(tf["src"], tf["doc_row"], tf["s1_row"]),
                      "score": tf["score"].to_numpy(np.float32), "rank": tf["rank"].to_numpy(np.int8),
                      "cos_tf": tf_cos.astype(np.float32), "in_tfidf": np.int8(1)})
    b = pd.DataFrame({"key": pair_key(dense["src"], dense["doc_row"], dense["s1_row"]),
                      "cos_d": dense["cos"].to_numpy(np.float32),
                      "drank_rec": dense["drank_rec"].to_numpy(np.int8),
                      "drank_s1": dense["drank_s1"].to_numpy(np.int8)})
    m = a.merge(b, on="key", how="outer", sort=True)
    key = m["key"].to_numpy(np.int64)
    out = pd.DataFrame({
        "src": (key >> 43).astype(np.int8),
        "doc_row": ((key >> 21) & ((1 << 22) - 1)).astype(np.int32),
        "s1_row": (key & ((1 << 21) - 1)).astype(np.int32),
        "score": m["score"].fillna(0).to_numpy(np.float32),
        "rank": m["rank"].fillna(ABSENT_RANK).to_numpy(np.int8),
        "in_tfidf": m["in_tfidf"].fillna(0).to_numpy(np.int8),
        "cos": m["cos_tf"].fillna(m["cos_d"]).to_numpy(np.float32),
    })
    drec = m["drank_rec"].fillna(-1).to_numpy(np.int8)
    ds1 = m["drank_s1"].fillna(-1).to_numpy(np.int8)
    out["in_dense"] = ((drec >= 0) | (ds1 >= 0)).astype(np.int8)
    out["drank_rec"] = np.where(drec >= 0, drec, ABSENT_RANK).astype(np.int8)
    out["drank_s1"] = np.where(ds1 >= 0, ds1, ABSENT_RANK).astype(np.int8)
    out["n_retrievers"] = (out["in_tfidf"] + (drec >= 0) + (ds1 >= 0)).astype(np.int8)
    return out


def dense_group_features(cand: pd.DataFrame) -> pd.DataFrame:
    """Cosine gaps / ranks within the record's and the S1's (per source) union candidates."""
    rec = cand.groupby(["src", "doc_row"], sort=False)["cos"]
    cand["dgap_rec"] = (rec.transform("max") - cand["cos"]).astype(np.float32)
    cand["drank_rec_all"] = (rec.rank(ascending=False, method="first") - 1).astype(np.int16)
    s1 = cand.groupby(["s1_row", "src"], sort=False)["cos"]
    cand["dgap_s1"] = (s1.transform("max") - cand["cos"]).astype(np.float32)
    cand["drank_s1_all"] = (s1.rank(ascending=False, method="first") - 1).astype(np.int16)
    return cand


def build(split: str, force: bool = False, feature_chunk: int = 500_000) -> None:
    """Union candidates + features for every country of ``split``."""
    from .baseline import truth_parents
    from .io import load_truth_pairs
    from .neural.common import v5cfg

    v = v5cfg()
    truth = load_truth_pairs() if split == "train" else None
    for country in split_countries(split):
        out = artifact_path(vdir("union"), split, f"{country}.parquet")
        if os.path.exists(out) and not force:
            print(f"[union {split}/{country}] exists, skipping", flush=True)
            continue
        t0 = time.time()
        tf = pq.read_table(artifact_path(vdir("tfidf"), split, f"{country}.parquet"),
                           columns=KEY_COLS + ["score", "rank"]).to_pandas()
        tf_cos = np.load(artifact_path(vdir("dense"), split, f"{country}_tfidf_cos.npy"))
        if len(tf_cos) != len(tf):
            raise ValueError(f"{country}: {len(tf_cos)} TF-IDF cosines for {len(tf)} TF-IDF rows")
        dense = pd.read_parquet(artifact_path(vdir("dense"), split, f"{country}_dense.parquet"))
        cand = merge_candidates(tf, tf_cos, dense)
        del tf, tf_cos, dense
        n_tf = int(cand["in_tfidf"].sum())
        cand = retrieval_features(cand)            # TF-IDF-score group features over the union
        cand = dense_group_features(cand)
        store = country_store(split, country)
        if split == "train":
            parents = truth_parents(store, truth)
            y = np.zeros(len(cand), dtype=np.int8)
            for src in (2, 3):
                m = (cand["src"] == src).to_numpy()
                y[m] = parents[src][cand.loc[m, "doc_row"].to_numpy()] == cand.loc[m, "s1_row"].to_numpy()
            cand["label"] = y
            k = sum(int((p >= 0).sum()) for p in parents.values())
            print(f"    pair recall: TF-IDF {cand.loc[cand['in_tfidf'] == 1, 'label'].sum() / k:.4f} | "
                  f"union {y.sum() / k:.4f}", flush=True)
        print(f"[union {split}/{country}] {n_tf:,} TF-IDF + {len(cand) - n_tf:,} dense-only = "
              f"{len(cand):,} pairs ({len(cand) / max(store.n(1), 1):.2f}/S1) {mem_str()}", flush=True)
        ensure_parent(out)
        write_features(cand, store, out + ".tmp", feature_chunk, log=lambda s: print(s, flush=True))
        os.replace(out + ".tmp", out)
        shutil.copy2(artifact_path(vdir("tfidf"), split, f"{country}_entities.parquet"),
                     artifact_path(vdir("union"), split, f"{country}_entities.parquet"))
        print(f"[union {split}/{country}] done in {time.time() - t0:.0f}s {mem_str()}", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(args.split, args.force)


if __name__ == "__main__":
    main()
