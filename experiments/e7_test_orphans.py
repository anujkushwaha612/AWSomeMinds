"""E7a: how many test records are orphans (no S1 parent)? Mixture estimate from best-candidate scores.

Per record (S2/S3), the best TF-IDF score and best name similarity over its candidates. On train, records
with a true parent among their candidates ("matched") and the rest ("unmatched": orphans + the few whose
parent was never retrieved) have different distributions; the test distribution is fitted as a mixture
a * unmatched + (1 - a) * matched (least squares over histogram bins, per country and source).
Train has a = its true unmatched share; a larger test a means more decoy records per S1 in test.
Output: artifacts/experiments/E7_test_orphans.json
Run:  python experiments/e7_test_orphans.py
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import nnls

from ber.config import artifact_path

BINS = np.linspace(0, 1, 41)
RUN = "baseline_gen"


def record_best(split: str, country: str) -> pd.DataFrame:
    """Best score / name similarity per record, and (train) whether its parent is among the candidates."""
    cols = ["src", "doc_row", "score", "name_tset", "addr_tset"] + (["label"] if split == "train" else [])
    t = pq.read_table(artifact_path(RUN, split, f"{country}.parquet"), columns=cols).to_pandas()
    agg = {"score": "max", "name_tset": "max", "addr_tset": "max"}
    if split == "train":
        agg["label"] = "max"
    return t.groupby(["src", "doc_row"], sort=False).agg(agg).reset_index()


def hist(x: np.ndarray) -> np.ndarray:
    return np.histogram(np.clip(x, 0, 1), BINS)[0] / max(len(x), 1)


out = {}
test_countries = sorted(f[:-8] for f in os.listdir(artifact_path(RUN, "test"))
                        if f.endswith(".parquet") and not f.endswith("_entities.parquet"))
train = {c: record_best("train", c) for c in ("India", "US")}
for country in test_countries:
    te = record_best("test", country)
    ref = train.get(country, pd.concat(train.values()))       # France: pooled train reference
    for src in (2, 3):
        tr = ref[ref.src == src]
        ts = te[te.src == src]
        res = {}
        for feat in ("score", "name_tset"):
            scale = 100.0 if feat != "score" else 1.0
            hm = hist(tr.loc[tr.label == 1, feat].to_numpy() / scale)
            hu = hist(tr.loc[tr.label == 0, feat].to_numpy() / scale)
            ht = hist(ts[feat].to_numpy() / scale)
            w, _ = nnls(np.column_stack([hu, hm]), ht)
            res[feat] = {"test_unmatched_share": float(w[0] / w.sum()), "fit_mass": float(w.sum())}
        n1_tr = pq.ParquetFile(artifact_path(RUN, "train", f"{country if country in train else 'US'}_entities.parquet")).metadata.num_rows
        n1_te = pq.ParquetFile(artifact_path(RUN, "test", f"{country}_entities.parquet")).metadata.num_rows
        out[f"{country}/S{src}"] = {
            "train_unmatched_share": float((tr.label == 0).mean()),
            "records_per_s1_train": len(tr) / n1_tr, "records_per_s1_test": len(ts) / n1_te,
            **{f"test_unmatched_share_by_{k}": round(v["test_unmatched_share"], 4) for k, v in res.items()},
            "fit_mass_score": round(res["score"]["fit_mass"], 3)}
        print(f"{country}/S{src}: train unmatched {out[f'{country}/S{src}']['train_unmatched_share']:.3f} | "
              f"test estimate (score) {res['score']['test_unmatched_share']:.3f}, (name) "
              f"{res['name_tset']['test_unmatched_share']:.3f} | records/S1 train "
              f"{out[f'{country}/S{src}']['records_per_s1_train']:.2f} test {out[f'{country}/S{src}']['records_per_s1_test']:.2f}",
              flush=True)
os.makedirs(artifact_path("experiments"), exist_ok=True)
json.dump(out, open(artifact_path("experiments", "E7_test_orphans.json"), "w"), indent=2)
