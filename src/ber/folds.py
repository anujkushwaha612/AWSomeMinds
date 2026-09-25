"""Entity-level fold assignment for full-universe cross-fitting (plan.md Step 4).

Every train S1 entity gets a fold in 0..n_folds-1, stratified by
country x k-bucket. A candidate pair inherits the fold of its S1 entity;
orphan S2/S3 records have no fold and only ever appear as negatives.

Run:  python -m ber.folds
"""

import json

import numpy as np
import pandas as pd

from .config import artifact_path, ensure_parent, load_config
from .eval.scorer import k_bucket
from .io import load_source, load_truth_pairs


def assign_folds(s1: pd.DataFrame, truth: pd.DataFrame, n_folds: int, seed: int,
                 top_bucket: int = 6) -> pd.DataFrame:
    """Return ``s1, country, k, k_bucket, fold`` for every S1 entity.

    Within each (country, k_bucket) stratum entities are shuffled with ``seed``
    and dealt round-robin, so every fold gets an equal share of every stratum.
    """
    k = truth.groupby("s1").size()
    df = pd.DataFrame({"s1": s1["entity_id"].to_numpy(), "country": s1["country"].to_numpy()})
    df["k"] = df["s1"].map(k).fillna(0).astype(np.int64)
    df["k_bucket"] = k_bucket(df["k"], top_bucket)
    rng = np.random.default_rng(seed)
    df["_r"] = rng.random(len(df))
    df = df.sort_values(["country", "k_bucket", "_r"], kind="mergesort")
    df["fold"] = df.groupby(["country", "k_bucket"]).cumcount() % n_folds
    return df.drop(columns="_r").sort_index().reset_index(drop=True)


def main() -> None:
    """Build ``artifacts/folds.parquet`` and print the per-fold balance."""
    cfg = load_config()
    val = cfg["validation"]
    folds = assign_folds(load_source("train", 1), load_truth_pairs(),
                         val["n_folds"], cfg["seed"], max(val["k_buckets"]))
    out = artifact_path("folds.parquet")
    ensure_parent(out)
    folds.to_parquet(out, index=False)

    table = pd.crosstab([folds["country"], folds["k_bucket"]], folds["fold"])
    print(table.to_string())
    summary = {
        "n_entities": int(len(folds)),
        "per_fold": folds["fold"].value_counts().sort_index().to_dict(),
        "mean_k_per_fold": folds.groupby("fold")["k"].mean().round(4).to_dict(),
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
