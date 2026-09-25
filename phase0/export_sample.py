"""Export a small, structurally faithful slice of the real data.

Why this exists: the agent sandbox can only reach ``github.com`` (Drive, S3,
Dropbox and Hugging Face are all blocked by the egress filter), so the only way
to get real text in front of it is to commit a sample to this repository.

The sample is taken by **hash bucket, not by row**. An S1 entity is kept when
``hash(entity_id) % denom == 0``; a record is kept when its parent entity was
kept, or - for orphans - when its own hash falls in the bucket. That keeps the
slice a closed, self-consistent world: the records-per-S1 density, the orphan
rate, the k histogram, the per-source caps and the country mix all survive,
which row sampling destroys. The one thing it cannot preserve is distractor
pressure, which drops by ``denom``; absolute recall measured on the sample will
therefore be optimistic, exactly as in phase0_report.md section 5.

Defaults give ~1/25 of train (~88k S1, ~413k records, roughly 30MB gzipped) and
~1/25 of test including France.

Run:
    python -m phase0.export_sample --denom 25 --out sample_pack
    git add sample_pack && git commit -m "real data sample" && git push
"""

import argparse
import gzip
import hashlib
import json
import os

import numpy as np
import pandas as pd

from ber.io import load_source, load_truth_pairs


def bucket(ids: pd.Series, denom: int) -> np.ndarray:
    """Stable hash bucket membership for a Series of ids.

    Uses blake2b rather than Python's ``hash``, which is salted per process and
    would give a different sample on every run.
    """
    return np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                                    "big") % denom == 0 for s in ids], dtype=bool)


def export_split(split: str, denom: int, out_dir: str, truth: pd.DataFrame | None) -> dict:
    """Write the gzipped slice of one split and return its marginals."""
    os.makedirs(out_dir, exist_ok=True)
    s1 = load_source(split, 1)
    keep_s1 = s1[bucket(s1["entity_id"], denom)].reset_index(drop=True)
    kept_ids = set(keep_s1["entity_id"])

    parent = (truth.set_index("rid")["s1"] if truth is not None
              else pd.Series(dtype=str))
    stats = {"split": split, "denom": denom, "n_s1": len(keep_s1),
             "countries": keep_s1["country"].value_counts().to_dict()}

    frames = {1: keep_s1}
    for src in (2, 3):
        rec = load_source(split, src)
        par = rec["entity_id"].map(parent) if truth is not None else pd.Series(
            [None] * len(rec), index=rec.index)
        has_parent = par.notna()
        # a record joins the slice if its parent did, or if it is an orphan whose
        # own hash lands in the bucket - this is what preserves the orphan rate
        keep = np.where(has_parent, par.isin(kept_ids),
                        bucket(rec["entity_id"], denom))
        frames[src] = rec[keep].reset_index(drop=True)
        stats[f"n_s{src}"] = len(frames[src])

    for src, df in frames.items():
        path = os.path.join(out_dir, f"{split}_source{src}.tsv.gz")
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            df.to_csv(f, sep="\t", index=False)
        stats[f"bytes_s{src}"] = os.path.getsize(path)

    if truth is not None:
        sub = truth[truth["s1"].isin(kept_ids)]
        lists = sub.groupby("s1")["rid"].agg(",".join)
        gt = pd.DataFrame({"source1_entity_id": keep_s1["entity_id"]})
        gt["matched_entity_ids"] = gt["source1_entity_id"].map(lists).fillna("")
        path = os.path.join(out_dir, f"{split}_ground_truth.tsv.gz")
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            gt.to_csv(f, sep="\t", index=False)
        k = sub.groupby("s1").size().reindex(keep_s1["entity_id"], fill_value=0)
        n_rec = stats["n_s2"] + stats["n_s3"]
        stats.update({
            "gt_pairs": len(sub), "mean_k": round(float(k.mean()), 3),
            "share_k0": round(float((k == 0).mean()), 4),
            "records_per_s1": round(n_rec / len(keep_s1), 3),
            "orphan_share": round(1 - len(sub) / n_rec, 4),
            "max_s2_per_s1": int(sub[sub["rid"].str.startswith("S2")]
                                 .groupby("s1").size().max()),
            "max_s3_per_s1": int(sub[sub["rid"].str.startswith("S3")]
                                 .groupby("s1").size().max()),
            "bytes_gt": os.path.getsize(path)})
    else:
        n_rec = stats["n_s2"] + stats["n_s3"]
        stats["records_per_s1"] = round(n_rec / len(keep_s1), 3)
    return stats


def main() -> None:
    """CLI: export train (+ GT) and test slices, and a manifest of their marginals."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--denom", type=int, default=25,
                    help="keep 1 of every N entities (25 -> ~30MB gzipped)")
    ap.add_argument("--out", default="sample_pack")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    manifest = {}
    for split in args.splits:
        truth = load_truth_pairs() if split == "train" else None
        manifest[split] = export_split(split, args.denom, args.out, truth)
        print(json.dumps(manifest[split], indent=2))

    total = sum(v for s in manifest.values() for k, v in s.items()
                if k.startswith("bytes_"))
    manifest["total_bytes"] = total
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\ntotal {total / 1e6:.1f} MB in {args.out}/")
    if total > 90e6:
        print("WARNING: >90MB. GitHub rejects single files over 100MB; "
              "re-run with a larger --denom.")


if __name__ == "__main__":
    main()
