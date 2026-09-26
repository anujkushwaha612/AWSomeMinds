"""E6: sample 20k fold-0 entities with their matched raw records -> <out>/pairs.parquet (experiments.md E6)."""
import sys, numpy as np, pandas as pd
OUT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/experiments"
folds = pd.read_parquet("artifacts/folds.parquet")
f0 = folds[folds.fold == 0]
rng = np.random.default_rng(0)
samp = f0.sample(20000, random_state=0)
gt = pd.read_csv("data/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt.source1_entity_id.isin(set(samp.s1))]
gt["m"] = gt.matched_entity_ids.str.split(",")
e = gt.explode("m")[["source1_entity_id", "m"]].rename(columns={"source1_entity_id": "s1"})
e = e[e.m.notna() & (e.m != "")]
want = {1: set(samp.s1), 2: set(e.m[e.m.str.startswith("S2")]), 3: set(e.m[e.m.str.startswith("S3")])}
rows = {}
for s in (1, 2, 3):
    keep = []
    for ch in pd.read_csv(f"data/dataset/train/train_source{s}.tsv", sep="\t", dtype=str,
                          keep_default_na=False, chunksize=500_000):
        keep.append(ch[ch.entity_id.isin(want[s])])
    rows[s] = pd.concat(keep).set_index("entity_id")
    print(s, len(rows[s]))
s1 = rows[1].add_prefix("s1_")
rec = pd.concat([rows[2], rows[3]]).add_prefix("r_")
p = e.join(s1, on="s1").join(rec, on="m")
p = p.merge(samp[["s1", "k"]], on="s1")
p.to_parquet(OUT + "/pairs.parquet", index=False)
rows[1].reset_index().merge(samp[["s1","k"]], left_on="entity_id", right_on="s1").to_parquet(OUT + "/s1_sample.parquet", index=False)
print(len(p), "pairs;", (samp.k == 0).mean(), "singleton share")
