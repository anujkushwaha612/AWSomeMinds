"""E6: mine the generator's noise vocabulary (name tokens it inserts / deletes) from fold-3 true pairs.

Tokens of ``name_tr`` (Latin for every script) that appear in a record but have no fuzzy counterpart in
its S1 parent (inserted), and the reverse (deleted). Fold 3 only, so the report fold (0) is never used.
Output: artifacts/experiments/E6_vocab.json (edit rates; ber.features.NOISE_TOKENS is curated from them).
Run:  python experiments/e6_mine_vocab.py
"""
import collections, json, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from ber.features import fuzzy_in

folds = pd.read_parquet("artifacts/folds.parquet")
samp = folds[folds.fold == 3].sample(40000, random_state=0)
gt = pd.read_csv("data/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt.source1_entity_id.isin(set(samp.s1)) & (gt.matched_entity_ids != "")]
e = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")[["source1_entity_id", "m"]]
names = {}
for s, ids in ((1, set(samp.s1)), (2, set(e.m)), (3, set(e.m))):
    t = pq.read_table(f"artifacts/norm/train_source{s}.parquet", columns=["entity_id", "name_tr"])
    t = t.filter(pc.is_in(t["entity_id"], value_set=pa.array(list(ids))))
    names.update(zip(t["entity_id"].to_pylist(), t["name_tr"].to_pylist()))
ins, dele, occ, n = collections.Counter(), collections.Counter(), collections.Counter(), 0
for s1, r in zip(e.source1_entity_id, e.m):
    a, b = names[s1].split(), names[r].split()
    ins.update(t for t in set(b) if not fuzzy_in(t, a))
    dele.update(t for t in set(a) if not fuzzy_in(t, b))
    occ.update(set(a)); occ.update(set(b))
    n += 1
MIN = max(20, n // 5000)               # ~0.02% of pairs: generator vocabulary, not entity-specific words
# filler = tokens the generator edits far more often than a content word is dropped by chance
rate = {t: (ins[t] + dele[t]) / occ[t] for t in occ if ins[t] + dele[t] >= MIN and len(t) > 1}
base = float(np.median(list(rate.values())))
RATE = 0.30
vocab = sorted(t for t, r in rate.items() if r >= RATE)
os.makedirs("artifacts/experiments", exist_ok=True)
json.dump({"pairs": n, "min_count": MIN, "edit_rate_min": RATE, "median_rate": base, "vocab": vocab,
           "rates": sorted(((t, round(r, 3)) for t, r in rate.items()), key=lambda x: -x[1]), "ins_top": ins.most_common(60),
           "del_top": dele.most_common(60)}, open("artifacts/experiments/E6_vocab.json", "w"), indent=1)
print(n, "pairs; min count", MIN, "; median edit rate", round(base, 3), "; vocab", len(vocab)); print(vocab)
r = sorted(rate.items(), key=lambda x: -x[1]); print("rates around the cut:", [(t, round(v, 2)) for t, v in r if 0.15 < v < 0.45][:60])
