"""Paired bootstrap between two v5-style runs on fold-0 entities, from their saved per-entity F0.5 arrays.

Works across different candidate sets (e.g. v5_lite on TF-IDF only vs v5 on the TF-IDF + dense union): the
per-entity arrays follow the S1 row order of the fold-0 entities, which is the same for every run.
Run:  python experiments/compare_runs.py artifacts/v5_lite/per_entity_stage2.npy artifacts/v5/per_entity_stage2_noce.npy
"""
import json
import sys

import numpy as np

from ber.config import load_config
from ber.eval.scorer import paired_bootstrap

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"different entity counts: {len(a)} vs {len(b)}")
boot = load_config()["bootstrap"]
bs = paired_bootstrap(a, b, boot["n_resamples"], boot["alpha"])
keep = bs["ci_low"] > 0 and bs["delta"] >= boot["materiality"]
print(f"A {a.mean():.5f} | B {b.mean():.5f} | delta {bs['delta']:+.5f}  CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]"
      f"  on {len(a):,} fold-0 entities -> {'B BETTER (KEEP)' if keep else 'NO CLEAR GAIN'}")
print(json.dumps(bs))
