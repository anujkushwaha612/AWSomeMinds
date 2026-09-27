"""E15: re-predict test with the France-only vocabulary / address canonicalization (ber.features.country_vocab).

1. back up the France union test file and the v5 test arrays (*.orig), rebuild artifacts/union/test/France.parquet
2. stage-1 test p1 with the saved stage-1 models (India / US rows are bit-identical: same features)
3. stage-2 (noce) test prediction -> subs/<name>/ (validated), and label-free France diagnostics before vs after
Run:  BER_CONFIG=artifacts/free/pipeline.yaml python experiments/e15_france_repredict.py sub_v5_fr1
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from ber import v5 as V
from ber.config import artifact_path

NAME = sys.argv[1] if len(sys.argv) > 1 else "sub_v5_fr1"
TAG = "noce"


def backup(path):
    if os.path.exists(path) and not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")


def france_stats(label):
    keys, ents = V.read_keys("test")
    kept = np.load(V.run_path(f"kept_test_{TAG}.npy"))
    p2 = np.load(V.run_path(f"p2_test_{TAG}.npy"))
    keep = np.load(V.run_path(f"keep_test_{TAG}.npy"))
    kk = keys[kept].reset_index(drop=True)
    out = {}
    for code, c in enumerate(["France", "India", "US"]):
        m = (kk["country"] == code).to_numpy()
        n1 = int((ents["country"] == code).sum())
        out[c] = {"borderline_0.2_0.9": float(((p2[m] > 0.2) & (p2[m] < 0.9)).mean()),
                  "pred_per_s1": float(keep[m].sum() / max(n1, 1)), "kept_pairs": int(m.sum())}
    print(label, json.dumps({c: {k: round(v, 4) for k, v in d.items()} for c, d in out.items()}), flush=True)
    return out


if __name__ == "__main__":
    before = france_stats("BEFORE")
    fr = artifact_path("union", "test", "France.parquet")
    for p in [fr] + [V.run_path(f) for f in ("p1_test.npy", f"p2_test_{TAG}.npy", f"kept_test_{TAG}.npy", f"keep_test_{TAG}.npy")]:
        backup(p)
    os.remove(fr)
    env = dict(os.environ)
    subprocess.run([sys.executable, "-u", "-m", "ber.union", "--split", "test"], check=True, env=env)
    res1 = json.load(open(V.run_path("stage1.json")))
    models = V.load_models("stage1")
    tkeys, _ = V.read_keys("test")
    p1 = V.predict_stream(models, "test", None, len(tkeys), res1["features"])
    old = np.load(V.run_path("p1_test.npy.orig"))
    ch = np.abs(p1 - old) > 1e-6
    for code, c in enumerate(["France", "India", "US"]):
        m = (tkeys["country"] == code).to_numpy()
        print(f"p1 changed on {c}: {ch[m].mean():.4f} of pairs", flush=True)
    np.save(V.run_path("p1_test.npy"), p1)
    del tkeys, p1, old
    subprocess.run([sys.executable, "-u", "-m", "ber.v5", "predict", "--tag", TAG, "--name", NAME], check=True, env=env)
    after = france_stats("AFTER ")
