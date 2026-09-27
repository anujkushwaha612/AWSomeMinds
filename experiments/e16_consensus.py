"""E16: record-consensus features (ber.consensus) in stage 2, on the 0.980 run's stage 1.

  python experiments/e16_consensus.py build --split train|test   -> artifacts/v5/cons_{split}.npy (pruned rows)
  python experiments/e16_consensus.py stage2 [--tag cons]         -> stage2_<tag>.json, per_entity_stage2_<tag>.npy
  python experiments/e16_consensus.py predict --name <sub> [--tag cons]

Stage 2 is ber.v5.stage2 unchanged except that the STAGE2_REL slot carries CONS_FEATURES (the pruned rows'
consensus matrix is appended to the stage-2 matrix); predict reuses ber.v5.predict the same way.
BER_CONFIG must point at the 0.980 run (artifacts/free/pipeline.yaml).
"""
import argparse
import os
import sys
import time

import numpy as np

from ber import v5 as V
from ber.consensus import CONS_FEATURES, consensus_features

log = V.log


def build(split: str) -> None:
    t0 = time.time()
    keys, _ = V.read_keys(split)
    p1 = np.load(V.run_path(f"p1_{split}.npy"))
    tau = V.vcfg()["prune_tau"]
    kept = p1 >= tau
    kk = keys[kept].reset_index(drop=True)
    p1k = p1[kept]
    del keys, p1
    out = np.zeros((len(kk), len(CONS_FEATURES)), np.float32)
    for code, cname in enumerate(V.split_countries(split)):
        idx = np.flatnonzero((kk["country"] == code).to_numpy())
        st = V.country_store(split, cname, cols=["name_n", "addr_n"])
        s1r = kk["s1_row"].to_numpy()[idx]
        src = kk["src"].to_numpy()[idx]
        doc = kk["doc_row"].to_numpy()[idx]

        def rec(col):
            a2, a3 = st.numpy(2, col), st.numpy(3, col)
            return np.where(src == 2, a2[np.where(src == 2, doc, 0)], a3[np.where(src == 3, doc, 0)])

        f = consensus_features(s1r, p1k[idx], st.numpy(1, "name_n")[s1r], st.numpy(1, "addr_n")[s1r],
                               rec("name_n"), rec("addr_n"))
        out[idx] = f.to_numpy(np.float32)
        del st, f
        log.info(f"[e16] {split} {cname}: {len(idx):,} pairs, {time.time() - t0:.0f}s {V.mem_str()}")
    np.save(V.run_path(f"cons_{split}.npy"), out)
    log.info(f"[e16] cons_{split}.npy {out.shape}; means " +
             ", ".join(f"{n} {m:.3f}" for n, m in zip(CONS_FEATURES, out.mean(0))))


# "consr": only counts over CONFIDENT partners (p1 >= 0.5). The plain counts over any other candidate are fooled on
# test, whose extra orphans come in groups (records of deleted S1s) that share their own house number: on test US the
# pairs `cons` newly accepts have cn_sh_num 1.30 but cn_sh_num_hi 0.16 (train fold 0: 1.18 / 1.00).
ROBUST = ["cn_dev_num", "cn_sh_num_hi", "cn_miss_num", "cn_dev_atok", "cn_sh_atok_hi", "cn_dev_tok", "cn_sh_tok_hi",
          "cn_n_other_hi"]
SUBSETS = {"cons": CONS_FEATURES, "consr": ROBUST}


def install(tag: str = "cons") -> None:
    """Route the STAGE2_REL slot of ber.v5 to the consensus matrix of the pruned rows (feature subset of ``tag``)."""
    feats = list(SUBSETS.get(tag, CONS_FEATURES))
    cols = [CONS_FEATURES.index(f) for f in feats]
    cache = {}
    n_kept = {}
    for split in ("train", "test"):
        path = V.run_path(f"cons_{split}.npy")
        if os.path.exists(path):
            cache[split] = np.load(path, mmap_mode="r")
            n_kept[len(cache[split])] = split

    def rel_features(kk, p):
        split = n_kept.get(len(kk))
        if split is None:
            raise ValueError(f"no consensus matrix with {len(kk)} rows (have {list(n_kept)})")
        import pandas as pd
        return pd.DataFrame(np.asarray(cache[split])[:, cols], columns=feats)

    orig = V.vcfg
    V.vcfg = lambda: {**orig(), "stage2_rel": True}
    V.STAGE2_REL = feats
    V.stage2_rel_features = rel_features


def fr_odds(name: str, tag: str, odds: float) -> None:
    """Re-decide the saved test p2 of ``tag`` with France-only stage-2 odds x ``odds``; India / US unchanged."""
    import json

    from ber.submit import finalize_submission

    res = json.load(open(V.run_path(f"stage2{V.sfx(tag)}.json")))
    keys, _ = V.read_keys("test")
    kept = np.load(V.run_path(f"kept_test{V.sfx(tag)}.npy"))
    p2 = np.load(V.run_path(f"p2_test{V.sfx(tag)}.npy")).astype(np.float64)
    kk = keys[kept].reset_index(drop=True)
    del keys
    fr = (kk["country"] == V.split_countries("test").index("France")).to_numpy()
    q = p2.copy()
    q[fr] = p2[fr] * odds / (p2[fr] * odds + 1 - p2[fr])
    keep = V.apply_rule(kk, q.astype(np.float32), res["rules"])
    base = np.load(V.run_path(f"keep_test{V.sfx(tag)}.npy"))
    if (keep[~fr] != base[~fr]).any():
        raise RuntimeError("India / US decisions changed: the France-only re-decision must leave them identical")
    log.info(f"[e16] France odds x{odds}: France matches {int(base[fr].sum()):,} -> {int(keep[fr].sum()):,} "
             f"({keep[fr].sum() / max(base[fr].sum(), 1) - 1:+.2%}); India / US identical")
    mpath, cpath, n_m, n_c = V.write_outputs(kk, keep, f"fr_odds{odds}")
    finalize_submission(name, mpath, cpath, offline_metrics=res["report"], n_match_pairs=n_m, n_candidate_pairs=n_c,
                        notes=f"stage2_{tag} (E16 record consensus) with France-only stage-2 odds x{odds}; "
                              f"India / US identical to the plain {tag} submission")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", required=True)
    s = sub.add_parser("stage2")
    s.add_argument("--tag", default="cons")
    p = sub.add_parser("predict")
    p.add_argument("--name", required=True)
    p.add_argument("--tag", default="cons")
    f = sub.add_parser("fr-odds")
    f.add_argument("--name", required=True)
    f.add_argument("--tag", default="cons")
    f.add_argument("--odds", type=float, required=True)
    a = ap.parse_args()
    V.setup_logging()
    if a.cmd == "build":
        build(a.split)
    elif a.cmd == "stage2":
        install(a.tag)
        V.stage2(a.tag, False)
    elif a.cmd == "fr-odds":
        fr_odds(a.name, a.tag, a.odds)
    else:
        install(a.tag)
        V.predict(a.name, a.tag)


if __name__ == "__main__":
    sys.exit(main())
