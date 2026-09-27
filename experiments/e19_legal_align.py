"""E19: legal forms are whole-word edits, never character mutations (ber.edit_ops.SKIP_LEGAL_ALIGN).

The E9 descriptors aligned a deleted S1 token with the closest record token, so a legal-form swap became letter edits:
French forms are prefixes of each other (sa -> sas = "tok_appended:s", log-LR +3.3; sasu -> sas +2.95), so on France
(unseen in training) same-address legal-form swaps got mean p2 0.37 although they are 93-98% true in India / US train.
Here the EDITOP_FEATURES of every pruned pair are recomputed with the fix; stage 2 = the consr stage 2 (0.980 run's
stage 1, ROBUST consensus features) retrained on them.

  python experiments/e19_legal_align.py check                 # fix OFF must reproduce the stored union features
  python experiments/e19_legal_align.py build --split train|test   -> artifacts/v5/eo_fix_{split}.npy (pruned rows)
  python experiments/e19_legal_align.py stage2 --tag legal
  python experiments/e19_legal_align.py predict --tag legal --name sub_v5_legal [--france-only-name sub_v5_consr_frlegal]
"""
import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EO = ["eo_llr", "eo_decoy_rep", "eo_ocr_rep", "eo_legal_ins", "eo_appended", "eo_dba", "eo_house_rel", "eo_num_rel"]


def _work(args):
    a_tr, b_tr, a_dig, b_dig, b_addr, skip = args
    from ber import edit_ops
    edit_ops.SKIP_LEGAL_ALIGN = skip
    f = edit_ops.edit_op_features(a_tr, b_tr, a_dig, b_dig, b_addr)
    return np.column_stack([f[k].astype(np.float32) for k in EO])


def eo_block(pool, st, src, s1r, doc, skip, workers):
    a_tr, b_tr = st.strings(1, "name_tr", s1r), st.strings(src, "name_tr", doc)
    a_dg, b_dg = st.strings(1, "addr_digits", s1r), st.strings(src, "addr_digits", doc)
    b_ad = st.strings(src, "addr_n", doc)
    n = len(s1r)
    bounds = np.linspace(0, n, 4 * workers + 1).astype(int)
    parts = [(a_tr[s:e], b_tr[s:e], a_dg[s:e], b_dg[s:e], b_ad[s:e], skip) for s, e in zip(bounds[:-1], bounds[1:])
             if e > s]
    return np.vstack(list(pool.map(_work, parts))) if parts else np.zeros((0, len(EO)), np.float32)


def pruned(split):
    from ber import v5 as V
    keys, ents = V.read_keys(split)
    p1 = np.load(V.run_path(f"p1_{split}.npy"))
    kept = p1 >= V.vcfg()["prune_tau"]
    return keys, ents, p1, kept


def build(split, skip=True, limit=None, workers=12):
    from ber import v5 as V
    t0 = time.time()
    keys, _, _, kept = pruned(split)
    kk = keys[kept].reset_index(drop=True)
    del keys
    out = np.zeros((len(kk), len(EO)), np.float32)
    with ProcessPoolExecutor(workers) as pool:
        for code, cname in enumerate(V.split_countries(split)):
            st = V.country_store(split, cname, cols=["name_tr", "addr_digits", "addr_n"])
            for s in (2, 3):
                idx = np.flatnonzero(((kk["country"] == code) & (kk["src"] == s)).to_numpy())
                if limit:
                    idx = idx[:limit]
                for a in range(0, len(idx), 1_000_000):
                    b = idx[a:a + 1_000_000]
                    out[b] = eo_block(pool, st, s, kk["s1_row"].to_numpy()[b], kk["doc_row"].to_numpy()[b], skip,
                                      workers)
                V.log.info(f"[e19] {split} {cname} S{s}: {len(idx):,} pairs {time.time() - t0:.0f}s {V.mem_str()}")
            del st
            gc.collect()
    return kk, out


def check():
    """Fix OFF on a sample of each (split, country, source) must equal the stored union columns."""
    from ber import v5 as V
    for split in ("train", "test"):
        _, _, _, kept = pruned(split)
        kk, new = build(split, skip=False, limit=20_000)
        stored = V.read_matrix(split, kept, EO)
        rows = np.flatnonzero(np.abs(new).sum(1) > 0)
        diff = np.abs(new[rows] - stored[rows]).max(0)
        V.log.info(f"[e19] check {split}: {len(rows):,} recomputed rows, max |diff| per feature "
                   + ", ".join(f"{n} {d:.4g}" for n, d in zip(EO, diff)))


def install_fixed_editops():
    """consr stage 2 exactly as E16 runs it (``E16.install('consr')`` + ``ber.v5.stage2`` / ``predict``), except that
    ``v5.read_matrix`` returns the fixed EDITOP columns for the pruned rows of each split."""
    import e16_consensus as E16
    from ber import v5 as V
    E16.install("consr")
    orig = V.read_matrix
    eo = {}
    for split in ("train", "test"):
        path = V.run_path(f"eo_fix_{split}.npy")
        if os.path.exists(path):
            eo[split] = np.load(path, mmap_mode="r")

    def read_matrix(split, rows_mask, names, batch=1_000_000):
        X = orig(split, rows_mask, names, batch)
        cols = [(names.index(f), j) for j, f in enumerate(EO) if f in names]
        if cols:
            if split not in eo or len(eo[split]) != len(X):
                raise ValueError(f"no fixed EDITOP matrix for {split} with {len(X)} rows")
            for i, j in cols:
                X[:, i] = eo[split][:, j]
            V.log.info(f"[e19] {split}: {len(cols)} EDITOP columns replaced by the fixed ones ({len(X):,} rows)")
        return X

    V.read_matrix = read_matrix


def mix_france(tag, fr_name):
    """India / US decisions of consr, France decisions of ``tag`` (same pruned rows)."""
    from ber import v5 as V
    from ber.submit import finalize_submission
    keys, _ = V.read_keys("test")
    kept = np.load(V.run_path(f"kept_test{V.sfx(tag)}.npy"))
    if (np.load(V.run_path("kept_test_consr.npy")) != kept).any():
        raise RuntimeError("consr and this run prune differently")
    kk = keys[kept].reset_index(drop=True)
    del keys
    keep, base_keep = np.load(V.run_path(f"keep_test{V.sfx(tag)}.npy")), np.load(V.run_path("keep_test_consr.npy"))
    for code, c in enumerate(V.split_countries("test")):
        m = (kk["country"] == code).to_numpy()
        n1 = V.country_store("test", c, cols=["entity_id"]).n(1)
        V.log.info(f"[e19] test {c}: matches/S1 consr {base_keep[m].sum() / n1:.4f} -> {tag} {keep[m].sum() / n1:.4f} "
                   f"(added {(keep & ~base_keep)[m].sum() / n1:.4f}, removed {(~keep & base_keep)[m].sum() / n1:.4f} per S1)")
    fr = (kk["country"] == V.split_countries("test").index("France")).to_numpy()
    mpath, cpath, n_m, n_c = V.write_outputs(kk, np.where(fr, keep, base_keep), f"predict{V.sfx(tag)}_fr_only")
    finalize_submission(fr_name, mpath, cpath,
                        offline_metrics=json.load(open(V.run_path("stage2_consr.json")))["report"],
                        n_match_pairs=n_m, n_candidate_pairs=n_c,
                        notes=f"India / US = sub_v5_consr decisions; France = stage2_{tag} (E19 legal-form fix)")


def main():
    from ber import v5 as V
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    b = sub.add_parser("build")
    b.add_argument("--split", required=True)
    s = sub.add_parser("stage2")
    s.add_argument("--tag", default="legal")
    p = sub.add_parser("predict")
    p.add_argument("--tag", default="legal")
    p.add_argument("--name", required=True)
    p.add_argument("--france-only-name", default="")
    a = ap.parse_args()
    V.setup_logging()
    if a.cmd == "check":
        check()
    elif a.cmd == "build":
        _, out = build(a.split)
        np.save(V.run_path(f"eo_fix_{a.split}.npy"), out)
        V.log.info(f"[e19] eo_fix_{a.split}.npy {out.shape}; means " + ", ".join(f"{n} {m:.3f}" for n, m in zip(EO, out.mean(0))))
    elif a.cmd == "stage2":
        install_fixed_editops()
        V.stage2(a.tag, False)
    else:
        install_fixed_editops()
        V.predict(a.name, a.tag)
        if a.france_only_name:
            mix_france(a.tag, a.france_only_name)


if __name__ == "__main__":
    main()
