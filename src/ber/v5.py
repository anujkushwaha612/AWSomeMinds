"""Part B of strategy_v5.md: union-feature GBDT (stage 1) -> pruning -> stage 2 -> decision (CPU).

Leakage rule (§2): GBDT models are cross-fitted on ``v5.gbdt_folds`` (default 0, 1, 2);
the encoder was trained on ``v5.encoder_folds`` (3, 4), which the GBDT never trains on.
Pairs of encoder-fold entities get predictions from the mean of the fold models (they
are needed as competitors for arbitration and stage-2 features, never as training rows).
Decision rules are tuned on the gbdt folds except ``report_fold`` and reported on it.

Stages (checkpointed under artifacts/v5/):
  stage1   LightGBM on FEATURES_V5            -> p1 (train OOF/mean, test mean)
  stage2   prune pairs with p1 < prune_tau   -> candidate set of the final model;
           stage-2 features from p1 (record / S1 competition, counts vs caps, margins)
           [+ cross-encoder logit ``ce`` and its record margin, if v5.cross_encoder.enabled]
           LightGBM on [FEATURES_V5, p1, STAGE2 (, CE)] -> p2; decision: (T_first, T_rest) grid
           vs expected-F0.5 prefix on isotonic-calibrated p2 (the better on tuning folds wins)
  predict  test: stage 1 -> prune -> stage 2 -> decision -> both TSVs, validated, snapshot
  compare  paired bootstrap on fold-0 entities: every stage vs the baseline run, and
           stage2 vs stage2_noce (the cross-encoder ablation) when both exist

``--tag`` keeps several stage-2 variants side by side (files get a ``_<tag>`` suffix);
``--no-ce`` trains stage 2 without the cross-encoder features.

Run:  python -u -m ber.v5 stage1
      python -u -m ber.v5 stage2 --tag noce --no-ce
      python -u -m ber.v5 stage2
      python -u -m ber.v5 predict --name sub03_v5 [--tag noce]
      python -u -m ber.v5 compare
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .baseline import Decider, EntityScorer, entity_key, write_lists
from .config import REPO_ROOT, artifact_path, ensure_parent, load_config
from .eval.scorer import k_bucket, paired_bootstrap
from .features import KEY_COLS
from .memory import mem_str
from .neural.common import country_store, vdir
from .store import split_countries
from .union import FEATURES_V5

STAGE2 = ["p1", "rec_pmax", "rec_p2", "p_minus_rec_other", "rec_prank", "rec_n",
          "s1src_pmax", "s1src_prank", "s1src_n05", "s1src_psum", "p_minus_s1src_other",
          "s1_n05", "s1_pmax_other_src", "s1_n_best"]
CE_FEATURES = ["ce", "ce_minus_rec_other"]
log = logging.getLogger("ber.v5")


# ------------------------------------------------------------------ paths / io
def vcfg() -> dict:
    return load_config()["v5"]


def run_path(*parts: str) -> str:
    """Outputs of this run: artifacts/<BER_V5_RUN or 'v5'>/."""
    return artifact_path(os.environ.get("BER_V5_RUN") or vdir("v5"), *parts)


def sfx(tag: str) -> str:
    """File suffix of a tagged stage-2 variant ('' for the main one)."""
    return f"_{tag}" if tag else ""


def ce_enabled() -> bool:
    return bool((vcfg().get("cross_encoder") or {}).get("enabled", False))


def feat_path(split: str, country: str, entities: bool = False) -> str:
    return artifact_path(vdir("union"), split,
                         f"{country}{'_entities' if entities else ''}.parquet")


def setup_logging() -> None:
    path = run_path("log.txt")
    ensure_parent(path)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def read_keys(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keys (+label on train) of every union pair and the entity tables, all countries."""
    keys, ents = [], []
    cols = KEY_COLS + (["label"] if split == "train" else [])
    for code, country in enumerate(split_countries(split)):
        k = pq.read_table(feat_path(split, country), columns=cols).to_pandas()
        k["country"] = np.int8(code)
        e = pd.read_parquet(feat_path(split, country, entities=True))
        e["country"] = np.int8(code)
        e["country_name"] = country
        keys.append(k)
        ents.append(e)
    return pd.concat(keys, ignore_index=True), pd.concat(ents, ignore_index=True)


def resolve_features() -> list[str]:
    """FEATURES_V5 columns present in every union file of both splits.

    Union files built before the number-conflict features existed lack them; the run
    then goes on without them (loud warning) instead of failing.
    """
    have = None
    for split in ("train", "test"):
        for country in split_countries(split):
            names = set(pq.ParquetFile(feat_path(split, country)).schema_arrow.names)
            have = names if have is None else have & names
    missing = [f for f in FEATURES_V5 if f not in have]
    if missing:
        log.warning(f"WARNING: union files lack {missing}; rebuild them (run_all -Redo union_train, "
                    f"then union_test) to use these features. Continuing without them.")
    return [f for f in FEATURES_V5 if f in have]


def stage1_features() -> list[str]:
    """The feature list stage 1 was trained with (stage-2 inputs must match it)."""
    res = json.load(open(run_path("stage1.json")))
    return res.get("features", FEATURES_V5)


def read_matrix(split: str, rows_mask: np.ndarray, names: list[str], batch: int = 1_000_000) -> np.ndarray:
    """Columns ``names`` of the selected rows (global row order), as one float32 matrix."""
    X = np.empty((int(rows_mask.sum()), len(names)), dtype=np.float32)
    off = fill = 0
    for country in split_countries(split):
        pf = pq.ParquetFile(feat_path(split, country))
        for rb in pf.iter_batches(batch_size=batch, columns=names):
            sel = rows_mask[off:off + rb.num_rows]
            n = int(sel.sum())
            if n:
                M = np.column_stack([rb.column(i).to_numpy(zero_copy_only=False).astype(np.float32)
                                     for i in range(rb.num_columns)])
                X[fill:fill + n] = M[sel]
                fill += n
            off += rb.num_rows
    return X


def pair_folds(keys: pd.DataFrame, ents: pd.DataFrame) -> np.ndarray:
    """Fold of every pair's S1 entity (via per-country entity arrays)."""
    out = np.empty(len(keys), dtype=np.int8)
    for code in np.unique(keys["country"]):
        em = (ents["country"] == code).to_numpy()
        pm = (keys["country"] == code).to_numpy()
        out[pm] = ents.loc[em, "fold"].to_numpy()[keys.loc[pm, "s1_row"].to_numpy()]
    return out


def encoder_seen(keys: pd.DataFrame) -> np.ndarray:
    """True for pairs whose record was a positive in the encoder's training data."""
    path = artifact_path(vdir("neural"), "train_pairs.parquet")
    if not os.path.exists(path):
        return np.zeros(len(keys), dtype=bool)
    tp = pd.read_parquet(path, columns=["country", "src", "doc_row"])
    seen = set(((tp["country"].to_numpy(np.int64) << 40) | (tp["src"].to_numpy(np.int64) << 32)
                | tp["doc_row"].to_numpy(np.int64)).tolist())
    rec = (keys["country"].to_numpy(np.int64) << 40) | (keys["src"].to_numpy(np.int64) << 32) \
        | keys["doc_row"].to_numpy(np.int64)
    return pd.Series(rec).isin(seen).to_numpy()


def lgb_params() -> dict:
    c = vcfg()
    return {"objective": "binary", "verbosity": -1, "num_threads": os.cpu_count(),
            "seed": load_config()["seed"], **{k: v for k, v in c["lgb"].items()
                                               if k not in ("num_rounds", "early_stopping")}}


def entity_uniform(ent_keys: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic U(0,1) per entity key (all rows of one entity get the same value)."""
    x = (ent_keys.astype(np.uint64) + np.uint64(salt)) * np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def cross_fit(Xtr: np.ndarray, ytr: np.ndarray, fsub: np.ndarray, ent: np.ndarray,
              names: list[str], tag: str) -> list:
    """One LightGBM per gbdt fold f, trained on the given rows with fold != f.

    Early stopping uses ``v5.early_stop_share`` of the training ENTITIES (all their rows),
    so fit and validation rows never share an entity. ``Xtr`` is binned once; pass it
    without keeping another reference so its memory is released during boosting.
    """
    c = vcfg()
    params = lgb_params()
    full = lgb.Dataset(Xtr, ytr, feature_name=names, params=params, free_raw_data=True)
    full.construct()
    del Xtr
    gc.collect()
    es_u = entity_uniform(ent, load_config()["seed"] + 7)
    models = []
    for f in c["gbdt_folds"]:
        t = time.time()
        tr = fsub != f
        es = tr & (es_u < c.get("early_stop_share", 0.1))
        booster = lgb.train(params, full.subset(np.flatnonzero(tr & ~es)), c["lgb"]["num_rounds"],
                            valid_sets=[full.subset(np.flatnonzero(es))],
                            callbacks=[lgb.early_stopping(c["lgb"]["early_stopping"], verbose=False)])
        booster.save_model(run_path("models", f"{tag}_fold{f}.txt"))
        booster.free_dataset()
        models.append(booster)
        gc.collect()
        log.info(f"[{tag}] fold {f}: {int((tr & ~es).sum()):,} rows, {booster.best_iteration} trees, "
                 f"{time.time() - t:.0f}s {mem_str()}")
        if len(models) == 1:                  # which features carry the model (evidence for keeping them)
            g = booster.feature_importance("gain")
            share = g / max(g.sum(), 1e-12)
            top = np.argsort(-share)[:20]
            log.info(f"[{tag}] gain share: " + ", ".join(f"{names[i]} {share[i]:.3f}" for i in top))
    del full
    return models


def cross_predict_block(models, Xb: np.ndarray, fb: np.ndarray | None) -> np.ndarray:
    """OOF p for gbdt-fold rows (model of their fold), mean of the models elsewhere / on test."""
    preds = np.stack([m.predict(Xb, num_threads=os.cpu_count()) for m in models])
    out = preds.mean(0)
    if fb is not None:
        for i, f in enumerate(vcfg()["gbdt_folds"]):
            out = np.where(fb == f, preds[i], out)
    return out.astype(np.float32)


def predict_cross(models, X: np.ndarray, fold: np.ndarray | None, batch: int = 2_000_000) -> np.ndarray:
    """:func:`cross_predict_block` over an in-memory matrix, in batches."""
    p = np.empty(len(X), dtype=np.float32)
    for a in range(0, len(X), batch):
        p[a:a + batch] = cross_predict_block(models, X[a:a + batch],
                                             None if fold is None else fold[a:a + batch])
    return p


def predict_stream(models, split: str, fold: np.ndarray | None, n: int, names: list[str],
                   batch: int = 1_000_000) -> np.ndarray:
    """:func:`cross_predict_block` streamed over the union feature files (low memory)."""
    p = np.empty(n, dtype=np.float32)
    off = 0
    for country in split_countries(split):
        pf = pq.ParquetFile(feat_path(split, country))
        for rb in pf.iter_batches(batch_size=batch, columns=names):
            M = np.column_stack([rb.column(i).to_numpy(zero_copy_only=False).astype(np.float32)
                                 for i in range(rb.num_columns)])
            p[off:off + len(M)] = cross_predict_block(models, M, None if fold is None else fold[off:off + len(M)])
            off += len(M)
    return p


def load_models(tag: str) -> list:
    return [lgb.Booster(model_file=run_path("models", f"{tag}_fold{f}.txt")) for f in vcfg()["gbdt_folds"]]


# ------------------------------------------------------------------ stage 2 features
def group_stats(key: np.ndarray, p: np.ndarray, extra: np.ndarray | None = None) -> dict:
    """Per-row statistics of ``p`` within groups of equal ``key`` (sort-based, no pandas).

    Returns arrays aligned with the input: rank (0 = highest p), max, second (0 if the
    group has one row), n (group size), sum, and sum of ``extra`` if given.
    """
    n = len(key)
    o = np.lexsort((-p, key))
    ks = key[o]
    start = np.ones(n, dtype=bool)
    start[1:] = ks[1:] != ks[:-1]
    starts = np.flatnonzero(start)
    grp = np.cumsum(start) - 1
    size = np.diff(np.append(starts, n))
    ps = p[o]
    second = np.where(size > 1, ps[np.minimum(starts + 1, n - 1)], 0.0)
    stats = {"rank": np.arange(n) - starts[grp], "max": ps[starts][grp], "second": second[grp],
             "n": size[grp], "sum": np.add.reduceat(ps, starts)[grp]}
    if extra is not None:
        stats["extra_sum"] = np.add.reduceat(extra[o], starts)[grp]
    inv = np.empty(n, dtype=np.int64)
    inv[o] = np.arange(n)
    return {k: v[inv] for k, v in stats.items()}


def stage2_features(keys: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Competition / count features from stage-1 p, aligned with ``keys`` rows.

    Record side (a record has at most one parent): best / 2nd-best p, margin to the best
    *other* S1. S1 side per source (true matches <= 5 in S2, <= 6 in S3): best p, rank,
    records with p > 0.5, sum of p, margin to the best other record. Across sources:
    records with p > 0.5, best p in the other source, and how many records have this S1
    as their best candidate (E4: 72% of false positives are orphan records; many orphans
    piling onto one S1 is a signal).
    """
    c = keys["country"].to_numpy(np.int64)
    src = keys["src"].to_numpy(np.int64)
    doc = keys["doc_row"].to_numpy(np.int64)
    s1 = keys["s1_row"].to_numpy(np.int64)
    p = np.asarray(p, dtype=np.float64)
    hi = (p > 0.5).astype(np.float64)
    r = group_stats((c << 40) | (src << 32) | doc, p)
    s = group_stats((c << 40) | (src << 32) | s1, p, hi)
    a = group_stats((c << 40) | s1, p, hi)
    out = pd.DataFrame({
        "p1": p.astype(np.float32),
        "rec_pmax": r["max"].astype(np.float32), "rec_p2": r["second"].astype(np.float32),
        "p_minus_rec_other": np.where(r["rank"] == 0, p - r["second"], p - r["max"]).astype(np.float32),
        "rec_prank": r["rank"].astype(np.int16), "rec_n": r["n"].astype(np.int16),
        "s1src_pmax": s["max"].astype(np.float32), "s1src_prank": s["rank"].astype(np.int16),
        "s1src_n05": s["extra_sum"].astype(np.int16), "s1src_psum": s["sum"].astype(np.float32),
        "p_minus_s1src_other": np.where(s["rank"] == 0, p - s["second"], p - s["max"]).astype(np.float32),
        "s1_n05": a["extra_sum"].astype(np.int16),
    })
    # best p of this S1 in the other source (2 <-> 3)
    s1src_key = (c << 40) | (src << 32) | s1
    per = pd.Series(s["max"]).groupby(s1src_key).max()
    other_key = (c << 40) | ((5 - src) << 32) | s1
    out["s1_pmax_other_src"] = per.reindex(other_key).fillna(0).to_numpy(np.float32)
    # number of records whose best candidate is this S1
    best_rows = r["rank"] == 0
    cnt = pd.Series(((c << 40) | s1)[best_rows]).value_counts()
    out["s1_n_best"] = cnt.reindex((c << 40) | s1).fillna(0).to_numpy(np.int16)
    return out[STAGE2]


def ce_features(keys: pd.DataFrame, ce: np.ndarray) -> pd.DataFrame:
    """Cross-encoder logit and its margin to the record's best OTHER scored S1.

    NaN where the pair was not scored (p1 outside the band) or the record has no other
    scored candidate; LightGBM routes missing values on its own.
    """
    ce = np.asarray(ce, dtype=np.float64)
    margin = np.full(len(ce), np.nan)
    ok = np.isfinite(ce)
    if ok.any():
        rec = (keys["country"].to_numpy(np.int64) << 40) | (keys["src"].to_numpy(np.int64) << 32) \
            | keys["doc_row"].to_numpy(np.int64)
        g = group_stats(rec[ok], ce[ok])
        v = ce[ok]
        margin[ok] = np.where(g["n"] > 1, np.where(g["rank"] == 0, v - g["second"], v - g["max"]), np.nan)
    return pd.DataFrame({"ce": ce.astype(np.float32), "ce_minus_rec_other": margin.astype(np.float32)})


# ------------------------------------------------------------------ decision rules
def expected_f_keep(keys: pd.DataFrame, q: np.ndarray, rec_best: np.ndarray, beta2: float = 0.25) -> np.ndarray:
    """Per entity, keep the prefix of its arbitrated candidates (by calibrated q) that
    maximizes expected F0.5 ~ (1+b2) S_m / (b2 E[k] + m); the empty prediction scores
    P(no true match) = prod(1 - q). Candidates are assumed independent."""
    idx = np.flatnonzero(rec_best)
    keep = np.zeros(len(q), dtype=bool)
    if not len(idx):
        return keep
    ent = entity_key(keys["country"].to_numpy()[idx], keys["s1_row"].to_numpy()[idx])
    qq = np.clip(q[idx].astype(np.float64), 1e-6, 1 - 1e-6)
    o = np.lexsort((-qq, ent))
    e, qs = ent[o], qq[o]
    start = np.ones(len(o), dtype=bool)
    start[1:] = e[1:] != e[:-1]
    starts = np.flatnonzero(start)
    grp = np.cumsum(start) - 1
    pos = np.arange(len(o)) - starts[grp]
    cs = np.cumsum(qs)
    cs = cs - np.concatenate([[0.0], cs[starts[1:] - 1]])[grp]
    ek = np.add.reduceat(qs, starts)[grp]
    score = (1 + beta2) * cs / (beta2 * ek + pos + 1)
    s0 = np.exp(np.add.reduceat(np.log1p(-qs), starts))
    best = np.maximum.reduceat(score, starts)
    # m* = first position reaching the group max (1-based), 0 if empty is better
    is_best = score >= best[grp] - 1e-12
    first_best = np.full(len(starts), -1)
    pos_best = np.where(is_best, pos, np.iinfo(np.int64).max)
    first_best = np.minimum.reduceat(pos_best, starts)
    m_star = np.where(s0 >= best, 0, first_best + 1)
    keep[idx[o[pos < m_star[grp]]]] = True
    return keep


def tune_rules(keys, ents, p, label, rep) -> dict:
    """Pick (T_first, T_rest) and compare with the expected-F rule on tuning folds."""
    from sklearn.isotonic import IsotonicRegression

    v = vcfg()
    folds = np.array(v["gbdt_folds"])
    tune_mask = np.isin(ents["fold"].to_numpy(), folds[folds != rep])
    rep_mask = (ents["fold"] == rep).to_numpy()
    dec = Decider(keys, p)
    tsc, rsc = EntityScorer(ents, keys, tune_mask), EntityScorer(ents, keys, rep_mask)
    lo, hi, step = v["grid"]
    grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    best = (-1.0, None, None)
    for t1 in grid:
        for t2 in grid:
            f = tsc.macro(dec.keep(t1, t2))
            if f > best[0]:
                best = (f, float(t1), float(t2))
    thr = {"rule": "threshold", "t_first": best[1], "t_rest": best[2], "tune_f05": best[0],
           "report_f05": rsc.macro(dec.keep(best[1], best[2]))}
    # isotonic calibration on tuning-fold pairs that won arbitration
    pk = pair_folds(keys, ents)
    cal_rows = dec.rec_best & np.isin(pk, folds[folds != rep])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[cal_rows], label[cal_rows])
    q = iso.predict(p).astype(np.float32)
    keep_e = expected_f_keep(keys, q, dec.rec_best)
    exp = {"rule": "expected_f", "tune_f05": tsc.macro(keep_e), "report_f05": rsc.macro(keep_e)}
    chosen = exp if exp["tune_f05"] > thr["tune_f05"] else thr
    return {"threshold": thr, "expected_f": exp, "chosen": chosen["rule"],
            "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist()}


def apply_rule(keys, p, rules: dict) -> np.ndarray:
    """Predicted-pair mask under the chosen decision rule."""
    dec = Decider(keys, p)
    if rules["chosen"] == "threshold":
        t = rules["threshold"]
        return dec.keep(t["t_first"], t["t_rest"])
    q = np.interp(p, rules["iso_x"], rules["iso_y"]).astype(np.float32)
    return expected_f_keep(keys, q, dec.rec_best)


def report(tag: str, keys, ents, keep, label=None) -> dict:
    """Fold-0 macro F0.5 + oracle + candidates/S1 + per-country / per-k for a kept mask."""
    rep = vcfg()["report_fold"]
    rep_mask = (ents["fold"] == rep).to_numpy()
    sc = EntityScorer(ents, keys, rep_mask)
    f = sc.per_entity(keep)
    orc = sc.per_entity(keys["label"].to_numpy().astype(bool))
    re = ents[rep_mask]
    n = len(re)
    kb = k_bucket(re["k"])
    out = {"macro_f05": float(f.mean()), "oracle_f05": float(orc.mean()),
           "candidates_per_s1": float(len(sc.pairs) / n),
           "predicted_per_s1": float(keep[sc.pairs].sum() / n),
           "per_country": {c: float(f[(re["country_name"] == c).to_numpy()].mean()) for c in re["country_name"].unique()},
           "per_k": {int(b): float(f[kb == b].mean()) for b in np.unique(kb)}}
    log.info(f"[{tag}] fold-{rep} macro F0.5 {out['macro_f05']:.5f} | oracle {out['oracle_f05']:.5f} | "
             f"{out['candidates_per_s1']:.2f} cand/S1 | {out['predicted_per_s1']:.2f} pred/S1 | "
             f"{json.dumps({k: round(x, 4) for k, x in out['per_country'].items()})}")
    np.save(run_path(f"per_entity_{tag}.npy"), f)
    return out


# ------------------------------------------------------------------ stage 1
def stage1() -> None:
    """Cross-fitted stage-1 LightGBM on the union features; p1 for train and test."""
    v = vcfg()
    os.makedirs(run_path("models"), exist_ok=True)
    keys, ents = read_keys("train")
    y = keys["label"].to_numpy()
    fold = pair_folds(keys, ents)
    in_gbdt = np.isin(fold, v["gbdt_folds"])
    if v["drop_encoder_records"]:
        in_gbdt &= ~encoder_seen(keys)
    rng = np.random.default_rng(load_config()["seed"])
    ent_u = {}
    for code in np.unique(keys["country"]):
        ent_u[code] = rng.random(int((ents["country"] == code).sum()))
    u = np.empty(len(keys), dtype=np.float64)
    for code, arr in ent_u.items():
        pm = (keys["country"] == code).to_numpy()
        u[pm] = arr[keys.loc[pm, "s1_row"].to_numpy()]
    frac = v["train_frac"]
    n_avail = int((in_gbdt & (u < frac)).sum())
    if n_avail > v["max_train_rows"]:          # memory cap: fewer entities, all their candidates
        frac *= v["max_train_rows"] / n_avail
    train_mask = in_gbdt & (u < frac)
    log.info(f"[stage1] {len(keys):,} union pairs; training rows {int(train_mask.sum()):,} "
             f"(gbdt folds {v['gbdt_folds']}, entity share {frac:.3f}) {mem_str()}")
    ent_all = entity_key(keys["country"], keys["s1_row"])
    names = resolve_features()
    models = cross_fit(read_matrix("train", train_mask, names), y[train_mask], fold[train_mask],
                       ent_all[train_mask], names, "stage1")
    gc.collect()
    p1 = predict_stream(models, "train", fold, len(keys), names)
    np.save(run_path("p1_train.npy"), p1)
    dec = Decider(keys, p1)
    rules = tune_rules(keys, ents, p1, y, v["report_fold"])
    keep = apply_rule(keys, p1, rules)
    res = {"features": names, "stage1_rules": rules, "stage1_report": report("stage1", keys, ents, keep)}
    # pruning curve on fold 0 (oracle loss vs candidate count)
    rep_mask = (ents["fold"] == v["report_fold"]).to_numpy()
    sc = EntityScorer(ents, keys, rep_mask)
    lab = y.astype(bool)
    curve = []
    for tau in (0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        kept = p1 >= tau
        curve.append({"tau": tau, "oracle_f05": float(sc.per_entity(lab & kept).mean()),
                      "cand_per_s1": float(kept[sc.pairs].sum() / rep_mask.sum())})
    res["prune_curve"] = curve
    log.info("[stage1] prune curve " + json.dumps(curve))
    del dec
    tkeys, _ = read_keys("test")                                       # test p1: mean of fold models
    np.save(run_path("p1_test.npy"), predict_stream(models, "test", None, len(tkeys), names))
    json.dump(res, open(run_path("stage1.json"), "w"), indent=2, default=float)
    log.info(f"[stage1] done {mem_str()}")


# ------------------------------------------------------------------ stage 2
def load_ce(split: str, n: int) -> np.ndarray:
    """Cross-encoder logits of every union pair of ``split`` (NaN outside the band)."""
    path = run_path(f"ce_{split}.npy")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run `python -m ber.neural.cross_encoder score --split {split}`, "
                         f"or set v5.cross_encoder.enabled: false (or pass --no-ce)")
    ce = np.load(path)
    if len(ce) != n:
        raise ValueError(f"{path} has {len(ce)} rows for {n} union pairs: rerun the cross-encoder scoring")
    return ce


def stage2_matrix(split: str, keys: pd.DataFrame, p1: np.ndarray, kept: np.ndarray,
                  base: list[str], ce: np.ndarray | None = None) -> np.ndarray:
    """[base features, STAGE2 (, CE_FEATURES)] for the pruned rows.

    STAGE2 and the CE margin are computed among the pruned rows only (the rows the
    final model scores).
    """
    kk = keys[kept].reset_index(drop=True)
    parts = [read_matrix(split, kept, base), stage2_features(kk, p1[kept]).to_numpy(np.float32)]
    if ce is not None:
        parts.append(ce_features(kk, ce[kept]).to_numpy(np.float32))
    return np.hstack(parts)


def stage2(tag: str = "", use_ce: bool | None = None) -> None:
    """Pruning + stage-2 LightGBM + decision rule; report and save for predict."""
    v = vcfg()
    use_ce = ce_enabled() if use_ce is None else use_ce
    keys, ents = read_keys("train")
    p1 = np.load(run_path("p1_train.npy"))
    if len(p1) != len(keys):
        raise ValueError(f"p1_train has {len(p1)} rows for {len(keys)} train pairs: rerun stage1")
    base = stage1_features()
    names = base + STAGE2 + (CE_FEATURES if use_ce else [])
    ce = load_ce("train", len(keys)) if use_ce else None
    kept = p1 >= v["prune_tau"]
    kk = keys[kept].reset_index(drop=True)
    y = kk["label"].to_numpy()
    fold = pair_folds(kk, ents)
    in_gbdt = np.isin(fold, v["gbdt_folds"])
    if v["drop_encoder_records"]:
        in_gbdt &= ~encoder_seen(kk)
    X = stage2_matrix("train", keys, p1, kept, base, ce)
    del ce
    name = "stage2" + sfx(tag)
    log.info(f"[{name}] pruned to {len(kk):,} pairs (tau {v['prune_tau']}); matrix {X.shape}; "
             f"cross-encoder features {'ON' if use_ce else 'OFF'} {mem_str()}")
    models = cross_fit(X[in_gbdt], y[in_gbdt], fold[in_gbdt],
                       entity_key(kk["country"], kk["s1_row"])[in_gbdt], names, name)
    p2 = predict_cross(models, X, fold)
    del X
    np.save(run_path(f"p2_train{sfx(tag)}.npy"), p2)
    np.save(run_path(f"kept_train{sfx(tag)}.npy"), kept)
    rules = tune_rules(kk, ents, p2, y, v["report_fold"])
    keep = apply_rule(kk, p2, rules)
    res = {"prune_tau": v["prune_tau"], "tag": tag, "use_ce": bool(use_ce), "base_features": base,
           "features": names, "rules": rules, "report": report(name, kk, ents, keep)}
    log.info(f"[{name}] chosen rule {rules['chosen']}: threshold tune {rules['threshold']['tune_f05']:.5f} "
             f"report {rules['threshold']['report_f05']:.5f} | expected-F tune "
             f"{rules['expected_f']['tune_f05']:.5f} report {rules['expected_f']['report_f05']:.5f}")
    json.dump(res, open(run_path(f"stage2{sfx(tag)}.json"), "w"), indent=2, default=float)


# ------------------------------------------------------------------ predict
def predict(name: str, tag: str = "") -> None:
    """Test: prune by p1, stage-2 p2 (mean of fold models), decision, write + validate."""
    from .submit import finalize_submission

    res = json.load(open(run_path(f"stage2{sfx(tag)}.json")))
    keys, _ = read_keys("test")
    p1 = np.load(run_path("p1_test.npy"))
    if len(p1) != len(keys):
        raise ValueError(f"p1_test has {len(p1)} rows for {len(keys)} test pairs: rerun stage1")
    use_ce = res.get("use_ce", False)
    kept = p1 >= res["prune_tau"]                 # the tau stage 2 was trained with
    kk = keys[kept].reset_index(drop=True)
    X = stage2_matrix("test", keys, p1, kept, res.get("base_features", FEATURES_V5),
                      load_ce("test", len(keys)) if use_ce else None)
    p2 = predict_cross(load_models("stage2" + sfx(tag)), X, None)
    del X
    keep = apply_rule(kk, p2, res["rules"])
    np.save(run_path(f"p2_test{sfx(tag)}.npy"), p2)
    np.save(run_path(f"kept_test{sfx(tag)}.npy"), kept)
    np.save(run_path(f"keep_test{sfx(tag)}.npy"), keep)
    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    mpath, cpath = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    n_s1 = n_m = n_c = 0
    with open(mpath, "w", encoding="utf-8", newline="") as fm, open(cpath, "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for code, country in enumerate(split_countries("test")):
            st = country_store("test", country, cols=["entity_id"])
            pm = (kk["country"] == code).to_numpy()
            src, doc = kk.loc[pm, "src"].to_numpy(), kk.loc[pm, "doc_row"].to_numpy()
            rid = np.where(src == 2, st.numpy(2, "entity_id")[np.where(src == 2, doc, 0)],
                           st.numpy(3, "entity_id")[np.where(src == 3, doc, 0)])
            a, b = write_lists(fm, fc, st.numpy(1, "entity_id"), rid, kk.loc[pm, "s1_row"].to_numpy(), keep[pm])
            n_s1 += st.n(1)
            n_m += a
            n_c += b
    log.info(f"[predict{sfx(tag)}] {n_s1:,} S1: {n_c:,} candidates ({n_c / n_s1:.2f}/S1), {n_m:,} matches "
             f"({n_m / n_s1:.2f}/S1) {mem_str()}")
    finalize_submission(name, str(mpath), str(cpath), offline_metrics=res["report"],
                        n_match_pairs=n_m, n_candidate_pairs=n_c,
                        notes="v5: TF-IDF + fine-tuned e5 union, stage1 GBDT, prune, stage2 GBDT"
                              + (" + cross-encoder feature" if use_ce else " (no cross-encoder)"))


# ------------------------------------------------------------------ compare
def compare(baseline_run: str | None = None) -> None:
    """Paired bootstrap on fold-0 entities: every v5 stage vs the baseline, and the CE ablation."""
    import glob

    baseline_run = baseline_run or vdir("tfidf")
    boot = load_config()["bootstrap"]
    base_dir = artifact_path(baseline_run)
    bm = json.load(open(os.path.join(base_dir, "metrics.json")))
    os.environ["BER_RUN"] = baseline_run
    os.environ["BER_FEATURES"] = baseline_run
    from . import baseline as B
    bkeys, bents = B.read_keys("train")
    bd = bm["decision"]
    bkeep = Decider(bkeys, np.load(os.path.join(base_dir, "oof.npy"))).keep(bd["t_first"], bd["t_rest"], bd["arbitrate"])
    rep = vcfg()["report_fold"]
    f_base = EntityScorer(bents, bkeys, (bents["fold"] == rep).to_numpy()).per_entity(bkeep)
    per = {os.path.basename(pth)[len("per_entity_"):-4]: np.load(pth)
           for pth in sorted(glob.glob(run_path("per_entity_*.npy")))}
    out = {}

    def verdict(bs):
        if bs["ci_low"] > 0 and bs["delta"] >= boot["materiality"]:
            return "KEEP (CI > 0 and delta >= materiality)"
        return "WORSE (CI < 0)" if bs["ci_high"] < 0 else "NO CLEAR DIFFERENCE"

    for tag, f in per.items():
        bs = paired_bootstrap(f_base, f, boot["n_resamples"], boot["alpha"])
        bs["verdict"] = verdict(bs)
        out[f"{tag}_vs_baseline"] = bs
        log.info(f"[compare] {tag} vs {baseline_run}: {f.mean():.5f} vs {f_base.mean():.5f}  "
                 f"delta {bs['delta']:+.5f}  CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]  {bs['verdict']}")
    if "stage2" in per and "stage2_noce" in per:           # the cross-encoder's own contribution
        bs = paired_bootstrap(per["stage2_noce"], per["stage2"], boot["n_resamples"], boot["alpha"])
        bs["verdict"] = verdict(bs)
        out["ce_ablation_stage2_vs_stage2_noce"] = bs
        log.info(f"[compare] cross-encoder ablation: stage2 {per['stage2'].mean():.5f} vs stage2_noce "
                 f"{per['stage2_noce'].mean():.5f}  delta {bs['delta']:+.5f}  "
                 f"CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]  {bs['verdict']}")
    json.dump(out, open(run_path("compare.json"), "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1")
    s2 = sub.add_parser("stage2")
    s2.add_argument("--tag", default="", help="variant name: outputs get a _<tag> suffix")
    s2.add_argument("--no-ce", action="store_true", help="without the cross-encoder features")
    p = sub.add_parser("predict")
    p.add_argument("--name", required=True)
    p.add_argument("--tag", default="", help="which stage-2 variant to use")
    c = sub.add_parser("compare")
    c.add_argument("--baseline", default=None, help="baseline run folder (default: v5.tfidf_run)")
    args = ap.parse_args()
    setup_logging()
    if args.cmd == "stage1":
        stage1()
    elif args.cmd == "stage2":
        stage2(args.tag, False if args.no_ce else None)
    elif args.cmd == "predict":
        predict(args.name, args.tag)
    else:
        compare(args.baseline)


if __name__ == "__main__":
    main()
