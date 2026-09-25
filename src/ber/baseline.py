"""Baseline end-to-end pipeline (Sub #1), memory-lean: retrieval -> features -> LightGBM -> decision.

Stages (each checkpointed under ``artifacts/<run>/``; re-running skips work
already done, so a crash resumes instead of restarting):

  build   --split train|test   per country: word 1+2-gram record-side retrieval
                               (top-k S1 per S2/S3 record), pair features written
                               chunk by chunk to parquet, labels on train
  train                        5-fold entity-level cross-fitting on the FULL train
                               universe -> out-of-fold p for every train pair;
                               decision rule tuned on folds != report_fold and
                               reported on report_fold
  predict --name sub01         test p = mean of the 5 fold models (streamed) ->
                               decision -> matching_results.tsv +
                               candidate_pairs.tsv, validated, snapshotted

Memory design (see the audit in experiments.md):
  * text lives only in the Arrow store of the current country (ber.store);
  * candidates are compact integer/float arrays, never text;
  * features are streamed to/from parquet in batches;
  * LightGBM trains on a sample of *entities* (all their candidates, so the
    natural pair distribution is kept; no negative subsampling) and predicts
    every pair in batches;
  * the decision layer works on compact key arrays (~17 bytes per pair).

Decision rule (plan.md Step 8, D2 + hard arbitration): each S2/S3 record goes to
its highest-p S1 only; an entity's best remaining candidate is kept if
p >= T_first, further ones if p >= T_rest.

Run:  python -u -m ber.baseline build --split train
      python -u -m ber.baseline build --split test
      python -u -m ber.baseline train
      python -u -m ber.baseline predict --name sub01_baseline
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .blocking.word_retrieval import retrieve_country
from .config import REPO_ROOT, artifact_path, ensure_parent, load_config
from .eval.scorer import f05_from_counts, k_bucket
from .features import FEATURES, KEY_COLS, retrieval_features, write_features
from .memory import mem_str
from .store import CountryStore, split_countries

log = logging.getLogger("ber.baseline")


# ------------------------------------------------------------------ utilities
def cfg() -> dict:
    """The ``baseline`` section of configs/pipeline.yaml."""
    return load_config()["baseline"]


def run_path(*parts: str) -> str:
    """Models / OOF / metrics / log of this run: ``artifacts/<BER_RUN>/`` (default ``baseline``)."""
    return artifact_path(os.environ.get("BER_RUN", cfg().get("run_name", "baseline")), *parts)


def feature_path(*parts: str) -> str:
    """Candidates + features: ``artifacts/<BER_FEATURES>/`` (defaults to the run folder).

    Lets a model experiment (new ``BER_RUN``) reuse features built once by another run.
    """
    run = os.environ.get("BER_FEATURES") or os.environ.get("BER_RUN", cfg().get("run_name", "baseline"))
    return artifact_path(run, *parts)


def setup_logging() -> None:
    """Log to stdout (unbuffered) and to artifacts/<run>/log.txt."""
    path = run_path("log.txt")
    ensure_parent(path)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def entity_key(country, s1_row) -> np.ndarray:
    """Unique int64 key per (country, S1 row)."""
    return (np.asarray(country, dtype=np.int64) << 32) | np.asarray(s1_row, dtype=np.int64)


def feature_batches(split: str, country: str, columns, batch: int):
    """Stream ``columns`` of a country's feature file as NumPy float32 matrices."""
    pf = pq.ParquetFile(feature_path(split, f"{country}.parquet"))
    for rb in pf.iter_batches(batch_size=batch, columns=list(columns)):
        yield np.column_stack([rb.column(i).to_numpy(zero_copy_only=False).astype(np.float32)
                               for i in range(rb.num_columns)])


def read_keys(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compact keys of every pair (+label on train) and the entity tables, all countries."""
    keys, ents = [], []
    cols = KEY_COLS + (["label"] if split == "train" else [])
    for code, country in enumerate(split_countries(split)):
        k = pq.read_table(feature_path(split, f"{country}.parquet"), columns=cols).to_pandas()
        k["country"] = np.int8(code)
        e = pd.read_parquet(feature_path(split, f"{country}_entities.parquet"))
        e["country"] = np.int8(code)
        e["country_name"] = country
        keys.append(k)
        ents.append(e)
    return pd.concat(keys, ignore_index=True), pd.concat(ents, ignore_index=True)


# ------------------------------------------------------------------ build
def truth_parents(store: CountryStore, truth: pd.DataFrame) -> dict:
    """Per source: array mapping doc row -> true parent S1 row (-1 = orphan)."""
    s1_index = pd.Index(store.numpy(1, "entity_id"))
    parents = {}
    for src in (2, 3):
        doc_index = pd.Index(store.numpy(src, "entity_id"))
        tt = truth[truth["rid"].str.startswith(f"S{src}-")]
        rows = doc_index.get_indexer(tt["rid"])
        ok = rows >= 0
        parent = np.full(len(doc_index), -1, dtype=np.int32)
        parent[rows[ok]] = s1_index.get_indexer(tt["s1"].to_numpy()[ok])
        parents[src] = parent
    return parents


def build(split: str, limit: int | None = None, force: bool = False) -> None:
    """Candidates + features (+ labels on train) for every country of ``split``."""
    from .io import load_truth_pairs

    c = cfg()
    truth = load_truth_pairs() if split == "train" else None
    folds = pd.read_parquet(artifact_path("folds.parquet")) if split == "train" else None
    with ProcessPoolExecutor(max_workers=c["workers"]) as pool:
        for country in split_countries(split):
            out = feature_path(split, f"{country}.parquet")
            ents_out = feature_path(split, f"{country}_entities.parquet")
            if os.path.exists(out) and os.path.exists(ents_out) and not force:
                log.info(f"[build {split}/{country}] exists, skipping")
                continue
            t0 = time.time()
            store = CountryStore(split, country, limit=limit)
            log.info(f"[build {split}/{country}] S1 {store.n(1):,}  S2 {store.n(2):,}  "
                     f"S3 {store.n(3):,}; text store {store.nbytes() / 2**30:.2f} GB {mem_str()}")
            cand = retrieve_country(store, c["k_rec"], c["max_df"], c["min_score"], pool,
                                    c["chunk"], c["window"], log=log.info)
            assert cand["src"].is_monotonic_increasing
            log.info(f"    {len(cand):,} candidate pairs ({len(cand) / max(store.n(1), 1):.2f} per S1) "
                     f"{mem_str()}")
            cand = retrieval_features(cand)

            ents = pd.DataFrame({"s1_row": np.arange(store.n(1), dtype=np.int32),
                                 "entity_id": store.numpy(1, "entity_id")})
            if split == "train":
                parents = truth_parents(store, truth)
                y = np.zeros(len(cand), dtype=np.int8)
                for src in (2, 3):
                    m = (cand["src"] == src).to_numpy()
                    y[m] = parents[src][cand.loc[m, "doc_row"].to_numpy()] == cand.loc[m, "s1_row"].to_numpy()
                cand["label"] = y
                k = np.zeros(store.n(1), dtype=np.int64)
                for src in (2, 3):
                    p = parents[src]
                    k += np.bincount(p[p >= 0], minlength=store.n(1))
                ents["k"] = k.astype(np.int16)
                ents["fold"] = pd.Index(folds["s1"]).get_indexer(ents["entity_id"])
                ents["fold"] = folds["fold"].to_numpy()[ents["fold"].to_numpy()].astype(np.int8)
                log.info(f"    labels: {int(y.sum()):,} positives = pair recall "
                         f"{y.sum() / max(k.sum(), 1):.4f} (orphan records: label 0)")
            ensure_parent(out)
            tmp = out + ".tmp"
            write_features(cand, store, tmp, c["feature_chunk"], log=log.info)
            os.replace(tmp, out)
            ents.to_parquet(ents_out, index=False)
            del cand, store, ents
            gc.collect()
            log.info(f"[build {split}/{country}] done in {time.time() - t0:.0f}s {mem_str()}")


# ------------------------------------------------------------------ decision
class Decider:
    """Precomputes arbitration and within-entity order once; evaluates rules fast."""

    def __init__(self, keys: pd.DataFrame, p: np.ndarray):
        self.p = p
        rec = (keys["country"].to_numpy(np.int64) << 40) | (keys["src"].to_numpy(np.int64) << 32) \
            | keys["doc_row"].to_numpy(np.int64)
        order = np.lexsort((-p, rec))
        first = np.ones(len(order), dtype=bool)
        first[1:] = rec[order][1:] != rec[order][:-1]
        del rec
        self.rec_best = np.zeros(len(p), dtype=bool)
        self.rec_best[order[first]] = True             # a record goes to its best-p S1 only
        del order, first
        ent = entity_key(keys["country"], keys["s1_row"])
        idx = np.flatnonzero(self.rec_best)
        o = idx[np.lexsort((-p[idx], ent[idx]))]
        e_sorted = ent[o]
        start = np.ones(len(o), dtype=bool)
        start[1:] = e_sorted[1:] != e_sorted[:-1]
        grp = np.cumsum(start) - 1
        starts = np.flatnonzero(start)
        self.pos = np.full(len(p), -1, dtype=np.int32)
        self.pos[o] = np.arange(len(o)) - starts[grp]
        self.top_p = np.zeros(len(p), dtype=np.float32)
        self.top_p[o] = p[o][starts][grp]

    def keep(self, t_first: float, t_rest: float, arbitrate: bool = True) -> np.ndarray:
        """Mask of predicted pairs under (T_first, T_rest)."""
        if not arbitrate:
            return self.p >= t_first
        return self.rec_best & (
            ((self.pos == 0) & (self.p >= t_first))
            | ((self.pos > 0) & (self.p >= t_rest) & (self.top_p >= t_first)))


class EntityScorer:
    """Macro F0.5 over a fixed entity subset; the pair -> entity map is built once."""

    def __init__(self, ents: pd.DataFrame, keys: pd.DataFrame, ent_mask: np.ndarray):
        ekeys = entity_key(ents["country"], ents["s1_row"])[ent_mask]
        order = np.argsort(ekeys)
        keys_sorted = ekeys[order]
        pk = entity_key(keys["country"], keys["s1_row"])
        pos = np.clip(np.searchsorted(keys_sorted, pk), 0, len(keys_sorted) - 1)
        self.pairs = np.flatnonzero(keys_sorted[pos] == pk)      # pairs of in-subset entities
        self.eidx = order[pos[self.pairs]]
        self.lab = keys["label"].to_numpy()[self.pairs].astype(np.float64)
        self.k = ents["k"].to_numpy()[ent_mask]

    def per_entity(self, keep: np.ndarray) -> np.ndarray:
        """Per-entity F0.5 when the pairs in ``keep`` are predicted."""
        kk = keep[self.pairs]
        e = self.eidx[kk]
        m = np.bincount(e, minlength=len(self.k))
        tp = np.bincount(e, weights=self.lab[kk], minlength=len(self.k))
        return f05_from_counts(self.k, m, tp)

    def macro(self, keep: np.ndarray) -> float:
        """Macro F0.5 when the pairs in ``keep`` are predicted."""
        return float(self.per_entity(keep).mean())


def tune(dec: Decider, scorer: EntityScorer) -> dict:
    """Grid-search (T_first, T_rest) with arbitration, plus two simpler references."""
    lo, hi, step = cfg()["grid"]
    grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    best = {"f05": -1.0}
    for t1 in grid:
        for t2 in grid:
            f = scorer.macro(dec.keep(t1, t2))
            if f > best["f05"]:
                best = {"f05": f, "t_first": float(t1), "t_rest": float(t2), "arbitrate": True}
    glob = max((scorer.macro(dec.keep(t, t)), float(t)) for t in grid)
    noarb = max((scorer.macro(dec.keep(t, t, False)), float(t)) for t in grid)
    best["reference"] = {"global_T_arbitrated": {"f05": glob[0], "T": glob[1]},
                         "global_T_no_arbitration": {"f05": noarb[0], "T": noarb[1]}}
    return best


# ------------------------------------------------------------------ train
def train(overrides: dict | None = None) -> None:
    """Cross-fitted LightGBM (entity-sampled training, streamed OOF), tuning, report.

    ``overrides`` replaces baseline config keys for this run (``train_frac``, and
    LightGBM keys under ``lgb``); they are recorded in metrics.json.
    """
    c = dict(cfg())
    c["lgb"] = dict(c["lgb"])
    for k, val in (overrides or {}).items():
        if val is None:
            continue
        if k in c["lgb"]:
            c["lgb"][k] = val
        else:
            c[k] = val
    v = load_config()["validation"]
    keys, ents = read_keys("train")
    names = split_countries("train")
    log.info(f"[train] {len(keys):,} pairs, {len(ents):,} entities, "
             f"positives {int(keys['label'].sum()):,} {mem_str()}")

    # fold and training-sample flag per pair, via per-country entity arrays (no string maps)
    rng = np.random.default_rng(load_config()["seed"])
    pair_fold = np.empty(len(keys), dtype=np.int8)
    pair_samp = np.empty(len(keys), dtype=bool)
    for code in range(len(names)):
        em = (ents["country"] == code).to_numpy()
        pm = (keys["country"] == code).to_numpy()
        e_fold = ents.loc[em, "fold"].to_numpy()
        e_u = rng.random(int(em.sum()))
        rows = keys.loc[pm, "s1_row"].to_numpy()
        pair_fold[pm] = e_fold[rows]
        pair_samp[pm] = e_u[rows] < c["train_frac"]
    y_all = keys["label"].to_numpy()

    # sample pass: read features of sampled entities only, into one preallocated matrix
    # (no list-of-batches + concatenate, which would briefly hold two copies)
    n_samp = int(pair_samp.sum())
    Xs = np.empty((n_samp, len(FEATURES)), dtype=np.float32)
    offset = filled = 0
    for code, country in enumerate(names):
        for X in feature_batches("train", country, FEATURES, c["predict_batch"]):
            sel = pair_samp[offset:offset + len(X)]
            n = int(sel.sum())
            Xs[filled:filled + n] = X[sel]
            filled += n
            offset += len(X)
    ys, fs = y_all[pair_samp], pair_fold[pair_samp]
    log.info(f"[train] training sample: {len(Xs):,} pairs ({c['train_frac']:.0%} of entities, "
             f"all their candidates; {int(ys.sum()):,} positives), "
             f"{Xs.nbytes / 2**30:.2f} GB {mem_str()}")

    params = {"objective": "binary", "verbosity": -1, "num_threads": os.cpu_count(),
              "seed": load_config()["seed"], **{k: v_ for k, v_ in c["lgb"].items()
                                                  if k not in ("num_rounds", "early_stopping")}}
    # Bin the sample once, free the float matrix, and cut each fold's train / early-stop
    # sets as subsets of the binned data (no per-fold copies of the raw features).
    t = time.time()
    full = lgb.Dataset(Xs, ys, feature_name=FEATURES, params=params, free_raw_data=True)
    full.construct()
    del Xs
    gc.collect()
    log.info(f"[train] binned training data in {time.time() - t:.0f}s {mem_str()}")
    model_dir = run_path("models")
    os.makedirs(model_dir, exist_ok=True)
    es_u = np.random.default_rng(load_config()["seed"] + 1).random(len(ys))
    boosters = []
    for f in range(v["n_folds"]):
        t = time.time()
        tr = fs != f
        es = tr & (es_u < 0.1)                       # 10% of the sample for early stopping
        fit = tr & ~es
        booster = lgb.train(params, full.subset(np.flatnonzero(fit)), c["lgb"]["num_rounds"],
                            valid_sets=[full.subset(np.flatnonzero(es))],
                            callbacks=[lgb.early_stopping(c["lgb"]["early_stopping"], verbose=False)])
        booster.save_model(os.path.join(model_dir, f"fold{f}.txt"))
        boosters.append(booster)
        log.info(f"[train] fold {f}: {int(fit.sum()):,} rows, {booster.best_iteration} trees, "
                 f"{time.time() - t:.0f}s {mem_str()}")
    del full, ys, fs, es_u
    gc.collect()

    # streamed out-of-fold prediction for every pair
    oof = np.empty(len(keys), dtype=np.float32)
    offset = 0
    for code, country in enumerate(names):
        for X in feature_batches("train", country, FEATURES, c["predict_batch"]):
            fb = pair_fold[offset:offset + len(X)]
            for f in range(v["n_folds"]):
                m = fb == f
                if m.any():
                    oof[offset + np.flatnonzero(m)] = boosters[f].predict(
                        X[m], num_threads=os.cpu_count())
            offset += len(X)
    np.save(run_path("oof.npy"), oof)
    log.info(f"[train] out-of-fold predictions done {mem_str()}")

    dec = Decider(keys, oof)
    rep = v["report_fold"]
    tune_mask = (ents["fold"] != rep).to_numpy()
    rep_mask = ~tune_mask
    t = time.time()
    best = tune(dec, EntityScorer(ents, keys, tune_mask))
    log.info(f"[train] decision tuned on folds != {rep} in {time.time() - t:.0f}s")
    rep_scorer = EntityScorer(ents, keys, rep_mask)
    keep = dec.keep(best["t_first"], best["t_rest"], best["arbitrate"])
    f_rep = rep_scorer.per_entity(keep)
    oracle = rep_scorer.per_entity(y_all.astype(bool))
    rep_ents = ents[rep_mask]
    in_rep = np.zeros(len(keys), dtype=bool)
    in_rep[rep_scorer.pairs] = True
    metrics = {
        "report_fold": rep, "decision": best,
        "macro_f05": float(f_rep.mean()),
        "oracle_f05_candidates": float(oracle.mean()),
        "pair_recall_candidates": float(y_all[in_rep].sum() / max(rep_ents["k"].sum(), 1)),
        "mean_candidates_per_s1": float(in_rep.sum() / max(len(rep_ents), 1)),
        "mean_predicted_per_s1": float((keep & in_rep).sum() / max(len(rep_ents), 1)),
        "per_country": {n: float(f_rep[(rep_ents["country_name"] == n).to_numpy()].mean())
                        for n in rep_ents["country_name"].unique()},
        "per_k_bucket": {int(b): float(f_rep[k_bucket(rep_ents["k"]) == b].mean())
                         for b in np.unique(k_bucket(rep_ents["k"]))},
        "features": FEATURES, "config": c,
    }
    with open(run_path("metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("[train] fold-%d macro F0.5 %.5f | oracle %.5f | pair recall %.4f | "
             "%.2f candidates/S1 | %.2f predicted/S1", rep, metrics["macro_f05"],
             metrics["oracle_f05_candidates"], metrics["pair_recall_candidates"],
             metrics["mean_candidates_per_s1"], metrics["mean_predicted_per_s1"])
    log.info("[train] decision %s", json.dumps(best))
    log.info("[train] per country %s | per k %s %s", metrics["per_country"],
             metrics["per_k_bucket"], mem_str())


# ------------------------------------------------------------------ predict
def write_lists(fh_match, fh_cand, s1_ids, rid_ids, s1_rows, keep) -> tuple[int, int]:
    """Append one country's rows to both submission files (every S1 gets a row)."""
    order = np.argsort(s1_rows, kind="stable")
    rows_sorted = s1_rows[order]
    bounds = np.searchsorted(rows_sorted, np.arange(len(s1_ids) + 1))
    rid_sorted = rid_ids[order]
    keep_sorted = keep[order]
    n_m = n_c = 0
    for i, s1 in enumerate(s1_ids):
        a, b = bounds[i], bounds[i + 1]
        cands = rid_sorted[a:b]
        matches = cands[keep_sorted[a:b]]
        fh_cand.write(f"{s1}\t{','.join(cands)}\n")
        fh_match.write(f"{s1}\t{','.join(matches)}\n")
        n_c += len(cands)
        n_m += len(matches)
    return n_m, n_c


def predict(name: str) -> None:
    """Score test candidates with the fold models (streamed), decide, write + validate."""
    from .submit import finalize_submission

    c = cfg()
    with open(run_path("metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    d = metrics["decision"]
    keys, _ = read_keys("test")
    names = split_countries("test")
    model_dir = run_path("models")
    boosters = [lgb.Booster(model_file=os.path.join(model_dir, f))
                for f in sorted(os.listdir(model_dir)) if f.startswith("fold")]
    p = np.empty(len(keys), dtype=np.float32)
    offset = 0
    for country in names:
        for X in feature_batches("test", country, FEATURES, c["predict_batch"]):
            p[offset:offset + len(X)] = np.mean(
                [b.predict(X, num_threads=os.cpu_count()) for b in boosters], axis=0)
            offset += len(X)
    log.info(f"[predict] scored {len(keys):,} test pairs with {len(boosters)} models {mem_str()}")
    keep = Decider(keys, p).keep(d["t_first"], d["t_rest"], d["arbitrate"])

    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    mpath, cpath = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    n_s1 = n_match = n_cand = 0
    with open(mpath, "w", encoding="utf-8", newline="") as fm, \
            open(cpath, "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for code, country in enumerate(names):
            store = CountryStore("test", country, cols=["entity_id"])
            pm = (keys["country"] == code).to_numpy()
            src = keys.loc[pm, "src"].to_numpy()
            doc = keys.loc[pm, "doc_row"].to_numpy()
            rid = np.where(src == 2, store.numpy(2, "entity_id")[np.where(src == 2, doc, 0)],
                           store.numpy(3, "entity_id")[np.where(src == 3, doc, 0)])
            nm, nc = write_lists(fm, fc, store.numpy(1, "entity_id"), rid,
                                 keys.loc[pm, "s1_row"].to_numpy(), keep[pm])
            n_s1 += store.n(1)
            n_match += nm
            n_cand += nc
            del store, rid
    log.info(f"[predict] wrote {n_s1:,} S1 rows: {n_cand:,} candidates ({n_cand / n_s1:.2f}/S1), "
             f"{n_match:,} matches ({n_match / n_s1:.2f}/S1) {mem_str()}")
    del keys, p, keep
    gc.collect()
    offline = {k: metrics[k] for k in ("macro_f05", "oracle_f05_candidates",
                                       "mean_candidates_per_s1", "per_country", "decision")}
    finalize_submission(name, str(mpath), str(cpath), offline_metrics=offline,
                        n_match_pairs=n_match, n_candidate_pairs=n_cand,
                        notes="baseline: word1+2 TF-IDF record-side top-k, LightGBM, D2 + arbitration")


def compare(run_a: str, run_b: str) -> None:
    """Paired-bootstrap comparison of two runs on the same fold-0 entities.

    Both runs must share the feature files (same candidates, same row order).
    Each run is scored with its own tuned decision rule, as it would be submitted.
    """
    from .eval.scorer import paired_bootstrap

    keys, ents = read_keys("train")
    v, boot = load_config()["validation"], load_config()["bootstrap"]
    rep_mask = (ents["fold"] == v["report_fold"]).to_numpy()
    scorer = EntityScorer(ents, keys, rep_mask)
    per, summary = {}, {}
    for run in (run_a, run_b):
        base = artifact_path(run)
        oof = np.load(os.path.join(base, "oof.npy"))
        with open(os.path.join(base, "metrics.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        d = m["decision"]
        per[run] = scorer.per_entity(Decider(keys, oof).keep(d["t_first"], d["t_rest"], d["arbitrate"]))
        summary[run] = {"macro_f05": float(per[run].mean()), "t_first": d["t_first"],
                        "t_rest": d["t_rest"], "train_frac": m["config"]["train_frac"]}
    bs = paired_bootstrap(per[run_a], per[run_b], boot["n_resamples"], boot["alpha"])
    rep_ents = ents[rep_mask]
    by_country = {n: float(per[run_b][(rep_ents["country_name"] == n).to_numpy()].mean()
                           - per[run_a][(rep_ents["country_name"] == n).to_numpy()].mean())
                  for n in rep_ents["country_name"].unique()}
    kb = k_bucket(rep_ents["k"])
    by_k = {int(b): float(per[run_b][kb == b].mean() - per[run_a][kb == b].mean()) for b in np.unique(kb)}
    material = bs["ci_low"] > 0 and bs["delta"] >= boot["materiality"]
    verdict = ("KEEP (better, CI > 0 and >= materiality)" if material
               else "WORSE (CI < 0)" if bs["ci_high"] < 0 else "NO CLEAR DIFFERENCE (keep simpler run)")
    out = {"a": summary[run_a], "b": summary[run_b], "delta": bs, "delta_by_country": by_country,
           "delta_by_k": by_k, "verdict": verdict}
    with open(os.path.join(artifact_path(run_b), f"compare_vs_{run_a}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info(f"[compare] {run_a}: {summary[run_a]['macro_f05']:.5f}   {run_b}: {summary[run_b]['macro_f05']:.5f}")
    log.info(f"[compare] delta {bs['delta']:+.5f}  95% CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]  "
             f"on {bs['n_entities']:,} fold-0 entities -> {verdict}")
    log.info(f"[compare] delta by country {json.dumps({k: round(x, 5) for k, x in by_country.items()})}")
    log.info(f"[compare] delta by k {json.dumps({k: round(x, 5) for k, x in by_k.items()})}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", required=True, choices=["train", "test"])
    b.add_argument("--limit", type=int, default=None, help="rows per source per country (smoke tests)")
    b.add_argument("--force", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--train-frac", type=float, default=None, help="share of entities to train on")
    t.add_argument("--num-leaves", type=int, default=None)
    t.add_argument("--learning-rate", type=float, default=None)
    t.add_argument("--num-rounds", type=int, default=None)
    t.add_argument("--min-data-in-leaf", type=int, default=None)
    p = sub.add_parser("predict")
    p.add_argument("--name", required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("run_a")
    cmp_.add_argument("run_b")
    args = ap.parse_args()
    setup_logging()
    if args.cmd == "build":
        build(args.split, args.limit, args.force)
    elif args.cmd == "train":
        train({"train_frac": args.train_frac, "num_leaves": args.num_leaves,
               "learning_rate": args.learning_rate, "num_rounds": args.num_rounds,
               "min_data_in_leaf": args.min_data_in_leaf})
    elif args.cmd == "compare":
        compare(args.run_a, args.run_b)
    else:
        predict(args.name)


if __name__ == "__main__":
    main()
