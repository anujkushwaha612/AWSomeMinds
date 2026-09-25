"""Baseline end-to-end pipeline (Sub #1): retrieval -> features -> LightGBM -> decision.

Stages (each checkpointed under ``artifacts/baseline/``; re-running skips work
already done, so a crash or notebook restart resumes instead of restarting):

  build   --split train|test   per country: word 1+2-gram record-side retrieval
                               (top-k S1 per S2/S3 record), pair features, labels
  train                        5-fold entity-level cross-fitting on the FULL train
                               universe -> out-of-fold p for every train pair;
                               decision rule tuned on folds != report_fold and
                               reported on report_fold (macro F0.5, oracle ceiling,
                               mean candidates per S1)
  predict --name sub01         test p = mean of the 5 fold models -> decision ->
                               matching_results.tsv + candidate_pairs.tsv, validated
                               and snapshotted to subs/<name>/

Decision rule (plan.md Step 8, D2 + hard arbitration): every S2/S3 record is
given only to its highest-p S1 (each record has at most one parent); an entity's
best remaining candidate is kept if p >= T_first, further ones if p >= T_rest.

Run:  python -u -m ber.baseline build --split train
      python -u -m ber.baseline build --split test
      python -u -m ber.baseline train
      python -u -m ber.baseline predict --name sub01_baseline
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking.word_retrieval import retrieve_country
from .config import artifact_path, ensure_parent, load_config
from .eval.scorer import f05_from_counts, k_bucket
from .features import FEATURES, retrieval_features, string_features
from .io import load_truth_pairs
from .normalize import load_normalized

NORM_COLS = ["entity_id", "country", "name_n", "addr_n", "name_legal", "name_nospace",
             "addr_digits", "script", "name_tr"]
log = logging.getLogger("ber.baseline")


# ------------------------------------------------------------------ utilities
def setup_logging() -> None:
    """Log to stdout (unbuffered) and to artifacts/baseline/log.txt."""
    path = run_path("log.txt")
    ensure_parent(path)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def cfg() -> dict:
    """The ``baseline`` section of configs/pipeline.yaml."""
    return load_config()["baseline"]


def run_path(*parts: str) -> str:
    """Path under ``artifacts/<run_name>/`` (separate run names keep smoke tests apart)."""
    return artifact_path(os.environ.get("BER_RUN", cfg().get("run_name", "baseline")), *parts)


def read_country(split: str, source: int, country: str, limit: int | None = None) -> pd.DataFrame:
    """Normalized rows of one source restricted to one country (only needed columns)."""
    path = artifact_path("norm", f"{split}_source{source}.parquet")
    if not os.path.exists(path):
        load_normalized(split, source)          # builds the cache once
    df = pd.read_parquet(path, columns=NORM_COLS, filters=[("country", "==", country)])
    df = df.reset_index(drop=True)
    return df.iloc[:limit].reset_index(drop=True) if limit else df


def countries(split: str) -> list[str]:
    """Every country present in S1 of ``split`` (open set: France appears in test)."""
    path = artifact_path("norm", f"{split}_source1.parquet")
    if not os.path.exists(path):
        load_normalized(split, 1)
    return sorted(pd.read_parquet(path, columns=["country"])["country"].unique())


# ------------------------------------------------------------------ build
def build(split: str, limit: int | None = None, force: bool = False) -> None:
    """Candidates + features (+ labels on train) for every country of ``split``."""
    c = cfg()
    truth = load_truth_pairs() if split == "train" else None
    folds = pd.read_parquet(artifact_path("folds.parquet")) if split == "train" else None
    with ProcessPoolExecutor(max_workers=c["workers"]) as pool:
        for country in countries(split):
            out = run_path(split, f"{country}.parquet")
            if os.path.exists(out) and not force:
                log.info(f"[build {split}/{country}] exists, skipping")
                continue
            t0 = time.time()
            t = {s: read_country(split, s, country, limit) for s in (1, 2, 3)}
            log.info(f"[build {split}/{country}] S1 {len(t[1]):,}  S2 {len(t[2]):,}  S3 {len(t[3]):,}")
            texts = {s: t[s]["name_n"] + " " + t[s]["addr_n"] for s in (1, 2, 3)}
            cand = retrieve_country(texts, c["k_rec"], c["max_df"], c["min_score"], pool,
                                    c["chunk"], log=log.info)
            del texts
            log.info(f"    {len(cand):,} candidate pairs "
                     f"({len(cand) / max(len(t[1]), 1):.2f} per S1)")
            cand = retrieval_features(cand)
            ts = time.time()
            cand = string_features(cand, t[1], {2: t[2], 3: t[3]})
            log.info(f"    features in {time.time() - ts:.0f}s")

            ents = pd.DataFrame({"s1_row": np.arange(len(t[1]), dtype=np.int32),
                                 "entity_id": t[1]["entity_id"].to_numpy()})
            if split == "train":
                cand["label"] = labels(cand, t, truth)
                s1_index = pd.Index(t[1]["entity_id"])
                tt = truth[truth["s1"].isin(s1_index)]
                ents["k"] = np.bincount(s1_index.get_indexer(tt["s1"]), minlength=len(t[1]))
                ents["fold"] = ents["entity_id"].map(folds.set_index("s1")["fold"]).astype(np.int8)
                pos = cand["label"].sum()
                log.info(f"    labels: {pos:,} positives = pair recall "
                         f"{pos / max(ents['k'].sum(), 1):.4f}")
            ensure_parent(out)
            cand.to_parquet(out, index=False)
            ents.to_parquet(run_path(split, f"{country}_entities.parquet"), index=False)
            log.info(f"[build {split}/{country}] done in {time.time() - t0:.0f}s -> {out}")


def labels(cand: pd.DataFrame, t: dict, truth: pd.DataFrame) -> np.ndarray:
    """1 if the candidate's S1 is the record's true parent (integer row lookups)."""
    s1_index = pd.Index(t[1]["entity_id"])
    y = np.zeros(len(cand), dtype=np.int8)
    for src in (2, 3):
        doc_index = pd.Index(t[src]["entity_id"])
        tt = truth[truth["rid"].str.startswith(f"S{src}-")]
        rows = doc_index.get_indexer(tt["rid"])
        ok = rows >= 0
        parent = np.full(len(doc_index), -1, dtype=np.int32)
        parent[rows[ok]] = s1_index.get_indexer(tt["s1"].to_numpy()[ok])
        m = (cand["src"] == src).to_numpy()
        y[m] = parent[cand.loc[m, "doc_row"].to_numpy()] == cand.loc[m, "s1_row"].to_numpy()
    return y


# ------------------------------------------------------------------ load helpers
def load_split(split: str):
    """Concatenate every country's candidates and entities, with a country code."""
    cands, ents = [], []
    for code, country in enumerate(countries(split)):
        c = pd.read_parquet(run_path(split, f"{country}.parquet"))
        e = pd.read_parquet(run_path(split, f"{country}_entities.parquet"))
        c["country"] = np.int8(code)
        e["country"] = np.int8(code)
        e["country_name"] = country
        cands.append(c)
        ents.append(e)
    return pd.concat(cands, ignore_index=True), pd.concat(ents, ignore_index=True)


def entity_key(country, s1_row) -> np.ndarray:
    """Unique int64 key per (country, S1 row)."""
    return (np.asarray(country, dtype=np.int64) << 32) | np.asarray(s1_row, dtype=np.int64)


# ------------------------------------------------------------------ decision
class Decider:
    """Precomputes arbitration and within-entity order once; evaluates rules fast."""

    def __init__(self, cand: pd.DataFrame, p: np.ndarray):
        self.p = p
        rec = (cand["country"].to_numpy(np.int64) << 40) | (cand["src"].to_numpy(np.int64) << 32) \
            | cand["doc_row"].to_numpy(np.int64)
        order = np.lexsort((-p, rec))
        first = np.ones(len(order), dtype=bool)
        first[1:] = rec[order][1:] != rec[order][:-1]
        self.rec_best = np.zeros(len(p), dtype=bool)
        self.rec_best[order[first]] = True             # record goes to its best-p S1 only
        self.ent = entity_key(cand["country"], cand["s1_row"])
        idx = np.flatnonzero(self.rec_best)
        o = idx[np.lexsort((-p[idx], self.ent[idx]))]
        e_sorted = self.ent[o]
        start = np.ones(len(o), dtype=bool)
        start[1:] = e_sorted[1:] != e_sorted[:-1]
        grp = np.cumsum(start) - 1
        self.pos = np.full(len(p), -1, dtype=np.int32)
        self.pos[o] = np.arange(len(o)) - np.flatnonzero(start)[grp]
        top = np.zeros(len(p), dtype=np.float32)
        top[o] = p[o][np.flatnonzero(start)][grp]
        self.top_p = top

    def keep(self, t_first: float, t_rest: float, arbitrate: bool = True) -> np.ndarray:
        """Mask of predicted pairs under (T_first, T_rest)."""
        if not arbitrate:
            return self.p >= t_first
        return self.rec_best & (
            ((self.pos == 0) & (self.p >= t_first))
            | ((self.pos > 0) & (self.p >= t_rest) & (self.top_p >= t_first)))


class EntityScorer:
    """Macro F0.5 over a fixed entity subset; the pair -> entity map is built once."""

    def __init__(self, ents: pd.DataFrame, cand: pd.DataFrame, ent_mask: np.ndarray):
        keys = entity_key(ents["country"], ents["s1_row"])[ent_mask]
        order = np.argsort(keys)
        keys_sorted = keys[order]
        pk = entity_key(cand["country"], cand["s1_row"])
        pos = np.clip(np.searchsorted(keys_sorted, pk), 0, len(keys_sorted) - 1)
        self.pairs = np.flatnonzero(keys_sorted[pos] == pk)      # pairs of in-subset entities
        self.eidx = order[pos[self.pairs]]
        self.lab = cand["label"].to_numpy()[self.pairs].astype(np.float64)
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
def train() -> None:
    """Cross-fitted LightGBM, decision tuning, fold-0 report."""
    c, v = cfg(), load_config()["validation"]
    cand, ents = load_split("train")
    log.info(f"[train] {len(cand):,} pairs, {len(ents):,} entities, positives {int(cand['label'].sum()):,}")
    ent_fold = dict(zip(entity_key(ents["country"], ents["s1_row"]), ents["fold"]))
    pair_fold = pd.Series(entity_key(cand["country"], cand["s1_row"])).map(ent_fold).to_numpy()
    X = cand[FEATURES].astype(np.float32)
    y = cand["label"].to_numpy()
    rng = np.random.default_rng(load_config()["seed"])
    ent_u = rng.random(len(ents))                              # entity-level subsample draw
    pair_u = pd.Series(entity_key(cand["country"], cand["s1_row"])).map(
        dict(zip(entity_key(ents["country"], ents["s1_row"]), ent_u))).to_numpy()
    params = {"objective": "binary", "verbosity": -1, "num_threads": os.cpu_count(),
              "seed": load_config()["seed"], **{k: v_ for k, v_ in c["lgb"].items()
                                                  if k not in ("num_rounds", "early_stopping")}}
    oof = np.zeros(len(cand), dtype=np.float32)
    model_dir = run_path("models")
    os.makedirs(model_dir, exist_ok=True)
    for f in range(v["n_folds"]):
        t = time.time()
        tr = (pair_fold != f) & (pair_u < c["train_frac"])
        es = tr & (pair_u < c["train_frac"] * 0.1)             # 10% of sampled entities: early stop
        fit = tr & ~es
        booster = lgb.train(params, lgb.Dataset(X[fit], y[fit]), c["lgb"]["num_rounds"],
                            valid_sets=[lgb.Dataset(X[es], y[es])],
                            callbacks=[lgb.early_stopping(c["lgb"]["early_stopping"], verbose=False)])
        oof[pair_fold == f] = booster.predict(X[pair_fold == f], num_threads=os.cpu_count())
        booster.save_model(os.path.join(model_dir, f"fold{f}.txt"))
        log.info(f"[train] fold {f}: {int(fit.sum()):,} rows, {booster.best_iteration} trees, "
                 f"{time.time() - t:.0f}s")
    np.save(run_path("oof.npy"), oof)

    dec = Decider(cand, oof)
    rep = v["report_fold"]
    tune_mask = (ents["fold"] != rep).to_numpy()
    rep_mask = ~tune_mask
    t = time.time()
    best = tune(dec, EntityScorer(ents, cand, tune_mask))
    log.info(f"[train] decision tuned on folds != {rep} in {time.time() - t:.0f}s")
    rep_scorer = EntityScorer(ents, cand, rep_mask)
    keep = dec.keep(best["t_first"], best["t_rest"], best["arbitrate"])
    f_rep = rep_scorer.per_entity(keep)
    oracle = rep_scorer.per_entity(cand["label"].to_numpy().astype(bool))
    rep_ents = ents[rep_mask]
    rep_keys = set(entity_key(rep_ents["country"], rep_ents["s1_row"]))
    in_rep = pd.Series(entity_key(cand["country"], cand["s1_row"])).isin(rep_keys).to_numpy()
    metrics = {
        "report_fold": rep, "decision": best,
        "macro_f05": float(f_rep.mean()),
        "oracle_f05_candidates": float(oracle.mean()),
        "pair_recall_candidates": float(cand["label"].to_numpy()[in_rep].sum()
                                        / max(rep_ents["k"].sum(), 1)),
        "mean_candidates_per_s1": float(in_rep.sum() / max(len(rep_ents), 1)),
        "per_country": {n: float(f_rep[(rep_ents["country_name"] == n).to_numpy()].mean())
                        for n in rep_ents["country_name"].unique()},
        "per_k_bucket": {int(b): float(f_rep[k_bucket(rep_ents["k"]) == b].mean())
                         for b in np.unique(k_bucket(rep_ents["k"]))},
        "features": FEATURES, "config": c,
    }
    out = run_path("metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("[train] fold-%d macro F0.5 %.5f | oracle %.5f | pair recall %.4f | %.2f candidates/S1",
             rep, metrics["macro_f05"], metrics["oracle_f05_candidates"],
             metrics["pair_recall_candidates"], metrics["mean_candidates_per_s1"])
    log.info("[train] decision %s", json.dumps(best))
    log.info("[train] per country %s | per k %s", metrics["per_country"], metrics["per_k_bucket"])


# ------------------------------------------------------------------ predict
def predict(name: str) -> None:
    """Score test candidates with the fold models, decide, write + validate submission."""
    from .submit import make_submission

    with open(run_path("metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    d = metrics["decision"]
    cand, _ = load_split("test")
    X = cand[FEATURES].astype(np.float32)
    model_dir = run_path("models")
    files = sorted(f for f in os.listdir(model_dir) if f.startswith("fold"))
    p = np.mean([lgb.Booster(model_file=os.path.join(model_dir, f)).predict(
        X, num_threads=os.cpu_count()) for f in files], axis=0).astype(np.float32)
    keep = Decider(cand, p).keep(d["t_first"], d["t_rest"], d["arbitrate"])

    ids = {}
    for code, country in enumerate(countries("test")):
        for s in (1, 2, 3):
            ids[(code, s)] = read_country("test", s, country)["entity_id"].to_numpy()
    s1 = np.empty(len(cand), dtype=object)
    rid = np.empty(len(cand), dtype=object)
    for (code, s), arr in ids.items():
        m = (cand["country"] == code).to_numpy()
        if s == 1:
            s1[m] = arr[cand.loc[m, "s1_row"].to_numpy()]
        else:
            ms = m & (cand["src"] == s).to_numpy()
            rid[ms] = arr[cand.loc[ms, "doc_row"].to_numpy()]
    pairs = pd.DataFrame({"s1": s1, "rid": rid})
    n_s1 = sum(len(arr) for (code, s), arr in ids.items() if s == 1)
    log.info(f"[predict] {len(pairs):,} candidate pairs ({len(pairs) / n_s1:.2f} per S1), "
             f"{int(keep.sum()):,} predicted matches ({keep.sum() / n_s1:.2f} per S1)")
    offline = {k: metrics[k] for k in ("macro_f05", "oracle_f05_candidates",
                                       "mean_candidates_per_s1", "per_country", "decision")}
    make_submission(name, pairs[keep], pairs, offline_metrics=offline,
                    notes="baseline: word1+2 TF-IDF record-side top-k, LightGBM, D2 + arbitration")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", required=True, choices=["train", "test"])
    b.add_argument("--limit", type=int, default=None, help="rows per source per country (smoke tests)")
    b.add_argument("--force", action="store_true")
    sub.add_parser("train")
    p = sub.add_parser("predict")
    p.add_argument("--name", required=True)
    args = ap.parse_args()
    setup_logging()
    if args.cmd == "build":
        build(args.split, args.limit, args.force)
    elif args.cmd == "train":
        train()
    else:
        predict(args.name)


if __name__ == "__main__":
    main()
