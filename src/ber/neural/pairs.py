"""A1: leakage-safe training triplets and the recall-gate sample (CPU, ~5 min).

Leakage rule (strategy_v5.md §2): the encoder may only see labels of entities in
``v5.encoder_folds``. Therefore
  * positives  = (record, its true S1 parent) with the parent in the encoder folds;
  * hard negative = the best-ranked TF-IDF candidate of that record that is not its
    parent AND whose S1 is also in the encoder folds (a negative S1 from folds 0-2
    would train on the label of a pair that fold-0 evaluation later scores);
    fallback: a random encoder-fold S1 of the same country;
  * the recall-gate sample is drawn from fold-0 records only.

Outputs (artifacts/neural/): ``train_pairs.parquet`` (country, src, doc_row, pos_s1,
neg_s1, neg_kind: 0 tfidf, 1 same name, 2 same first word, 3 random) and ``eval_records.parquet`` (country, src, doc_row, parent_s1,
tfidf_rank of the parent, -1 if not retrieved, script).

Run:  python -m ber.neural.pairs
"""

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import artifact_path, ensure_parent, load_config
from ..io import load_truth_pairs
from ..store import split_countries
from .common import country_store, ncfg, v5cfg, vdir


def truth_parents_arrays(store, truth: pd.DataFrame) -> dict:
    """Per source: doc row -> parent S1 row (-1 = orphan)."""
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


NEG_TFIDF, NEG_SAME_NAME, NEG_FIRST_WORD, NEG_RANDOM = 0, 1, 2, 3
NEG_KIND_NAMES = {NEG_TFIDF: "tfidf", NEG_SAME_NAME: "same_name", NEG_FIRST_WORD: "same_first_word",
                  NEG_RANDOM: "random"}


def sibling_in_group(keys: pd.Series, allowed: np.ndarray) -> np.ndarray:
    """For every row: another ``allowed`` row with the same key (cyclic next in the group), else -1.

    Used to mine look-alike negatives (same name = chain branch) among encoder-fold S1s only.
    """
    rows = np.flatnonzero(allowed)
    k = keys.to_numpy()[rows]
    ok = k != ""
    rows, k = rows[ok], k[ok]
    order = np.argsort(k, kind="stable")
    rows, k = rows[order], k[order]
    out = np.full(len(keys), -1, dtype=np.int64)
    if not len(rows):
        return out
    start = np.ones(len(k), dtype=bool)
    start[1:] = k[1:] != k[:-1]
    starts = np.flatnonzero(start)
    grp = np.cumsum(start) - 1
    size = np.diff(np.append(starts, len(k)))[grp]
    pos = np.arange(len(k)) - starts[grp]
    nxt = starts[grp] + (pos + 1) % size
    multi = size > 1
    out[rows[multi]] = rows[nxt[multi]]
    return out


def build_country(code: int, country: str, truth: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training triplets and fold-0 evaluation records for one train country."""
    v = v5cfg()
    enc_folds = np.array(v["encoder_folds"])
    tfidf_dir = artifact_path(vdir("tfidf"), "train")
    ents = pd.read_parquet(os.path.join(tfidf_dir, f"{country}_entities.parquet"))
    fold = ents["fold"].to_numpy()
    in_enc_s1 = np.isin(fold, enc_folds)
    enc_s1_rows = np.flatnonzero(in_enc_s1)

    store = country_store("train", country, cols=["entity_id", "script", "name_n"])
    if store.n(1) != len(fold):
        raise ValueError(f"{country}: store has {store.n(1):,} S1 rows but the TF-IDF run has "
                         f"{len(fold):,} entities (row universes differ; check v5.limit)")
    parents = truth_parents_arrays(store, truth)
    cand = pq.read_table(os.path.join(tfidf_dir, f"{country}.parquet"),
                         columns=["src", "doc_row", "s1_row", "rank", "label"]).to_pandas()
    # look-alike negatives among encoder-fold S1s: same full name (chains), else same first word
    names = pd.Series(store.strings(1, "name_n"))
    first = names.str.split(n=1).str[0].fillna("")
    same_name = sibling_in_group(names, in_enc_s1)
    same_first = sibling_in_group(first, in_enc_s1)

    trip, evals = [], []
    for src in (2, 3):
        par = parents[src]
        c = cand[cand["src"] == src]
        # hard negatives: non-parent candidates whose S1 is in the encoder folds, best rank first
        neg = c[(c["label"] == 0) & in_enc_s1[c["s1_row"].to_numpy()]]
        neg = neg.sort_values(["doc_row", "rank"]).drop_duplicates("doc_row")
        neg_of = pd.Series(neg["s1_row"].to_numpy(), index=neg["doc_row"].to_numpy())

        docs = np.flatnonzero((par >= 0) & in_enc_s1[np.maximum(par, 0)])
        pos = par[docs]
        hard = neg_of.reindex(docs).to_numpy()
        rand = enc_s1_rows[rng.integers(0, len(enc_s1_rows), len(docs))]
        # a random negative must not be the parent itself
        clash = rand == pos
        rand[clash] = enc_s1_rows[(np.searchsorted(enc_s1_rows, rand[clash]) + 1) % len(enc_s1_rows)]
        # priority: TF-IDF non-parent > same-name S1 > same-first-word S1 > random
        kind = np.full(len(docs), NEG_RANDOM, dtype=np.int8)
        negs = rand.astype(np.int64)
        for k, cand_neg in ((NEG_FIRST_WORD, same_first[pos]), (NEG_SAME_NAME, same_name[pos])):
            ok = cand_neg >= 0
            negs[ok], kind[ok] = cand_neg[ok], k
        ok = ~np.isnan(hard)
        negs[ok], kind[ok] = hard[ok].astype(np.int64), NEG_TFIDF
        trip.append(pd.DataFrame({"country": np.int8(code), "src": np.int8(src),
                                  "doc_row": docs.astype(np.int32), "pos_s1": pos.astype(np.int32),
                                  "neg_s1": negs.astype(np.int32), "neg_kind": kind}))

        # fold-0 evaluation records with a parent
        e_docs = np.flatnonzero((par >= 0) & (fold[np.maximum(par, 0)] == v["report_fold"]))
        pos_rank = c[c["label"] == 1].set_index("doc_row")["rank"]
        evals.append(pd.DataFrame({"country": np.int8(code), "src": np.int8(src),
                                   "doc_row": e_docs.astype(np.int32),
                                   "parent_s1": par[e_docs].astype(np.int32),
                                   "tfidf_rank": pos_rank.reindex(e_docs).fillna(-1).astype(np.int8).to_numpy(),
                                   "script": store.numpy(src, "script")[e_docs]}))
    return pd.concat(trip, ignore_index=True), pd.concat(evals, ignore_index=True)


def main() -> None:
    """Build and save the triplets and the evaluation sample."""
    c, seed = ncfg(), load_config()["seed"]
    rng = np.random.default_rng(seed)
    truth = load_truth_pairs()
    trips, evals = [], []
    for code, country in enumerate(split_countries("train")):
        t, e = build_country(code, country, truth, rng)
        n_eval = min(c["eval_records"], len(e))
        evals.append(e.sample(n_eval, random_state=seed))
        trips.append(t)
        print(f"{country}: {len(t):,} positives from encoder folds "
              f"(negatives: {', '.join(f'{NEG_KIND_NAMES[k]} {v:.1%}' for k, v in t['neg_kind'].value_counts(normalize=True).sort_index().items())}); "
              f"{n_eval:,} fold-0 eval records", flush=True)
    trips = pd.concat(trips, ignore_index=True)
    if c["n_pairs"] and c["n_pairs"] < len(trips):
        trips = trips.sample(c["n_pairs"], random_state=seed)
    trips = trips.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)   # shuffle
    out = artifact_path(vdir("neural"), "train_pairs.parquet")
    ensure_parent(out)
    trips.to_parquet(out, index=False)
    pd.concat(evals, ignore_index=True).to_parquet(artifact_path(vdir("neural"), "eval_records.parquet"), index=False)
    print(f"wrote {len(trips):,} training triplets -> {out}")


if __name__ == "__main__":
    main()
