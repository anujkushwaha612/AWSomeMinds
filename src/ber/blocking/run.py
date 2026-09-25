"""Production candidate-generation driver (blocking_strategy.md, stages 0-4).

Shards the work by (country, source), probes every S2/S3 record against that
country's S1 index, runs the selection cascade, and writes
``candidate_pairs.tsv`` plus an audit JSON.

Memory strategy, because a single India shard is 810k S1 rows against 2.3M
records:

* the index is fitted once per (country, view) and reused by both sources;
* IDF is fitted on a sample of the corpus (``idf_sample``) - IDF is a corpus
  statistic and a 500k-document sample estimates it to three decimals;
* records are probed in chunks and each chunk is immediately cut to its top
  ``k_keep`` S1 per record, so peak memory is one chunk, not the whole shard;
* the reverse rank ``r_ent`` is *derived* from the forward table by default
  (rank of this record among all records that proposed the same S1) instead of
  running a second full retrieval. That removes the largest memory peak and
  ~40% of the retrieval cost; ``--rev-mode exact`` runs the real S1 -> record
  pass when you can afford it.

Run:
    python -m ber.blocking.run --split train                  # + full scorecard
    python -m ber.blocking.run --split test --out output/candidate_pairs.tsv
    python -m ber.blocking.run --split train --country India --sample-records 0.1
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ..config import artifact_path, ensure_parent, load_config
from ..io import load_source, load_truth_pairs, write_id_lists
from ..normalize import load_normalized
from . import metrics as bm
from .probe import ProbeIndex, pair_table
from .select import SelectPolicy, group_rank, select

# Text views. NA is the backbone; A carries the Indic pairs whose names share no
# tokens with their S1 partner (address Jaccard median 0.65-0.76 on those); N
# carries records whose address is empty or garbled (~3% have no address).
VIEWS = {
    "NA": lambda d: (d["name_legal"] + " " + d["addr_n"]).str.strip(),
    "A": lambda d: d["addr_n"],
    "N": lambda d: d["name_legal"],
    "T": lambda d: (d["name_tr"] + " " + d["addr_tr"]).str.strip(),
    "NS": lambda d: (d["name_nospace"] + " " + d["addr_n"]).str.strip(),
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def derive_r_ent(pairs: pd.DataFrame) -> np.ndarray:
    """Rank of each record among all records proposing the same (S1, source).

    A cheap stand-in for the true reverse rank: a record that never proposed the
    S1 could not be a candidate anyway, so ranking only over the proposers keeps
    the signal the reciprocal filter needs - "is this record among the best that
    want this entity?" - at zero extra retrieval cost.
    """
    codes = pd.factorize(pairs["s1_row"].astype(np.int64) * 4 + pairs["src"], sort=False)[0]
    return np.minimum(group_rank(codes, pairs["score"].to_numpy(dtype=np.float64)),
                      np.iinfo(np.int16).max).astype(np.int16)


def build_indices(tab: dict, views, cfg: dict, seed: int = 0) -> dict:
    """Fit one :class:`ProbeIndex` per view over this country's S1 rows."""
    rng = np.random.default_rng(seed)
    corpus_parts = []
    for s in (1, 2, 3):
        txt = VIEWS["NA"](tab[s])
        n = cfg.get("idf_sample") or len(txt)
        if len(txt) > n:
            txt = txt.iloc[np.sort(rng.choice(len(txt), int(n), replace=False))]
        corpus_parts.append(txt)
    corpus = pd.concat(corpus_parts, ignore_index=True)

    out = {}
    for v in views:
        t0 = time.time()
        out[v] = ProbeIndex(ngram=tuple(cfg["ngram"]), min_df=cfg["min_df"],
                            max_index_df=cfg["max_index_df"],
                            sketch_terms=cfg["sketch_terms"]).fit(VIEWS[v](tab[1]), corpus)
        _log(f"    index {v}: {out[v].stats} ({time.time() - t0:.0f}s)")
    return out


def probe_shard(tab: dict, src: int, indices: dict, cfg: dict, policy: SelectPolicy,
                rev_mode: str = "derived", sample_records: float = 1.0,
                seed: int = 0) -> tuple[pd.DataFrame, int]:
    """Probe every record of one (country, source) shard; return the pair table.

    The returned table is already cut to ``k_keep`` candidates per record, which
    is a strict superset of anything :func:`~ber.blocking.select.select` can
    keep (``k_keep`` > ``a_max``), so the cut costs no recall downstream.

    Returns ``(pairs, probed_ids)``. ``probed_ids`` is what makes
    ``--sample-records`` honest: the audit restricts ground truth and the
    per-record denominator to the records actually queried, so pairs
    completeness and candidates-per-record stay exact rather than being scaled
    down by the sampling rate.
    """
    rec = tab[src]
    rows = np.arange(len(rec))
    if sample_records < 1.0:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(rec), max(1, int(len(rec) * sample_records)),
                                  replace=False))
        rec = rec.iloc[rows].reset_index(drop=True)

    texts = {v: VIEWS[v](rec) for v in indices}
    s1_ids = tab[1]["entity_id"].to_numpy()
    rec_ids = rec["entity_id"].to_numpy()
    chunk = int(cfg["query_chunk"])
    k_keep = int(cfg.get("k_keep", 5))

    acc = []
    for start in range(0, len(rec), chunk):
        sl = slice(start, start + chunk)
        probes = {v: indices[v].query(texts[v].iloc[sl], cfg["k_forward"]) for v in indices}
        pt = pair_table(s1_ids, rec_ids[sl], src, probes)
        # per-record pre-cut: bounds the accumulated table at k_keep x n_records
        codes = pd.factorize(pt["rid"], sort=False)[0]
        pt = pt[group_rank(codes, pt["score"].to_numpy(dtype=np.float64)) < k_keep]
        acc.append(pt.reset_index(drop=True))
        if (start // chunk) % 20 == 0:
            _log(f"    S{src}: {min(start + chunk, len(rec)):,}/{len(rec):,} records")

    pairs = pd.concat(acc, ignore_index=True) if acc else pair_table(s1_ids, rec_ids, src, {})
    if pairs.empty:
        return pairs, rec_ids

    pairs["s1_row"] = pd.factorize(pairs["s1"], sort=False)[0]
    if rev_mode == "exact":
        rev = indices["NA"].query_from_index(
            int(cfg["k_reverse"]), indices["NA"].transform(texts["NA"]))
        idx = rev[0]
        rmap = pd.DataFrame({
            "s1": s1_ids[np.repeat(np.arange(idx.shape[0]), idx.shape[1])[idx.ravel() >= 0]],
            "rid": rec_ids[idx.ravel()[idx.ravel() >= 0]],
            "r_ent_x": np.tile(np.arange(idx.shape[1]),
                               idx.shape[0])[idx.ravel() >= 0].astype(np.int16)})
        rmap = rmap.groupby(["s1", "rid"], sort=False)["r_ent_x"].min().reset_index()
        pairs = pairs.drop(columns="r_ent").merge(rmap, on=["s1", "rid"], how="left")
        pairs["r_ent"] = pairs.pop("r_ent_x").fillna(-1).astype(np.int16)
    else:
        pairs["r_ent"] = derive_r_ent(pairs)
    return pairs.drop(columns="s1_row"), rec_ids


def run(split: str, countries=None, views=None, rev_mode: str = "derived",
        sample_records: float = 1.0, policy: SelectPolicy | None = None,
        save_pairs: bool = True) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Run the cascade over every (country, source) shard of ``split``.

    Returns ``(candidates, report, probed_record_ids)``. Raw probe tables are
    cached per shard under ``artifacts/blocking/<split>/`` so policy tuning can
    re-run stage 3 alone.
    """
    cfg = load_config()["blocking"]
    policy = policy or SelectPolicy.from_config()
    views = views or cfg["views"]
    norm = {s: load_normalized(split, s) for s in (1, 2, 3)}
    countries = countries or sorted(norm[1]["country"].unique())

    selected, probed = [], []
    report = {"shards": {}, "policy": policy.__dict__, "views": list(views)}
    for country in countries:
        t0 = time.time()
        _log(f"  country {country}")
        tab = {s: norm[s][norm[s]["country"] == country].reset_index(drop=True)
               for s in (1, 2, 3)}
        indices = build_indices(tab, views, cfg)
        for src in (2, 3):
            t1 = time.time()
            pairs, probed_ids = probe_shard(tab, src, indices, cfg, policy, rev_mode,
                                            sample_records)
            probed.append(probed_ids)
            n_probed = len(probed_ids)
            if save_pairs and not pairs.empty:
                path = artifact_path("blocking", split, f"pairs_{country}_S{src}.parquet")
                ensure_parent(path)
                pairs.to_parquet(path, index=False)
            out = select(pairs, policy)
            selected.append(out)
            secs = time.time() - t1
            shard = {
                "n_s1": len(tab[1]), "n_records": len(tab[src]),
                "n_probed": int(n_probed),
                "raw_pairs": int(len(pairs)), "selected": int(len(out)),
                # normalise by records actually probed, so the rate is unbiased
                # under --sample-records and comparable across shards
                "cand_per_record": round(len(out) / max(1, n_probed), 4),
                "seconds": round(secs, 1)}
            if n_probed < len(tab[src]):
                shard["projected_full_seconds"] = round(secs * len(tab[src]) / n_probed, 1)
            report["shards"][f"{country}/S{src}"] = shard
            _log(f"    S{src}: {len(pairs):,} raw -> {len(out):,} selected "
                 f"({shard['cand_per_record']}/record, {secs:.0f}s"
                 + (f", full run ~{shard['projected_full_seconds'] / 60:.0f} min"
                    if "projected_full_seconds" in shard else "") + ")")
        del indices
        _log(f"  {country} done in {time.time() - t0:.0f}s")

    cand = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(
        columns=["s1", "rid", "src", "score", "r_rec", "r_ent", "n_views", "tier"])
    all_probed = np.concatenate(probed) if probed else np.array([], dtype=object)
    return cand, report, all_probed


def main() -> None:
    """CLI: run blocking, write candidate_pairs.tsv, and audit it against GT."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--country", nargs="*", default=None)
    ap.add_argument("--views", nargs="*", default=None)
    ap.add_argument("--rev-mode", default="derived", choices=["derived", "exact"])
    ap.add_argument("--sample-records", type=float, default=1.0,
                    help="probe a fraction of records (index stays full). "
                         "PC and candidates-per-record stay unbiased; "
                         "oracle F0.5 is only valid at 1.0")
    ap.add_argument("--out", default=None, help="path for candidate_pairs.tsv")
    args = ap.parse_args()

    t0 = time.time()
    cand, report, probed = run(args.split, args.country, args.views, args.rev_mode,
                               args.sample_records)
    report["total_seconds"] = round(time.time() - t0, 1)
    proj = [v.get("projected_full_seconds") for v in report["shards"].values()]
    if any(proj):
        report["projected_full_run_minutes"] = round(sum(p or 0 for p in proj) / 60, 1)
        _log(f"EXTRAPOLATION: a full run of these shards is about "
             f"{report['projected_full_run_minutes']:.0f} minutes "
             f"(index build is already included once per country)")

    s1 = load_source(args.split, 1)
    records = pd.concat([load_source(args.split, s) for s in (2, 3)])
    if args.country:
        # restrict BOTH sides, or every per-record and reduction-ratio figure is
        # divided by the whole dataset while only one country was blocked
        s1 = s1[s1["country"].isin(args.country)]
        records = records[records["country"].isin(args.country)]
    universe = bm.universe_sizes(s1, records)
    if args.sample_records < 1.0:
        # score only what was actually probed, so PC and candidates-per-record
        # are exact for the sample instead of being scaled down by the rate
        universe["n_records"] = len(probed)

    out = args.out or artifact_path("blocking", args.split, "candidate_pairs.tsv")
    write_id_lists(cand, s1["entity_id"], out, kind="candidate")
    _log(f"wrote {out}")

    if args.split == "train":
        truth = load_truth_pairs()
        if args.sample_records < 1.0:
            truth = truth[truth["rid"].isin(set(probed))]
        scorecard = bm.report(cand, truth, s1["entity_id"], universe, name="cascade")
        if args.sample_records < 1.0:
            scorecard["NOTE"] = ("records sampled: pairs_completeness, pairs_quality "
                                 "and C_per_record_mean are exact for the probed "
                                 "records; oracle_f05 and C_per_s1 are downward-"
                                 "biased because each entity only had a fraction of "
                                 "its records probed - re-measure those at 1.0")
        report["scorecard"] = scorecard
        print("\n== blocking scorecard")
        for k, v in scorecard.items():
            print(f"  {k}: {v}")
        strata = s1.set_index("entity_id")[["country"]]
        print("\n== per stratum")
        print(bm.by_stratum(cand, truth, s1["entity_id"], strata).round(4)
              .to_string(index=False))
    else:
        report["scorecard"] = bm.size_stats(cand, s1["entity_id"], len(records))
        report["scorecard"]["projected_C_per_s1_from_records"] = round(
            report["scorecard"]["C_per_record_mean"] * len(records) / len(s1), 3)
        print("\n== size report (no labels on test)")
        for k, v in report["scorecard"].items():
            print(f"  {k}: {v}")

    path = artifact_path("blocking", args.split, "audit.json")
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    _log(f"wrote {path}  (total {report['total_seconds']}s)")


if __name__ == "__main__":
    main()
