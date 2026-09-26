"""Part C of strategy_v5.md: where does the offline -> leaderboard gap come from? (CPU)

  stress    Density stress. Test has ~5.8 S2/S3 records per S1 vs 4.68 in train (E4), so
            a random ``v5.stress_fraction`` (0.19) of S1 entities is removed from the train
            universe (their candidate rows are dropped; their records become orphans).
            Fold-0 entities that remain are re-scored with the same p and decision rule,
            and with thresholds re-tuned under stress. Approximation: p and the features
            are not recomputed after removal (a removed S1 cannot be replaced by the next
            retrieval candidate), so this *under*-states the real density effect.
  france    Label-free diagnostics of test predictions per country vs train fold 0.
  probe     Copy of output/matching_results.tsv with France rows emptied (an optional
            leaderboard probe: LB(full) - LB(emptied) = country share x (F_country - singleton share of country)).

Run:  python -m ber.gap stress --run v5        (or --run baseline)
      python -m ber.gap france
      python -m ber.gap probe --name probe_france_empty
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from .baseline import Decider, EntityScorer, entity_key
from .config import REPO_ROOT, artifact_path, load_config
from .store import split_countries
from .neural.common import country_store


def load_run(run: str):
    """(keys, ents, p, decision-mask function) for the train universe of ``run``."""
    if run == "baseline":
        from . import baseline as B
        name = load_config()["v5"]["tfidf_run"]              # the TF-IDF baseline run folder
        os.environ["BER_RUN"] = os.environ["BER_FEATURES"] = name
        keys, ents = B.read_keys("train")
        p = np.load(artifact_path(name, "oof.npy"))
        d = json.load(open(artifact_path(name, "metrics.json")))["decision"]
        return keys, ents, p, lambda k, pp: Decider(k, pp).keep(d["t_first"], d["t_rest"], d["arbitrate"])
    from . import v5 as V
    keys, ents = V.read_keys("train")
    kept = np.load(V.run_path("kept_train.npy"))
    keys = keys[kept].reset_index(drop=True)
    p = np.load(V.run_path("p2_train.npy"))
    rules = json.load(open(V.run_path("stage2.json")))["rules"]
    return keys, ents, p, lambda k, pp: V.apply_rule(k, pp, rules)


def stress(run: str, seeds: int = 3) -> dict:
    """Fold-0 macro F0.5 with and without the density stress (same rule; re-tuned T)."""
    cfg = load_config()["v5"]
    frac, rep = cfg["stress_fraction"], cfg["report_fold"]
    keys, ents, p, decide = load_run(run)
    rep_mask = (ents["fold"] == rep).to_numpy()
    tune_folds = [f for f in (cfg["gbdt_folds"] if run == "v5" else range(load_config()["validation"]["n_folds"]))
                  if f != rep]
    tune_mask = np.isin(ents["fold"].to_numpy(), tune_folds)       # thresholds never chosen on fold 0
    base = EntityScorer(ents, keys, rep_mask).macro(decide(keys, p))
    ekey = entity_key(ents["country"], ents["s1_row"])
    pkey = entity_key(keys["country"], keys["s1_row"])
    lo, hi, step = cfg["grid"]
    grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    res = []
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        removed = rng.random(len(ents)) < frac
        keep_rows = ~pd.Series(pkey).isin(set(ekey[removed].tolist())).to_numpy()
        k2 = keys[keep_rows].reset_index(drop=True)
        p2 = p[keep_rows]
        sc = EntityScorer(ents, k2, rep_mask & ~removed)
        tsc = EntityScorer(ents, k2, tune_mask & ~removed)
        same = sc.macro(decide(k2, p2))
        dec = Decider(k2, p2)
        best = max((tsc.macro(dec.keep(t1, t2)), float(t1), float(t2)) for t1 in grid for t2 in grid)
        retuned = sc.macro(dec.keep(best[1], best[2]))              # tuned on tune folds, scored on fold 0
        res.append({"seed": seed, "f05_same_rule": same, "f05_retuned": retuned,
                    "t_first": best[1], "t_rest": best[2]})
        print(f"  seed {seed}: same rule {same:.5f} | re-tuned {retuned:.5f} at T=({best[1]}, {best[2]})",
              flush=True)
    out = {"run": run, "stress_fraction": frac, "unstressed_f05": base,
           "stressed_same_rule_mean": float(np.mean([r["f05_same_rule"] for r in res])),
           "stressed_retuned_mean": float(np.mean([r["f05_retuned"] for r in res])), "seeds": res}
    print(f"[stress {run}] unstressed {base:.5f} -> stressed {out['stressed_same_rule_mean']:.5f} "
          f"(same rule), {out['stressed_retuned_mean']:.5f} (re-tuned)", flush=True)
    json.dump(out, open(artifact_path("experiments", f"C1_stress_{run}.json"), "w"), indent=2)
    return out


def france() -> dict:
    """Per-country test prediction statistics next to train fold 0 (v5 run)."""
    from . import v5 as V
    rep = load_config()["v5"]["report_fold"]
    out = {}
    for split in ("train", "test"):
        keys, ents = V.read_keys(split)
        kept = np.load(V.run_path(f"kept_{split}.npy"))
        keys = keys[kept].reset_index(drop=True)
        p = np.load(V.run_path(f"p2_{split}.npy"))
        if split == "train":
            rules = json.load(open(V.run_path("stage2.json")))["rules"]
            keep = V.apply_rule(keys, p, rules)
            sel_e = (ents["fold"] == rep).to_numpy()
        else:
            keep = np.load(V.run_path("keep_test.npy"))
            sel_e = np.ones(len(ents), dtype=bool)
        rec = (keys["country"].to_numpy(np.int64) << 40) | (keys["src"].to_numpy(np.int64) << 32) \
            | keys["doc_row"].to_numpy(np.int64)
        best_p = pd.Series(p).groupby(rec).max()
        rec_country = pd.Series(keys["country"].to_numpy()).groupby(rec).first()
        for code, country in enumerate(split_countries(split)):
            em = sel_e & (ents["country"] == code).to_numpy()
            ek = set(entity_key(np.full(em.sum(), code), ents.loc[em, "s1_row"]).tolist())
            pk = pd.Series(entity_key(keys["country"], keys["s1_row"]))
            in_e = pk.isin(ek).to_numpy()
            n_e = int(em.sum())
            pred_per = pd.Series(keep[in_e]).groupby(pk[in_e].to_numpy()).sum()
            bp = best_p[rec_country == code].to_numpy()
            out[f"{split}/{country}"] = {
                "entities": n_e,
                "candidates_per_s1": float(in_e.sum() / n_e),
                "predicted_per_s1": float(keep[in_e].sum() / n_e),
                "share_s1_predicted_empty": float(1 - (pred_per > 0).sum() / n_e),
                "record_best_p_q10_q50_q90": np.quantile(bp, [0.1, 0.5, 0.9]).round(4).tolist(),
                "share_records_best_p_ge_0.5": float((bp >= 0.5).mean()),
            }
    print(pd.DataFrame(out).T.to_string())
    json.dump(out, open(artifact_path("experiments", "C2_france.json"), "w"), indent=2)
    return out


def probe(name: str, country: str) -> None:
    """Write subs/<name>/ = current matching results with every row of ``country`` emptied."""
    from .submit import finalize_submission

    if country not in split_countries("test"):
        raise SystemExit(f"{country!r} is not a test country: {split_countries('test')}")
    fr = set(country_store("test", country, cols=["entity_id"]).numpy(1, "entity_id").tolist())
    src = REPO_ROOT / "output" / "matching_results.tsv"
    tmp = REPO_ROOT / "output" / "matching_results_probe.tsv"
    n = 0
    with open(src, encoding="utf-8") as fi, open(tmp, "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            s1 = line.split("\t", 1)[0]
            if s1 in fr:
                fo.write(f"{s1}\t\n")
                n += 1
            else:
                fo.write(line)
    if n == 0:
        raise SystemExit(f"no {country} rows found in {src}")
    print(f"emptied {n:,} {country} rows")
    finalize_submission(name, str(tmp), str(REPO_ROOT / "output" / "candidate_pairs.tsv"),
                        notes=f"probe: {country} rows emptied")
    shutil.move(str(tmp), str(REPO_ROOT / "subs" / name / "matching_results_probe.tsv"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stress")
    s.add_argument("--run", default="v5", choices=["v5", "baseline"])
    sub.add_parser("france")
    pr = sub.add_parser("probe")
    pr.add_argument("--name", default="probe_country_empty")
    pr.add_argument("--country", default="France", help="test country whose rows are emptied")
    args = ap.parse_args()
    os.makedirs(artifact_path("experiments"), exist_ok=True)
    if args.cmd == "stress":
        stress(args.run)
    elif args.cmd == "france":
        france()
    else:
        probe(args.name, args.country)


if __name__ == "__main__":
    main()
