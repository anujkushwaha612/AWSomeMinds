"""Candidate-set audit: entity-level oracle F0.5 and |C| for any view/K mix.

A *component* is ``VIEW:fK`` (S1 -> top-K per source) or ``VIEW:rK`` (each S2/S3
record -> its top-K S1). A candidate configuration is a union of components.
For a configuration this reports, on fold-0 entities (plan.md Step 3/4):

  * oracle macro-F0.5: a perfect matcher on the candidates (P = 1, R = share of
    the entity's true matches retrieved; k = 0 entities score 1)
  * pair recall overall and on the hard strata (Indic-script partner, low name
    overlap)
  * candidate count per entity (mean / p90 / max) after the union

and the paired-bootstrap Δ of each extra component over a base configuration.

Run:  python -m ber.blocking.audit --country India
      python -m ber.blocking.audit --country India --base NA:f20 --add A:f10 NA:r3
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from ..config import artifact_path, ensure_parent, load_config
from ..eval.scorer import k_bucket, paired_bootstrap

DEFAULT_GRID_K_FWD = (5, 10, 20, 30, 50)
DEFAULT_GRID_K_REV = (1, 3, 5, 10)


def parse(spec: str) -> list[tuple[str, str, int]]:
    """``"NA:f20,A:r3"`` -> ``[("NA", "fwd", 20), ("A", "rev", 3)]``."""
    comps = []
    for tok in filter(None, (t.strip() for t in spec.split(","))):
        view, rest = tok.split(":")
        comps.append((view, {"f": "fwd", "r": "rev"}[rest[0]], int(rest[1:])))
    return comps


class Audit:
    """Evaluates candidate configurations for one country and one retrieval run."""

    def __init__(self, country: str, run: str = "train", fold: int | None = None):
        self.run_dir = artifact_path("retrieval", run, country)
        self.ids = np.load(os.path.join(self.run_dir, "ids.npz"))
        self.table = pd.read_parquet(artifact_path("rank_table", run, f"{country}.parquet"))
        self.sampled = run != "train"
        n1 = len(self.ids["s1"])

        # entity set: fold-0 S1 rows (full run) or the sampled S1 queries (timing run)
        if self.sampled:
            ents = np.sort(self.ids["q1"])
        else:
            fold = load_config()["validation"]["report_fold"] if fold is None else fold
            folds = pd.read_parquet(artifact_path("folds.parquet"))
            in_fold = set(folds.loc[folds["fold"] == fold, "s1"])
            ents = np.flatnonzero(pd.Series(self.ids["s1"]).isin(in_fold).to_numpy())
        self.ents = ents
        self.in_ents = np.zeros(n1, dtype=bool)
        self.in_ents[ents] = True
        self.pairs = self.table[self.in_ents[self.table["s1_row"].to_numpy()]].reset_index(drop=True)
        self.pos1 = np.full(n1, -1, dtype=np.int64)
        self.pos1[self.ids["q1"]] = np.arange(len(self.ids["q1"]))
        self.k = np.bincount(self.pairs["s1_row"], minlength=n1)[ents]
        self._res = {}

    # ------------------------------------------------------------------ helpers
    def _load(self, view: str):
        if view not in self._res:
            self._res[view] = np.load(os.path.join(self.run_dir, f"{view}.npz"))
        return self._res[view]

    def reachable(self, comps) -> np.ndarray:
        """Boolean per fold pair: retrieved by at least one component."""
        hit = np.zeros(len(self.pairs), dtype=bool)
        for view, direction, k in comps:
            r = self.pairs[f"{view}_{direction}"].to_numpy()
            hit |= (r >= 0) & (r < k)
        return hit

    def oracle(self, comps) -> np.ndarray:
        """Per-entity oracle F0.5 (aligned with ``self.ents``)."""
        n1 = len(self.in_ents)
        tp = np.bincount(self.pairs["s1_row"], weights=self.reachable(comps), minlength=n1)[self.ents]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(self.k > 0, tp / np.maximum(self.k, 1), 0.0)
            f = np.where(tp > 0, 1.25 * r / (0.25 + r), 0.0)
        return np.where(self.k == 0, 1.0, f)

    def candidate_counts(self, comps) -> np.ndarray:
        """Per-entity |C| of the union of ``comps`` (both sources)."""
        keys = []
        for view, direction, k in comps:
            res = self._load(view)
            for src in (2, 3):
                if direction == "fwd":
                    lists = res[f"fwd_s{src}_idx"][self.pos1[self.ents], :k]
                    s1 = np.repeat(self.ents, lists.shape[1])
                    doc = lists.ravel()
                else:
                    if self.sampled:
                        continue            # reverse |C| needs every doc queried
                    lists = res[f"rev_s{src}_idx"][:, :k]
                    doc = np.repeat(np.arange(lists.shape[0]), lists.shape[1])
                    s1 = lists.ravel()
                ok = doc >= 0
                ok &= s1 >= 0
                s1, doc = s1[ok].astype(np.int64), doc[ok].astype(np.int64)
                if direction == "rev":
                    m = self.in_ents[s1]
                    s1, doc = s1[m], doc[m]
                keys.append((s1 << 33) | (src << 32) | doc)
        if not keys:
            return np.zeros(len(self.ents), dtype=np.int64)
        uniq = np.unique(np.concatenate(keys))
        return np.bincount(uniq >> 33, minlength=len(self.in_ents))[self.ents]

    # ------------------------------------------------------------------ reports
    def evaluate(self, comps) -> dict:
        """Summary metrics for one configuration."""
        f = self.oracle(comps)
        hit = self.reachable(comps)
        c = self.candidate_counts(comps)
        p = self.pairs
        hard_script = (p["script"] > 0).to_numpy()
        low_name = (p["name_jacc"] < 0.25).to_numpy()
        return {
            "config": ",".join(f"{v}:{d[0]}{k}" for v, d, k in comps),
            "oracle_f05": round(float(f.mean()), 5),
            "pair_recall": round(float(hit.mean()), 5),
            "recall_indic_partner": round(float(hit[hard_script].mean()), 5) if hard_script.any() else None,
            "recall_low_name_overlap": round(float(hit[low_name].mean()), 5) if low_name.any() else None,
            "C_mean": round(float(c.mean()), 1),
            "C_p90": int(np.percentile(c, 90)),
            "C_max": int(c.max()),
        }

    def strata(self, comps) -> pd.DataFrame:
        """Oracle F by k-bucket and pair recall by source x script group."""
        f = self.oracle(comps)
        rows = [{"stratum": "ALL", "n": len(f), "value": f.mean(), "metric": "oracle_f05"}]
        kb = k_bucket(self.k)
        for b in np.unique(kb):
            m = kb == b
            rows.append({"stratum": f"k_bucket={b}", "n": int(m.sum()),
                         "value": f[m].mean(), "metric": "oracle_f05"})
        hit = self.reachable(comps)
        group = np.select([self.pairs["script"] == 0, self.pairs["script"] == 1],
                          ["latin", "devanagari"], "other_indic")
        for (src, g), idx in pd.Series(np.arange(len(hit))).groupby(
                [self.pairs["src"].to_numpy(), group]):
            rows.append({"stratum": f"S{src}/{g}", "n": len(idx),
                         "value": hit[idx.to_numpy()].mean(), "metric": "pair_recall"})
        return pd.DataFrame(rows)


def main() -> None:
    """CLI: single-component curves, marginal gains over a base, strata table."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--country", required=True)
    ap.add_argument("--run", default="train")
    ap.add_argument("--base", default="NA:f20")
    ap.add_argument("--add", nargs="*", default=None,
                    help="components to test on top of --base (default: every stored view)")
    args = ap.parse_args()

    a = Audit(args.country, args.run)
    views = sorted({c.split("_")[0] for c in a.table.columns if c.endswith("_fwd")})
    boot = load_config()["bootstrap"]
    print(f"{args.country} [{args.run}]: {len(a.ents):,} entities, {len(a.pairs):,} true pairs")

    curves = []
    for v in views:
        for k in DEFAULT_GRID_K_FWD:
            curves.append(a.evaluate([(v, "fwd", k)]))
        if not a.sampled:
            for k in DEFAULT_GRID_K_REV:
                curves.append(a.evaluate([(v, "rev", k)]))
    curves = pd.DataFrame(curves)
    print("\n== single components\n" + curves.to_string(index=False))

    base = parse(args.base)
    f_base = a.oracle(base)
    adds = args.add
    if adds is None:
        adds = [f"{v}:f10" for v in views] + ([] if a.sampled else [f"{v}:r3" for v in views])
    marg = []
    for spec in adds:
        comps = base + parse(spec)
        res = a.evaluate(comps)
        bs = paired_bootstrap(f_base, a.oracle(comps), boot["n_resamples"], boot["alpha"])
        res.update({"added": spec, "delta_f05": round(bs["delta"], 5),
                    "ci": [round(bs["ci_low"], 5), round(bs["ci_high"], 5)]})
        marg.append(res)
    marg = pd.DataFrame(marg)
    print(f"\n== marginal gain over base {args.base} ({a.evaluate(base)})\n"
          + marg[["added", "delta_f05", "ci", "pair_recall", "recall_indic_partner",
                  "recall_low_name_overlap", "C_mean", "C_p90"]].to_string(index=False))

    strata = a.strata(base)
    print(f"\n== strata for base {args.base}\n" + strata.round(5).to_string(index=False))

    out = artifact_path("audit", args.run, f"{args.country}.json")
    ensure_parent(out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"curves": curves.to_dict("records"), "marginal": marg.to_dict("records"),
                   "strata": strata.to_dict("records")}, fh, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
