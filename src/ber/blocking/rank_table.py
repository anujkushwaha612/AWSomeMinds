"""Rank of every true pair in every stored retrieval run (plan.md Step 3).

For one train country, each ground-truth pair (s1, rid) gets, per view:
  <view>_fwd  rank of rid in the S1 row's top-k list (0-based; -1 = not retrieved)
  <view>_rev  rank of s1 in the rid row's top-k list
plus pair attributes used for stratified recall: source, partner script,
name token-Jaccard. With this table any view / direction / K subset can be
evaluated without re-running retrieval.

Run:  python -m ber.blocking.rank_table --country India
"""

import argparse
import os

import numpy as np
import pandas as pd

from ..config import artifact_path, ensure_parent
from ..io import load_truth_pairs
from ..normalize import load_normalized

NOT_FOUND = -1
UNKNOWN = -2     # query row not in a sampled run


def token_jaccard(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Token-set Jaccard of aligned string Series (0 when both are empty)."""
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        u = len(sx | sy)
        out[i] = len(sx & sy) / u if u else 0.0
    return out


def find_rank(lists: np.ndarray, rows: np.ndarray, target: np.ndarray,
              row_pos: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """Rank of ``target[i]`` inside ``lists[row_pos[rows[i]]]``.

    ``row_pos`` maps a full-table row to its row in ``lists`` (-1 if the row was
    not queried, giving UNKNOWN). Returns int16 ranks, NOT_FOUND if absent.
    """
    out = np.full(len(rows), UNKNOWN, dtype=np.int16)
    pos = row_pos[rows]
    known = np.flatnonzero(pos >= 0)
    for s in range(0, len(known), chunk):
        sel = known[s:s + chunk]
        hit = lists[pos[sel]] == target[sel, None]
        any_hit = hit.any(axis=1)
        out[sel] = np.where(any_hit, hit.argmax(axis=1), NOT_FOUND)
    return out


def build(country: str, run_dir: str) -> pd.DataFrame:
    """Build the rank table for ``country`` from retrieval outputs in ``run_dir``."""
    ids = np.load(os.path.join(run_dir, "ids.npz"))
    s1_index = pd.Index(ids["s1"])
    truth = load_truth_pairs()
    truth = truth[truth["s1"].isin(s1_index)].reset_index(drop=True)
    truth["src"] = truth["rid"].str[1].astype(np.int8)

    norm = {s: load_normalized("train", s) for s in (1, 2, 3)}
    norm = {s: df[df["country"] == country].reset_index(drop=True) for s, df in norm.items()}

    parts = []
    for src in (2, 3):
        t = truth[truth["src"] == src].reset_index(drop=True)
        doc_index = pd.Index(ids[f"s{src}"])
        t["s1_row"] = s1_index.get_indexer(t["s1"]).astype(np.int32)
        t["doc_row"] = doc_index.get_indexer(t["rid"]).astype(np.int32)
        if (t["doc_row"] < 0).any():
            raise ValueError(f"{int((t['doc_row'] < 0).sum())} partners outside {country}")
        partner = norm[src].iloc[t["doc_row"].to_numpy()]
        anchor = norm[1].iloc[t["s1_row"].to_numpy()]
        t["script"] = partner["script"].to_numpy()
        t["name_jacc"] = token_jaccard(anchor["name_n"], partner["name_n"])
        t["addr_jacc"] = token_jaccard(anchor["addr_n"], partner["addr_n"])
        parts.append(t)
    table = pd.concat(parts, ignore_index=True)

    pos = {}
    for s in (1, 2, 3):
        p = np.full(len(ids[f"s{s}"]), -1, dtype=np.int64)
        p[ids[f"q{s}"]] = np.arange(len(ids[f"q{s}"]))
        pos[s] = p

    views = sorted(f[:-4] for f in os.listdir(run_dir) if f.endswith(".npz") and f != "ids.npz")
    for view in views:
        res = np.load(os.path.join(run_dir, f"{view}.npz"))
        fwd = np.full(len(table), UNKNOWN, dtype=np.int16)
        rev = np.full(len(table), UNKNOWN, dtype=np.int16)
        for src in (2, 3):
            m = (table["src"] == src).to_numpy()
            s1_row = table.loc[m, "s1_row"].to_numpy()
            doc_row = table.loc[m, "doc_row"].to_numpy()
            fwd[m] = find_rank(res[f"fwd_s{src}_idx"], s1_row, doc_row, pos[1])
            rev[m] = find_rank(res[f"rev_s{src}_idx"], doc_row, s1_row, pos[src])
        table[f"{view}_fwd"] = fwd
        table[f"{view}_rev"] = rev
    return table


def main() -> None:
    """CLI: build and save the rank table for one train country."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--country", required=True)
    ap.add_argument("--run", default="train", help="retrieval run folder, e.g. train_sample0.01")
    args = ap.parse_args()
    run_dir = artifact_path("retrieval", args.run, args.country)
    table = build(args.country, run_dir)
    out = artifact_path("rank_table", args.run, f"{args.country}.parquet")
    ensure_parent(out)
    table.to_parquet(out, index=False)
    print(f"{len(table):,} true pairs -> {out}")


if __name__ == "__main__":
    main()
