"""Synthetic 3-source dataset calibrated to the Phase-0 marginals.

The real 2.4GB challenge data does not live in this repo, so the blocking
cascade is developed and regression-tested against a generator that reproduces
the *structure* Phase-0 measured, at 1/100th the scale:

  * countries and their S1 / S2 / S3 row ratios (test-set proportions)
  * the k histogram (0: 5.6%, 1: 5.4%, 2: 17%, 3: 24%, 4: 22%, 5: 14.6%, 6: 7.5%)
  * per-S1 caps of <=5 S2 and <=6 S3 matches, and exactly one S1 parent per record
  * ~26% orphan S2/S3 records
  * noise: typos, token drops, token reordering, abbreviation swaps, truncation,
    legal-suffix churn, ~3% empty addresses, ~4% domain-style names
  * chain pressure: 27% (US) / 37% (India) duplicate normalised S1 names
  * Indic scripts on 13.4% (S2) / 7.5% (S3) of India names, S1 always Latin

**These are not the real data.** Numbers produced from it validate that the
pipeline runs and that the size/recall mechanics behave as designed; the
absolute recall and F0.5 values mean nothing about the leaderboard. Anything
reported as a result must be re-measured on the real files.

Run:  python -m phase0.simulate --scale 0.01 --out artifacts/sim
"""

import argparse
import os
import string

import numpy as np
import pandas as pd

# row counts per split, phase0_report.md section 1. Train is US + India only and
# has 4.68 records per S1; test adds France and has 5.75.
SPLITS = {
    "train": {
        "US":    {"s1": 1_323_633, "s2": 3_016_817, "s3": 3_170_056, "dup": 0.27},
        "India": {"s1":   883_188, "s2": 2_017_799, "s3": 2_115_547, "dup": 0.37},
    },
    "test": {
        "US":     {"s1": 663_106, "s2": 1_871_330, "s3": 1_945_701, "dup": 0.27},
        "India":  {"s1": 809_986, "s2": 2_312_565, "s3": 2_405_000, "dup": 0.37},
        "France": {"s1": 259_452, "s2": 703_378,   "s3": 731_615,   "dup": 0.26},
    },
}
COUNTRIES = SPLITS["test"]      # back-compat for callers that predate --split
K_HIST = {0: 0.0558, 1: 0.0540, 2: 0.1700, 3: 0.2406, 4: 0.2194, 5: 0.1459, 6: 0.0747,
          7: 0.0200, 8: 0.0120, 9: 0.0076}
CAP = {2: 5, 3: 6}
INDIC_SHARE = {2: 0.134 + 0.102, 3: 0.075 + 0.057}      # India only

HEAD = ["golden", "royal", "sunrise", "blue", "green", "prime", "metro", "global", "star",
        "united", "national", "silver", "classic", "modern", "apex", "crown", "delta",
        "orient", "pacific", "summit", "vertex", "harbour", "ivory", "jade", "kestrel"]
CORE = ["trading", "hospitality", "logistics", "textiles", "motors", "pharma", "foods",
        "constructions", "enterprises", "solutions", "industries", "medicals", "traders",
        "engineering", "packaging", "chemicals", "electricals", "agencies", "printers"]
SUFFIX = {"US": ["inc", "llc", "corp", "co", "ltd"],
          "India": ["pvt ltd", "private limited", "llp", "and sons", "traders"],
          "France": ["sarl", "sas", "sa", "eurl", "sasu"]}
STREET = ["main", "market", "church", "park", "station", "mill", "temple", "oak", "hill",
          "river", "lake", "grand", "north", "south", "victory", "liberty", "rue de la paix"]
STYPE = {"US": ["street", "avenue", "road", "boulevard", "lane"],
         "India": ["road", "marg", "nagar", "cross", "layout"],
         "France": ["rue", "avenue", "boulevard", "place", "impasse"]}
CITY = {"US": ["springfield", "riverside", "fairview", "georgetown", "salem"],
        "India": ["pune", "indore", "surat", "kochi", "nagpur", "jaipur"],
        "France": ["lyon", "nantes", "rennes", "dijon", "toulouse"]}
ABBREV = {"street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd", "lane": "ln",
          "private limited": "pvt ltd", "corporation": "corp", "and": "&", "nagar": "ngr"}
DEVANAGARI = "अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _brand(rng) -> str:
    """A pronounceable invented brand token, so the name space is not tiny."""
    cv = ["ba", "ke", "vi", "no", "ra", "mi", "tu", "sa", "le", "dro", "pra", "shi",
          "vel", "kor", "tan", "fer", "gil", "mar", "zen", "oro"]
    return "".join(rng.choice(cv) for _ in range(int(rng.integers(2, 4))))


def _name(rng, country: str) -> str:
    """A plausible business name for ``country``."""
    parts = [_brand(rng) if rng.random() < 0.65 else rng.choice(HEAD), rng.choice(CORE)]
    if rng.random() < 0.3:
        parts.insert(0, rng.choice(HEAD))
    if rng.random() < 0.55:
        parts.append(rng.choice(SUFFIX[country]))
    return " ".join(parts)


def _address(rng, country: str) -> str:
    """A plausible address for ``country``."""
    return (f"{rng.integers(1, 999)} {rng.choice(STREET)} {rng.choice(STYPE[country])}, "
            f"{rng.choice(CITY[country])} {rng.integers(10000, 99999)}")


def _typo(rng, s: str) -> str:
    """One character substitution / deletion / transposition."""
    if len(s) < 4:
        return s
    i = int(rng.integers(1, len(s) - 1))
    op = rng.integers(3)
    if op == 0:
        return s[:i] + rng.choice(list(string.ascii_lowercase)) + s[i + 1:]
    if op == 1:
        return s[:i] + s[i + 1:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _to_indic(rng, s: str) -> str:
    """Replace the Latin tokens of ``s`` with same-length Devanagari strings.

    A crude stand-in for the real transliteration variants: it reproduces the
    property that matters for blocking (the name shares no characters with its
    S1 partner, so only the address can carry the pair).
    """
    return " ".join("".join(rng.choice(list(DEVANAGARI), size=len(t))) for t in s.split()[:3])


def _noisy(rng, name: str, addr: str, country: str, src: int) -> tuple[str, str]:
    """Apply the Phase-0 noise cocktail to one (name, address) pair."""
    toks = name.split()
    if rng.random() < 0.25 and len(toks) > 1:                 # drop a token
        toks.pop(int(rng.integers(len(toks))))
    if rng.random() < 0.15 and len(toks) > 1:                 # reorder
        rng.shuffle(toks)
    name = " ".join(toks)
    if rng.random() < 0.30:
        name = _typo(rng, name)
    if rng.random() < 0.10:                                   # suffix churn
        name = f"{name} {rng.choice(SUFFIX[country])}"
    if rng.random() < 0.04:                                   # domain-style
        name = name.replace(" ", "") + rng.choice([".com", ".in", ".net"])
    if country == "India" and rng.random() < INDIC_SHARE[src]:
        name = _to_indic(rng, name)

    for long, short in ABBREV.items():                        # abbreviations
        if rng.random() < 0.5:
            addr = addr.replace(long, short)
    if rng.random() < 0.20:                                   # drop the postcode
        addr = " ".join(addr.split()[:-1])
    if rng.random() < 0.35:
        addr = _typo(rng, addr)
    if rng.random() < 0.03:
        addr = rng.choice(["", "null", "NA"])
    return name, addr


def generate(scale: float = 0.01, seed: int = 7, k_scale: float = 1.0,
             split: str = "test") -> dict:
    """Build the synthetic split; returns ``{"s1", "s2", "s3", "truth"}`` frames.

    ``scale`` multiplies every real row count, so ``scale=0.01`` gives ~17k S1
    entities and ~100k S2/S3 records - small enough to run the whole cascade in
    a couple of minutes on two cores.

    ``split`` picks the row counts: "train" is US + India at 4.68 records per S1,
    "test" adds France at 5.75 - the same shift the real data has.

    ``k_scale`` resolves the open question R4 in phase0_report.md: test has 5.75
    S2/S3 records per S1 against train's 4.68, and it is not known whether the
    extra records are matches or orphans. ``k_scale=1.0`` keeps the train k
    histogram, so the surplus becomes orphans (39% orphan rate); ``k_scale=1.23``
    scales k up instead, holding the orphan rate at the measured 26%. Generate
    both and check that the chosen blocking policy survives either.
    """
    rng = _rng(seed)
    ks = np.array(list(K_HIST)), np.array(list(K_HIST.values()))
    kvals, kprob = ks[0], ks[1] / ks[1].sum()

    s1_rows, rec_rows, truth = [], {2: [], 3: []}, []
    uid = iter(range(10_000_000, 99_999_999))

    for country, spec in SPLITS[split].items():
        n1 = max(50, int(spec["s1"] * scale))
        # chain pressure: exactly ``dup`` of the entities reuse another's name,
        # so the realised duplicate-name rate matches the measured one
        n_uniq = max(1, int(round(n1 * (1 - spec["dup"]))))
        pool = [_name(rng, country) for _ in range(n_uniq)]
        names = pool + [pool[int(i)] for i in rng.integers(0, n_uniq, n1 - n_uniq)]
        rng.shuffle(names)
        addrs = [_address(rng, country) for _ in range(n1)]

        ids1 = [f"S1-{next(uid)}" for _ in range(n1)]
        s1_rows.append(pd.DataFrame({"entity_id": ids1, "business_name": names,
                                     "business_address": addrs, "country": country}))

        # match counts, split across the two sources under the measured caps
        k = rng.choice(kvals, size=n1, p=kprob)
        if k_scale != 1.0:
            k = np.minimum(np.rint(k * k_scale).astype(int), CAP[2] + CAP[3])
        for i in range(n1):
            k2 = min(int(rng.binomial(k[i], 0.48)), CAP[2])
            k3 = min(int(k[i]) - k2, CAP[3])
            for src, cnt in ((2, k2), (3, k3)):
                for _ in range(cnt):
                    nm, ad = _noisy(rng, names[i], addrs[i], country, src)
                    rid = f"S{src}-{next(uid)}"
                    rec_rows[src].append((rid, nm, ad, country))
                    truth.append((ids1[i], rid))

        # orphans: pad each source to the measured record/S1 density with records
        # that have no S1 parent (they are noisy versions of *unused* businesses)
        for src in (2, 3):
            target = int(spec[f"s{src}"] * scale)
            have = sum(1 for r in rec_rows[src] if r[3] == country)
            for _ in range(max(0, target - have)):
                nm, ad = _noisy(rng, _name(rng, country), _address(rng, country), country, src)
                rec_rows[src].append((f"S{src}-{next(uid)}", nm, ad, country))

    cols = ["entity_id", "business_name", "business_address", "country"]
    out = {"s1": pd.concat(s1_rows, ignore_index=True)}
    for src in (2, 3):
        df = pd.DataFrame(rec_rows[src], columns=cols)
        out[f"s{src}"] = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["truth"] = pd.DataFrame(truth, columns=["s1", "rid"])
    return out


def profile(data: dict) -> pd.DataFrame:
    """Marginals of the generated data, to compare against phase0_report.md."""
    s1, truth = data["s1"], data["truth"]
    k = truth.groupby("s1").size().reindex(s1["entity_id"], fill_value=0)
    n_rec = len(data["s2"]) + len(data["s3"])
    rows = [
        ("S1 rows", len(s1)),
        ("S2+S3 rows", n_rec),
        ("records per S1", round(n_rec / len(s1), 3)),
        ("GT pairs", len(truth)),
        ("mean k", round(k.mean(), 3)),
        ("share k=0", round((k == 0).mean(), 4)),
        ("orphan record share", round(1 - len(truth) / n_rec, 4)),
        ("S1 duplicate-name share", round(s1["business_name"].duplicated().mean(), 4)),
        ("max S2 per S1", int(truth[truth["rid"].str.startswith("S2")]
                              .groupby("s1").size().max())),
        ("max S3 per S1", int(truth[truth["rid"].str.startswith("S3")]
                              .groupby("s1").size().max())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_dataset(data: dict, root: str, split: str) -> None:
    """Write the split in exact challenge format, so pipeline code can be tested.

    Produces ``<root>/<split>/{split}_source{1,2,3}.tsv`` and, for train, the
    ``{split}_ground_truth.tsv`` with one row per S1 entity (empty list allowed).
    """
    import csv

    d = os.path.join(root, split)
    os.makedirs(d, exist_ok=True)
    for src in (1, 2, 3):
        data[f"s{src}" if src > 1 else "s1"].to_csv(
            os.path.join(d, f"{split}_source{src}.tsv"), sep="\t", index=False,
            quoting=csv.QUOTE_NONE, escapechar=None, encoding="utf-8")
    if split == "train":
        lists = data["truth"].groupby("s1")["rid"].agg(",".join)
        gt = pd.DataFrame({"source1_entity_id": data["s1"]["entity_id"]})
        gt["matched_entity_ids"] = gt["source1_entity_id"].map(lists).fillna("")
        gt.to_csv(os.path.join(d, f"{split}_ground_truth.tsv"), sep="\t", index=False,
                  quoting=csv.QUOTE_NONE, encoding="utf-8")


def main() -> None:
    """CLI: write the synthetic split to ``--out`` as parquet and print a profile."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k-scale", type=float, default=1.0,
                    help="1.0 = train k histogram (surplus becomes orphans); "
                         "1.23 = hold the 26%% orphan rate and raise k instead")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", default="artifacts/sim")
    ap.add_argument("--as-dataset", default=None, metavar="DIR",
                    help="also write challenge-format TSVs under DIR/<split>/")
    args = ap.parse_args()

    data = generate(args.scale, args.seed, args.k_scale, args.split)
    os.makedirs(args.out, exist_ok=True)
    for name, df in data.items():
        df.to_parquet(os.path.join(args.out, f"{name}.parquet"), index=False)
    print(profile(data).to_string(index=False))
    print(f"\nwrote {args.out}")
    if args.as_dataset:
        write_dataset(data, args.as_dataset, args.split)
        print(f"wrote challenge-format TSVs to {args.as_dataset}/{args.split}")


if __name__ == "__main__":
    main()
