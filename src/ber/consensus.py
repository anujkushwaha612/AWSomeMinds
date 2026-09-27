"""Record-consensus features for stage 2 (experiments.md E16).

The S1 is itself a noisy copy of a latent entity: when the generator edited the S1's
house number or street, the true records agree with EACH OTHER against the S1, while a
decoy (orphan) record's deviation is unique to it. For every pruned pair (s, r), with
"others" = the other pruned candidates of the same S1:

  cn_dev_num     address numbers of r missing from s (zero-stripped)
  cn_sh_num      max over those numbers of the others that also carry it
  cn_sh_num_hi   same, counting only others with p1 >= 0.5
  cn_miss_num    address numbers of s missing from r
  cn_have_miss   max over those numbers of the others that carry it (s is not the odd one out)
  cn_dev_atok / cn_sh_atok / cn_sh_atok_hi   same for non-numeric address tokens
  cn_dev_tok / cn_sh_tok / cn_sh_tok_hi      same for content name tokens (len >= 3, no noise words)
  cn_n_other     others of the S1; cn_n_other_hi  others with p1 >= 0.5

Language-agnostic by construction (no vocabulary), so it applies to France unchanged.
"""

import re

import numpy as np
import pandas as pd

from .features import NOISE_TOKENS

CONS_FEATURES = ["cn_dev_num", "cn_sh_num", "cn_sh_num_hi", "cn_miss_num", "cn_have_miss",
                 "cn_dev_atok", "cn_sh_atok", "cn_sh_atok_hi", "cn_dev_tok", "cn_sh_tok", "cn_sh_tok_hi",
                 "cn_n_other", "cn_n_other_hi"]
DIG = re.compile(r"\d+")


def _num_set(s: str) -> set:
    return {m.lstrip("0") or "0" for m in DIG.findall(s)}


def _atok_set(s: str) -> set:
    return {t for t in s.split() if len(t) >= 3 and not t.isdigit()}


def _ntok_set(s: str) -> set:
    return {t for t in s.split() if len(t) >= 3 and t not in NOISE_TOKENS and not t.isdigit()}


def _explode(sets, rows):
    lens = np.fromiter((len(x) for x in sets), dtype=np.int64, count=len(sets))
    r = np.repeat(rows, lens)
    h = np.fromiter((hash(t) for x in sets for t in x), dtype=np.int64, count=int(lens.sum()))
    return r, h


def _consensus(group, hi, rec_sets, dev_sets):
    """(max over dev tokens of other rows carrying it, same counting only hi rows)."""
    n = len(group)
    rr, rh = _explode(rec_sets, np.arange(n))
    a = pd.DataFrame({"g": group[rr], "t": rh, "hi": hi[rr].astype(np.int32)})
    cnt = a.groupby(["g", "t"], sort=False).agg(c=("hi", "size"), ch=("hi", "sum"))
    del a
    dr, dh = _explode(dev_sets, np.arange(n))
    d = pd.DataFrame({"g": group[dr], "t": dh, "r": dr})
    d = d.join(cnt, on=["g", "t"])
    d["c"] = d["c"].fillna(0)
    d["ch"] = d["ch"].fillna(0)
    # subtract the row itself when it carries the token (dev tokens of r are r's own tokens,
    # "missing" tokens are the S1's and r does not carry them)
    own = rec_sets
    self_has = np.fromiter((t in own[i] for i, t in zip(dr, _tokens(dev_sets))), dtype=bool, count=len(dr))
    d["c"] -= self_has
    d["ch"] -= self_has & hi[dr]
    m = d.groupby("r").agg(c=("c", "max"), ch=("ch", "max"))
    out, out_hi = np.zeros(n, np.float32), np.zeros(n, np.float32)
    out[m.index.to_numpy()] = m["c"].to_numpy()
    out_hi[m.index.to_numpy()] = m["ch"].to_numpy()
    return out, out_hi


def _tokens(sets):
    return (t for x in sets for t in x)


def consensus_features(s1_row: np.ndarray, p1: np.ndarray, s1_name, s1_addr, r_name, r_addr) -> pd.DataFrame:
    """CONS_FEATURES for the pairs of ONE country (aligned arrays; texts are normalized name_n / addr_n)."""
    g = np.asarray(s1_row, dtype=np.int64)
    hi = np.asarray(p1) >= 0.5
    out = {}
    rn, sn = [_num_set(x) for x in r_addr], [_num_set(x) for x in s1_addr]
    dev = [a - b for a, b in zip(rn, sn)]
    miss = [b - a for a, b in zip(rn, sn)]
    out["cn_dev_num"] = np.fromiter((len(x) for x in dev), np.float32, len(dev))
    out["cn_sh_num"], out["cn_sh_num_hi"] = _consensus(g, hi, rn, dev)
    out["cn_miss_num"] = np.fromiter((len(x) for x in miss), np.float32, len(miss))
    out["cn_have_miss"], _ = _consensus(g, hi, rn, miss)
    del rn, sn, dev, miss
    ra, sa = [_atok_set(x) for x in r_addr], [set(x.split()) for x in s1_addr]
    dev = [a - b for a, b in zip(ra, sa)]
    out["cn_dev_atok"] = np.fromiter((len(x) for x in dev), np.float32, len(dev))
    out["cn_sh_atok"], out["cn_sh_atok_hi"] = _consensus(g, hi, ra, dev)
    del ra, sa, dev
    rt, stt = [_ntok_set(x) for x in r_name], [set(x.split()) for x in s1_name]
    dev = [a - b for a, b in zip(rt, stt)]
    out["cn_dev_tok"] = np.fromiter((len(x) for x in dev), np.float32, len(dev))
    out["cn_sh_tok"], out["cn_sh_tok_hi"] = _consensus(g, hi, rt, dev)
    del rt, stt, dev
    s = pd.Series(g)
    out["cn_n_other"] = (s.map(s.value_counts()).to_numpy() - 1).astype(np.float32)
    nh = pd.Series(hi.astype(np.int32)).groupby(g).transform("sum").to_numpy()
    out["cn_n_other_hi"] = (nh - hi).astype(np.float32)
    return pd.DataFrame(out)[CONS_FEATURES]
