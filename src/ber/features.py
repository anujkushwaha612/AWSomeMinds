"""Pair features for candidate (S1, S2/S3) pairs (plan.md Step 6, baseline set).

Three groups, all computed identically on train and test and never from labels:
  retrieval  - TF-IDF score, record-side rank / gap to the record's best S1,
               S1-side rank / gap within the same source, candidate counts
  string     - rapidfuzz similarities on name / legal-stripped name / no-space
               name / transliterated name / address (vectorized, all cores)
  structure  - address digit overlap, house-number agreement, script flag,
               missing address, length ratio, source

No country feature: France is unseen in training (plan.md §1).
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURES = [
    "score", "rank", "gap_rec", "n_cand_rec", "rank_s1", "gap_s1", "n_cand_s1", "n_cand_s1_src",
    "name_ratio", "name_tset", "name_partial", "name_jw", "legal_ratio", "nospace_ratio",
    "translit_ratio", "addr_ratio", "addr_tset", "addr_partial",
    "digit_jacc", "house_state", "n_digits_s1", "n_digits_rec",
    "src", "rec_indic", "rec_addr_empty", "name_len_ratio", "addr_len_ratio",
]


def _sim(a, b, scorer) -> np.ndarray:
    """Element-wise similarity of two aligned string lists (0-100), all cores."""
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _digit_features(a: list[str], b: list[str]):
    """Digit-token Jaccard, house-number state (1 match, -1 conflict, 0 missing), counts."""
    n = len(a)
    jacc = np.zeros(n, dtype=np.float32)
    house = np.zeros(n, dtype=np.int8)
    na = np.zeros(n, dtype=np.int16)
    nb = np.zeros(n, dtype=np.int16)
    for i, (x, y) in enumerate(zip(a, b)):
        tx, ty = x.split(), y.split()
        na[i], nb[i] = len(tx), len(ty)
        if tx and ty:
            sx, sy = set(tx), set(ty)
            jacc[i] = len(sx & sy) / len(sx | sy)
            house[i] = 1 if tx[0] == ty[0] else -1
    return jacc, house, na, nb


def retrieval_features(cand: pd.DataFrame) -> pd.DataFrame:
    """Add record-side and S1-side retrieval features to one country's candidates."""
    rec = cand.groupby(["src", "doc_row"])["score"]
    cand["gap_rec"] = (rec.transform("max") - cand["score"]).astype(np.float32)
    cand["n_cand_rec"] = rec.transform("size").astype(np.int16)
    s1src = cand.groupby(["s1_row", "src"])["score"]
    cand["rank_s1"] = (s1src.rank(ascending=False, method="first") - 1).astype(np.int16)
    cand["gap_s1"] = (s1src.transform("max") - cand["score"]).astype(np.float32)
    cand["n_cand_s1_src"] = s1src.transform("size").astype(np.int16)
    cand["n_cand_s1"] = cand.groupby("s1_row")["score"].transform("size").astype(np.int16)
    return cand


def string_features(cand: pd.DataFrame, s1: pd.DataFrame, docs: dict) -> pd.DataFrame:
    """Add string/structure features. ``s1``/``docs[src]`` are normalized tables."""
    out = []
    for src in (2, 3):
        c = cand[cand["src"] == src].copy()
        if c.empty:
            continue
        a = s1.iloc[c["s1_row"].to_numpy()]
        b = docs[src].iloc[c["doc_row"].to_numpy()]
        an, bn = a["name_n"].tolist(), b["name_n"].tolist()
        aa, ba = a["addr_n"].tolist(), b["addr_n"].tolist()
        c["name_ratio"] = _sim(an, bn, fuzz.ratio)
        c["name_tset"] = _sim(an, bn, fuzz.token_set_ratio)
        c["name_partial"] = _sim(an, bn, fuzz.partial_ratio)
        c["name_jw"] = _sim(an, bn, JaroWinkler.normalized_similarity)
        c["legal_ratio"] = _sim(a["name_legal"].tolist(), b["name_legal"].tolist(), fuzz.ratio)
        c["nospace_ratio"] = _sim(a["name_nospace"].tolist(), b["name_nospace"].tolist(), fuzz.ratio)
        c["translit_ratio"] = _sim(a["name_tr"].tolist(), b["name_tr"].tolist(), fuzz.token_set_ratio)
        c["addr_ratio"] = _sim(aa, ba, fuzz.ratio)
        c["addr_tset"] = _sim(aa, ba, fuzz.token_set_ratio)
        c["addr_partial"] = _sim(aa, ba, fuzz.partial_ratio)
        (c["digit_jacc"], c["house_state"], c["n_digits_s1"],
         c["n_digits_rec"]) = _digit_features(a["addr_digits"].tolist(), b["addr_digits"].tolist())
        c["rec_indic"] = (b["script"].to_numpy() > 0).astype(np.int8)
        c["rec_addr_empty"] = (b["addr_n"].to_numpy() == "").astype(np.int8)
        la, lb = a["name_n"].str.len().to_numpy(), b["name_n"].str.len().to_numpy()
        c["name_len_ratio"] = (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
        la, lb = a["addr_n"].str.len().to_numpy(), b["addr_n"].str.len().to_numpy()
        c["addr_len_ratio"] = (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
        out.append(c)
    return pd.concat(out, ignore_index=True)
