"""Pair features for candidate (S1, S2/S3) pairs (plan.md Step 6, baseline set).

Three groups, all computed identically on train and test and never from labels:
  retrieval  - TF-IDF score, record-side rank / gap to the record's best S1,
               S1-side rank / gap within the same source, candidate counts
  string     - rapidfuzz similarities on name / legal-stripped name / no-space
               name / transliterated name / address (vectorized, native threads)
  structure  - address digit overlap, house-number agreement, script flag,
               missing address, length ratio, source

No country feature: France is unseen in training (plan.md §1).

Memory-lean: retrieval features are computed on the compact candidate arrays;
string features run chunk by chunk (text looked up from the Arrow store for
that chunk only) and every chunk is appended to a parquet file.
"""

import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURES = [
    "score", "rank", "gap_rec", "n_cand_rec", "rank_s1", "gap_s1", "n_cand_s1", "n_cand_s1_src",
    "name_ratio", "name_tset", "name_partial", "name_jw", "legal_ratio", "nospace_ratio",
    "translit_ratio", "addr_ratio", "addr_tset", "addr_partial",
    "digit_jacc", "house_state", "n_digits_s1", "n_digits_rec",
    "src", "rec_indic", "rec_addr_empty", "name_len_ratio", "addr_len_ratio",
]
KEY_COLS = ["src", "doc_row", "s1_row"]
# Number-conflict features (v5 union only; the baseline feature files do not have them).
# house_state compares only the FIRST number; these compare the full number sets, so
# "suite 400" vs "suite 500" or a differing PIN/ZIP registers as a conflict even when
# the house number and every string similarity agree.
STRUCT_FEATURES = ["num_only_s1", "num_only_rec", "num_conflict", "num_conflict_len",
                   "name_num_conflict", "name_num_diff"]
DIGIT_RUN = re.compile(r"\d+")


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


def _number_features(a_digits: list[str], b_digits: list[str],
                     a_name: list[str], b_name: list[str]) -> dict:
    """Set-level number agreement between S1 (a) and record (b).

    Address numbers (``addr_digits`` tokens): how many are only in S1 / only in the
    record; a *conflict* is when both sides have an unmatched number (a missing
    component only makes one side non-empty, which is benign), and its length is the
    shorter of the two sides' longest unmatched number (1-digit unit vs 6-digit PIN).
    Name numbers (digit runs in ``name_n``): the same conflict count, plus the total
    number of unmatched name numbers.
    """
    n = len(a_digits)
    only_a = np.zeros(n, dtype=np.int16)
    only_b = np.zeros(n, dtype=np.int16)
    conflict = np.zeros(n, dtype=np.int16)
    conflict_len = np.zeros(n, dtype=np.int16)
    name_conflict = np.zeros(n, dtype=np.int16)
    name_diff = np.zeros(n, dtype=np.int16)
    for i in range(n):
        x, y = a_digits[i], b_digits[i]
        if x or y:
            sx, sy = set(x.split()), set(y.split())
            ox, oy = sx - sy, sy - sx
            only_a[i], only_b[i] = len(ox), len(oy)
            if ox and oy:
                conflict[i] = min(len(ox), len(oy))
                conflict_len[i] = min(max(map(len, ox)), max(map(len, oy)))
        nx, ny = DIGIT_RUN.findall(a_name[i]), DIGIT_RUN.findall(b_name[i])
        if nx or ny:
            sx, sy = set(nx), set(ny)
            ox, oy = len(sx - sy), len(sy - sx)
            name_conflict[i] = min(ox, oy)
            name_diff[i] = ox + oy
    return {"num_only_s1": only_a, "num_only_rec": only_b, "num_conflict": conflict,
            "num_conflict_len": conflict_len, "name_num_conflict": name_conflict,
            "name_num_diff": name_diff}


def _len_ratio(a: list[str], b: list[str]) -> np.ndarray:
    la = np.fromiter((len(x) for x in a), dtype=np.int32, count=len(a))
    lb = np.fromiter((len(x) for x in b), dtype=np.int32, count=len(b))
    return (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)


def retrieval_features(cand: pd.DataFrame) -> pd.DataFrame:
    """Add record-side and S1-side retrieval features (numeric columns only)."""
    rec = cand.groupby(["src", "doc_row"], sort=False)["score"]
    cand["gap_rec"] = (rec.transform("max") - cand["score"]).astype(np.float32)
    cand["n_cand_rec"] = rec.transform("size").astype(np.int32)
    s1src = cand.groupby(["s1_row", "src"], sort=False)["score"]
    cand["rank_s1"] = (s1src.rank(ascending=False, method="first") - 1).astype(np.int32)
    cand["gap_s1"] = (s1src.transform("max") - cand["score"]).astype(np.float32)
    cand["n_cand_s1_src"] = s1src.transform("size").astype(np.int32)
    cand["n_cand_s1"] = cand.groupby("s1_row", sort=False)["score"].transform("size").astype(np.int32)
    return cand


def string_block(store, src: int, s1_rows: np.ndarray, doc_rows: np.ndarray,
                 extra: bool = False) -> dict:
    """String/structure features for aligned (S1 row, doc row) arrays of one source.

    ``extra`` adds :data:`STRUCT_FEATURES` (used by the v5 union files only).
    """
    def pair(col):
        return store.strings(1, col, s1_rows), store.strings(src, col, doc_rows)

    out = {}
    an, bn = pair("name_n")
    out["name_ratio"] = _sim(an, bn, fuzz.ratio)
    out["name_tset"] = _sim(an, bn, fuzz.token_set_ratio)
    out["name_partial"] = _sim(an, bn, fuzz.partial_ratio)
    out["name_jw"] = _sim(an, bn, JaroWinkler.normalized_similarity)
    out["name_len_ratio"] = _len_ratio(an, bn)
    if extra:
        ad, bd = pair("addr_digits")
        out.update(_number_features(ad, bd, an, bn))
        del ad, bd
    del an, bn
    a, b = pair("name_legal")
    out["legal_ratio"] = _sim(a, b, fuzz.ratio)
    a, b = pair("name_nospace")
    out["nospace_ratio"] = _sim(a, b, fuzz.ratio)
    a, b = pair("name_tr")
    out["translit_ratio"] = _sim(a, b, fuzz.token_set_ratio)
    aa, ba = pair("addr_n")
    out["addr_ratio"] = _sim(aa, ba, fuzz.ratio)
    out["addr_tset"] = _sim(aa, ba, fuzz.token_set_ratio)
    out["addr_partial"] = _sim(aa, ba, fuzz.partial_ratio)
    out["addr_len_ratio"] = _len_ratio(aa, ba)
    out["rec_addr_empty"] = np.fromiter((x == "" for x in ba), dtype=np.int8, count=len(ba))
    del aa, ba
    a, b = pair("addr_digits")
    (out["digit_jacc"], out["house_state"], out["n_digits_s1"],
     out["n_digits_rec"]) = _digit_features(a, b)
    del a, b
    script = store.numpy(src, "script")[doc_rows]
    out["rec_indic"] = (script > 0).astype(np.int8)
    return out


def write_features(cand: pd.DataFrame, store, path: str, chunk: int = 500_000,
                   log=print, extra: bool = False) -> int:
    """Compute string features chunk by chunk and append everything to ``path``.

    ``cand`` holds keys, retrieval features and (on train) ``label``. Returns the
    number of rows written. Row order in the file = row order of ``cand`` (which must
    be sorted by ``src`` within each chunk). ``extra``: see :func:`string_block`.
    """
    writer = None
    n = len(cand)
    for start in range(0, n, chunk):
        part = cand.iloc[start:start + chunk]
        blocks = []
        for src in (2, 3):
            m = (part["src"] == src).to_numpy()
            if not m.any():
                continue
            sub = part[m].reset_index(drop=True)
            feats = string_block(store, src, sub["s1_row"].to_numpy(), sub["doc_row"].to_numpy(), extra)
            blocks.append(pd.concat([sub, pd.DataFrame(feats)], axis=1))
        table = pa.Table.from_pandas(pd.concat(blocks, ignore_index=True), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        del blocks, table
        if log and (start // chunk) % 10 == 0:
            log(f"    features {min(start + chunk, n):,}/{n:,}")
    if writer is not None:
        writer.close()
    return n
