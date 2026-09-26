"""Generator edit operations (experiments.md E9): decoy mutations vs genuine record noise.

A pair (S1, record) is broken into atomic descriptors: character edit ops between each S1 name token and its
closest record token (``name_tr``), whole-word insertions / deletions, and the relation of the zero-stripped
house number and number set. E9 measured, in the confusable zone, which descriptors are decoy fingerprints
(e.g. an inserted ``pvt`` / ``private``, an appended letter, ``d>p`` / ``s>z`` swaps, a house number with a
digit dropped mid-way) and which are genuine noise (``dba`` / ``formerly``, OCR swaps ``o>0`` / ``l>1``).

Features (``EDITOP_FEATURES``, v5 union only):
  eo_llr        naive-Bayes sum of log P(d | decoy) / P(d | true) over the pair's descriptors, from the
                versioned table ``configs/E9_lr_table.json`` (learned on folds 3-4 only by experiments/e9_edit_ops.py)
  eo_decoy_rep  letter replacements from the decoy set;  eo_ocr_rep  OCR-style swaps (0/o, 1/l, 5/s, 8/b ...)
  eo_legal_ins  legal words inserted by the record that the S1 lacks (pvt, private, llp, ltd, limited ...)
  eo_appended   S1 token with 1-2 letters appended;  eo_dba  dba / formerly / aka / doing-business-as marker
  eo_house_rel  house-number relation code (NUM_REL order);  eo_num_rel  worst relation over record numbers
"""

import json
import os
import re

import numpy as np
from rapidfuzz.distance import Levenshtein

from .config import REPO_ROOT
from .features import NOISE_TOKENS, phonetic_key

EDITOP_FEATURES = ["eo_llr", "eo_decoy_rep", "eo_ocr_rep", "eo_legal_ins", "eo_appended", "eo_dba",
                   "eo_house_rel", "eo_num_rel"]
NUM_REL = ["none", "equal", "digit_dropped_first", "digit_dropped_last", "digit_dropped_mid", "digit_added",
           "digit_replaced", "digits_transposed", "shift<=10", "shift<=100", "unrelated"]
OCR = {frozenset(p) for p in (("0", "o"), ("1", "l"), ("1", "i"), ("l", "i"), ("5", "s"), ("8", "b"), ("2", "z"),
                              ("6", "g"))}
# decoy letter replacements with LR >= 10 in E9 (support >= 60)
DECOY_REP = {("d", "p"), ("s", "z"), ("k", "c"), ("a", "i"), ("a", "x"), ("i", "u"), ("l", "t"), ("t", "l"),
             ("u", "e"), ("l", "n"), ("l", "i"), ("e", "o"), ("c", "d"), ("p", "d"), ("n", "l")}
LEGAL = frozenset({"pvt", "private", "llp", "ltd", "limited", "llc", "inc", "corp", "corporation", "co", "company",
                   "lp", "pllc", "pc", "incorporated", "praivet", "piraivet"})
DBA = frozenset({"dba", "formerly", "fka", "aka", "doing"})
LR_TABLE = "E9_lr_table.json"
LR_TABLE_PATH = os.path.join(REPO_ROOT, "configs", LR_TABLE)
_lr_cache = {}


def lr_table() -> dict:
    """descriptor -> log LR, from the versioned table (a missing table is an error, not silent zeros)."""
    if "t" not in _lr_cache:
        if not os.path.exists(LR_TABLE_PATH):
            raise FileNotFoundError(f"{LR_TABLE_PATH} is missing: run experiments/e9_edit_ops.py")
        _lr_cache["t"] = json.load(open(LR_TABLE_PATH))["log_lr"]
    return _lr_cache["t"]


def token_ops(x: str, y: str) -> set:
    """Descriptors of the edit x (S1 token) -> y (record token)."""
    d = set()
    ops = Levenshtein.editops(x, y)
    d.add(f"tok_edit_n={min(len(ops), 3)}")
    if phonetic_key(x) == phonetic_key(y):
        d.add("tok_phonetic_equal")
    if len(x) == len(y) and sorted(x) == sorted(y) and len(ops) == 2:
        d.add("tok_transposition")
    if y.startswith(x) and len(y) - len(x) <= 2:
        d.add(f"tok_appended:{y[len(x):]}" if len(y) - len(x) == 1 else "tok_appended_2")
    if x.startswith(y) and len(x) - len(y) <= 2:
        d.add("tok_truncated")
    if re.sub(r"(.)\1", r"\1", y) == x or re.sub(r"(.)\1", r"\1", x) == y:
        d.add("tok_double_letter")
    for op, i, j in ops:
        a = x[i] if op != "insert" and i < len(x) else ""
        b = y[j] if op != "delete" and j < len(y) else ""
        if op == "replace":
            d.add(f"rep:{a}>{b}")
            if frozenset((a, b)) in OCR:
                d.add("rep_ocr")
            if a.isdigit() or b.isdigit():
                d.add("rep_digit_letter")
            if a in "aeiou" and b in "aeiou":
                d.add("rep_vowel_vowel")
        elif op == "insert":
            d.add(f"ins:{b}")
            d.add("ins_pos:" + ("end" if j >= len(x) else ("start" if j == 0 else "mid")))
        else:
            d.add(f"del:{a}")
    return d


def name_desc(a: str, b: str) -> set:
    """Descriptors of the name edit S1 ``a`` -> record ``b`` (``name_tr`` strings)."""
    A, B = a.split(), b.split()
    sa, sb = set(A), set(B)
    d = set()
    if A == B:
        return {"name_identical"}
    if sa == sb:
        d.add("name_reordered_only")
    left_b = sb - sa
    for x in sorted(sa - sb):                   # sorted: set order depends on the process hash seed
        lim = max(2, len(x) // 3)
        best, bd = None, lim + 1
        for y in sorted(left_b):
            e = Levenshtein.distance(x, y, score_cutoff=lim)
            if e < bd:
                best, bd = y, e
        if best is None:
            d.add("del_word_filler" if x in NOISE_TOKENS else ("del_word_short" if len(x) <= 2 else "del_word_content"))
        else:
            left_b.discard(best)
            d |= token_ops(x, best)
    for y in sorted(left_b):
        d.add("ins_word_filler" if y in NOISE_TOKENS else ("ins_word_short" if len(y) <= 2 else "ins_word_content"))
        if y in NOISE_TOKENS:
            d.add(f"ins_filler:{y}")
    if len(B) > len(sb):
        d.add("name_dup_word")
    return d


def num_rel(x: str, y: str) -> str:
    """Relation of record number ``y`` to S1 number ``x`` (both zero-stripped)."""
    if x == y:
        return "equal"
    if len(y) == len(x) - 1 and any(x[:i] + x[i + 1:] == y for i in range(len(x))):
        return "digit_dropped_first" if x[1:] == y else ("digit_dropped_last" if x[:-1] == y else "digit_dropped_mid")
    if len(y) == len(x) + 1 and any(y[:i] + y[i + 1:] == x for i in range(len(y))):
        return "digit_added"
    if len(x) == len(y) and Levenshtein.distance(x, y) == 1:
        return "digit_replaced"
    if len(x) == len(y) and sorted(x) == sorted(y):
        return "digits_transposed"
    dlt = abs(int(x[:15]) - int(y[:15]))
    return "shift<=10" if dlt <= 10 else ("shift<=100" if dlt <= 100 else "unrelated")


def addr_desc(a_digits: str, b_digits: str, b_addr: str) -> tuple[set, str, str]:
    """(descriptors, house relation, worst number relation) of the address edit."""
    if b_addr.replace("null", "").strip() == "":
        return {"addr_empty"}, "none", "none"
    A = [t.lstrip("0") or "0" for t in a_digits.split()]
    raw = b_digits.split()
    B = [t.lstrip("0") or "0" for t in raw]
    d = {"rec_zero_padded"} if raw != B else set()
    if not B:
        return d | {"rec_no_number"}, "none", "none"
    if not A:
        return d | {"s1_no_number"}, "none", "none"
    house = num_rel(A[0], B[0])
    d.add("house:" + house)
    rels = [min((num_rel(x, y) for x in A), key=NUM_REL.index) for y in B]
    for r in set(rels):
        d.add("anynum:" + r)
    if len(B) > len(A):
        d.add("rec_more_numbers")
    elif len(B) < len(A):
        d.add("rec_fewer_numbers")
    return d, house, max(rels, key=NUM_REL.index)


def edit_op_features(a_names: list[str], b_names: list[str], a_digits: list[str], b_digits: list[str],
                     b_addr: list[str]) -> dict:
    """EDITOP_FEATURES for aligned pair lists (S1 = a, record = b)."""
    table = lr_table()
    n = len(a_names)
    out = {"eo_llr": np.zeros(n, np.float32), "eo_decoy_rep": np.zeros(n, np.int8), "eo_ocr_rep": np.zeros(n, np.int8),
           "eo_legal_ins": np.zeros(n, np.int8), "eo_appended": np.zeros(n, np.int8), "eo_dba": np.zeros(n, np.int8),
           "eo_house_rel": np.zeros(n, np.int8), "eo_num_rel": np.zeros(n, np.int8)}
    for i in range(n):
        nd = name_desc(a_names[i], b_names[i])
        ad, house, worst = addr_desc(a_digits[i], b_digits[i], b_addr[i])
        d = nd | ad
        out["eo_llr"][i] = sum(table.get(x, 0.0) for x in d)
        out["eo_decoy_rep"][i] = sum(1 for x in d if x.startswith("rep:") and tuple(x[4:].split(">")) in DECOY_REP)
        out["eo_ocr_rep"][i] = 1 if "rep_ocr" in d else 0
        sa = set(a_names[i].split())
        out["eo_legal_ins"][i] = sum(1 for t in set(b_names[i].split()) - sa if t in LEGAL)
        out["eo_appended"][i] = sum(1 for x in d if x.startswith("tok_appended"))
        out["eo_dba"][i] = 1 if DBA & set(b_names[i].split()) else 0
        out["eo_house_rel"][i] = NUM_REL.index(house)
        out["eo_num_rel"][i] = NUM_REL.index(worst)
    return out
