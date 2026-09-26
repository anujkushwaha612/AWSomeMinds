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
from rapidfuzz.distance import JaroWinkler, Levenshtein

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
# Generator-aware features (experiments.md E6; v5 union only). The generator zero-pads house numbers
# (20% of rejected true pairs vs 5% of all), drops / replaces a digit or shifts the number slightly,
# and edits names with a fixed vocabulary of filler words; decoy records instead swap a CONTENT word
# (15% of false positives vs 4% of true pairs).
GEN_FEATURES = ["zhouse_state", "znum_only_rec", "znum_min_edit", "znum_min_rel", "znum_affix",
                "nm_ins", "nm_del", "nm_sub", "nm_shared", "nm_content_cov", "nm_concat_sim"]
# Filler tokens the generator inserts into / deletes from names (``name_tr``). Curated from the
# edit rates that experiments/e6_mine_vocab.py measures on fold-3 true pairs (never the report fold):
# legal suffixes (named in the problem statement; incl. their transliterations and French forms,
# France being in test), honorifics, DBA / web markers, and the generic words inserted at edit rates
# >= 0.3 (median token 0.145). Content words stay out, so a swapped content word still counts.
NOISE_TOKENS = frozenset((
    "llc", "inc", "lnc", "incorporated", "ltd", "limited", "limitet", "limittad", "pvt", "private",
    "praivet", "piraivet", "praibhet", "praivatt", "corp", "corporation", "co", "company", "llp", "lp",
    "pllc", "pc", "plc", "elaelapi", "elel", "elelpi", "sarl", "sas", "sa", "eurl", "sci", "snc", "cie",
    "dr", "mr", "mrs", "ms", "smt", "shri", "shree", "sri", "sree",
    "dba", "fka", "aka", "formerly", "known", "doing", "as", "com", "c0m", "www", "net", "org", "id",
    "the", "and", "of", "center", "centre", "service", "services", "5ervices", "sarvisas", "sarvisej",
    "partners",
))
DIGIT_RUN = re.compile(r"\d+")


def fuzzy_in(t: str, toks) -> bool:
    """True if some token of ``toks`` is within the typo budget of ``t`` (exact up to 3 chars, then 1-2 edits)."""
    d = 0 if len(t) <= 3 else (1 if len(t) < 6 else 2)
    return any(Levenshtein.distance(t, u, score_cutoff=d) <= d for u in toks)


# Tested in experiments.md E6 and NOT kept: +0.0008 alone, negative on top of CHAIN_FEATURES.
ADDR_FEATURES = ["ad_cov", "ad_extra", "ad_tr_tset"]
# Name ambiguity (E6: +0.0039 fold-0 F0.5 on top of GEN_FEATURES): S1s sharing the S1's / record's name.
CHAIN_FEATURES = ["s1_name_freq", "rec_name_freq"]
_US_STATES = ("al alabama ak alaska az arizona ar arkansas ca california co colorado ct connecticut de delaware "
              "fl florida ga georgia hi hawaii id idaho il illinois in indiana ia iowa ks kansas ky kentucky "
              "la louisiana me maine md maryland ma massachusetts mi michigan mn minnesota ms mississippi "
              "mo missouri mt montana ne nebraska nv nevada nh newhampshire nj newjersey nm newmexico "
              "ny newyork nc northcarolina nd northdakota oh ohio ok oklahoma or oregon pa pennsylvania "
              "ri rhodeisland sc southcarolina sd southdakota tn tennessee tx texas ut utah vt vermont "
              "va virginia wa washington wv westvirginia wi wisconsin wy wyoming dc districtofcolumbia").split()
_IN_STATES = ("mh maharashtra dl delhi ka karnataka tn tamilnadu tg telangana ap andhrapradesh gj gujarat "
              "rj rajasthan up uttarpradesh wb westbengal kl kerala keralam hr haryana pb punjab mp madhyapradesh "
              "br bihar or odisha jh jharkhand cg chhattisgarh as assam uk uttarakhand hp himachalpradesh "
              "ga goa jk jammuandkashmir").split()
# street-type / direction abbreviations (USPS-style) and India's old city names, to one canonical token
ADDR_CANON = {**{w: a for a, w in zip(_US_STATES[0::2], _US_STATES[1::2])},
              **{w: a for a, w in zip(_IN_STATES[0::2], _IN_STATES[1::2])},
              "street": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr",
              "court": "ct", "lane": "ln", "place": "pl", "boulevard": "blvd", "highway": "hwy",
              "parkway": "pkwy", "circle": "cir", "trail": "trl", "terrace": "ter", "square": "sq",
              "north": "n", "south": "s", "east": "e", "west": "w", "apartment": "apt", "suite": "ste",
              "building": "bldg", "bldg": "bldg", "floor": "fl", "near": "nr", "opposite": "opp",
              "heights": "hts", "height": "hts", "knoll": "knl", "ridge": "rdg", "mount": "mt",
              "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata", "bangalore": "bengaluru",
              "gurgaon": "gurugram", "poona": "pune"}
ADDR_FILLER = frozenset({"null", "no", "h", "hn", "house", "door", "unit", "cdp", "city", "town", "of",
                         "the", "po", "box", "p", "o", "post", "office", "plot", "flat", "shop", "fl"})


def _addr_tokens(s: str) -> list[str]:
    """Canonical non-numeric address tokens (two-word state names are joined first)."""
    t = s.split()
    joined, i = [], 0
    while i < len(t):
        if i + 1 < len(t) and (t[i] + t[i + 1]) in ADDR_CANON:
            joined.append(t[i] + t[i + 1]); i += 2
        else:
            joined.append(t[i]); i += 1
    return [ADDR_CANON.get(x, x) for x in joined if not x.isdigit() and x not in ADDR_FILLER]


def _addr_edit_features(a_addr: list[str], b_addr: list[str]) -> dict:
    """Order-free address agreement after abbreviation / state canonicalization (``addr_tr``).

    ad_cov: share of S1 address words found (fuzzily) in the record (-1 = record address empty);
    ad_extra: record words with no counterpart in S1; ad_tr_tset: token-set ratio on the canonical text.
    """
    n = len(a_addr)
    cov = np.full(n, -1.0, dtype=np.float32)
    extra = np.zeros(n, dtype=np.int8)
    ta_all, tb_all = [], []
    for i in range(n):
        ta, tb = _addr_tokens(a_addr[i]), _addr_tokens(b_addr[i])
        ta_all.append(" ".join(ta)); tb_all.append(" ".join(tb))
        if not tb:
            continue
        sa, sb = set(ta), set(tb)
        cov[i] = (sum(1 for t in sa if t in sb or fuzzy_in(t, sb)) / len(sa)) if sa else 1.0
        extra[i] = min(sum(1 for t in sb if t not in sa and not fuzzy_in(t, sa)), 127)
    return {"ad_cov": cov, "ad_extra": extra, "ad_tr_tset": _sim(ta_all, tb_all, fuzz.token_set_ratio)}


def _gen_number_features(a_digits: list[str], b_digits: list[str]) -> dict:
    """Number agreement after stripping leading zeros, and how far the unmatched record numbers are.

    znum_min_edit / znum_min_rel: smallest digit edit distance / relative difference between an
    unmatched record number and any S1 number (0 = all matched, -1 = nothing to compare);
    znum_affix: an unmatched record number is a prefix or suffix of an S1 number or vice versa
    (the generator's dropped digit: 21346 -> 1346, 2120 -> 120).
    """
    n = len(a_digits)
    house = np.zeros(n, dtype=np.int8)
    only_b = np.zeros(n, dtype=np.int16)
    min_edit = np.full(n, -1, dtype=np.int8)
    min_rel = np.full(n, -1.0, dtype=np.float32)
    affix = np.zeros(n, dtype=np.int8)
    for i in range(n):
        ta = [x.lstrip("0") or "0" for x in a_digits[i].split()]
        tb = [x.lstrip("0") or "0" for x in b_digits[i].split()]
        if not ta or not tb:
            continue
        house[i] = 1 if ta[0] == tb[0] else -1
        sa = set(ta)
        miss = [x for x in set(tb) if x not in sa]
        only_b[i] = len(miss)
        if not miss:
            min_edit[i], min_rel[i] = 0, 0.0
            continue
        me, mr, af = 99, 9.0, 0
        for x in miss:
            ix = int(x[:15])
            for y in sa:
                me = min(me, Levenshtein.distance(x, y))
                iy = int(y[:15])
                mr = min(mr, abs(ix - iy) / max(ix, iy, 1))
                if y.startswith(x) or y.endswith(x) or x.startswith(y) or x.endswith(y):
                    af = 1
        min_edit[i], min_rel[i], affix[i] = min(me, 99), mr, af
    return {"zhouse_state": house, "znum_only_rec": only_b, "znum_min_edit": min_edit,
            "znum_min_rel": min_rel, "znum_affix": affix}


def _name_edit_features(a_names: list[str], b_names: list[str]) -> dict:
    """Token edits between S1 and record names (``name_tr``), filler vocabulary excluded.

    nm_ins / nm_del: record / S1 content tokens with no fuzzy counterpart on the other side;
    nm_sub = min of the two (a swapped content word: the decoy pattern); nm_shared: record tokens
    matched exactly or fuzzily; nm_content_cov: share of S1 content tokens found in the record;
    nm_concat_sim: similarity of the concatenated content tokens (domain forms, glued words).
    """
    n = len(a_names)
    ins = np.zeros(n, dtype=np.int8)
    dele = np.zeros(n, dtype=np.int8)
    shared = np.zeros(n, dtype=np.int8)
    cov = np.zeros(n, dtype=np.float32)
    concat = np.zeros(n, dtype=np.float32)
    drop = NOISE_TOKENS | {"com", "www"}
    for i in range(n):
        sa, sb = set(a_names[i].split()), set(b_names[i].split())
        la, lb = sa - sb, sb - sa
        fa = {t for t in la if fuzzy_in(t, lb)}
        fb = {t for t in lb if fuzzy_in(t, la)}
        ins[i] = min(sum(1 for t in lb - fb if t not in NOISE_TOKENS and len(t) > 1), 127)
        dele[i] = min(sum(1 for t in la - fa if t not in NOISE_TOKENS and len(t) > 1), 127)
        shared[i] = min(len(sa & sb) + len(fb), 127)
        ca = [t for t in sa if t not in NOISE_TOKENS and len(t) > 1]
        cov[i] = (sum(1 for t in ca if t in sb or t in fa) / len(ca)) if ca else 1.0
        ja = "".join(t for t in a_names[i].split() if t not in drop)
        jb = "".join(t for t in b_names[i].split() if t not in drop)
        concat[i] = Levenshtein.normalized_similarity(ja, jb) if ja and jb else 0.0
    return {"nm_ins": ins, "nm_del": dele, "nm_sub": np.minimum(ins, dele), "nm_shared": shared,
            "nm_content_cov": cov, "nm_concat_sim": concat}


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


def _py_extra(args) -> dict:
    """The Python-loop extra features for one slice of pairs (runs in a worker process)."""
    from .edit_ops import edit_op_features      # local import: edit_ops imports this module
    a_tr, b_tr, a_dig, b_dig, b_addr = args
    out = _name_edit_features(a_tr, b_tr)
    out.update(_decoy_name_features(a_tr, b_tr))
    out.update(_gen_number_features(a_dig, b_dig))
    out.update(edit_op_features(a_tr, b_tr, a_dig, b_dig, b_addr))
    return out


def py_extra_features(a_tr, b_tr, a_dig, b_dig, b_addr, pool=None) -> dict:
    """GEN (numbers, name edits) + DECOY + EDITOP features; split across ``pool`` workers when given."""
    n = len(a_tr)
    if pool is None or n < 20_000:
        return _py_extra((a_tr, b_tr, a_dig, b_dig, b_addr))
    k = 2 * pool._max_workers
    bounds = np.linspace(0, n, k + 1).astype(int)
    parts = [(a_tr[s:e], b_tr[s:e], a_dig[s:e], b_dig[s:e], b_addr[s:e])
             for s, e in zip(bounds[:-1], bounds[1:]) if e > s]
    res = list(pool.map(_py_extra, parts))
    return {key: np.concatenate([r[key] for r in res]) for key in res[0]}


def string_block(store, src: int, s1_rows: np.ndarray, doc_rows: np.ndarray,
                 extra: bool = False, pool=None) -> dict:
    """String/structure features for aligned (S1 row, doc row) arrays of one source.

    ``extra`` adds :data:`STRUCT_FEATURES`, :data:`GEN_FEATURES`, :data:`CHAIN_FEATURES`, :data:`DECOY_FEATURES` and
    ``ber.edit_ops.EDITOP_FEATURES`` (v5 union only); the Python-loop part runs on ``pool`` when given.
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
        out.update(store_chain_features(store, an, bn))
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
    if extra:
        a, b = pair("name_tr")
        ad, bd = pair("addr_digits")
        out.update(py_extra_features(a, b, ad, bd, store.strings(src, "addr_n", doc_rows), pool))
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
    pool = None
    if extra:                                  # Python-loop extra features: all but two cores
        import os
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=max(1, min(12, (os.cpu_count() or 4) - 2)))
    for start in range(0, n, chunk):
        part = cand.iloc[start:start + chunk]
        blocks = []
        for src in (2, 3):
            m = (part["src"] == src).to_numpy()
            if not m.any():
                continue
            sub = part[m].reset_index(drop=True)
            feats = string_block(store, src, sub["s1_row"].to_numpy(), sub["doc_row"].to_numpy(), extra, pool)
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
    if pool is not None:
        pool.shutdown()
    return n


def chain_features(s1_names: list[str], s1_rows_name: list[str], rec_names: list[str]) -> dict:
    """How ambiguous a name is: S1s of the country sharing the S1's / the record's exact ``name_n``.

    ``s1_names``: every S1 ``name_n`` of the country (the count table); the other two are aligned per pair.
    """
    counts = pd.Series(s1_names).value_counts()
    return {"s1_name_freq": pd.Series(s1_rows_name).map(counts).fillna(0).to_numpy(np.int32),
            "rec_name_freq": pd.Series(rec_names).map(counts).fillna(0).to_numpy(np.int32)}


def store_chain_features(store, s1_names: list[str], rec_names: list[str]) -> dict:
    """:func:`chain_features` with the country's S1 name counts cached on the store."""
    if getattr(store, "_s1_name_counts", None) is None:
        store._s1_name_counts = pd.Series(store.strings(1, "name_n")).value_counts()
    counts = store._s1_name_counts
    return {"s1_name_freq": pd.Series(s1_names).map(counts).fillna(0).to_numpy(np.int32),
            "rec_name_freq": pd.Series(rec_names).map(counts).fillna(0).to_numpy(np.int32)}


# Decoy signature (experiments.md E7b): orphan records that are false positives carry a PHONETIC respelling
# of a content word of the S1 name (quantyn -> kwantyn, halcify -> halkify) in 17.5% of cases, true pairs in
# 0.1-0.2%; generator noise on true records is typo-style instead. v5 union only.
DECOY_FEATURES = ["nm_phonetic", "nm_typo"]
_PHON = (("qu", "k"), ("kw", "k"), ("ck", "k"), ("ph", "f"), ("kh", "k"), ("x", "ks"), ("c", "k"), ("z", "s"), ("y", "i"),
         ("w", "v"))
_REPEAT = re.compile(r"(.)\1+")
_TRAIL_VOWELS = re.compile(r"[aeiou]+$")


def phonetic_key(t: str) -> str:
    """Crude English phonetic key: c/k/q, ph/f, z/s, y/i, w/v merged, doubled letters and trailing vowels dropped."""
    for a, b in _PHON:
        t = t.replace(a, b)
    t = _REPEAT.sub(r"\1", t)
    return _TRAIL_VOWELS.sub("", t) or t


def _decoy_name_features(a_names: list[str], b_names: list[str]) -> dict:
    """Per pair: S1 content tokens whose closest record token differs only phonetically / by a typo (``name_tr``)."""
    n = len(a_names)
    phon = np.zeros(n, dtype=np.int8)
    typo = np.zeros(n, dtype=np.int8)
    for i in range(n):
        A = {t for t in a_names[i].split() if t not in NOISE_TOKENS and len(t) > 2}
        B = [t for t in b_names[i].split() if t not in NOISE_TOKENS and len(t) > 2]
        if not A or not B:
            continue
        sb = set(B)
        p = q = 0
        for x in sorted(A - sb):              # sorted: set order depends on the process hash seed
            lim = max(2, len(x) // 3)
            best, bd = None, lim + 1
            for y in sorted(sb):
                d = Levenshtein.distance(x, y, score_cutoff=lim)
                if d < bd:
                    best, bd = y, d
            if best is None:
                continue
            if phonetic_key(x) == phonetic_key(best):
                p += 1
            else:
                q += 1
        phon[i], typo[i] = min(p, 127), min(q, 127)
    return {"nm_phonetic": phon, "nm_typo": typo}
