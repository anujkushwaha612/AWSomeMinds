"""Record normalization (plan.md Step 2): additive views, never destructive.

Views produced per record (raw fields are kept alongside):
  name_n / addr_n   lowercase, Latin diacritics folded to ASCII, Indic digits to
                    ASCII, punctuation -> space, literal "null"-like values -> "".
                    Indic letters are kept as-is (they are not ASCII-foldable).
  name_legal        name_n without legal-form tokens (llc, pvt, ltd, sarl, ...).
  name_nospace      name_n with spaces and web affixes (www, .com, ...) removed, so
                    "kairossons.com" and "kairos sons" compare equal-ish.
  addr_digits       digit runs of the address, space-joined.
  script            0 = no Indic letters, 1..9 = first Indic block seen in the name
                    (see INDIC_BLOCKS).
  name_tr / addr_tr [EXP] all-Indic -> Latin transliteration of name_n / addr_n
                    (Latin text passes through unchanged).

Everything here is pure string processing on the provided data: no external
dictionaries, services or lookups.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

# Unicode Indic blocks are laid out in parallel (ISCII-derived), 0x80 apart.
INDIC_BLOCKS = ["Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym"]
INDIC_START, INDIC_END = 0x0900, 0x0D80

NULL_VALUES = {"", "null", "none", "nan", "n/a", "na", "nil", "-", "--", "---", "0", "."}

LEGAL_TOKENS = frozenset("""
    llc inc incorporated corp corporation co company cos ltd limited pvt private
    llp plc lp lllp pllc pc pa sarl sas sasu sa eurl snc sci gmbh ag pte bv nv
    the and of
    praivet praivat prayvet piraivet praibhet pra li
    limitad limitet limittad limited elaelapi elelapi elelpi
""".split())

WEB_AFFIXES = re.compile(r"^(?:www )|(?: (?:com|in|net|org|co|biz|info|fr|us))+$")


def _latin_fold_table() -> dict:
    """Translate table: Latin letters with diacritics -> ASCII, Indic digits -> ASCII.

    Built once from Unicode decompositions of U+00C0..U+024F plus a few letters
    that do not decompose (oe, ae, ss, o-slash, ...).
    """
    table = {}
    for cp in range(0x00C0, 0x0250):
        ch = chr(cp)
        base = "".join(c for c in unicodedata.normalize("NFKD", ch)
                       if not unicodedata.combining(c))
        if base and base.isascii() and base != ch:
            table[cp] = base
    table.update({ord(k): v for k, v in {
        "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ø": "o", "Ø": "o",
        "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "ı": "i", "þ": "th", "ð": "d",
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "＆": "&",
    }.items()})
    for b in range(len(INDIC_BLOCKS)):          # Indic digits share offsets 0x66..0x6F
        for d in range(10):
            table[INDIC_START + 0x80 * b + 0x66 + d] = str(d)
    return table


FOLD = _latin_fold_table()
NON_TOKEN = re.compile(r"[^0-9a-zऀ-ൿ]+")
DIGIT_RUN = re.compile(r"\d+")
INDIC_CHAR = re.compile(r"[ऀ-ൿ]")


def clean_text(s: str) -> str:
    """Normalize one field: null-like -> "", fold, lowercase, punctuation -> space."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(FOLD).lower()
    if s.strip() in NULL_VALUES:
        return ""
    s = s.replace("&", " and ")
    return " ".join(NON_TOKEN.sub(" ", s).split())


def strip_legal(name_n: str) -> str:
    """Drop legal-form and stop tokens; fall back to the input if nothing is left."""
    kept = [t for t in name_n.split() if t not in LEGAL_TOKENS]
    return " ".join(kept) if kept else name_n


def no_space(name_n: str) -> str:
    """Remove web affixes (www, trailing com/in/net/...) and all spaces."""
    return WEB_AFFIXES.sub("", name_n).replace(" ", "")


def script_code(s: str) -> int:
    """0 if ``s`` has no Indic letters, else 1 + index of its first Indic block."""
    m = INDIC_CHAR.search(s)
    return 0 if m is None else 1 + (ord(m.group()) - INDIC_START) // 0x80


# ---------------------------------------------------------------- transliteration
# Offsets within an Indic block (identical across the 9 blocks).
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f",
    0x5F: "y",
}
_VOWELS = {  # independent vowels
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_MATRAS = {  # dependent vowel signs (replace the inherent "a")
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x57: "au", 0x62: "li", 0x63: "li",
}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}   # candrabindu, anusvara, visarga, tippi
_VIRAMA = 0x4D
# Malayalam chillu letters (consonants without inherent vowel) and Bengali khanda ta
_FINAL_CONS = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l",
               0x0D7F: "k", 0x09CE: "t"}
# Word-final inherent "a" is silent in these scripts (Deva, Beng, Guru, Gujr, Orya)
# but pronounced in Tamil/Telugu/Kannada/Malayalam, which mark silence with virama.
_SCHWA_DELETING_BLOCKS = {0, 1, 2, 3, 4}
_PRE_REPLACE = {"റ്റ": "ട്ട"}  # Malayalam rra-rra reads "tt"


def transliterate(s: str) -> str:
    """Romanize Indic letters in ``s`` (all 9 blocks); leave everything else as is.

    Simplified ISO-15919-style mapping tuned for matching English-origin business
    words: retroflex/dental pairs collapse (t/d/n/l), inherent "a" is added
    between consonants and, for north-Indian scripts, dropped at word end
    (schwa deletion); nukta and gemination marks are ignored.
    """
    if not INDIC_CHAR.search(s):
        return s
    for k, v in _PRE_REPLACE.items():
        s = s.replace(k, v)
    out = []
    pending = False            # last output was a consonant still carrying inherent "a"
    block = 0                  # Indic block of the last Indic character
    for ch in s:
        cp = ord(ch)
        if INDIC_START <= cp < INDIC_END:
            block = (cp - INDIC_START) // 0x80
            if cp in _FINAL_CONS:
                if pending:
                    out.append("a")
                out.append(_FINAL_CONS[cp])
                pending = False
                continue
            off = (cp - INDIC_START) % 0x80
            if off in _CONS:
                if pending:
                    out.append("a")
                out.append(_CONS[off])
                pending = True
            elif off in _MATRAS:
                out.append(_MATRAS[off])
                pending = False
            elif off == _VIRAMA:
                pending = False
            elif off in _VOWELS:
                if pending:
                    out.append("a")
                out.append(_VOWELS[off])
                pending = False
            elif off in _NASAL:
                if pending:
                    out.append("a")
                out.append(_NASAL[off])
                pending = False
            # nukta (0x3C), avagraha, danda, addak, etc. are dropped
        else:
            if cp in (0x200C, 0x200D):              # ZWNJ / ZWJ: not a word boundary
                continue
            if pending and block not in _SCHWA_DELETING_BLOCKS:
                out.append("a")
            pending = False
            out.append(ch)
    if pending and block not in _SCHWA_DELETING_BLOCKS:
        out.append("a")
    return " ".join("".join(out).split())


# ---------------------------------------------------------------- frame-level API
def normalize_frame(df: pd.DataFrame, translit: bool = True) -> pd.DataFrame:
    """Return the normalized views for a source table (columns of :mod:`ber.io`)."""
    name_n = [clean_text(s) for s in df["business_name"]]
    addr_n = [clean_text(s) for s in df["business_address"]]
    out = pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "country": df["country"].to_numpy(),
        "name_n": name_n,
        "addr_n": addr_n,
        "name_legal": [strip_legal(s) for s in name_n],
        "name_nospace": [no_space(s) for s in name_n],
        "addr_digits": [" ".join(DIGIT_RUN.findall(s)) for s in addr_n],
        "script": np.array([script_code(s) for s in name_n], dtype=np.int8),
    })
    if translit:
        out["name_tr"] = [transliterate(s) for s in name_n]
        out["addr_tr"] = [transliterate(s) for s in addr_n]
    return out


def normalize_parallel(df: pd.DataFrame, n_jobs: int | None = None,
                       chunk: int = 200_000) -> pd.DataFrame:
    """:func:`normalize_frame` over row chunks in a process pool (same output).

    In-memory convenience for small frames; full tables go through
    :func:`build_normalized`, which streams to parquet.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    parts = [df.iloc[i:i + chunk] for i in range(0, len(df), chunk)]
    if n_jobs == 1 or len(parts) == 1:
        return normalize_frame(df)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return pd.concat(list(pool.map(normalize_frame, parts)), ignore_index=True)


def build_normalized(split: str, source: int, n_jobs: int = 8, chunk: int = 200_000) -> str:
    """Stream ``{split}_source{source}.tsv`` -> normalized parquet; return its path.

    Memory stays bounded: the raw TSV is read ``chunk`` rows at a time and at
    most ``2 * n_jobs`` chunks are in flight in the worker pool; each finished
    chunk is appended to the parquet file (one row group) in input order.
    """
    import os
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    from .config import artifact_path, data_path, ensure_parent

    path = artifact_path("norm", f"{split}_source{source}.parquet")
    tmp = path + ".tmp"
    ensure_parent(path)
    reader = pd.read_csv(data_path(split, f"{split}_source{source}.tsv"), sep="\t", dtype=str,
                         keep_default_na=False, na_filter=False, chunksize=chunk)
    writer, window = None, deque()
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        def drain(limit):
            nonlocal writer
            while len(window) > limit:
                table = pa.Table.from_pandas(window.popleft().result(), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table)
        for part in reader:
            window.append(pool.submit(normalize_frame, part))
            drain(2 * n_jobs)
        drain(0)
    if writer is not None:
        writer.close()
    os.replace(tmp, path)
    return path


def load_normalized(split: str, source: int) -> pd.DataFrame:
    """Normalized views of ``{split}_source{source}`` as pandas (built once, streamed)."""
    import os

    from .config import artifact_path

    path = artifact_path("norm", f"{split}_source{source}.parquet")
    if not os.path.exists(path):
        build_normalized(split, source)
    return pd.read_parquet(path)


def main() -> None:
    """Build every normalized cache and print a short profile per table."""
    import time

    import os

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from .config import artifact_path
    from .memory import mem_str

    for split in ("train", "test"):
        for source in (1, 2, 3):
            t = time.time()
            path = artifact_path("norm", f"{split}_source{source}.parquet")
            if not os.path.exists(path):
                build_normalized(split, source)
            tab = pq.read_table(path, columns=["addr_n", "script"])
            empty_addr = pc.mean(pc.equal(tab["addr_n"], "").cast("int8")).as_py()
            indic = pc.mean(pc.greater(tab["script"], 0).cast("int8")).as_py()
            print(f"{split}_source{source}: {tab.num_rows:,} rows, empty addr {empty_addr:.2%}, "
                  f"Indic names {indic:.2%}, {time.time() - t:.0f}s {mem_str()}", flush=True)


if __name__ == "__main__":
    main()
