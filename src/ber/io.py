"""Reading challenge TSVs and writing submission files.

All sources are read as strings with pandas' NA conversion disabled: the data
contains literal "null"/"NA"-like strings that must stay text (cleanup happens
in normalization, not in the reader). Quoted fields use standard CSV quoting,
which the default parser handles.
"""

import csv

import pandas as pd

from .config import artifact_path, data_path, ensure_parent, is_s3

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def read_tsv(path: str) -> pd.DataFrame:
    """Read a challenge TSV with every column as ``str`` and no NA conversion."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)


def _cached(name: str, loader) -> pd.DataFrame:
    """Return ``loader()`` via a parquet cache under ``artifacts/cache/``."""
    cache = artifact_path("cache", f"{name}.parquet")
    try:
        return pd.read_parquet(cache)
    except (FileNotFoundError, OSError):
        df = loader()
        ensure_parent(cache)
        df.to_parquet(cache, index=False)
        return df


def load_source(split: str, source: int) -> pd.DataFrame:
    """Load ``{split}_source{source}.tsv`` (split in {"train", "test"}, source 1-3)."""
    name = f"{split}_source{source}"
    df = _cached(name, lambda: read_tsv(data_path(split, f"{name}.tsv")))
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    return df


def load_truth_pairs() -> pd.DataFrame:
    """Load the train ground truth as long-form pairs with columns ``s1, rid``.

    Entities with no matches do not appear here; use the S1 table for the full
    entity list.
    """

    def _load():
        gt = read_tsv(data_path("train", "train_ground_truth.tsv"))
        gt["rid"] = gt["matched_entity_ids"].str.split(",")
        pairs = gt[["source1_entity_id", "rid"]].explode("rid")
        pairs = pairs[pairs["rid"].notna() & (pairs["rid"] != "")]
        return pairs.rename(columns={"source1_entity_id": "s1"}).reset_index(drop=True)

    return _cached("train_truth_pairs", _load)


def write_id_lists(pairs: pd.DataFrame, s1_ids, path: str, kind: str = "matching") -> None:
    """Write ``pairs`` (columns ``s1, rid``) as a submission-style TSV.

    Every id in ``s1_ids`` gets exactly one row (empty list when it has no pairs);
    duplicate pairs are dropped. ``kind`` is "matching" or "candidate" and picks
    the header the validator expects.
    """
    header = MATCHING_HEADER if kind == "matching" else CANDIDATE_HEADER
    s1_index = pd.Index(pd.unique(pd.Series(list(s1_ids), dtype=str)), name="s1")
    pairs = pairs[["s1", "rid"]].drop_duplicates()
    unknown = ~pairs["s1"].isin(s1_index)
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} pairs reference S1 ids outside s1_ids")
    bad_prefix = ~pairs["rid"].str.match(r"^S[23]-")
    if bad_prefix.any():
        raise ValueError(f"{int(bad_prefix.sum())} pairs have non S2-/S3- ids")
    lists = pairs.groupby("s1")["rid"].agg(",".join).reindex(s1_index, fill_value="")
    out = pd.DataFrame({header[0]: lists.index, header[1]: lists.values})
    ensure_parent(path)
    # the scorer expects bare comma-joined lists, never quoted fields
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, encoding="utf-8")
    if not is_s3(path):
        _check_written(path, len(s1_index))


def _check_written(path: str, n_expected: int) -> None:
    """Sanity-check a written submission file's row count."""
    with open(path, encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1
    if n != n_expected:
        raise RuntimeError(f"{path}: wrote {n} rows, expected {n_expected}")
