"""Compact, Arrow-backed text store for one split x country (memory-lean pipeline).

Normalized columns are held as pyarrow arrays (contiguous UTF-8 buffers: about
1 byte per character) instead of pandas object columns (a Python object per
string, ~50-80 bytes of overhead each). Strings become Python objects only
for the rows of one chunk, when a library (hashing, rapidfuzz) needs them.
Candidate tables never carry text: only integer row ids into this store.
"""

import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import artifact_path

TEXT_COLS = ["entity_id", "name_n", "addr_n", "name_legal", "name_nospace",
             "addr_digits", "script", "name_tr"]


def norm_path(split: str, source: int) -> str:
    """Parquet path of the normalized table (built on first use)."""
    path = artifact_path("norm", f"{split}_source{source}.parquet")
    if not os.path.exists(path):
        from .normalize import build_normalized
        build_normalized(split, source)
    return path


def split_countries(split: str) -> list[str]:
    """Every country present in S1 of ``split`` (open set: France appears in test)."""
    col = pq.read_table(norm_path(split, 1), columns=["country"])["country"]
    return sorted(pc.unique(col).to_pylist())


class CountryStore:
    """Normalized S1/S2/S3 columns of one split and country, as Arrow arrays."""

    def __init__(self, split: str, country: str, cols=TEXT_COLS, limit: int | None = None,
                 batch: int = 200_000):
        # Filter batch by batch: reading the whole file with ``filters=`` first
        # materializes every country's rows and leaves the freed memory in Arrow's
        # pool (measured on train India: 4.07 GB process vs 1.66 GB this way).
        self.tables = {}
        for s in (1, 2, 3):
            pf = pq.ParquetFile(norm_path(split, s))
            parts, kept = [], 0
            for rb in pf.iter_batches(batch_size=batch, columns=list(cols) + ["country"]):
                rb = rb.filter(pc.equal(rb.column("country"), country))
                parts.append(pa.Table.from_batches([rb]).drop(["country"]))
                kept += rb.num_rows
                if limit and kept >= limit:
                    break
            t = pa.concat_tables(parts).combine_chunks()
            del parts
            self.tables[s] = t.slice(0, limit) if limit else t
        pa.default_memory_pool().release_unused()

    def n(self, s: int) -> int:
        """Number of rows of source ``s``."""
        return self.tables[s].num_rows

    def column(self, s: int, name: str) -> pa.Array:
        """Arrow array of one column."""
        return self.tables[s].column(name).chunk(0) if self.tables[s].num_rows else pa.array([])

    def strings(self, s: int, name: str, rows: np.ndarray | None = None,
                start: int = 0, stop: int | None = None) -> list[str]:
        """Python strings for ``rows`` (fancy index) or the slice [start, stop)."""
        col = self.column(s, name)
        if rows is not None:
            col = col.take(pa.array(rows, type=pa.int64()))
        else:
            col = col.slice(start, (stop if stop is not None else len(col)) - start)
        return col.to_pylist()

    def na_text(self, s: int, start: int, stop: int) -> list[str]:
        """Retrieval view text (name + " " + address) for rows [start, stop)."""
        name = self.column(s, "name_n").slice(start, stop - start)
        addr = self.column(s, "addr_n").slice(start, stop - start)
        # separator typed like the columns: pandas >= 3 stores text as large_string
        return pc.binary_join_element_wise(name, addr, pa.scalar(" ", type=name.type)).to_pylist()

    def numpy(self, s: int, name: str) -> np.ndarray:
        """A numeric (or id) column as a NumPy array."""
        return self.column(s, name).to_numpy(zero_copy_only=False)

    def nbytes(self) -> int:
        """Arrow buffer bytes held by this store."""
        return sum(t.nbytes for t in self.tables.values())
