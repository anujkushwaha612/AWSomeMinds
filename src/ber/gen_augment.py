"""E6: add the generator-aware features (GEN_FEATURES + CHAIN_FEATURES) to baseline feature files (CPU).

Reads artifacts/<src_run>/<split>/<country>.parquet batch by batch, computes GEN_FEATURES from the
normalized store (``name_tr``, ``addr_digits``) and writes artifacts/<dst_run>/<split>/<country>.parquet
with every original column + GEN_FEATURES, same row order (entity files are copied). The candidates
are unchanged, so the baseline and the augmented run compare pair by pair. Resumable per file.

Then train / predict the baseline on them:  BER_RUN=<dst_run> BER_GEN=1 python -m ber.baseline train
Run:  python -m ber.gen_augment --split train   (and --split test)
"""

import argparse
import os
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import artifact_path, ensure_parent
from .features import CHAIN_FEATURES, GEN_FEATURES, _gen_number_features, _name_edit_features, store_chain_features
from .memory import mem_str
from .store import CountryStore, split_countries


def gen_block(store: CountryStore, src: np.ndarray, s1: np.ndarray, doc: np.ndarray) -> dict:
    """GEN_FEATURES + CHAIN_FEATURES for aligned (src, s1_row, doc_row) arrays (both record sources)."""
    out = {}
    for s in (2, 3):
        m = src == s
        if not m.any():
            continue
        g = _gen_number_features(store.strings(1, "addr_digits", s1[m]), store.strings(s, "addr_digits", doc[m]))
        g.update(_name_edit_features(store.strings(1, "name_tr", s1[m]), store.strings(s, "name_tr", doc[m])))
        g.update(store_chain_features(store, store.strings(1, "name_n", s1[m]), store.strings(s, "name_n", doc[m])))
        for k, v in g.items():
            if k not in out:
                out[k] = np.zeros(len(src), dtype=v.dtype)
            out[k][m] = v
    return out


def augment(split: str, src_run: str, dst_run: str, batch: int = 500_000) -> None:
    """Augmented copies of every country's feature file of ``split``."""
    for country in split_countries(split):
        src_path = artifact_path(src_run, split, f"{country}.parquet")
        dst_path = artifact_path(dst_run, split, f"{country}.parquet")
        ents = f"{country}_entities.parquet"
        if os.path.exists(dst_path):
            print(f"[gen {split}/{country}] exists, skipping", flush=True)
            continue
        t0 = time.time()
        store = CountryStore(split, country, cols=["name_tr", "addr_digits", "name_n"])
        pf = pq.ParquetFile(src_path)
        n, done, writer = pf.metadata.num_rows, 0, None
        ensure_parent(dst_path)
        for rb in pf.iter_batches(batch_size=batch):
            g = gen_block(store, rb.column("src").to_numpy(), rb.column("s1_row").to_numpy(),
                          rb.column("doc_row").to_numpy())
            t = pa.Table.from_batches([rb])
            for k in GEN_FEATURES + CHAIN_FEATURES:
                t = t.append_column(k, pa.array(g.get(k, np.zeros(rb.num_rows, dtype=np.float32))))
            if writer is None:
                writer = pq.ParquetWriter(dst_path + ".tmp", t.schema, compression="zstd")
            writer.write_table(t)
            done += rb.num_rows
            if (done // batch) % 10 == 0 or done == n:
                el = time.time() - t0
                print(f"    {split}/{country}: {done:,}/{n:,} pairs  {el / 60:.1f} min, "
                      f"ETA {el / done * (n - done) / 60:.0f} min {mem_str()}", flush=True)
        writer.close()
        assert pq.ParquetFile(dst_path + ".tmp").metadata.num_rows == n
        os.replace(dst_path + ".tmp", dst_path)
        shutil.copy2(artifact_path(src_run, split, ents), artifact_path(dst_run, split, ents))
        del store
        print(f"[gen {split}/{country}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--src-run", default="baseline")
    ap.add_argument("--dst-run", default="baseline_gen")
    args = ap.parse_args()
    augment(args.split, args.src_run, args.dst_run)


if __name__ == "__main__":
    main()
