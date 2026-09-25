"""E1: how many hashing worker processes? Time and total memory for one full pass.

Hashes all train-India S2 records (word 1+2-gram, the retrieval pass-1 workload)
with 4 / 8 / 12 / 15 workers and samples the resident memory of the whole
process tree (main + workers) every 0.2 s.

Run:  python experiments/e1_worker_sweep.py
"""

import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ber.blocking.word_retrieval import N_FEATURES, iter_counts
from ber.memory import current_gb
from ber.store import CountryStore


def main():
    store = CountryStore("train", "India")
    base = current_gb()
    print(f"store loaded: S2 {store.n(2):,} rows, arrow {store.nbytes() / 2**30:.2f} GB, "
          f"process {base:.2f} GB", flush=True)
    for workers in (4, 8, 12, 15):
        peak = [0.0]
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], current_gb())
                time.sleep(0.2)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        t = time.time()
        df = np.zeros(N_FEATURES, dtype=np.int64)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for _, counts in iter_counts(store, 2, pool, chunk=50_000, window=2 * workers):
                df += np.bincount(counts.indices, minlength=N_FEATURES)
        secs = time.time() - t
        stop.set()
        th.join()
        print(f"workers {workers:>2}: {secs:5.1f} s  ({store.n(2) / secs / 1e3:.0f}k rec/s)  "
              f"peak tree memory {peak[0]:.2f} GB (+{peak[0] - base:.2f} GB over the store)",
              flush=True)


if __name__ == "__main__":
    main()
