"""Laptop side of the free-GPU plan (KAGGLE.md): pack what the Kaggle GPU stages read into one zip.

The GPU stages (env_check, train_biencoder, dense_retrieve) only read
  * artifacts/norm/<split>_source<s>.parquet     -> columns country, name_n, addr_n
  * artifacts/baseline/<split>/<country>.parquet -> columns src, doc_row, s1_row
  * artifacts/neural/train_pairs.parquet, eval_records.parquet
so only those columns are copied (row order kept: row ids index into these files),
together with src/, configs/ and kaggle/run_gpu.py. Upload the zip as a private Kaggle dataset.

Run:  .venv/Scripts/python kaggle/make_bundle.py        -> artifacts/kaggle_bundle.zip
"""

import os
import sys
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ber.config import artifact_path  # noqa: E402
from ber.neural.common import vdir  # noqa: E402

NORM_COLS = ["country", "name_n", "addr_n"]
TFIDF_COLS = ["src", "doc_row", "s1_row"]


def slim_copy(src: str, dst: str, cols: list[str]) -> None:
    """Copy ``cols`` of a parquet file batch by batch (same row order, low memory)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pf = pq.ParquetFile(src)
    writer = None
    for rb in pf.iter_batches(batch_size=500_000, columns=cols):
        if writer is None:
            writer = pq.ParquetWriter(dst, rb.schema, compression="zstd")
        writer.write_batch(rb)
    writer.close()
    assert pq.ParquetFile(dst).metadata.num_rows == pf.metadata.num_rows, dst


def add_code(z: zipfile.ZipFile) -> None:
    """src/, configs/, kaggle/ and pyproject.toml under code/ in the zip."""
    for sub in ("src", "configs", "kaggle"):
        for f in sorted((REPO / sub).rglob("*")):
            skip = any(p == "__pycache__" or p.endswith(".egg-info") for p in f.parts)
            if f.is_file() and not skip:
                z.write(f, ("code/" + str(f.relative_to(REPO))).replace(os.sep, "/"))
    z.write(REPO / "pyproject.toml", "code/pyproject.toml")


def main() -> None:
    """Build artifacts/kaggle_bundle/ and zip it (``--code-only``: just the code, as a small patch)."""
    if "--code-only" in sys.argv:
        zpath = Path(artifact_path("kaggle_code.zip"))
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            add_code(z)
        print(f"wrote {zpath} ({zpath.stat().st_size / 1e3:.0f} kB)")
        return
    out = Path(artifact_path("kaggle_bundle"))
    files = []                                          # (path on disk, path in zip)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            src = artifact_path("norm", f"{split}_source{s}.parquet")
            dst = out / "artifacts" / "norm" / f"{split}_source{s}.parquet"
            if not dst.exists():
                slim_copy(src, str(dst), NORM_COLS)
            files.append(dst)
        tdir = Path(artifact_path(vdir("tfidf"), split))
        for f in sorted(tdir.glob("*.parquet")):
            if f.name.endswith("_entities.parquet"):
                continue
            dst = out / "artifacts" / vdir("tfidf") / split / f.name
            if not dst.exists():
                slim_copy(str(f), str(dst), TFIDF_COLS)
            files.append(dst)
    for name in ("train_pairs.parquet", "eval_records.parquet"):
        f = Path(artifact_path(vdir("neural"), name))
        if not f.exists():
            raise SystemExit(f"{f} is missing: run `python -m ber.neural.pairs` first")
        files.append(f)

    zpath = Path(artifact_path("kaggle_bundle.zip"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:     # parquet is already compressed
        for f in files:
            f = Path(f)
            arc = f.relative_to(out) if out in f.parents else Path("artifacts") / vdir("neural") / f.name
            z.write(f, str(arc).replace(os.sep, "/"))
        add_code(z)
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e9:.2f} GB)")
    for f in files:
        print(f"  {Path(f).stat().st_size / 1e6:8.1f} MB  {f}")


if __name__ == "__main__":
    main()
