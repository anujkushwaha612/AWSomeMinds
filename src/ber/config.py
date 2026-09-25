"""Configuration loading and local/S3-agnostic path handling."""

import os
import posixpath
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


def is_s3(path: str) -> bool:
    """Return True if ``path`` is an ``s3://`` URI."""
    return str(path).startswith("s3://")


def join(root: str, *parts: str) -> str:
    """Join path parts under ``root``, using '/' for S3 URIs and OS rules otherwise.

    Relative local roots are resolved against the repo root, so scripts behave the
    same whatever directory they are launched from.
    """
    if is_s3(root):
        return posixpath.join(root, *parts)
    base = Path(root)
    if not base.is_absolute():
        base = REPO_ROOT / base
    return str(base.joinpath(*parts))


def ensure_parent(path: str) -> None:
    """Create the parent directory of a local ``path`` (no-op for S3)."""
    if not is_s3(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict:
    """Load the pipeline config from ``path``, ``$BER_CONFIG`` or the default file."""
    path = path or os.environ.get("BER_CONFIG") or str(DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(*parts: str) -> str:
    """Path to a raw data file, e.g. ``data_path("train", "train_source1.tsv")``."""
    return join(load_config()["data_root"], *parts)


def artifact_path(*parts: str) -> str:
    """Path to a pipeline artifact under ``artifacts_root``."""
    return join(load_config()["artifacts_root"], *parts)
