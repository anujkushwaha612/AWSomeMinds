"""Submission packaging: write both TSVs, run the official validator, snapshot.

Every leaderboard upload must go through :func:`make_submission` so that the
"version history of all submissions" rule is met (plan.md Step 0): each call
writes ``subs/<name>/`` with the commit hash, config, offline metrics and the
paths of the two output files.
"""

import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, data_path, load_config
from .io import load_source, write_id_lists

VALIDATOR = REPO_ROOT / "data" / "utils" / "validate_submission.py"


def run_validator(matching: str, candidate: str, test_dir: str) -> tuple[list, list]:
    """Run the organizers' ``validate()`` and return ``(errors, warnings)``."""
    spec = importlib.util.spec_from_file_location("validate_submission", VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate(matching, candidate, test_dir, check_ids=False)


def _git_commit() -> str:
    """Current commit hash, with a ``-dirty`` suffix if the tree has changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                      text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                        text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_submission(name: str, matches: pd.DataFrame, candidates: pd.DataFrame,
                    offline_metrics: dict | None = None, notes: str = "") -> Path:
    """Write, validate and snapshot one submission.

    ``matches`` and ``candidates`` are long-form ``s1, rid`` pairs on the TEST
    set. Raises if the validator reports errors or matches are not a subset of
    candidates. Returns the snapshot directory ``subs/<name>/``.
    """
    missing = matches.merge(candidates, on=["s1", "rid"], how="left", indicator=True)
    if (missing["_merge"] == "left_only").any():
        raise ValueError("matches must be a subset of candidates")

    out_dir = REPO_ROOT / "output"
    matching_path = str(out_dir / "matching_results.tsv")
    candidate_path = str(out_dir / "candidate_pairs.tsv")
    s1_ids = load_source("test", 1)["entity_id"]
    write_id_lists(matches, s1_ids, matching_path, "matching")
    write_id_lists(candidates, s1_ids, candidate_path, "candidate")

    errors, warnings = run_validator(matching_path, candidate_path, data_path("test"))
    for w in warnings:
        print("WARNING:", w)
    if errors:
        raise RuntimeError("validator failed:\n" + "\n".join(errors))

    snap = REPO_ROOT / "subs" / name
    snap.mkdir(parents=True, exist_ok=True)
    for p in (matching_path, candidate_path):
        shutil.copy2(p, snap / Path(p).name)
    meta = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "config": load_config(),
        "offline_metrics": offline_metrics or {},
        "n_match_pairs": int(len(matches)),
        "n_candidate_pairs": int(len(candidates)),
        "leaderboard_score": None,  # fill in after upload
        "notes": notes,
    }
    (snap / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"PASS - submission snapshot at {snap}")
    return snap
