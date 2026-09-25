"""Memory measurement helpers: current / peak resident memory of this process tree.

Used to *measure* (not assume) the RAM footprint of every pipeline stage. Peak is
the OS-reported peak working set of the main process (Windows) or max RSS
(Linux); child worker processes are added to the current figure.
"""

import os

try:
    import psutil
except ImportError:          # memory logging is optional
    psutil = None


def _gb(nbytes: float) -> float:
    return nbytes / 2 ** 30


def current_gb(include_children: bool = True) -> float:
    """Resident memory (GB) of this process, plus its worker processes."""
    if psutil is None:
        return float("nan")
    p = psutil.Process(os.getpid())
    total = p.memory_info().rss
    if include_children:
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
    return _gb(total)


def peak_gb() -> float:
    """Peak resident memory (GB) of the main process so far."""
    if psutil is None:
        return float("nan")
    info = psutil.Process(os.getpid()).memory_info()
    if hasattr(info, "peak_wset"):                   # Windows
        return _gb(info.peak_wset)
    import resource                                   # Linux / macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def available_gb() -> float:
    """Free physical memory on the machine (GB)."""
    return _gb(psutil.virtual_memory().available) if psutil else float("nan")


def mem_str() -> str:
    """Short memory status for log lines."""
    return f"[mem {current_gb():.2f} GB now, main peak {peak_gb():.2f} GB, free {available_gb():.1f} GB]"
