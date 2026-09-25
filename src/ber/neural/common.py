"""Shared pieces of the bi-encoder: config, text format, device, encoder wrapper.

The text format must be identical in training, evaluation and search:
``"query: <name_n> | <addr_n>"`` on the normalized views (Indic letters kept, Latin
accents folded). e5 models use the "query: " prefix on both sides for symmetric tasks.
Token lengths of this format were measured in experiments.md E4 (p99 <= 56).
"""

import os
import time

import numpy as np

from ..config import artifact_path, load_config

PREFIX = "query: "


def v5cfg() -> dict:
    """The ``v5`` section of configs/pipeline.yaml."""
    return load_config()["v5"]


def ncfg() -> dict:
    """The ``v5.neural`` section."""
    return v5cfg()["neural"]


def country_store(split: str, country: str, cols=None):
    """CountryStore with the v5 row limit (``v5.limit``, smoke tests only; normally None).

    Every v5 module must index rows through this so that row numbers agree with the
    TF-IDF candidate files they were built from.
    """
    from ..store import TEXT_COLS, CountryStore
    return CountryStore(split, country, cols=cols or TEXT_COLS, limit=v5cfg().get("limit"))


def vdir(name: str) -> str:
    """Artifact folder name for ``name`` (neural / dense / union / v5 / tfidf), per v5.dirs."""
    if name == "tfidf":
        return v5cfg()["tfidf_run"]
    return v5cfg().get("dirs", {}).get(name, name)


def model_dir() -> str:
    """Where the fine-tuned encoder is saved."""
    return artifact_path(vdir("neural"), "biencoder")


def make_texts(names: list[str], addrs: list[str]) -> list[str]:
    """Encoder input strings for aligned name / address lists."""
    return [f"{PREFIX}{n} | {a}" for n, a in zip(names, addrs)]


def store_texts(store, s: int, rows=None) -> list[str]:
    """Encoder input strings for rows of source ``s`` (all rows if ``rows`` is None)."""
    if rows is None:
        return make_texts(store.strings(s, "name_n"), store.strings(s, "addr_n"))
    return make_texts(store.strings(s, "name_n", rows), store.strings(s, "addr_n", rows))


def tile_rows(n_cols: int, tile_gb: float, bytes_per: int = 2, lo: int = 64, hi: int = 16384) -> int:
    """Rows per similarity tile so that a (rows x n_cols) matrix fits in ``tile_gb``."""
    rows = int(tile_gb * 1e9 / max(1, n_cols * bytes_per))
    return int(max(lo, min(hi, rows)))


def emb_store_mode() -> str:
    """Where a country's embeddings live during search: ``gpu`` (all resident, needs ~12 GB VRAM
    for US train) or ``disk`` (float16 memmaps under ``emb_dir``; two sources on the GPU at a time,
    for 16-24 GB cards / low-RAM machines). Config ``v5.neural.emb_store``, default ``gpu``."""
    return str(ncfg().get("emb_store", "gpu")).lower()


def emb_dir(split: str) -> str:
    """Folder of the on-disk embeddings (``v5.neural.emb_dir``, default artifacts/<dense>/emb/<split>).
    An absolute path (e.g. a fast local disk on Colab) is used as is."""
    root = ncfg().get("emb_dir")
    return os.path.join(root, split) if root else artifact_path(vdir("dense"), "emb", split)


def open_memmap(path: str, n_rows: int, dim: int, mode: str = "r"):
    """float16 (n_rows x dim) memmap. ``mode='w+'`` creates or truncates the file."""
    return np.memmap(path, dtype=np.float16, mode=mode, shape=(n_rows, dim))


def length_order(texts: list[str]) -> np.ndarray:
    """Indices sorting texts by length (less padding per batch); inverse via argsort."""
    return np.argsort(np.fromiter((len(t) for t in texts), dtype=np.int32, count=len(texts)),
                      kind="stable")


def device():
    """The torch device: CUDA if available, else CPU (slow; smoke tests only)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype():
    """bfloat16 where supported (Ampere+), else float16."""
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class Encoder:
    """Tokenizer + transformer with mean pooling and L2 normalization (e5 convention)."""

    def __init__(self, name_or_path: str | None = None, max_len: int | None = None,
                 grad_checkpointing: bool | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = name_or_path or (model_dir() if os.path.isdir(model_dir()) else ncfg()["model"])
        self.path = path
        self.max_len = max_len or ncfg()["max_len"]
        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.dev)
        self.torch = torch
        # Activation checkpointing trades ~30% speed for ~5x less activation memory: it is what
        # lets batch 256 (768 texts / step) train on a 16-24 GB card (config v5.neural.grad_checkpointing).
        gc_on = ncfg().get("grad_checkpointing", False) if grad_checkpointing is None else grad_checkpointing
        if gc_on and getattr(self.model, "supports_gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    @property
    def dim(self) -> int:
        """Embedding size."""
        return int(self.model.config.hidden_size)

    def forward(self, texts: list[str]):
        """Normalized embeddings with gradients (training)."""
        t = self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                     return_tensors="pt").to(self.dev)
        out = self.model(**t).last_hidden_state
        mask = t["attention_mask"].unsqueeze(-1).to(out.dtype)
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.torch.nn.functional.normalize(emb, dim=-1)

    def encode(self, texts: list[str], batch: int | None = None, out=None, log=None):
        """Normalized float16 embeddings in input order (length-sorted batches).

        ``out=None``: returns a tensor on the device (all embeddings resident on the GPU).
        ``out=<np.memmap / ndarray (n x dim) float16>``: fills it in place and returns it, so
        a country's embeddings never have to fit in VRAM or RAM at once.
        """
        torch = self.torch
        batch = batch or ncfg()["encode_batch"]
        self.model.eval()
        order = length_order(texts)
        to_device = out is None
        if to_device:
            out = torch.empty((len(texts), self.dim), dtype=torch.float16, device=self.dev)
        elif tuple(out.shape) != (len(texts), self.dim):
            raise ValueError(f"out has shape {tuple(out.shape)}, expected {(len(texts), self.dim)}")
        use_amp = self.dev.type == "cuda"
        t0, n_done = time.time(), 0
        with torch.inference_mode(), torch.autocast(self.dev.type, dtype=amp_dtype(), enabled=use_amp):
            for i in range(0, len(texts), batch):
                idx = order[i:i + batch]
                emb = self.forward([texts[j] for j in idx]).to(torch.float16)
                if to_device:
                    out[torch.as_tensor(idx, device=self.dev)] = emb
                else:
                    out[np.sort(idx)] = emb.cpu().numpy()[np.argsort(idx)]   # sorted rows: sequential writes
                n_done += len(idx)
                if log and (i // batch) % 500 == 499:
                    rate = n_done / max(time.time() - t0, 1e-9)
                    log(f"      encoded {n_done:,}/{len(texts):,}  {rate:,.0f} texts/s  "
                        f"ETA {(len(texts) - n_done) / rate / 60:.0f} min")
        if not to_device and hasattr(out, "flush"):
            out.flush()
        return out

    def to_device(self, emb, slice_rows: int = 1 << 18):
        """A float16 tensor on the device from a device tensor (no-op) or a host array / memmap.

        Host data is copied slice by slice (``slice_rows`` x dim, ~400 MB at dim 768), so a
        multi-GB memmap never has to be materialized in RAM.
        """
        torch = self.torch
        if isinstance(emb, torch.Tensor):
            return emb.to(self.dev)
        out = torch.empty(tuple(emb.shape), dtype=torch.float16, device=self.dev)
        for a in range(0, emb.shape[0], slice_rows):
            out[a:a + slice_rows] = torch.from_numpy(np.array(emb[a:a + slice_rows], dtype=np.float16))
        return out
