"""Shared pieces of the bi-encoder: config, text format, device, encoder wrapper.

The text format must be identical in training, evaluation and search:
``"query: <name_n> | <addr_n>"`` on the normalized views (Indic letters kept, Latin
accents folded). e5 models use the "query: " prefix on both sides for symmetric tasks.
Token lengths of this format were measured in experiments.md E4 (p99 <= 56).
"""

import os

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


def length_order(texts: list[str]) -> np.ndarray:
    """Indices sorting texts by length (less padding per batch); inverse via argsort."""
    return np.argsort(np.fromiter((len(t) for t in texts), dtype=np.int32, count=len(texts)),
                      kind="stable")


def device():
    """The torch device: CUDA if available, else CPU (slow; smoke tests only)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype():
    """bfloat16 on GPUs with native bf16 tensor cores (compute capability >= 8.0), else float16.

    ``torch.cuda.is_bf16_supported()`` is also True on Turing (T4) where bf16 is emulated and
    slow, so the capability is checked instead.
    """
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


class Encoder:
    """Tokenizer + transformer with mean pooling and L2 normalization (e5 convention)."""

    def __init__(self, name_or_path: str | None = None, max_len: int | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = name_or_path or (model_dir() if os.path.isdir(model_dir()) else ncfg()["model"])
        self.path = path
        self.max_len = max_len or ncfg()["max_len"]
        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.dev)
        self.torch = torch

    def tokenize(self, texts: list[str]):
        """CPU tokenization (pinned memory when a GPU is used, for async copies)."""
        t = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        if self.dev.type == "cuda":
            t = {k: v.pin_memory() for k, v in t.items()}
        return t

    def embed(self, t):
        """Mean-pooled, L2-normalized embeddings from a tokenized batch (float32 math)."""
        t = {k: v.to(self.dev, non_blocking=True) for k, v in t.items()}
        out = self.model(**t).last_hidden_state
        mask = t["attention_mask"].unsqueeze(-1).to(out.dtype)
        emb = (out * mask).sum(1).float() / mask.sum(1).float().clamp(min=1e-6)
        return self.torch.nn.functional.normalize(emb, dim=-1)

    def forward(self, texts: list[str]):
        """Normalized embeddings with gradients (training)."""
        return self.embed(self.tokenize(texts))

    def encode(self, texts: list[str], batch: int | None = None):
        """Normalized float16 embeddings on the device, in input order.

        Batches are length-sorted (less padding) and the next batch is tokenized on a CPU
        thread while the GPU runs the current one (HF fast tokenizers release the GIL).
        """
        from concurrent.futures import ThreadPoolExecutor

        torch = self.torch
        batch = batch or ncfg()["encode_batch"]
        self.model.eval()
        order = length_order(texts)
        out = torch.empty((len(texts), self.model.config.hidden_size), dtype=torch.float16,
                          device=self.dev)
        starts = list(range(0, len(texts), batch))
        use_amp = self.dev.type == "cuda"
        with ThreadPoolExecutor(max_workers=1) as ex, torch.inference_mode(), \
                torch.autocast(self.dev.type, dtype=amp_dtype(), enabled=use_amp):
            nxt = ex.submit(self.tokenize, [texts[j] for j in order[0:batch]]) if starts else None
            for n, i in enumerate(starts):
                t = nxt.result()
                if n + 1 < len(starts):
                    a = starts[n + 1]
                    nxt = ex.submit(self.tokenize, [texts[j] for j in order[a:a + batch]])
                idx = torch.as_tensor(order[i:i + batch], device=self.dev)
                out[idx] = self.embed(t).to(torch.float16)
        return out
