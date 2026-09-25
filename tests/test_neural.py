"""Neural pieces that can run on CPU with a tiny random XLM-R model built on the fly (no hub access).

Covers the memory paths added for small GPUs: encode() into a memmap must equal encode() on
the device; disk-mode search (memmap tiles moved to the device) must equal all-resident search;
gradient checkpointing must not change the forward pass; training steps run and resume.
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ber.neural import common as C  # noqa: E402
from ber.neural.dense_retrieve import pair_cos, search_source, topk_tiles  # noqa: E402

DIM = 16


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A 2-layer random XLM-R with a 300-token fast tokenizer, saved like a hub checkpoint."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

    d = tmp_path_factory.mktemp("tiny_xlmr")
    tk = Tokenizer(models.WordPiece(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    corpus = ["query: krishna business private limited | s no 861 nashik maharashtra",
              "query: ferrari block company | 12 main street springfield",
              "query: lv assignments com | ", "query: कृष्णा बिजनेस | नाशिक"] * 5
    tk.train_from_iterator(corpus, trainers.WordPieceTrainer(
        vocab_size=300, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>",
                                   unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                   cls_token="<s>", sep_token="</s>", model_max_length=64)
    fast.save_pretrained(d)
    cfg = XLMRobertaConfig(vocab_size=fast.vocab_size, hidden_size=DIM, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=32, max_position_embeddings=80,
                           pad_token_id=fast.pad_token_id, hidden_dropout_prob=0.0,
                           attention_probs_dropout_prob=0.0)   # deterministic in train mode
    torch.manual_seed(0)
    XLMRobertaModel(cfg).save_pretrained(d)
    return str(d)


@pytest.fixture
def cfg(monkeypatch):
    """Small neural config; every module reads it through ncfg()."""
    conf = {"model": None, "n_pairs": 0, "max_len": 24, "batch_size": 8, "same_country_batches": True, "lr": 1e-3,
            "warmup": 0.0, "epochs": 1, "scale": 20.0, "encode_batch": 7, "k_rec": 3, "k_s1": 2,
            "tile_gb": 1e-6, "grad_checkpointing": False, "ckpt_every": 2, "emb_store": "gpu"}
    monkeypatch.setattr(C, "ncfg", lambda: conf)
    return conf


TEXTS = [f"query: shop {i} | {i * 7 % 13} road" for i in range(45)]


def test_encode_memmap_equals_device(tiny_model, cfg, tmp_path):
    enc = C.Encoder(tiny_model)
    on_dev = enc.encode(TEXTS).cpu().numpy()
    mm = C.open_memmap(str(tmp_path / "e.f16"), len(TEXTS), enc.dim, mode="w+")
    out = enc.encode(TEXTS, out=mm)
    assert out is mm
    np.testing.assert_allclose(np.asarray(mm, dtype=np.float32), on_dev.astype(np.float32), atol=2e-3)
    assert np.allclose(np.linalg.norm(on_dev.astype(np.float32), axis=1), 1.0, atol=1e-2)   # normalized
    with pytest.raises(ValueError):
        enc.encode(TEXTS, out=np.zeros((3, enc.dim), np.float16))


def test_topk_tiles_matches_torch_and_accepts_host_tiles(tiny_model, cfg):
    enc = C.Encoder(tiny_model)
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((37, DIM)).astype(np.float16)
    D = rng.standard_normal((29, DIM)).astype(np.float16)
    Dt = torch.from_numpy(D)
    ref_s, ref_i = torch.topk(torch.from_numpy(Q) @ Dt.T, 3, dim=1)
    for q in (torch.from_numpy(Q), Q):                       # device tensor, then host array
        idx, sim = topk_tiles(q, Dt, 3, tile_gb=1e-6, enc=enc)   # tiny tile -> many tiles
        assert idx.shape == (37, 3)
        np.testing.assert_array_equal(idx, ref_i.numpy())
        np.testing.assert_allclose(sim, ref_s.float().numpy(), atol=1e-3)


def test_search_source_disk_equals_gpu(tiny_model, cfg, tmp_path):
    """Same candidates whether the embeddings come from memmaps or device tensors."""
    enc = C.Encoder(tiny_model)
    rng = np.random.default_rng(2)
    E1 = torch.nn.functional.normalize(torch.from_numpy(rng.standard_normal((40, DIM))).half().float(), dim=1).half()
    E2 = torch.nn.functional.normalize(torch.from_numpy(rng.standard_normal((90, DIM))).half().float(), dim=1).half()
    gpu = search_source(enc, E1, E2, 2, 3, 2, 1e-6)
    m1 = C.open_memmap(str(tmp_path / "s1.f16"), 40, DIM, "w+"); m1[:] = E1.numpy()
    m2 = C.open_memmap(str(tmp_path / "s2.f16"), 90, DIM, "w+"); m2[:] = E2.numpy()
    disk = search_source(enc, enc.to_device(m1), enc.to_device(m2), 2, 3, 2, 1e-6)
    key = ["src", "doc_row", "s1_row"]
    a, b = gpu.sort_values(key).reset_index(drop=True), disk.sort_values(key).reset_index(drop=True)
    assert a[key + ["drank_rec", "drank_s1"]].equals(b[key + ["drank_rec", "drank_s1"]])
    np.testing.assert_allclose(a["cos"], b["cos"], atol=1e-3)
    # every record has k_rec record-side rows; every S1 has k_s1 S1-side rows
    assert (gpu[gpu.drank_rec >= 0].groupby("doc_row").size() == 3).all() and gpu.doc_row.nunique() == 90
    assert (gpu[gpu.drank_s1 >= 0].groupby("s1_row").size() == 2).all()
    rows_a, rows_b = np.array([0, 5, 89]), np.array([3, 3, 39])
    ref = (E2[rows_a].float() * E1[rows_b].float()).sum(1).numpy()
    np.testing.assert_allclose(pair_cos(E2, E1, rows_a, rows_b, chunk=2), ref, atol=1e-3)


def test_grad_checkpointing_same_forward(tiny_model, cfg):
    a = C.Encoder(tiny_model, grad_checkpointing=False)
    b = C.Encoder(tiny_model, grad_checkpointing=True)
    assert b.model.is_gradient_checkpointing and not a.model.is_gradient_checkpointing
    a.model.train(), b.model.train()                          # checkpointing is only active in train mode
    ea, eb = a.forward(TEXTS[:5]), b.forward(TEXTS[:5])
    torch.testing.assert_close(ea, eb, atol=1e-5, rtol=1e-4)
    eb.sum().backward()                                       # backward through checkpointed layers
    assert any(p.grad is not None for p in b.model.parameters())


def test_train_runs_and_resumes(tiny_model, cfg, tmp_path, monkeypatch):
    """Two steps, checkpoint, then a resumed run finishes the remaining step from the state file."""
    import pandas as pd

    from ber.neural import train_biencoder as T

    class Store:
        def __init__(self, n):
            self.names = [f"biz {i}" for i in range(n)]
        def strings(self, s, col, rows=None):
            base = self.names if col == "name_n" else ["addr"] * len(self.names)
            return [base[i] for i in rows] if rows is not None else base

    n = 24
    pairs = pd.DataFrame({"country": np.int8(0), "src": np.int8(2), "doc_row": np.arange(n),
                          "pos_s1": np.arange(n), "neg_s1": (np.arange(n) + 1) % n})
    pairs_path = tmp_path / "train_pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)
    ckpt = tmp_path / "biencoder"
    cfg["model"] = tiny_model
    monkeypatch.setattr(T, "artifact_path", lambda *p: str(pairs_path))
    monkeypatch.setattr(T, "model_dir", lambda: str(ckpt))
    monkeypatch.setattr(T, "split_countries", lambda split: ["India"])
    monkeypatch.setattr(T, "country_store", lambda split, c, cols=None: Store(n))
    monkeypatch.setattr(T, "load_config", lambda: {"seed": 0})
    monkeypatch.setattr(T, "ncfg", lambda: cfg)              # T binds ncfg at import time
    monkeypatch.setattr(C, "model_dir", lambda: str(ckpt))
    T.train(max_steps=2)                                     # 24 pairs / batch 8 = 3 batches; stop at 2
    state = json.load(open(ckpt / "train_state.json"))
    assert state["step"] == 2 and os.path.exists(ckpt / "optimizer.pt")
    T.train()                                                # resumes at step 2, finishes step 3
    assert json.load(open(ckpt / "train_state.json"))["step"] == 3
    cfg["model"] = "some/other-model"                        # a different base model must not resume silently
    with pytest.raises(RuntimeError, match="checkpoint"):
        T.train()
