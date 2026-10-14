"""ONNX artifact: embedding lookup ids (batch, seq) int64 -> embeddings (batch, seq, dim) float32.

The word2vec input table is stored as int8 with one float32 scale per row (symmetric absmax
quantization, a quarter of the float32 size); rows are dequantized after the lookup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from w2v.data import Vocab


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8: weight ~= q * scale, scale = row absmax / 127 (all-zero rows get scale 0)."""
    w = weight.detach().cpu().float()
    scale = w.abs().amax(dim=1, keepdim=True) / 127
    q = torch.round(w / torch.where(scale > 0, scale, torch.ones_like(scale))).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_int8(weight: torch.Tensor) -> torch.Tensor:
    """What the artifact returns for each row of weight."""
    q, scale = quantize_int8(weight)
    return q.float() * scale


class EmbeddingLookup(nn.Module):
    """The only thing inference needs: the input embedding table (int8 + row scales) and a lookup."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        q, scale = quantize_int8(weight)
        self.register_buffer("embeddings_int8", q)
        self.register_buffer("row_scales", scale)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.embeddings_int8).float() * F.embedding(ids, self.row_scales)


def export_onnx(weight: torch.Tensor, path) -> Path:
    path = Path(path)
    if weight.numel() + weight.shape[0] * 4 >= 2**31:
        raise ValueError(f"embedding matrix of {weight.numel()} int8 values reaches the 2 GB protobuf limit; "
                         "a single-file ONNX artifact is not possible")
    model = EmbeddingLookup(weight).eval()
    example = torch.zeros((2, 3), dtype=torch.long)
    dims = {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq")}
    torch.onnx.export(model, (example,), str(path), input_names=["ids"], output_names=["embeddings"],
                      dynamic_shapes={"ids": dims}, dynamo=True, external_data=False)
    return path


def verify_onnx(path, weight: torch.Tensor, atol: float = 1e-5) -> float:
    import onnxruntime as ort

    weight = dequantize_int8(weight)  # the artifact stores int8; compare with what it holds
    v = weight.shape[0]
    ids = np.random.default_rng(0).integers(0, v, size=(3, 17)).astype(np.int64)
    ids[0, 0], ids[0, 1] = 0, v - 1
    out = ort.InferenceSession(str(path)).run(None, {"ids": ids})[0]
    diff = float(np.abs(out - weight[torch.from_numpy(ids)].numpy()).max())
    if not np.isfinite(diff) or diff > atol:
        raise RuntimeError(f"ONNX/PyTorch mismatch: max abs diff {diff} > {atol}")
    return diff


def load_onnx_embeddings(run_dir, chunk: int = 65536) -> tuple[list[str], np.ndarray]:
    import onnxruntime as ort

    run_dir = Path(run_dir)
    vocab = Vocab.load(run_dir / "vocab.json")
    sess = ort.InferenceSession(str(run_dir / "model.onnx"))
    parts = [sess.run(None, {"ids": np.arange(s, min(s + chunk, len(vocab)), dtype=np.int64)[None]})[0][0]
             for s in range(0, len(vocab), chunk)]
    return vocab.words, np.concatenate(parts)


def write_vec(path, words: list[str], matrix: np.ndarray) -> None:
    if len(words) != matrix.shape[0]:
        raise ValueError(f"{len(words)} words but {matrix.shape[0]} matrix rows")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(words)} {matrix.shape[1]}\n")
        for w, row in zip(words, matrix):
            fh.write(w + " " + " ".join(f"{x:.6g}" for x in row) + "\n")
