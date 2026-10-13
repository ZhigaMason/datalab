"""ONNX artifact: embedding lookup ids (batch, seq) int64 -> embeddings (batch, seq, dim) float32."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from w2v.data import PAD_ID, Vocab


class EmbeddingLookup(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=PAD_ID)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


def export_onnx(weight: torch.Tensor, path) -> Path:
    path = Path(path)
    model = EmbeddingLookup(weight.detach().float().cpu()).eval()
    example = torch.zeros((2, 3), dtype=torch.long)
    dims = {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq")}
    torch.onnx.export(model, (example,), str(path), input_names=["ids"], output_names=["embeddings"],
                      dynamic_shapes={"ids": dims}, dynamo=True)
    return path


def verify_onnx(path, weight: torch.Tensor, atol: float = 1e-5) -> float:
    import onnxruntime as ort

    weight = weight.detach().float().cpu()
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
