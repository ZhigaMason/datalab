"""Vocabulary, corpus encoding and the training token stream."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

PAD_ID, UNK_ID = 0, 1
SPECIALS = ["<pad>", "<unk>"]


@dataclass
class Vocab:
    words: list[str]
    counts: list[int]
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {w: i for i, w in enumerate(self.words)}

    def __len__(self) -> int:
        return len(self.words)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"words": self.words, "counts": self.counts}, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["words"], d["counts"])

    @classmethod
    def build(cls, counter: Counter, max_size: int, min_count: int) -> "Vocab":
        items = sorted(((w, c) for w, c in counter.items() if c >= min_count), key=lambda wc: (-wc[1], wc[0]))
        items = items[: max_size - len(SPECIALS)]
        return cls(SPECIALS + [w for w, _ in items], [0] * len(SPECIALS) + [c for _, c in items])


def iter_sentences(text_dir, max_tokens: int | None = None, num_shards: int | None = None) -> Iterator[list[str]]:
    """Sentences from shard files in sorted order; stops once max_tokens have been yielded.

    With num_shards, reads exactly shard_0000..shard_{num_shards-1} (ignoring stale extra shards)."""
    if num_shards is not None:
        files = [Path(text_dir) / f"shard_{i:04d}.txt" for i in range(num_shards)]
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"{f} missing; run scripts.preprocess first")
    else:
        files = sorted(Path(text_dir).glob("shard_*.txt"))
    if not files:
        raise FileNotFoundError(f"no shard_*.txt in {text_dir}; run scripts.preprocess first")
    total = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                toks = line.split()
                if not toks:
                    continue
                yield toks
                total += len(toks)
                if max_tokens is not None and total >= max_tokens:
                    return


def encode_corpus(text_dir, vocab: Vocab, out_path, pad: int, max_tokens: int | None = None,
                  num_shards: int | None = None) -> dict:
    """Flat uint32 stream: `pad` PADs, then each sentence followed by `pad` PADs. OOV words are dropped."""
    n_tokens = n_sent = length = 0
    buf: list[int] = [PAD_ID] * pad
    with open(out_path, "wb") as fh:
        for sent in iter_sentences(text_dir, max_tokens, num_shards):
            ids = [vocab.index[w] for w in sent if w in vocab.index]
            if not ids:
                continue
            buf.extend(ids)
            buf.extend([PAD_ID] * pad)
            n_tokens += len(ids)
            n_sent += 1
            if len(buf) >= 1 << 20:
                np.asarray(buf, dtype=np.uint32).tofile(fh)
                length += len(buf)
                buf = []
        np.asarray(buf, dtype=np.uint32).tofile(fh)
        length += len(buf)
    return {"tokens": n_tokens, "length": length, "pad": pad, "sentences": n_sent}


class TokenStream:
    """Random contiguous chunks of a uint32 token memmap; batch(step) is deterministic in (seed, step)."""

    def __init__(self, path, seq_len: int, batch_size: int, seed: int):
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")
        self.num_tokens = len(self.tokens)
        if self.num_tokens < seq_len:
            raise ValueError(f"corpus {path} has {self.num_tokens} tokens, fewer than seq_len={seq_len}")
        self.seq_len, self.batch_size, self.seed = seq_len, batch_size, seed
        self._offsets = np.arange(seq_len)

    def steps_per_epoch(self) -> int:
        return max(1, self.num_tokens // (self.batch_size * self.seq_len))

    def batch(self, step: int) -> torch.Tensor:
        rng = np.random.default_rng([self.seed, step])
        starts = rng.integers(0, self.num_tokens - self.seq_len + 1, size=self.batch_size)
        return torch.from_numpy(self.tokens[starts[:, None] + self._offsets].astype(np.int64))


def keep_probs(counts, t: float) -> torch.Tensor:
    """word2vec frequent-word subsampling: keep with prob min(1, sqrt(t / f)); 0 for unseen ids."""
    c = torch.tensor(counts, dtype=torch.float64)
    seen = c > 0
    if t <= 0:
        return seen.float()
    f = c / c.sum()
    p = torch.sqrt(t / f.clamp(min=1e-30)).clamp(max=1.0)
    return torch.where(seen, p, torch.zeros_like(p)).float()


def sample_mask(x: torch.Tensor, keep: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.rand(x.shape, device=x.device, generator=generator) < keep[x]


def noise_dist(counts, power: float = 0.75) -> torch.Tensor:
    q = torch.tensor(counts, dtype=torch.float64).pow(power)
    return (q / q.sum()).float()
