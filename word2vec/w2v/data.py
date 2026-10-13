"""Vocabulary, corpus encoding and the training token stream."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

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


def iter_sentences(text_dir, max_tokens: int | None = None) -> Iterator[list[str]]:
    """Sentences from shard files in sorted order; stops once max_tokens have been yielded."""
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


def encode_corpus(text_dir, vocab: Vocab, out_path, pad: int, max_tokens: int | None = None) -> dict:
    """Flat uint32 stream: `pad` PADs, then each sentence followed by `pad` PADs. OOV words are dropped."""
    n_tokens = n_sent = length = 0
    buf: list[int] = [PAD_ID] * pad
    with open(out_path, "wb") as fh:
        for sent in iter_sentences(text_dir, max_tokens):
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
