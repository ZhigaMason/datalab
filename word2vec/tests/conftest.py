import json

import numpy as np
import pytest

from w2v.data import Vocab, encode_corpus


@pytest.fixture
def tiny_data(tmp_path):
    """Synthetic 'en' corpus with vocab/stats/tokens.bin and a toy similarity benchmark."""
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(50)]
    probs = 1 / np.arange(1, 51)
    probs /= probs.sum()
    text_dir = tmp_path / "data/en/text"
    text_dir.mkdir(parents=True)
    lines = [" ".join(rng.choice(words, size=8, p=probs)) for _ in range(400)]
    (text_dir / "shard_0000.txt").write_text("\n".join(lines) + "\n")
    counts = {w: 0 for w in words}
    for line in lines:
        for w in line.split():
            counts[w] += 1
    vocab = Vocab(["<pad>", "<unk>"] + words, [0, 0] + [counts[w] for w in words])
    vocab.save(tmp_path / "data/en/vocab.json")
    stats = encode_corpus(text_dir, vocab, tmp_path / "data/en/tokens.bin", pad=5)
    (tmp_path / "data/en/stats.json").write_text(json.dumps(stats))
    bench = tmp_path / "data/benchmarks/en"
    bench.mkdir(parents=True)
    (bench / "sim_toy.tsv").write_text("w0\tw1\t5\nw2\tw3\t3\nw4\tw5\t1\n")
    return tmp_path
