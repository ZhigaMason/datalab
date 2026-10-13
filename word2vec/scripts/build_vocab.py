"""Count tokens of a preprocessed language and write vocab.json + stats.json.

English is capped to the largest token count of other languages via --cap-from.

    python -m scripts.build_vocab --lang en --num-shards 64 --data-dir data \\
        --cap-from data/cs/stats.json data/uk/stats.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from w2v.data import Vocab, iter_sentences


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["en", "cs", "uk"])
    p.add_argument("--num-shards", type=int, required=True, help="expected number of preprocessed shards")
    p.add_argument("--max-vocab", type=int, default=200_000, help="including <pad> and <unk>")
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--cap-from", nargs="*", default=[], help="stats.json files; cap = max of their tokens_counted")
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)

    lang_dir = Path(a.data_dir) / a.lang
    text_dir = lang_dir / "text"
    missing = [f"shard_{i:04d}.done" for i in range(a.num_shards) if not (text_dir / f"shard_{i:04d}.done").exists()]
    if missing:
        raise FileNotFoundError(f"preprocessing incomplete in {text_dir}: missing {', '.join(missing[:5])}")

    cap = max((json.loads(Path(f).read_text())["tokens_counted"] for f in a.cap_from), default=None)
    counter: Counter = Counter()
    total = 0
    for sent in iter_sentences(text_dir, cap):
        counter.update(sent)
        total += len(sent)
    vocab = Vocab.build(counter, a.max_vocab, a.min_count)
    vocab.save(lang_dir / "vocab.json")
    stats = {"tokens_counted": total, "cap": cap, "vocab_size": len(vocab),
             "min_count": a.min_count, "max_vocab": a.max_vocab}
    (lang_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
