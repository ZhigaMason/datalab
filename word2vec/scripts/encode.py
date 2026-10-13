"""Encode a language's preprocessed text into tokens.bin (uint32) using vocab.json.

`--pad` PAD tokens separate sentences; it must be >= the training window.

    python -m scripts.encode --lang cs --pad 5 --data-dir data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from w2v.data import Vocab, encode_corpus


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["en", "cs", "uk"])
    p.add_argument("--pad", type=int, default=5)
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)

    lang_dir = Path(a.data_dir) / a.lang
    stats_path = lang_dir / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} missing; run scripts.build_vocab first")
    stats = json.loads(stats_path.read_text())
    vocab = Vocab.load(lang_dir / "vocab.json")
    tmp = lang_dir / "tokens.bin.part"
    stats.update(encode_corpus(lang_dir / "text", vocab, tmp, a.pad, stats.get("cap")))
    tmp.replace(lang_dir / "tokens.bin")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
