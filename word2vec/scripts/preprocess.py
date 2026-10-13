"""Clean + tokenize (en) or lemmatize (cs/uk) one shard of Wikipedia articles.

Parquet row groups (~1000 articles each) are assigned to shards round-robin, so each shard reads only its own
row groups.

    python -m scripts.preprocess --lang cs --shard 3 --num-shards 16 --workers 8 --data-dir data
"""
from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from w2v.textproc import UDPIPE_MODELS, WIKI_SNAPSHOT, clean_lines, make_lemmatizer, tokenize_en

_process = None


def _init_worker(lang: str, model_path: str | None) -> None:
    global _process
    if lang == "en":
        _process = tokenize_en
    else:
        lemmatizer = make_lemmatizer(lang, model_path)
        _process = lambda text: lemmatizer("\n\n".join(clean_lines(text)))  # noqa: E731


def _work(text: str) -> list[str]:
    return [" ".join(sent) for sent in _process(text)]


def iter_shard_texts(raw_dir: Path, shard: int, num_shards: int) -> Iterator[str]:
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {raw_dir}; run scripts.download first")
    unit = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            if unit % num_shards == shard:
                yield from pf.read_row_group(rg, columns=["text"]).column(0).to_pylist()
            unit += 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["en", "cs", "uk"])
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)
    if not 0 <= a.shard < a.num_shards:
        p.error(f"--shard must be in [0, {a.num_shards})")

    data_dir = Path(a.data_dir)
    out_dir = data_dir / a.lang / "text"
    out = out_dir / f"shard_{a.shard:04d}.txt"
    done = out.with_suffix(".done")
    if done.exists():
        print(f"{done} exists, skipping")
        return

    model_path = None
    if a.lang != "en":
        model_path = data_dir / "models" / UDPIPE_MODELS[a.lang][0]
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing; run scripts.download --lang {a.lang} first")
    raw_dir = data_dir / "raw" / f"{WIKI_SNAPSHOT}.{a.lang}"
    texts = iter_shard_texts(raw_dir, a.shard, a.num_shards)
    first = next(texts, None)  # surface a missing raw dir before starting workers
    out_dir.mkdir(parents=True, exist_ok=True)

    def all_texts():
        if first is not None:
            yield first
            yield from texts

    tmp = out.with_suffix(".part")
    n_articles = n_sentences = 0
    with get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(a.lang, str(model_path) if model_path else None)) as pool, \
            open(tmp, "w", encoding="utf-8") as fo:
        for lines in pool.imap(_work, all_texts(), chunksize=16):
            n_articles += 1
            for line in lines:
                fo.write(line + "\n")
            n_sentences += len(lines)
    tmp.replace(out)
    done.write_text(f"articles={n_articles} sentences={n_sentences}\n")
    print(f"{out}: {n_articles} articles, {n_sentences} sentences")


if __name__ == "__main__":
    main()
