import json
from collections import Counter

import numpy as np
import pytest

from scripts import build_vocab, encode
from w2v.data import Vocab, encode_corpus, iter_sentences


def _write_shards(text_dir, shards, done=True):
    text_dir.mkdir(parents=True, exist_ok=True)
    for i, lines in enumerate(shards):
        (text_dir / f"shard_{i:04d}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        if done:
            (text_dir / f"shard_{i:04d}.done").write_text("ok\n")


def test_vocab_build_orders_by_count_then_word_and_filters():
    v = Vocab.build(Counter({"a": 5, "b": 7, "c": 5, "d": 1}), max_size=4, min_count=2)
    assert v.words == ["<pad>", "<unk>", "b", "a"]
    assert v.counts == [0, 0, 7, 5]
    assert v.index["a"] == 3 and len(v) == 4


def test_vocab_roundtrip_non_ascii(tmp_path):
    v = Vocab(["<pad>", "<unk>", "м'ясо", "dům"], [0, 0, 3, 2])
    v.save(tmp_path / "vocab.json")
    w = Vocab.load(tmp_path / "vocab.json")
    assert w.words == v.words and w.counts == v.counts and w.index["dům"] == 3
    assert "м'ясо" in (tmp_path / "vocab.json").read_text(encoding="utf-8")


def test_iter_sentences_respects_cap(tmp_path):
    _write_shards(tmp_path, [["a b c", "", "d e"], ["f g h"]])
    assert list(iter_sentences(tmp_path)) == [["a", "b", "c"], ["d", "e"], ["f", "g", "h"]]
    assert list(iter_sentences(tmp_path, max_tokens=4)) == [["a", "b", "c"], ["d", "e"]]


def test_iter_sentences_no_shards_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_sentences(tmp_path))


def test_encode_corpus_pads_between_sentences_and_drops_oov(tmp_path):
    _write_shards(tmp_path / "text", [["a b", "zzz", "b a a"]])
    vocab = Vocab(["<pad>", "<unk>", "a", "b"], [0, 0, 3, 2])
    stats = encode_corpus(tmp_path / "text", vocab, tmp_path / "tokens.bin", pad=2)
    ids = np.fromfile(tmp_path / "tokens.bin", dtype=np.uint32).tolist()
    assert ids == [0, 0, 2, 3, 0, 0, 3, 2, 2, 0, 0]
    assert stats == {"tokens": 5, "length": 11, "pad": 2, "sentences": 2}


def _data(tmp_path, lang, lines, n_shards=1):
    _write_shards(tmp_path / lang / "text", [lines] + [["x"]] * (n_shards - 1))


def test_build_vocab_requires_all_done_markers(tmp_path):
    _data(tmp_path, "en", ["the cat"])
    with pytest.raises(FileNotFoundError, match="shard_0001"):
        build_vocab.main(["--lang", "en", "--num-shards", "2", "--data-dir", str(tmp_path)])


def test_build_vocab_and_encode_scripts(tmp_path):
    _data(tmp_path, "en", ["the cat", "the dog", "the cat"])
    build_vocab.main(["--lang", "en", "--num-shards", "1", "--min-count", "2", "--data-dir", str(tmp_path)])
    v = Vocab.load(tmp_path / "en/vocab.json")
    assert v.words == ["<pad>", "<unk>", "the", "cat"]
    stats = json.loads((tmp_path / "en/stats.json").read_text())
    assert stats["tokens_counted"] == 6 and stats["cap"] is None
    encode.main(["--lang", "en", "--pad", "1", "--data-dir", str(tmp_path)])
    stats = json.loads((tmp_path / "en/stats.json").read_text())
    assert stats["tokens"] == 5 and stats["length"] == 9 and stats["pad"] == 1 and stats["tokens_counted"] == 6
    assert len(np.fromfile(tmp_path / "en/tokens.bin", dtype=np.uint32)) == 9


def test_build_vocab_cap_from_other_languages(tmp_path):
    for lang, n in (("cs", 3), ("uk", 4)):
        (tmp_path / lang).mkdir()
        (tmp_path / lang / "stats.json").write_text(json.dumps({"tokens_counted": n}))
    _data(tmp_path, "en", ["a b", "c d", "e f", "g h"])
    build_vocab.main(["--lang", "en", "--num-shards", "1", "--min-count", "1", "--data-dir", str(tmp_path),
                      "--cap-from", str(tmp_path / "cs/stats.json"), str(tmp_path / "uk/stats.json")])
    stats = json.loads((tmp_path / "en/stats.json").read_text())
    assert stats["cap"] == 4 and stats["tokens_counted"] == 4
    assert Vocab.load(tmp_path / "en/vocab.json").words[2:] == ["a", "b", "c", "d"]
    encode.main(["--lang", "en", "--pad", "1", "--data-dir", str(tmp_path)])
    assert json.loads((tmp_path / "en/stats.json").read_text())["tokens"] == 4


def test_stale_shards_beyond_num_shards_are_ignored(tmp_path):
    _write_shards(tmp_path / "en/text", [["the cat"], ["the dog"]])
    (tmp_path / "en/text/shard_0002.txt").write_text("stale words here\n")
    build_vocab.main(["--lang", "en", "--num-shards", "2", "--min-count", "1", "--data-dir", str(tmp_path)])
    assert json.loads((tmp_path / "en/stats.json").read_text())["num_shards"] == 2
    assert not {"stale", "words", "here"} & set(Vocab.load(tmp_path / "en/vocab.json").words)
    encode.main(["--lang", "en", "--pad", "1", "--data-dir", str(tmp_path)])
    assert json.loads((tmp_path / "en/stats.json").read_text())["tokens"] == 4


def test_iter_sentences_missing_numbered_shard_errors(tmp_path):
    _write_shards(tmp_path, [["a b"]])
    with pytest.raises(FileNotFoundError, match="shard_0001"):
        list(iter_sentences(tmp_path, num_shards=2))
