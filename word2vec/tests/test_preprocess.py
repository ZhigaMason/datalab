import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import download
from scripts.preprocess import iter_shard_texts, main


def _write_raw(data_dir, lang, texts, files=2, row_group_size=None):
    raw = data_dir / "raw" / f"20231101.{lang}"
    raw.mkdir(parents=True)
    per = -(-len(texts) // files)
    for f in range(files):
        chunk = texts[f * per:(f + 1) * per]
        pq.write_table(pa.table({"text": chunk}), raw / f"train-{f:05d}.parquet", row_group_size=row_group_size)
    return raw


def test_iter_shard_texts_assigns_row_groups_round_robin(tmp_path):
    texts = [f"article {i}" for i in range(12)]
    raw = _write_raw(tmp_path, "en", texts, files=2, row_group_size=2)  # 6 row groups: 3 per file
    shards = [list(iter_shard_texts(raw, s, 4)) for s in range(4)]
    assert shards[0] == ["article 0", "article 1", "article 8", "article 9"]  # row groups 0 and 4
    assert shards[3] == ["article 6", "article 7"]  # row group 3 = first group of the second file
    assert sorted(sum(shards, [])) == sorted(texts)


def test_iter_shard_texts_reads_only_its_row_groups(tmp_path, monkeypatch):
    raw = _write_raw(tmp_path, "en", [f"a{i}" for i in range(8)], files=1, row_group_size=2)
    read = []
    orig = pq.ParquetFile.read_row_group
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", lambda self, i, **kw: read.append(i) or orig(self, i, **kw))
    assert list(iter_shard_texts(raw, 1, 2)) == ["a2", "a3", "a6", "a7"]
    assert read == [1, 3]


def test_iter_shard_texts_missing_dir_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        list(iter_shard_texts(tmp_path / "nope", 0, 1))


def _args(tmp_path, lang="en", shard=0, n=1):
    return ["--lang", lang, "--shard", str(shard), "--num-shards", str(n), "--workers", "1",
            "--data-dir", str(tmp_path)]


def test_preprocess_en_writes_shard_and_marker(tmp_path):
    _write_raw(tmp_path, "en", [
        "Heading\nThe cat sat on the mat. The dog ran away fast.",
        "Short\nBirds can fly over the tall trees.",
    ])
    main(_args(tmp_path))
    out = tmp_path / "en/text/shard_0000.txt"
    assert out.read_text(encoding="utf-8").splitlines() == [
        "the cat sat on the mat", "the dog ran away fast", "birds can fly over the tall trees",
    ]
    assert (tmp_path / "en/text/shard_0000.done").exists()
    mtime = out.stat().st_mtime_ns
    main(_args(tmp_path))  # idempotent: finished shard is skipped
    assert out.stat().st_mtime_ns == mtime


def test_preprocess_rejects_bad_shard(tmp_path):
    with pytest.raises(SystemExit):
        main(_args(tmp_path, shard=3, n=2))


def test_preprocess_cs_requires_model(tmp_path):
    _write_raw(tmp_path, "cs", ["Nějaký delší text o hlavním městě Praha."], files=1)
    with pytest.raises(FileNotFoundError, match="download"):
        main(_args(tmp_path, lang="cs"))


def test_download_model_rejects_html_response(tmp_path, monkeypatch):
    def fake_retrieve(url, dst):
        open(dst, "w").write("<!DOCTYPE html><html></html>")
    monkeypatch.setattr(download.urllib.request, "urlretrieve", fake_retrieve)
    with pytest.raises(RuntimeError, match="not a UDPipe model"):
        download.download_model("cs", tmp_path)
    assert not list((tmp_path / "models").glob("*.udpipe"))


def test_download_model_noop_for_english(tmp_path):
    download.download_model("en", tmp_path)
    assert not (tmp_path / "models").exists()


def test_select_files_evenly_spaced():
    files = [f"train-{i:05d}-of-00041.parquet" for i in range(41)]
    picked = download.select_files(files, 8)
    assert [int(f[6:11]) for f in picked] == [0, 6, 11, 17, 23, 29, 34, 40]
    assert download.select_files(files, 0) == files
    assert download.select_files(files, 100) == files
    assert download.select_files(files, 1) == files[:1]
