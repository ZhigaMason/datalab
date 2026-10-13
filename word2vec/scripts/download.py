"""Download Wikipedia parquet files and (cs/uk) the UDPipe model.

    python -m scripts.download --lang cs --data-dir data [--tiny]
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from w2v.textproc import UDPIPE_MODELS, WIKI_SNAPSHOT

WIKI_REPO = "wikimedia/wikipedia"
TINY_ROWS = 1000
MIN_MODEL_BYTES = 1_000_000


def download_model(lang: str, data_dir: Path) -> Path | None:
    if lang not in UDPIPE_MODELS:
        return None
    name, url = UDPIPE_MODELS[lang]
    dst = Path(data_dir) / "models" / name
    if dst.exists():
        print(f"model exists: {dst}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < MIN_MODEL_BYTES:
        tmp.unlink()
        raise RuntimeError(f"{url} returned {tmp.name} that is not a UDPipe model (too small; repository moved?)")
    tmp.replace(dst)
    print(f"downloaded {dst}")
    return dst


def download_wiki(lang: str, data_dir: Path, tiny: bool) -> Path:
    raw = Path(data_dir) / "raw" / f"{WIKI_SNAPSHOT}.{lang}"
    if tiny:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = sorted(fs.glob(f"datasets/{WIKI_REPO}/{WIKI_SNAPSHOT}.{lang}/*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files for {WIKI_SNAPSHOT}.{lang} on the hub")
        with fs.open(files[0], "rb") as fh:
            table = pq.ParquetFile(fh).read_row_group(0, columns=["text"]).slice(0, TINY_ROWS)
        raw.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, raw / "tiny.parquet")
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=WIKI_REPO, repo_type="dataset",
                          allow_patterns=[f"{WIKI_SNAPSHOT}.{lang}/*.parquet"],
                          local_dir=Path(data_dir) / "raw")
    print(f"wikipedia {lang}: {sorted(p.name for p in raw.glob('*.parquet'))}")
    return raw


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["en", "cs", "uk"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tiny", action="store_true", help=f"only the first {TINY_ROWS} articles")
    a = p.parse_args(argv)
    download_model(a.lang, Path(a.data_dir))
    download_wiki(a.lang, Path(a.data_dir), a.tiny)


if __name__ == "__main__":
    main()
