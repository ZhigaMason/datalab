"""Download Wikipedia parquet files and (cs/uk) the UDPipe model.

    python -m scripts.download --lang cs --data-dir data [--tiny] [--max-files N]

--max-files N downloads only N evenly spaced parquet files of the snapshot (spread over article ids) and removes
other parquet files already in the raw dir, so preprocessing sees exactly the selected subset.
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


def select_files(files: list[str], max_files: int) -> list[str]:
    """Evenly spaced subset of sorted files; all of them when max_files <= 0 or >= len(files)."""
    files = sorted(files)
    if max_files <= 0 or max_files >= len(files):
        return files
    if max_files == 1:
        return files[:1]
    return [files[round(i * (len(files) - 1) / (max_files - 1))] for i in range(max_files)]


def download_wiki(lang: str, data_dir: Path, tiny: bool, max_files: int = 0) -> Path:
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
        from huggingface_hub import HfApi, snapshot_download

        prefix = f"{WIKI_SNAPSHOT}.{lang}/"
        remote = [f for f in HfApi().list_repo_files(WIKI_REPO, repo_type="dataset")
                  if f.startswith(prefix) and f.endswith(".parquet")]
        if not remote:
            raise FileNotFoundError(f"no parquet files for {WIKI_SNAPSHOT}.{lang} on the hub")
        chosen = select_files(remote, max_files)
        if raw.exists():
            keep = {Path(f).name for f in chosen}
            for stale in raw.glob("*.parquet"):
                if stale.name not in keep:
                    print(f"removing unselected {stale}")
                    stale.unlink()
        snapshot_download(repo_id=WIKI_REPO, repo_type="dataset", allow_patterns=chosen,
                          local_dir=Path(data_dir) / "raw")
    print(f"wikipedia {lang}: {sorted(p.name for p in raw.glob('*.parquet'))}")
    return raw


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["en", "cs", "uk"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tiny", action="store_true", help=f"only the first {TINY_ROWS} articles")
    p.add_argument("--max-files", type=int, default=0, help="download N evenly spaced parquet files (0 = all)")
    a = p.parse_args(argv)
    download_model(a.lang, Path(a.data_dir))
    download_wiki(a.lang, Path(a.data_dir), a.tiny, a.max_files)


if __name__ == "__main__":
    main()
