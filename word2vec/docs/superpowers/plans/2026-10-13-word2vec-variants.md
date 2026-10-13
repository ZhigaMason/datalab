# Word2vec Variants (bidir / causal × NS / JEPA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PyTorch word2vec experiment suite (en/cs/uk × bidir/causal × negative-sampling/JEPA) that preprocesses Wikipedia, trains on MetaCentrum GPUs via PBS, and emits a validated ONNX embedding-lookup artifact plus intrinsic-evaluation metrics per run.

**Architecture:** Package `w2v/` holds all logic (text processing, data, model, losses, training, export, evaluation); `scripts/` are thin `python -m scripts.<name>` CLIs for the data pipeline; `jobs/` holds PBS scripts. Context aggregation is a fixed, closed-form averaging kernel applied with depthwise `conv1d`. Everything is unit-tested on CPU; a synthetic-corpus end-to-end test covers train → ONNX → eval.

**Tech Stack:** Python ≥3.10, uv (`pyproject.toml` + `uv.lock`), PyTorch 2.14 (cu126 index), onnx + onnxscript + onnxruntime, numpy, scipy, tensorboard; prep extra: ufal.udpipe, huggingface_hub, pyarrow; pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-word2vec-variants-design.md`

## Global Constraints

- All paths below are relative to `word2vec/` (the project root; the git repo root is its parent `datalab/`). Run every command from `word2vec/`.
- Run Python only via uv: `uv run python -m …`, `uv run pytest …`.
- `torch` comes from the `pytorch-cu126` index (`https://download.pytorch.org/whl/cu126`), declared explicit in `pyproject.toml`.
- Reserved ids: `0 = <pad>`, `1 = <unk>`. OOV words are dropped at encode time (never encoded as `<unk>`).
- Kernel (window radius k): bidir `([1]*k + [0] + [1]*k) / 2k`; causal `([1]*k + [0] + [0]*k) / k`. Non-trainable buffer.
- Loss positions: `m[t] == 1` AND at least one valid context token in the window.
- NS: K = 1024 shared negatives from unigram^0.75; negative term scaled by `5/K`.
- JEPA: target = `E_ema` (EMA of context encoder `E`), τ cosine 0.996 → 1.0; loss `1 − cos + λ_var·var + λ_cov·cov`, defaults λ_var = 1.0, λ_cov = 0.04.
- Defaults: dim 300, window 5, batch 64, seq_len 1024, lr 2e-3, warmup 2 %, cosine to 0, subsample t = 1e-4, epochs 1, vocab 200 000 (incl. specials), min_count 5.
- Run dir name: `<lang>-<mode>[-jepa]-d<dim>-s<seed>`.
- ONNX graph: input `ids` int64 (batch, seq) → output `embeddings` float32 (batch, seq, dim); parity with PyTorch `atol = 1e-5`.
- Wikipedia snapshot `20231101` from HF `wikimedia/wikipedia`.
- Every git commit message ends with the `Co-Authored-By` trailer from the session's attribution instructions.

## Review Focus

1. **Encoded sentence padding narrower than the training window** — training must refuse (`stats["pad"] < window` → `ValueError`), not silently leak context across sentences. Test in Task 10.
2. **Re-running into an existing run dir with different hyperparameters** — must refuse with a clear error instead of resuming a mismatched checkpoint. Test in Task 10.
3. **Benchmarks with poor or zero vocabulary coverage** — evaluation returns `None` scores with `coverage` reported, never crashes or emits NaN. Test in Task 8.
4. **Corpus shorter than `seq_len`** (tiny/local runs) — `TokenStream` raises a clear `ValueError` naming `seq_len`. Test in Task 4.
5. **Lemmatization collapsing benchmark items** (e.g. analogy `a == b` after lemmatizing, multi-token lemma) — such items are dropped and counted in the manifest, not evaluated. Test in Task 9.

---

## File Structure

```
word2vec/
  pyproject.toml, .python-version, .gitignore, README.md, uv.lock
  w2v/__init__.py
  w2v/textproc.py     token normalization, English tokenizer, CoNLL-U lemma parsing, UDPipe wrapper, model URLs
  w2v/data.py         Vocab, iter_sentences, encode_corpus, TokenStream, keep_probs, sample_mask, noise_dist
  w2v/model.py        context_kernel, W2V (embedding, conv context, NS head / JEPA predictor + EMA)
  w2v/losses.py       masked_mean, ns_loss, vicreg_terms, jepa_loss, normalized_std
  w2v/export.py       export_onnx, verify_onnx, load_onnx_embeddings, write_vec
  w2v/evaluate.py     load_benchmarks, eval_similarity, eval_analogy, evaluate_all, CLI
  w2v/train.py        CLI + training loop, checkpoint/resume, finalize (export + eval)
  scripts/__init__.py
  scripts/download.py, preprocess.py, build_vocab.py, encode.py, fetch_benchmarks.py, compare.py
  jobs/config.sh, setup_env.sh, download.pbs, prep.pbs, vocab.pbs, train.pbs, submit_all.sh
  tests/conftest.py, test_textproc.py, test_preprocess.py, test_vocab_encode.py, test_stream.py,
        test_model.py, test_losses.py, test_export.py, test_evaluate.py, test_fetch_benchmarks.py,
        test_train.py, test_compare.py
```

---

### Task 1: Project scaffold + text processing

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `w2v/__init__.py`, `scripts/__init__.py`, `w2v/textproc.py`
- Test: `tests/test_textproc.py`

**Interfaces:**
- Produces: `w2v.textproc.NUM = "<num>"`, `WIKI_SNAPSHOT = "20231101"`, `UDPIPE_MODELS: dict[str, tuple[str, str]]` (lang → (filename, url)), `normalize_token(tok: str) -> str | None`, `clean_lines(text: str) -> list[str]`, `tokenize_en(text: str) -> list[list[str]]`, `lemmas_from_conllu(conllu: str) -> list[list[str]]`, `class UdpipeLemmatizer(model_path)` with `__call__(text) -> list[list[str]]` and `lemma(word) -> str | None`.

- [ ] **Step 1: Create project files**

`pyproject.toml`:
```toml
[project]
name = "w2v"
version = "0.1.0"
description = "Word2vec variants: bidirectional/causal x negative-sampling/JEPA"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.9",
    "numpy>=1.26",
    "scipy>=1.11",
    "onnx>=1.17",
    "onnxscript>=0.3",
    "onnxruntime>=1.20",
    "tensorboard>=2.17",
]

[project.optional-dependencies]
prep = ["ufal.udpipe>=1.3", "huggingface_hub>=0.25", "pyarrow>=15"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["w2v"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`.python-version`:
```
3.12
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
data/
data-tiny/
runs/
results/
```

`w2v/__init__.py` and `scripts/__init__.py`: empty files.

- [ ] **Step 2: Lock and sync**

Run: `uv lock && uv sync --extra prep`
Expected: `uv.lock` created; `.venv` with torch 2.14.x+cu126, ufal.udpipe, pytest installed. (Downloads ~3 GB of CUDA wheels once.)

- [ ] **Step 3: Write the failing tests** — `tests/test_textproc.py`

```python
from pathlib import Path

import pytest

from w2v.textproc import NUM, UdpipeLemmatizer, clean_lines, lemmas_from_conllu, normalize_token, tokenize_en


def test_normalize_token_words_numbers_and_punct():
    assert normalize_token("Hello") == "hello"
    assert normalize_token("don’t") == "don't"
    assert normalize_token("м’ясо") == "м'ясо"
    assert normalize_token("мʼясо") == "м'ясо"
    assert normalize_token("well-known") == "well-known"
    assert normalize_token("1,234.5") == NUM
    assert normalize_token(".") is None
    assert normalize_token("_") is None
    assert normalize_token("abc123") is None


def test_clean_lines_drops_headings_and_short_lines():
    text = "History\n\nThe city was founded in 1200 by settlers.\nSee also\n"
    assert clean_lines(text) == ["The city was founded in 1200 by settlers."]


def test_tokenize_en_splits_sentences_and_lowercases():
    text = "The Cat sat on the mat. It was 3 years old! Really, it was."
    assert tokenize_en(text) == [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["it", "was", NUM, "years", "old"],
        ["really", "it", "was"],
    ]


def _row(i, form, lemma, upos):
    return "\t".join([i, form, lemma, upos] + ["_"] * 6)


def test_lemmas_from_conllu_skips_punct_multiword_and_comments():
    conllu = "\n".join([
        "# sent_id = 1",
        _row("1", "Byli", "být", "AUX"),
        _row("2", "jsme", "být", "AUX"),
        _row("3", "v", "v", "ADP"),
        _row("4", "lesích", "les", "NOUN"),
        _row("5", ".", ".", "PUNCT"),
        "",
        "# sent_id = 2",
        _row("1-2", "abychom", "_", "_"),
        _row("1", "aby", "aby", "SCONJ"),
        _row("2", "bychom", "být", "AUX"),
        _row("3", "Prahu", "Praha", "PROPN"),
        _row("4", "2024", "2024", "NUM"),
        "",
    ])
    assert lemmas_from_conllu(conllu) == [["být", "být", "v", "les"], ["aby", "být", "praha", NUM]]


CS_MODEL = Path("data/models/czech-pdt-ud-2.5-191206.udpipe")


@pytest.mark.skipif(not CS_MODEL.exists(), reason="UDPipe model not downloaded (Task 2 download)")
def test_udpipe_lemmatizer_czech():
    lem = UdpipeLemmatizer(CS_MODEL)
    assert lem("Byli jsme v lesích.") == [["být", "být", "v", "les"]]
    assert lem.lemma("lesích") == "les"


def test_udpipe_lemmatizer_missing_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        UdpipeLemmatizer(tmp_path / "missing.udpipe")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_textproc.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'w2v.textproc'`.

- [ ] **Step 5: Implement** — `w2v/textproc.py`

```python
"""Token normalization, English tokenization and UDPipe lemmatization."""
from __future__ import annotations

import re
from pathlib import Path

NUM = "<num>"
WIKI_SNAPSHOT = "20231101"
MIN_LINE_WORDS = 4

_LINDAT = "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams"
UDPIPE_MODELS: dict[str, tuple[str, str]] = {
    "cs": ("czech-pdt-ud-2.5-191206.udpipe", f"{_LINDAT}/124d9958-240e-44e6-b522-eea8447a8de6/content"),
    "uk": ("ukrainian-iu-ud-2.5-191206.udpipe", f"{_LINDAT}/72ba199d-b8fc-40a5-a12d-c5ea219de998/content"),
}

_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'"})
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9])")
_DROP_UPOS = {"PUNCT", "SYM"}


def normalize_token(tok: str) -> str | None:
    """Lowercased word, NUM for numbers, None for anything else."""
    tok = tok.translate(_APOSTROPHES)
    if _WORD_RE.fullmatch(tok):
        return tok.lower()
    if _NUM_RE.fullmatch(tok):
        return NUM
    return None


def clean_lines(text: str) -> list[str]:
    """Wikipedia article text -> paragraphs; drops headings and very short lines."""
    return [line.strip() for line in text.split("\n") if len(line.split()) >= MIN_LINE_WORDS]


def tokenize_en(text: str) -> list[list[str]]:
    sentences = []
    for para in clean_lines(text):
        for sent in _SENT_SPLIT_RE.split(para):
            toks = [normalize_token(t) for t in _TOKEN_RE.findall(sent.translate(_APOSTROPHES))]
            toks = [t for t in toks if t]
            if toks:
                sentences.append(toks)
    return sentences


def lemmas_from_conllu(conllu: str) -> list[list[str]]:
    sentences, cur = [], []
    for line in conllu.split("\n"):
        if not line:
            if cur:
                sentences.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0] or cols[3] in _DROP_UPOS:
            continue
        tok = normalize_token(cols[2])
        if tok:
            cur.append(tok)
    if cur:
        sentences.append(cur)
    return sentences


class UdpipeLemmatizer:
    """Tokenize + tag with a UDPipe 1 model; returns lemmatized, normalized sentences."""

    def __init__(self, model_path: str | Path):
        from ufal.udpipe import Model, Pipeline

        if not Path(model_path).exists():
            raise FileNotFoundError(f"UDPipe model not found: {model_path}; run scripts.download first")
        self._model = Model.load(str(model_path))
        if self._model is None:
            raise FileNotFoundError(f"cannot load UDPipe model {model_path}")
        self._pipe = Pipeline(self._model, "tokenize", "default", "none", "conllu")

    def __call__(self, text: str) -> list[list[str]]:
        from ufal.udpipe import ProcessingError

        err = ProcessingError()
        out = self._pipe.process(text, err)
        if err.occurred():
            raise RuntimeError(f"UDPipe failed: {err.message}")
        return lemmas_from_conllu(out)

    def lemma(self, word: str) -> str | None:
        toks = [t for sent in self(word) for t in sent]
        return toks[0] if len(toks) == 1 else None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_textproc.py -v`
Expected: all PASS except `test_udpipe_lemmatizer_czech` SKIPPED (model downloaded in Task 2).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore w2v/__init__.py scripts/__init__.py w2v/textproc.py tests/test_textproc.py
git commit -m "feat(w2v): project scaffold and text processing"
```

---

### Task 2: Download + preprocess scripts

**Files:**
- Create: `scripts/download.py`, `scripts/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `w2v.textproc.{UDPIPE_MODELS, WIKI_SNAPSHOT, UdpipeLemmatizer, clean_lines, tokenize_en}`.
- Produces: raw parquet at `<data>/raw/20231101.<lang>/*.parquet`; UDPipe models at `<data>/models/<filename>`; preprocess output `<data>/<lang>/text/shard_{i:04d}.txt` (one space-joined sentence per line) plus marker `shard_{i:04d}.done`. Functions `scripts.preprocess.iter_shard_texts(raw_dir: Path, shard: int, num_shards: int) -> Iterator[str]`, `scripts.preprocess.main(argv)`, `scripts.download.download_model(lang, data_dir)`, `scripts.download.main(argv)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_preprocess.py`

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import download
from scripts.preprocess import iter_shard_texts, main


def _write_raw(data_dir, lang, texts, files=2):
    raw = data_dir / "raw" / f"20231101.{lang}"
    raw.mkdir(parents=True)
    per = -(-len(texts) // files)
    for f in range(files):
        chunk = texts[f * per:(f + 1) * per]
        pq.write_table(pa.table({"text": chunk}), raw / f"train-{f:05d}.parquet")
    return raw


def test_iter_shard_texts_round_robin(tmp_path):
    texts = [f"article {i}" for i in range(10)]
    raw = _write_raw(tmp_path, "en", texts)
    shards = [list(iter_shard_texts(raw, s, 3)) for s in range(3)]
    assert shards[0] == ["article 0", "article 3", "article 6", "article 9"]
    assert sorted(sum(shards, [])) == sorted(texts)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: FAIL — `ImportError: cannot import name 'download' from 'scripts'` / missing module.

- [ ] **Step 3: Implement** — `scripts/download.py`

```python
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
```

- [ ] **Step 4: Implement** — `scripts/preprocess.py`

```python
"""Clean + tokenize (en) or lemmatize (cs/uk) one shard of Wikipedia articles.

Articles are assigned to shards round-robin (article index % num_shards).

    python -m scripts.preprocess --lang cs --shard 3 --num-shards 16 --workers 8 --data-dir data
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from w2v.textproc import UDPIPE_MODELS, WIKI_SNAPSHOT, UdpipeLemmatizer, clean_lines, tokenize_en

_process = None


def _init_worker(lang: str, model_path: str | None) -> None:
    global _process
    if lang == "en":
        _process = tokenize_en
    else:
        lemmatizer = UdpipeLemmatizer(model_path)
        _process = lambda text: lemmatizer("\n\n".join(clean_lines(text)))  # noqa: E731


def _work(text: str) -> list[str]:
    return [" ".join(sent) for sent in _process(text)]


def iter_shard_texts(raw_dir: Path, shard: int, num_shards: int) -> Iterator[str]:
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {raw_dir}; run scripts.download first")
    idx = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1024, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if idx % num_shards == shard:
                    yield text
                idx += 1


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
    with Pool(a.workers, initializer=_init_worker, initargs=(a.lang, str(model_path) if model_path else None)) as pool, \
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: 7 PASS.

- [ ] **Step 6: Fetch the Czech UDPipe model and re-run the UDPipe test (needs network)**

Run: `uv run python -c "from scripts.download import download_model; download_model('cs', 'data')" && uv run pytest tests/test_textproc.py -v`
Expected: `downloaded data/models/czech-pdt-ud-2.5-191206.udpipe`; `test_udpipe_lemmatizer_czech` now PASSES.

- [ ] **Step 7: Commit**

```bash
git add scripts/download.py scripts/preprocess.py tests/test_preprocess.py
git commit -m "feat(w2v): wikipedia download and sharded preprocessing"
```

---

### Task 3: Vocabulary, corpus encoding and their scripts

**Files:**
- Create: `w2v/data.py` (vocab/encode part), `scripts/build_vocab.py`, `scripts/encode.py`
- Test: `tests/test_vocab_encode.py`

**Interfaces:**
- Consumes: shard files `<data>/<lang>/text/shard_*.txt` + `.done` markers from Task 2.
- Produces (in `w2v/data.py`): `PAD_ID = 0`, `UNK_ID = 1`, `SPECIALS = ["<pad>", "<unk>"]`; `@dataclass Vocab(words: list[str], counts: list[int])` with `.index: dict[str,int]`, `__len__`, `save(path)`, `Vocab.load(path)`, `Vocab.build(counter, max_size, min_count)`; `iter_sentences(text_dir, max_tokens=None) -> Iterator[list[str]]`; `encode_corpus(text_dir, vocab, out_path, pad, max_tokens=None) -> dict` with keys `tokens, length, pad, sentences`.
- Files: `<data>/<lang>/vocab.json` (`{"words": [...], "counts": [...]}`), `<data>/<lang>/stats.json` (build_vocab writes `tokens_counted, cap, vocab_size, min_count, max_vocab`; encode adds `tokens, length, pad, sentences`), `<data>/<lang>/tokens.bin` (uint32).

- [ ] **Step 1: Write the failing tests** — `tests/test_vocab_encode.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vocab_encode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'w2v.data'`.

- [ ] **Step 3: Implement** — `w2v/data.py`

```python
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
```

- [ ] **Step 4: Implement** — `scripts/build_vocab.py`

```python
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
```

- [ ] **Step 5: Implement** — `scripts/encode.py`

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_vocab_encode.py -v`
Expected: 8 PASS.

- [ ] **Step 7: Commit**

```bash
git add w2v/data.py scripts/build_vocab.py scripts/encode.py tests/test_vocab_encode.py
git commit -m "feat(w2v): vocabulary building and corpus encoding"
```

---

### Task 4: Token stream, subsampling, noise distribution

**Files:**
- Modify: `w2v/data.py` (append)
- Test: `tests/test_stream.py`

**Interfaces:**
- Produces: `TokenStream(path, seq_len, batch_size, seed)` with `.batch(step: int) -> torch.LongTensor (B, T)` (deterministic in `(seed, step)`), `.steps_per_epoch() -> int`, `.num_tokens: int`; `keep_probs(counts: list[int], t: float) -> torch.FloatTensor (V,)`; `sample_mask(x: LongTensor, keep: FloatTensor, generator=None) -> BoolTensor`; `noise_dist(counts, power=0.75) -> torch.FloatTensor (V,)` (sums to 1, zero for count-0 ids).

- [ ] **Step 1: Write the failing tests** — `tests/test_stream.py`

```python
import numpy as np
import pytest
import torch

from w2v.data import TokenStream, keep_probs, noise_dist, sample_mask


def _corpus(tmp_path, n):
    p = tmp_path / "tokens.bin"
    np.arange(n, dtype=np.uint32).tofile(p)
    return p


def test_token_stream_deterministic_contiguous_chunks(tmp_path):
    p = _corpus(tmp_path, 100)
    s = TokenStream(p, seq_len=10, batch_size=4, seed=0)
    b = s.batch(3)
    assert b.shape == (4, 10) and b.dtype == torch.int64
    assert torch.equal(b, s.batch(3))
    assert torch.equal(b, TokenStream(p, 10, 4, seed=0).batch(3))
    assert not torch.equal(b, s.batch(4))
    assert not torch.equal(b, TokenStream(p, 10, 4, seed=1).batch(3))
    assert torch.equal(b[:, 1:] - b[:, :-1], torch.ones(4, 9, dtype=torch.long))
    assert int(b.max()) < 100
    assert s.steps_per_epoch() == 2 and s.num_tokens == 100


def test_token_stream_shorter_than_seq_len_errors(tmp_path):
    with pytest.raises(ValueError, match="seq_len"):
        TokenStream(_corpus(tmp_path, 5), seq_len=10, batch_size=2, seed=0)


def test_keep_probs():
    p = keep_probs([0, 0, 900, 100], t=0.01)
    assert p[0] == 0 and p[1] == 0
    assert p[2].item() == pytest.approx((0.01 / 0.9) ** 0.5, rel=1e-5)
    assert p[3].item() == pytest.approx((0.01 / 0.1) ** 0.5, rel=1e-5)
    assert keep_probs([0, 0, 1, 10**9], t=1e-4)[2].item() == 1.0
    assert keep_probs([0, 0, 900, 100], t=0).tolist() == [0.0, 0.0, 1.0, 1.0]


def test_sample_mask_follows_keep_probs():
    keep = torch.tensor([0.0, 0.0, 1.0, 0.5])
    g = torch.Generator().manual_seed(0)
    assert not sample_mask(torch.zeros(1, 50, dtype=torch.long), keep, g).any()
    assert sample_mask(torch.full((1, 50), 2), keep, g).all()
    assert sample_mask(torch.full((1, 20000), 3), keep, g).float().mean().item() == pytest.approx(0.5, abs=0.02)


def test_noise_dist():
    q = noise_dist([0, 0, 16, 1])
    assert q.tolist() == pytest.approx([0.0, 0.0, 8 / 9, 1 / 9])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stream.py -v`
Expected: FAIL — `ImportError: cannot import name 'TokenStream'`.

- [ ] **Step 3: Implement** — append to `w2v/data.py` (add `import torch` to the imports at the top)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stream.py tests/test_vocab_encode.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add w2v/data.py tests/test_stream.py
git commit -m "feat(w2v): token stream, subsampling and noise distribution"
```

---

### Task 5: Model — closed-form convolutional context

**Files:**
- Create: `w2v/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `w2v.data.PAD_ID`.
- Produces: `MODES = ("bidir", "causal")`; `context_kernel(window: int, mode: str) -> Tensor (2k+1,)`; `class W2V(nn.Module)` with `__init__(vocab_size, dim, window, mode, jepa)`, attributes `emb: nn.Embedding`, `out: nn.Linear(dim, V, bias=False)` (non-JEPA only), `predictor: nn.Linear(dim, dim)` and `emb_ema: nn.Embedding` (JEPA only, `requires_grad=False`), buffers `kernel`, `support`; `context(x: LongTensor (B,T), m: BoolTensor (B,T)) -> (ctx (B,T,d), has_ctx BoolTensor (B,T))`; `update_ema(tau: float)`. (`loss` is added in Task 6.)

- [ ] **Step 1: Write the failing tests** — `tests/test_model.py`

```python
import pytest
import torch

from w2v.model import W2V, context_kernel


def naive_context(e, m, window, mode):
    T, d = e.shape
    out, has = torch.zeros(T, d), torch.zeros(T, dtype=torch.bool)
    offsets = [o for o in range(-window, window + 1) if o != 0 and (mode == "bidir" or o < 0)]
    norm = 2 * window if mode == "bidir" else window
    for t in range(T):
        for o in offsets:
            j = t + o
            if 0 <= j < T and m[j]:
                out[t] += e[j]
                has[t] = True
        out[t] /= norm
    return out, has


def test_context_kernel_closed_form():
    assert context_kernel(2, "bidir").tolist() == [0.25, 0.25, 0.0, 0.25, 0.25]
    assert context_kernel(2, "causal").tolist() == [0.5, 0.5, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        context_kernel(2, "sideways")


@pytest.mark.parametrize("mode", ["bidir", "causal"])
def test_context_matches_naive_loop(mode):
    torch.manual_seed(0)
    model = W2V(20, 6, window=3, mode=mode, jepa=False)
    x = torch.randint(2, 20, (2, 15))
    m = torch.rand(2, 15) > 0.3
    x[0, :2] = 0
    m[0, :2] = False
    ctx, has = model.context(x, m)
    for b in range(2):
        exp, exp_has = naive_context(model.emb(x[b]).detach(), m[b], 3, mode)
        torch.testing.assert_close(ctx[b].detach(), exp)
        assert torch.equal(has[b], exp_has)


def test_causal_context_ignores_current_and_future_tokens():
    torch.manual_seed(0)
    model = W2V(20, 6, window=4, mode="causal", jepa=False)
    x = torch.randint(2, 20, (1, 20))
    m = torch.ones(1, 20, dtype=torch.bool)
    x2 = x.clone()
    x2[0, 10:] = (x[0, 10:] - 2 + 1) % 18 + 2  # different non-special ids
    ctx, _ = model.context(x, m)
    ctx2, _ = model.context(x2, m)
    torch.testing.assert_close(ctx[0, :11], ctx2[0, :11])
    assert not torch.allclose(ctx[0, 11:], ctx2[0, 11:])


def test_bidir_context_ignores_center_token():
    torch.manual_seed(0)
    model = W2V(20, 6, window=2, mode="bidir", jepa=False)
    x = torch.randint(2, 20, (1, 12))
    m = torch.ones(1, 12, dtype=torch.bool)
    x2 = x.clone()
    x2[0, 6] = (x[0, 6] - 2 + 1) % 18 + 2
    torch.testing.assert_close(model.context(x, m)[0][0, 6], model.context(x2, m)[0][0, 6])


def test_has_ctx_false_when_window_fully_masked():
    model = W2V(10, 4, window=2, mode="bidir", jepa=False)
    x = torch.full((1, 9), 3)
    m = torch.zeros(1, 9, dtype=torch.bool)
    m[0, 0] = m[0, 8] = True
    ctx, has = model.context(x, m)
    assert has[0].tolist() == [False, True, True, False, False, False, True, True, False]
    assert torch.count_nonzero(ctx[0, 4]) == 0
    causal = W2V(10, 4, window=2, mode="causal", jepa=False)
    assert not causal.context(x, torch.ones(1, 9, dtype=torch.bool))[1][0, 0]


def test_jepa_model_has_ema_copy_without_grad():
    model = W2V(30, 8, window=2, mode="bidir", jepa=True)
    assert torch.equal(model.emb_ema.weight, model.emb.weight)
    assert not model.emb_ema.weight.requires_grad
    assert not hasattr(model, "out")
    with torch.no_grad():
        model.emb.weight.add_(1.0)
    before = model.emb_ema.weight.clone()
    model.update_ema(0.9)
    torch.testing.assert_close(model.emb_ema.weight, 0.9 * before + 0.1 * model.emb.weight)


def test_non_jepa_out_is_zero_initialized():
    model = W2V(30, 8, window=2, mode="causal", jepa=False)
    assert model.out.weight.shape == (30, 8) and torch.count_nonzero(model.out.weight) == 0
    assert torch.count_nonzero(model.emb.weight[0]) == 0
    assert model.emb.weight.abs().max() <= 0.5 / 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'w2v.model'`.

- [ ] **Step 3: Implement** — `w2v/model.py`

```python
"""Word2vec with a fixed, closed-form convolutional context average."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from w2v.data import PAD_ID

MODES = ("bidir", "causal")


def context_kernel(window: int, mode: str) -> torch.Tensor:
    """bidir: ([1]*k + [0] + [1]*k) / 2k;  causal: ([1]*k + [0] + [0]*k) / k."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if window < 1:
        raise ValueError("window must be >= 1")
    right = torch.ones(window) if mode == "bidir" else torch.zeros(window)
    w = torch.cat([torch.ones(window), torch.zeros(1), right])
    return w / w.sum()


class W2V(nn.Module):
    def __init__(self, vocab_size: int, dim: int, window: int, mode: str, jepa: bool):
        super().__init__()
        self.dim, self.window, self.mode, self.jepa = dim, window, mode, jepa
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        nn.init.uniform_(self.emb.weight, -0.5 / dim, 0.5 / dim)
        with torch.no_grad():
            self.emb.weight[PAD_ID].zero_()
        kernel = context_kernel(window, mode)
        self.register_buffer("kernel", kernel, persistent=False)
        self.register_buffer("support", (kernel > 0).float(), persistent=False)
        if jepa:
            self.predictor = nn.Linear(dim, dim)
            self.emb_ema = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
            self.emb_ema.weight.data.copy_(self.emb.weight.data)
            self.emb_ema.weight.requires_grad_(False)
        else:
            self.out = nn.Linear(dim, vocab_size, bias=False)
            nn.init.zeros_(self.out.weight)

    def context(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B,T) ids, m (B,T) bool -> ctx (B,T,d) closed-form window average, has_ctx (B,T) bool."""
        e = self.emb(x) * m.unsqueeze(-1).to(self.emb.weight.dtype)
        w = self.kernel.view(1, 1, -1).expand(self.dim, 1, -1)
        ctx = F.conv1d(e.transpose(1, 2), w, padding=self.window, groups=self.dim).transpose(1, 2)
        n = F.conv1d(m.float().unsqueeze(1), self.support.view(1, 1, -1), padding=self.window).squeeze(1)
        return ctx, n > 0.5

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        self.emb_ema.weight.lerp_(self.emb.weight, 1.0 - tau)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add w2v/model.py tests/test_model.py
git commit -m "feat(w2v): closed-form convolutional context encoder"
```

---

### Task 6: Losses (negative sampling, JEPA + VICReg) and `W2V.loss`

**Files:**
- Create: `w2v/losses.py`
- Modify: `w2v/model.py` (add `loss` method and import)
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: `W2V.context`, `W2V.out`, `W2V.predictor`, `W2V.emb_ema` from Task 5.
- Produces: `NS_NEGATIVES = 5`; `masked_mean(v, mask) -> Tensor`; `ns_loss(ctx, targets, valid, out_weight, negatives) -> (loss, {"loss_pos"})`; `vicreg_terms(z) -> (var_loss, cov_loss)`; `jepa_loss(pred, tgt, valid, lambda_var, lambda_cov, n_vicreg) -> (loss, {"loss_cos","loss_var","loss_cov"})`; `normalized_std(z) -> Tensor` (mean per-dim std of L2-normalized rows). `W2V.loss(x, m, negatives=None, lambda_var=1.0, lambda_cov=0.04, n_vicreg=2048) -> (loss: Tensor, parts: dict[str, Tensor])`.

- [ ] **Step 1: Write the failing tests** — `tests/test_losses.py`

```python
import pytest
import torch
import torch.nn.functional as F

from w2v.losses import jepa_loss, normalized_std, ns_loss, vicreg_terms
from w2v.model import W2V


def test_ns_loss_matches_reference():
    torch.manual_seed(0)
    B, T, d, V, K = 1, 4, 3, 10, 6
    ctx, W = torch.randn(B, T, d), torch.randn(V, d)
    tgt = torch.randint(2, V, (B, T))
    negs = torch.randint(2, V, (K,))
    valid = torch.tensor([[True, False, True, True]])
    loss, parts = ns_loss(ctx, tgt, valid, W, negs)
    ref = []
    for t in (0, 2, 3):
        h = ctx[0, t]
        pos = F.logsigmoid(h @ W[tgt[0, t]])
        neg = sum(F.logsigmoid(-(h @ W[n])) for n in negs)
        ref.append(-(pos + 5 / K * neg))
    torch.testing.assert_close(loss, torch.stack(ref).mean())
    assert "loss_pos" in parts


def test_ns_loss_all_invalid_is_zero_not_nan():
    loss, _ = ns_loss(torch.randn(1, 3, 2), torch.ones(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.bool),
                      torch.randn(4, 2), torch.tensor([2, 3]))
    assert loss.item() == 0.0


def test_vicreg_terms():
    torch.manual_seed(0)
    var_l, _ = vicreg_terms(torch.randn(512, 8) * 3)
    assert var_l.item() == 0.0
    var_l, _ = vicreg_terms(torch.ones(64, 8))
    assert var_l.item() == pytest.approx(0.99, abs=1e-3)
    _, cov_l = vicreg_terms(torch.randn(4096, 4))
    assert cov_l.item() < 0.01
    z = torch.randn(4096, 4)
    z[:, 1] = z[:, 0]
    assert vicreg_terms(z)[1].item() > 0.3


def test_jepa_loss_cosine_part():
    torch.manual_seed(0)
    tgt = torch.randn(1, 5, 4)
    valid = torch.ones(1, 5, dtype=torch.bool)
    loss, parts = jepa_loss(2 * tgt, tgt, valid, 0.0, 0.0, 8)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    loss, parts = jepa_loss(-tgt, tgt, valid, 0.0, 0.0, 8)
    assert parts["loss_cos"].item() == pytest.approx(2.0, abs=1e-5)


def test_normalized_std():
    assert normalized_std(torch.ones(10, 4)).item() == pytest.approx(0.0, abs=1e-6)
    assert normalized_std(torch.randn(1000, 16)).item() > 0.1


def test_model_ns_loss_backprop():
    torch.manual_seed(0)
    model = W2V(30, 8, window=2, mode="bidir", jepa=False)
    x = torch.randint(2, 30, (2, 12))
    m = torch.rand(2, 12) > 0.2
    loss, parts = model.loss(x, m, torch.randint(2, 30, (16,)))
    loss.backward()
    assert torch.isfinite(loss) and model.out.weight.grad.abs().sum() > 0


def test_masked_target_does_not_contribute():
    torch.manual_seed(0)
    model = W2V(30, 8, window=2, mode="causal", jepa=False)
    torch.nn.init.normal_(model.out.weight)
    x = torch.randint(2, 30, (1, 12))
    m = torch.ones(1, 12, dtype=torch.bool)
    m[0, 5] = False
    negs = torch.randint(2, 30, (16,))
    x2 = x.clone()
    x2[0, 5] = (x[0, 5] - 2 + 1) % 28 + 2
    torch.testing.assert_close(model.loss(x, m, negs)[0], model.loss(x2, m, negs)[0])


def test_model_jepa_loss_grads_skip_ema():
    torch.manual_seed(0)
    model = W2V(30, 8, window=2, mode="causal", jepa=True)
    x = torch.randint(2, 30, (2, 12))
    m = torch.ones(2, 12, dtype=torch.bool)
    loss, parts = model.loss(x, m, None, n_vicreg=16)
    loss.backward()
    assert set(parts) == {"loss_cos", "loss_var", "loss_cov"}
    assert model.emb_ema.weight.grad is None
    assert model.emb.weight.grad is not None and model.predictor.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'w2v.losses'`.

- [ ] **Step 3: Implement** — `w2v/losses.py`

```python
"""Training objectives: word2vec negative sampling and JEPA latent prediction (+ VICReg)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

NS_NEGATIVES = 5  # word2vec's negatives per position; the shared-K term is rescaled to match


def masked_mean(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(v.dtype)
    return (v * mask).sum() / mask.sum().clamp(min=1)


def ns_loss(ctx, targets, valid, out_weight, negatives):
    """ctx (B,T,d), targets (B,T), valid (B,T) bool, out_weight (V,d), negatives (K,) shared by the batch."""
    pos = (ctx * out_weight[targets]).sum(-1).float()
    neg = (ctx @ out_weight[negatives].T).float()
    k = negatives.numel()
    per = -(F.logsigmoid(pos) + (NS_NEGATIVES / k) * F.logsigmoid(-neg).sum(-1))
    return masked_mean(per, valid), {"loss_pos": masked_mean(-F.logsigmoid(pos), valid).detach()}


def vicreg_terms(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.float()
    n, d = z.shape
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    zc = z - z.mean(dim=0)
    cov = zc.T @ zc / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return var_loss, off_diag.pow(2).sum() / d


def jepa_loss(pred, tgt, valid, lambda_var: float, lambda_cov: float, n_vicreg: int):
    """pred (B,T,d) predictor output, tgt (B,T,d) EMA target embeddings (no grad)."""
    loss_cos = masked_mean(1.0 - F.cosine_similarity(pred.float(), tgt.float(), dim=-1), valid)
    weights = valid.flatten().float() + 1e-12  # never all-zero, keeps shapes static
    idx = torch.multinomial(weights, n_vicreg, replacement=True)
    var_l, cov_l = vicreg_terms(pred.flatten(0, 1)[idx])
    loss = loss_cos + lambda_var * var_l + lambda_cov * cov_l
    return loss, {"loss_cos": loss_cos.detach(), "loss_var": var_l.detach(), "loss_cov": cov_l.detach()}


def normalized_std(z: torch.Tensor) -> torch.Tensor:
    """Mean per-dimension std of L2-normalized rows; ~0 means collapse, ~1/sqrt(d) is healthy."""
    return F.normalize(z.float(), dim=-1).std(dim=0).mean()
```

- [ ] **Step 4: Add `loss` to `W2V`** — in `w2v/model.py` add `from w2v.losses import jepa_loss, ns_loss` to the imports and this method to the class:

```python
    def loss(self, x, m, negatives=None, lambda_var: float = 1.0, lambda_cov: float = 0.04, n_vicreg: int = 2048):
        """Returns (scalar loss, dict of detached parts). Loss positions: real target with >= 1 context token."""
        ctx, has_ctx = self.context(x, m)
        valid = m & has_ctx
        if self.jepa:
            with torch.no_grad():
                tgt = self.emb_ema(x)
            return jepa_loss(self.predictor(ctx), tgt, valid, lambda_var, lambda_cov, n_vicreg)
        return ns_loss(ctx, x, valid, self.out.weight, negatives)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_losses.py tests/test_model.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add w2v/losses.py w2v/model.py tests/test_losses.py
git commit -m "feat(w2v): negative-sampling and JEPA objectives"
```

---

### Task 7: ONNX export, verification, loading

**Files:**
- Create: `w2v/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `w2v.data.{PAD_ID, Vocab}`.
- Produces: `export_onnx(weight: Tensor (V,d), path) -> Path`; `verify_onnx(path, weight, atol=1e-5) -> float` (raises `RuntimeError` on mismatch); `load_onnx_embeddings(run_dir, chunk=65536) -> (list[str], np.ndarray (V,d))` reading `run_dir/model.onnx` + `run_dir/vocab.json`; `write_vec(path, words, matrix)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_export.py`

```python
import numpy as np
import onnxruntime as ort
import pytest
import torch

from w2v.data import Vocab
from w2v.export import export_onnx, load_onnx_embeddings, verify_onnx, write_vec


def test_export_and_verify_with_dynamic_shapes(tmp_path):
    torch.manual_seed(0)
    w = torch.randn(40, 6)
    path = export_onnx(w, tmp_path / "m.onnx")
    assert verify_onnx(path, w) <= 1e-5
    sess = ort.InferenceSession(str(path))
    assert [i.name for i in sess.get_inputs()] == ["ids"]
    assert [o.name for o in sess.get_outputs()] == ["embeddings"]
    out = sess.run(None, {"ids": np.array([[0, 5, 39]], dtype=np.int64)})[0]
    assert out.shape == (1, 3, 6)
    np.testing.assert_allclose(out[0, 1], w[5].numpy(), atol=1e-6)
    assert sess.run(None, {"ids": np.zeros((4, 13), dtype=np.int64)})[0].shape == (4, 13, 6)


def test_verify_onnx_detects_mismatch(tmp_path):
    w = torch.randn(10, 3)
    path = export_onnx(w, tmp_path / "m.onnx")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_onnx(path, w + 1.0)


def test_load_onnx_embeddings_and_write_vec(tmp_path):
    vocab = Vocab(["<pad>", "<unk>", "a", "b", "c"], [0, 0, 3, 2, 1])
    w = torch.randn(5, 4)
    export_onnx(w, tmp_path / "model.onnx")
    vocab.save(tmp_path / "vocab.json")
    words, mat = load_onnx_embeddings(tmp_path, chunk=2)
    assert words == vocab.words
    np.testing.assert_allclose(mat, w.numpy(), atol=1e-6)
    write_vec(tmp_path / "v.vec", words, mat)
    lines = (tmp_path / "v.vec").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "5 4" and lines[3].startswith("a ") and len(lines) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'w2v.export'`.

- [ ] **Step 3: Implement** — `w2v/export.py`

```python
"""ONNX artifact: embedding lookup ids (batch, seq) int64 -> embeddings (batch, seq, dim) float32."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from w2v.data import PAD_ID, Vocab


class EmbeddingLookup(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=PAD_ID)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


def export_onnx(weight: torch.Tensor, path) -> Path:
    path = Path(path)
    model = EmbeddingLookup(weight.detach().float().cpu()).eval()
    example = torch.zeros((2, 3), dtype=torch.long)
    dims = {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq")}
    torch.onnx.export(model, (example,), str(path), input_names=["ids"], output_names=["embeddings"],
                      dynamic_shapes={"ids": dims}, dynamo=True)
    return path


def verify_onnx(path, weight: torch.Tensor, atol: float = 1e-5) -> float:
    import onnxruntime as ort

    weight = weight.detach().float().cpu()
    v = weight.shape[0]
    ids = np.random.default_rng(0).integers(0, v, size=(3, 17)).astype(np.int64)
    ids[0, 0], ids[0, 1] = 0, v - 1
    out = ort.InferenceSession(str(path)).run(None, {"ids": ids})[0]
    diff = float(np.abs(out - weight[torch.from_numpy(ids)].numpy()).max())
    if diff > atol:
        raise RuntimeError(f"ONNX/PyTorch mismatch: max abs diff {diff} > {atol}")
    return diff


def load_onnx_embeddings(run_dir, chunk: int = 65536) -> tuple[list[str], np.ndarray]:
    import onnxruntime as ort

    run_dir = Path(run_dir)
    vocab = Vocab.load(run_dir / "vocab.json")
    sess = ort.InferenceSession(str(run_dir / "model.onnx"))
    parts = [sess.run(None, {"ids": np.arange(s, min(s + chunk, len(vocab)), dtype=np.int64)[None]})[0][0]
             for s in range(0, len(vocab), chunk)]
    return vocab.words, np.concatenate(parts)


def write_vec(path, words: list[str], matrix: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(words)} {matrix.shape[1]}\n")
        for w, row in zip(words, matrix):
            fh.write(w + " " + " ".join(f"{x:.6g}" for x in row) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_export.py -v`
Expected: 3 PASS (export logs `[torch.onnx] ... ✅` lines).

- [ ] **Step 5: Commit**

```bash
git add w2v/export.py tests/test_export.py
git commit -m "feat(w2v): ONNX embedding-lookup export with parity check"
```

---

### Task 8: Intrinsic evaluation

**Files:**
- Create: `w2v/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `w2v.export.load_onnx_embeddings`.
- Benchmark dir format (written by Task 9): `<data>/benchmarks/<lang>/sim_<name>.tsv` lines `w1\tw2\tscore`; `analogy_<name>.tsv` lines `category\ta\tb\tc\td` (a:b :: c:d, predict d). Analogy categories starting with `gram` count as syntactic, the rest semantic.
- Produces: `load_benchmarks(bench_dir) -> {"sim": {name: [(w1,w2,score)]}, "analogy": {name: [(cat,a,b,c,d)]}}` (warns + empty if dir missing); `eval_similarity(pairs, index, matrix) -> {"spearman": float|None, "coverage": float, "n": int}`; `eval_analogy(items, index, matrix, batch=512) -> {"acc", "coverage", "n", optionally "semantic_acc", "syntactic_acc"}`; `evaluate_all(benchmarks, index, matrix) -> dict[str, float|int|None]` with flat keys `"<name>/<metric>"`; CLI `python -m w2v.evaluate RUN_DIR --data-dir data`.

- [ ] **Step 1: Write the failing tests** — `tests/test_evaluate.py`

```python
import pytest
import torch

from w2v.evaluate import eval_analogy, eval_similarity, evaluate_all, load_benchmarks

WORDS = ["<pad>", "<unk>", "king", "queen", "man", "woman", "apple"]
INDEX = {w: i for i, w in enumerate(WORDS)}
MATRIX = torch.tensor([
    [0.0, 0.0, 0.0], [0.3, 0.3, 0.3],
    [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.1], [0.0, -1.0, 0.1], [0.0, 0.0, 1.0],
])


def test_eval_analogy_3cosadd_with_groups_and_coverage():
    items = [("capital", "man", "king", "woman", "queen"),
             ("gram1", "man", "king", "woman", "apple"),
             ("capital", "man", "king", "banana", "queen")]
    r = eval_analogy(items, INDEX, MATRIX)
    assert r["n"] == 2 and r["coverage"] == pytest.approx(2 / 3)
    assert r["acc"] == 0.5 and r["semantic_acc"] == 1.0 and r["syntactic_acc"] == 0.0


def test_eval_similarity_spearman():
    index = {"a": 0, "b": 1, "c": 2}
    m = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
    r = eval_similarity([("a", "b", 10.0), ("a", "c", 1.0), ("b", "c", 2.0), ("a", "zzz", 5.0)], index, m)
    assert r["spearman"] == pytest.approx(1.0) and r["n"] == 3 and r["coverage"] == 0.75


def test_zero_coverage_gives_none_not_nan():
    assert eval_similarity([("x", "y", 1.0)], INDEX, MATRIX) == {"spearman": None, "coverage": 0.0, "n": 0}
    r = eval_analogy([("c", "x", "y", "z", "w")], INDEX, MATRIX)
    assert r == {"acc": None, "coverage": 0.0, "n": 0}
    assert eval_similarity([], INDEX, MATRIX)["coverage"] == 0.0


def test_load_benchmarks_and_evaluate_all(tmp_path):
    (tmp_path / "sim_toy.tsv").write_text("king\tqueen\t8\nman\twoman\t7\nking\tapple\t1\n", encoding="utf-8")
    (tmp_path / "analogy_toy.tsv").write_text("capital\tman\tking\twoman\tqueen\n", encoding="utf-8")
    b = load_benchmarks(tmp_path)
    assert b["sim"]["sim_toy"][0] == ("king", "queen", 8.0)
    assert b["analogy"]["analogy_toy"] == [("capital", "man", "king", "woman", "queen")]
    res = evaluate_all(b, INDEX, MATRIX)
    assert {"sim_toy/spearman", "sim_toy/coverage", "sim_toy/n", "analogy_toy/acc",
            "analogy_toy/semantic_acc"} <= set(res)
    assert res["analogy_toy/acc"] == 1.0


def test_load_benchmarks_missing_dir_warns(tmp_path):
    with pytest.warns(UserWarning, match="benchmarks"):
        assert load_benchmarks(tmp_path / "none") == {"sim": {}, "analogy": {}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'w2v.evaluate'`.

- [ ] **Step 3: Implement** — `w2v/evaluate.py`

```python
"""Intrinsic evaluation: word similarity (Spearman) and 3CosAdd analogies.

    python -m w2v.evaluate runs/cs-bidir-d300-s0 --data-dir data
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

N_SPECIAL = 2  # <pad>, <unk> are never analogy answers


def load_benchmarks(bench_dir) -> dict:
    bench_dir = Path(bench_dir)
    out: dict = {"sim": {}, "analogy": {}}
    if not bench_dir.is_dir():
        warnings.warn(f"no benchmarks at {bench_dir}; run scripts.fetch_benchmarks", UserWarning)
        return out
    for f in sorted(bench_dir.glob("sim_*.tsv")):
        rows = [line.split("\t") for line in f.read_text(encoding="utf-8").splitlines() if line]
        out["sim"][f.stem] = [(a, b, float(s)) for a, b, s in rows]
    for f in sorted(bench_dir.glob("analogy_*.tsv")):
        out["analogy"][f.stem] = [tuple(line.split("\t")) for line in f.read_text(encoding="utf-8").splitlines() if line]
    return out


def eval_similarity(pairs, index, matrix: torch.Tensor) -> dict:
    kept = [(index[a], index[b], s) for a, b, s in pairs if a in index and b in index]
    coverage = len(kept) / len(pairs) if pairs else 0.0
    if len(kept) < 2:
        return {"spearman": None, "coverage": coverage, "n": len(kept)}
    m = F.normalize(matrix.float(), dim=-1)
    ia, ib, gold = zip(*kept)
    sims = (m[list(ia)] * m[list(ib)]).sum(-1).cpu().numpy()
    rho = spearmanr(sims, gold)[0]
    return {"spearman": None if rho != rho else float(rho), "coverage": coverage, "n": len(kept)}


@torch.no_grad()
def eval_analogy(items, index, matrix: torch.Tensor, batch: int = 512) -> dict:
    kept = [(cat, *(index[w] for w in ws)) for cat, *ws in items if all(w in index for w in ws)]
    coverage = len(kept) / len(items) if items else 0.0
    if not kept:
        return {"acc": None, "coverage": coverage, "n": 0}
    m = F.normalize(matrix.float(), dim=-1)
    correct = []
    for s in range(0, len(kept), batch):
        chunk = kept[s:s + batch]
        a, b, c, d = (torch.tensor([row[i] for row in chunk], device=m.device) for i in range(1, 5))
        scores = (m[b] - m[a] + m[c]) @ m.T
        scores.scatter_(1, torch.stack([a, b, c], dim=1), float("-inf"))
        scores[:, :N_SPECIAL] = float("-inf")
        correct += (scores.argmax(dim=1) == d).tolist()
    res = {"acc": sum(correct) / len(correct), "coverage": coverage, "n": len(kept)}
    for group, is_group in (("semantic", lambda cat: not cat.startswith("gram")),
                            ("syntactic", lambda cat: cat.startswith("gram"))):
        hits = [ok for row, ok in zip(kept, correct) if is_group(row[0])]
        if hits:
            res[f"{group}_acc"] = sum(hits) / len(hits)
    return res


def evaluate_all(benchmarks: dict, index, matrix: torch.Tensor) -> dict:
    out = {}
    for name, pairs in benchmarks["sim"].items():
        out.update({f"{name}/{k}": v for k, v in eval_similarity(pairs, index, matrix).items()})
    for name, items in benchmarks["analogy"].items():
        out.update({f"{name}/{k}": v for k, v in eval_analogy(items, index, matrix).items()})
    return out


def main(argv=None):
    from w2v.export import load_onnx_embeddings

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)
    lang = json.loads((Path(a.run_dir) / "config.json").read_text())["lang"]
    words, matrix = load_onnx_embeddings(a.run_dir)
    res = evaluate_all(load_benchmarks(Path(a.data_dir) / "benchmarks" / lang),
                       {w: i for i, w in enumerate(words)}, torch.from_numpy(matrix))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add w2v/evaluate.py tests/test_evaluate.py
git commit -m "feat(w2v): similarity and analogy evaluation"
```

---

### Task 9: Benchmark fetching and normalization

**Files:**
- Create: `scripts/fetch_benchmarks.py`
- Test: `tests/test_fetch_benchmarks.py`

**Interfaces:**
- Consumes: `w2v.textproc.{normalize_token, UdpipeLemmatizer, UDPIPE_MODELS}`; UDPipe model in `<data>/models/` (from Task 2 download) for cs/uk.
- Produces: `<data>/benchmarks/raw/<files>`, `<data>/benchmarks/<lang>/{sim_*,analogy_*}.tsv` in the Task 8 format, `<data>/benchmarks/<lang>/manifest.json` (`{name: {"raw": int, "kept": int}}`). Functions: `parse_analogy_txt(text, keep=None) -> list[tuple]`, `parse_vecto_csv(text, strip_pos=False) -> list[tuple]`, `parse_ws353_cs(csv_text) -> list[tuple]`, `normalize_analogies(items, norm) -> list`, `normalize_pairs(pairs, norm) -> list`, `main(argv)`.

Benchmark sources and selection:

| lang | name | source | selection |
|---|---|---|---|
| en | `sim_simlex999`, `sim_ws353_sim`, `sim_ws353_rel`, `sim_men` | `https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/{simlex999,wordsim353-sim,wordsim353-rel,men}.csv` | all; MEN strips `-n/-v/-j` suffixes |
| en | `analogy_google` | `https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt` | all 14 categories (not lemmatized) |
| cs | `analogy_cz_semantic` | `https://raw.githubusercontent.com/Svobikl/cz_corpus/master/corpus/czech_emb_corpus_no_phrase.txt` | `Antonyms-nouns, Antonyms-adjectives, Antonyms-verbs, States-cities, Family-relations` |
| cs | `sim_ws353cs` | `https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/946adaf7-2070-4756-af55-496ad15bb480/content` (zip, member `WordSim353-cs/WordSim353-cs.csv`, cols `cs_word_1, cs_word_2, cs mean`) | all |
| uk | `analogy_uk_semantic` | `https://raw.githubusercontent.com/lang-uk/vecs/master/test/test_vocabulary.txt` (tab-separated) | `country-capital, country-region, family, country-nationality, opposite, currency` |

- [ ] **Step 1: Write the failing tests** — `tests/test_fetch_benchmarks.py`

```python
import json

from scripts.fetch_benchmarks import (normalize_analogies, normalize_pairs, parse_analogy_txt, parse_vecto_csv,
                                      parse_ws353_cs, write_benchmark)


def test_parse_analogy_txt_filters_categories_and_handles_tabs():
    text = ": country-capital\t\t\t\nНігерія\tАбуджа\tДанія\tКопенгаген\n: comparative\t\t\t\na\tb\tc\td\nbroken line\n"
    assert parse_analogy_txt(text) == [("country-capital", "Нігерія", "Абуджа", "Данія", "Копенгаген"),
                                       ("comparative", "a", "b", "c", "d")]
    assert parse_analogy_txt(text, keep={"country-capital"}) == [
        ("country-capital", "Нігерія", "Абуджа", "Данія", "Копенгаген")]


def test_parse_vecto_csv_and_pos_stripping():
    text = ",word1,word2,similarity\n0,sun-n,sunlight-n,50.0\n1,Old,new,0.5\n"
    assert parse_vecto_csv(text, strip_pos=True) == [("sun", "sunlight", 50.0), ("Old", "new", 0.5)]


def test_parse_ws353_cs():
    text = '"","cs_word_1","cs_word_2","cs mean"\n"2","vstupné","lístek",6.78\n'
    assert parse_ws353_cs(text) == [("vstupné", "lístek", 6.78)]


def test_normalize_drops_oov_and_collapsed_items():
    lemmas = {"Byli": "být", "byl": "být", "Praha": "praha", "Brno": "brno", "Česko": "česko", "Morava": "morava"}
    norm = lemmas.get
    items = [("c", "Praha", "Česko", "Brno", "Morava"),   # kept
             ("c", "Byli", "byl", "Brno", "Morava"),      # a == b after lemmatization -> dropped
             ("c", "Praha", "Česko", "Brno", "Xyz")]      # unknown word -> dropped
    assert normalize_analogies(items, norm) == [("c", "praha", "česko", "brno", "morava")]
    pairs = [("Praha", "Brno", 5.0), ("Byli", "byl", 9.0), ("Praha", "Xyz", 1.0)]
    assert normalize_pairs(pairs, norm) == [("praha", "brno", 5.0)]


def test_write_benchmark_and_manifest(tmp_path):
    manifest = {}
    write_benchmark(tmp_path, "sim_toy", [("a", "b", 1.0)], raw_count=3, manifest=manifest)
    write_benchmark(tmp_path, "analogy_toy", [("c", "a", "b", "c", "d")], raw_count=1, manifest=manifest)
    assert (tmp_path / "sim_toy.tsv").read_text() == "a\tb\t1.0\n"
    assert (tmp_path / "analogy_toy.tsv").read_text() == "c\ta\tb\tc\td\n"
    assert manifest == {"sim_toy": {"raw": 3, "kept": 1}, "analogy_toy": {"raw": 1, "kept": 1}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fetch_benchmarks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_benchmarks'`.

- [ ] **Step 3: Implement** — `scripts/fetch_benchmarks.py`

```python
"""Download word-similarity / analogy benchmarks and normalize them like the corpus.

en: lowercased word forms. cs/uk: lemmatized with the same UDPipe model as the corpus;
only semantic analogy categories are kept (lemmatization erases morphological ones).

    python -m scripts.fetch_benchmarks --lang cs --data-dir data
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

from w2v.textproc import UDPIPE_MODELS, UdpipeLemmatizer, normalize_token

VECTO = "https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en"
GOOGLE = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
CZ_ANALOGY = "https://raw.githubusercontent.com/Svobikl/cz_corpus/master/corpus/czech_emb_corpus_no_phrase.txt"
WS353_CS = "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/946adaf7-2070-4756-af55-496ad15bb480/content"
UK_ANALOGY = "https://raw.githubusercontent.com/lang-uk/vecs/master/test/test_vocabulary.txt"
CS_SEMANTIC = {"Antonyms-nouns", "Antonyms-adjectives", "Antonyms-verbs", "States-cities", "Family-relations"}
UK_SEMANTIC = {"country-capital", "country-region", "family", "country-nationality", "opposite", "currency"}
_POS_SUFFIX = re.compile(r"-[nvja]$")


def fetch(url: str, dest: Path) -> bytes:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        if not data or data.lstrip()[:15].lower().startswith(b"<!doctype html"):
            raise RuntimeError(f"{url} did not return the expected file (got HTML or nothing)")
        dest.write_bytes(data)
    return dest.read_bytes()


def parse_analogy_txt(text: str, keep: set[str] | None = None) -> list[tuple]:
    items, cat = [], None
    for line in text.splitlines():
        if line.startswith(":"):
            cat = line[1:].strip()
            continue
        parts = line.split()
        if len(parts) == 4 and cat is not None and (keep is None or cat in keep):
            items.append((cat, *parts))
    return items


def parse_vecto_csv(text: str, strip_pos: bool = False) -> list[tuple]:
    rows = list(csv.DictReader(io.StringIO(text)))
    fix = (lambda w: _POS_SUFFIX.sub("", w)) if strip_pos else (lambda w: w)
    return [(fix(r["word1"]), fix(r["word2"]), float(r["similarity"])) for r in rows]


def parse_ws353_cs(text: str) -> list[tuple]:
    return [(r["cs_word_1"], r["cs_word_2"], float(r["cs mean"])) for r in csv.DictReader(io.StringIO(text))]


def normalize_analogies(items, norm) -> list[tuple]:
    out = []
    for cat, *words in items:
        ws = [norm(w) for w in words]
        if None in ws or ws[0] == ws[1] or ws[2] == ws[3]:
            continue
        out.append((cat, *ws))
    return out


def normalize_pairs(pairs, norm) -> list[tuple]:
    out = []
    for a, b, s in pairs:
        na, nb = norm(a), norm(b)
        if na is None or nb is None or na == nb:
            continue
        out.append((na, nb, s))
    return out


def write_benchmark(out_dir: Path, name: str, rows, raw_count: int, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.tsv").write_text("".join("\t".join(map(str, r)) + "\n" for r in rows), encoding="utf-8")
    manifest[name] = {"raw": raw_count, "kept": len(rows)}
    print(f"{name}: kept {len(rows)}/{raw_count}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["en", "cs", "uk"])
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)
    data_dir = Path(a.data_dir)
    raw = data_dir / "benchmarks" / "raw"
    out_dir = data_dir / "benchmarks" / a.lang
    manifest: dict = {}

    if a.lang == "en":
        norm = normalize_token
        for name, file, strip in (("sim_simlex999", "simlex999", False), ("sim_ws353_sim", "wordsim353-sim", False),
                                  ("sim_ws353_rel", "wordsim353-rel", False), ("sim_men", "men", True)):
            pairs = parse_vecto_csv(fetch(f"{VECTO}/{file}.csv", raw / f"en_{file}.csv").decode(), strip)
            write_benchmark(out_dir, name, normalize_pairs(pairs, norm), len(pairs), manifest)
        items = parse_analogy_txt(fetch(GOOGLE, raw / "questions-words.txt").decode())
        write_benchmark(out_dir, "analogy_google", normalize_analogies(items, norm), len(items), manifest)
    else:
        lemmatizer = UdpipeLemmatizer(data_dir / "models" / UDPIPE_MODELS[a.lang][0])
        norm = lru_cache(maxsize=None)(lemmatizer.lemma)
        if a.lang == "cs":
            items = parse_analogy_txt(fetch(CZ_ANALOGY, raw / "czech_emb_corpus_no_phrase.txt").decode(), CS_SEMANTIC)
            write_benchmark(out_dir, "analogy_cz_semantic", normalize_analogies(items, norm), len(items), manifest)
            with zipfile.ZipFile(io.BytesIO(fetch(WS353_CS, raw / "WordSim353-cs.zip"))) as z:
                pairs = parse_ws353_cs(z.read("WordSim353-cs/WordSim353-cs.csv").decode("utf-8"))
            write_benchmark(out_dir, "sim_ws353cs", normalize_pairs(pairs, norm), len(pairs), manifest)
        else:
            items = parse_analogy_txt(fetch(UK_ANALOGY, raw / "uk_test_vocabulary.txt").decode(), UK_SEMANTIC)
            write_benchmark(out_dir, "analogy_uk_semantic", normalize_analogies(items, norm), len(items), manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fetch_benchmarks.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Real fetch for English (needs network)**

Run: `uv run python -m scripts.fetch_benchmarks --lang en --data-dir data && cat data/benchmarks/en/manifest.json`
Expected: kept counts ≈ simlex999 999, ws353_sim 203, ws353_rel 252, men 3000, google 19544.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_benchmarks.py tests/test_fetch_benchmarks.py
git commit -m "feat(w2v): benchmark fetching with lemmatized cs/uk items"
```

---

### Task 10: Training loop, checkpoint/resume, finalize

**Files:**
- Create: `w2v/train.py`, `tests/conftest.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: everything from Tasks 3–8: `Vocab`, `TokenStream`, `keep_probs`, `sample_mask`, `noise_dist`, `W2V`, `normalized_std`, `export_onnx`, `verify_onnx`, `load_onnx_embeddings`, `write_vec`, `load_benchmarks`, `evaluate_all`; data files `<data>/<lang>/{vocab.json, stats.json, tokens.bin}`.
- Produces: `run_name(lang, mode, jepa, dim, seed) -> str`; `lr_lambda(step, total, warmup) -> float`; `ema_tau(step, total, tau0) -> float`; `parse_args(argv) -> Namespace`; `train(args) -> Path` (run dir). Run dir contents: `config.json`, `metrics.jsonl`, `tb/`, `ckpt.pt`, and after completion `model.onnx`, `vocab.json`, `final_metrics.json` (+ `vectors.vec` with `--write-vec`). `--stop-after N` stops after N steps without finalizing (used for walltime splitting and tests).

- [ ] **Step 1: Write the shared fixture** — `tests/conftest.py`

```python
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
```

- [ ] **Step 2: Write the failing tests** — `tests/test_train.py`

```python
import json

import pytest
import torch

from w2v.train import ema_tau, lr_lambda, parse_args, run_name, train


def test_run_name():
    assert run_name("cs", "causal", True, 300, 0) == "cs-causal-jepa-d300-s0"
    assert run_name("en", "bidir", False, 100, 2) == "en-bidir-d100-s2"


def test_schedules():
    assert lr_lambda(0, 100, 10) == pytest.approx(0.1)
    assert lr_lambda(9, 100, 10) == pytest.approx(1.0)
    assert lr_lambda(100, 100, 10) == pytest.approx(0.0)
    assert ema_tau(0, 100, 0.996) == pytest.approx(0.996)
    assert ema_tau(100, 100, 0.996) == pytest.approx(1.0)


def _args(root, *extra, mode="bidir", jepa=False, steps=6):
    argv = ["--lang", "en", "--mode", mode, "--dim", "16", "--batch", "4", "--seq-len", "64", "--neg", "32",
            "--max-steps", str(steps), "--log-every", "2", "--eval-every", "3", "--ckpt-every", "3",
            "--n-vicreg", "64", "--device", "cpu", "--no-compile",
            "--data-dir", str(root / "data"), "--runs-dir", str(root / "runs"), *extra]
    if jepa:
        argv.append("--jepa")
    return parse_args(argv)


@pytest.mark.parametrize("mode", ["bidir", "causal"])
@pytest.mark.parametrize("jepa", [False, True])
def test_train_end_to_end(tiny_data, mode, jepa):
    run_dir = train(_args(tiny_data, "--write-vec", mode=mode, jepa=jepa))
    assert run_dir.name == run_name("en", mode, jepa, 16, 0)
    for f in ("config.json", "metrics.jsonl", "ckpt.pt", "model.onnx", "vocab.json", "final_metrics.json",
              "vectors.vec"):
        assert (run_dir / f).exists(), f
    final = json.loads((run_dir / "final_metrics.json").read_text())
    assert final["onnx_max_abs_diff"] <= 1e-5 and "sim_toy/spearman" in final and "collapse_warning" in final
    records = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_recs = [r for r in records if "loss" in r]
    assert [r["step"] for r in train_recs] == [2, 4, 6]
    assert all(r["loss"] == r["loss"] for r in train_recs)  # not NaN
    assert any("sim_toy/coverage" in r for r in records)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["dim"] == 16 and cfg["jepa"] == jepa and cfg["vocab_size"] == 52


def test_resume_matches_uninterrupted_run(tiny_data):
    straight = train(_args(tiny_data, "--run-dir", str(tiny_data / "runs/straight")))
    interrupted = tiny_data / "runs/interrupted"
    train(_args(tiny_data, "--run-dir", str(interrupted), "--stop-after", "3"))
    assert torch.load(interrupted / "ckpt.pt", weights_only=False)["step"] == 3
    assert not (interrupted / "model.onnx").exists()
    train(_args(tiny_data, "--run-dir", str(interrupted)))
    a = torch.load(straight / "ckpt.pt", weights_only=False)
    b = torch.load(interrupted / "ckpt.pt", weights_only=False)
    assert a["step"] == b["step"] == 6
    torch.testing.assert_close(a["model"]["emb.weight"], b["model"]["emb.weight"])


def test_resume_refuses_changed_config(tiny_data):
    run_dir = str(tiny_data / "runs/x")
    train(_args(tiny_data, "--run-dir", run_dir, "--stop-after", "1"))
    with pytest.raises(ValueError, match="different config"):
        train(_args(tiny_data, "--run-dir", run_dir, "--lr", "0.1"))


def test_window_wider_than_encoded_pad_is_rejected(tiny_data):
    with pytest.raises(ValueError, match="pad"):
        train(_args(tiny_data, "--window", "6"))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'w2v.train'`.

- [ ] **Step 4: Implement** — `w2v/train.py`

```python
"""Train one word2vec variant and export its ONNX embedding artifact.

    python -m w2v.train --lang cs --mode causal --jepa --dim 300 --seed 0 --data-dir data
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from w2v.data import TokenStream, Vocab, keep_probs, noise_dist, sample_mask
from w2v.evaluate import evaluate_all, load_benchmarks
from w2v.export import export_onnx, load_onnx_embeddings, verify_onnx, write_vec
from w2v.losses import normalized_std
from w2v.model import MODES, W2V

CONFIG_KEYS = ("lang", "mode", "jepa", "dim", "window", "neg", "subsample_t", "lr", "warmup_frac", "batch",
               "seq_len", "epochs", "max_steps", "ema_start", "lambda_var", "lambda_cov", "n_vicreg", "seed")
COLLAPSE_STD = 1e-3


def run_name(lang: str, mode: str, jepa: bool, dim: int, seed: int) -> str:
    return f"{lang}-{mode}{'-jepa' if jepa else ''}-d{dim}-s{seed}"


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_tau(step: int, total: int, tau0: float) -> float:
    return 1.0 - (1.0 - tau0) * (math.cos(math.pi * min(1.0, step / max(1, total))) + 1.0) / 2.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["en", "cs", "uk"])
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--jepa", action="store_true")
    p.add_argument("--dim", type=int, default=300)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--neg", type=int, default=1024, help="shared negatives per batch (non-JEPA)")
    p.add_argument("--subsample-t", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None, help="override the epoch-derived step count")
    p.add_argument("--ema-start", type=float, default=0.996)
    p.add_argument("--lambda-var", type=float, default=1.0)
    p.add_argument("--lambda-cov", type=float, default=0.04)
    p.add_argument("--n-vicreg", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000, help="0 disables periodic evaluation")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--stop-after", type=int, default=None, help="stop after N steps this invocation, no export")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-dir", default=None, help="explicit run directory (default: runs-dir/<run name>)")
    p.add_argument("--device", default="auto")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile the loss (default: on for CUDA)")
    p.add_argument("--write-vec", action="store_true", help="also write vectors.vec")
    return p.parse_args(argv)


def _save_ckpt(path: Path, model, opt, sched, step: int, collapse: bool) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "collapse": collapse}, tmp)
    os.replace(tmp, path)


def _prepare_run_dir(args, run_dir: Path, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = sorted(k for k in (*CONFIG_KEYS, "vocab_size", "total_steps") if old.get(k) != config.get(k))
        if diff:
            raise ValueError(f"run dir {run_dir} has a different config ({', '.join(diff)}); use another --run-dir")
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def train(args) -> Path:
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    use_compile = device.type == "cuda" if args.compile is None else args.compile
    lang_dir = Path(args.data_dir) / args.lang
    vocab = Vocab.load(lang_dir / "vocab.json")
    stats = json.loads((lang_dir / "stats.json").read_text())
    if stats["pad"] < args.window:
        raise ValueError(f"corpus was encoded with pad={stats['pad']} < window={args.window}; "
                         f"re-run scripts.encode with --pad >= {args.window}")
    stream = TokenStream(lang_dir / "tokens.bin", args.seq_len, args.batch, args.seed)
    total = args.max_steps or max(1, int(stream.steps_per_epoch() * args.epochs))
    warmup = max(1, int(total * args.warmup_frac))

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.runs_dir) / run_name(
        args.lang, args.mode, args.jepa, args.dim, args.seed)
    config = {k: getattr(args, k) for k in CONFIG_KEYS}
    config.update(vocab_size=len(vocab), total_steps=total, data_stats=stats)
    _prepare_run_dir(args, run_dir, config)

    torch.manual_seed(args.seed)
    model = W2V(len(vocab), args.dim, args.window, args.mode, args.jepa).to(device)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("predictor.")]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("predictor.")]
    opt = torch.optim.AdamW([{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": 0.01}],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, warmup))
    keep = keep_probs(vocab.counts, args.subsample_t).to(device)
    noise = noise_dist(vocab.counts).to(device)

    ckpt_path = run_dir / "ckpt.pt"
    start, collapse = 0, False
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, collapse = state["step"], state["collapse"]
        print(f"resumed {run_dir} at step {start}/{total}")

    loss_fn = torch.compile(model.loss) if use_compile else model.loss
    benchmarks = load_benchmarks(Path(args.data_dir) / "benchmarks" / args.lang)
    writer = SummaryWriter(str(run_dir / "tb"))
    end = total if args.stop_after is None else min(total, start + args.stop_after)
    t0, tokens_seen = time.time(), 0

    with open(run_dir / "metrics.jsonl", "a") as metrics_f:
        def log(record: dict) -> None:
            metrics_f.write(json.dumps(record) + "\n")
            metrics_f.flush()
            for k, v in record.items():
                if k != "step" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    writer.add_scalar(k, v, record["step"])

        for step in range(start, end):
            model.train()
            x = stream.batch(step).to(device, non_blocking=True)
            torch.manual_seed(args.seed * 1_000_003 + step)  # mask, negatives, VICReg sample: resumable
            m = sample_mask(x, keep)
            negs = None if args.jepa else torch.multinomial(noise, args.neg, replacement=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, parts = loss_fn(x, m, negs, args.lambda_var, args.lambda_cov, args.n_vicreg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if args.jepa:
                model.update_ema(ema_tau(step + 1, total, args.ema_start))
            tokens_seen += x.numel()
            done = step + 1

            if done % args.log_every == 0 or done == total:
                loss_v = loss.item()
                if not math.isfinite(loss_v):
                    raise FloatingPointError(f"non-finite loss {loss_v} at step {done}")
                record = {"step": done, "loss": loss_v, "lr": sched.get_last_lr()[0],
                          "tokens_per_s": tokens_seen / max(time.time() - t0, 1e-9)}
                record.update({k: v.item() for k, v in parts.items()})
                with torch.no_grad():
                    flat = x.flatten()[m.flatten()][:4096]
                    record["emb_std"] = normalized_std(model.emb(flat)).item()
                    if args.jepa:
                        ctx, _ = model.context(x, m)
                        record["pred_std"] = normalized_std(model.predictor(ctx.flatten(0, 1)[:4096])).item()
                if args.jepa and min(record["emb_std"], record["pred_std"]) < COLLAPSE_STD:
                    if not collapse:
                        warnings.warn(f"representation collapse suspected at step {done}")
                    collapse = True
                    record["collapse_warning"] = True
                log(record)
                t0, tokens_seen = time.time(), 0
            if args.eval_every and done % args.eval_every == 0:
                ev = evaluate_all(benchmarks, vocab.index, model.emb.weight.detach().float())
                log({"step": done, **ev})
            if done % args.ckpt_every == 0:
                _save_ckpt(ckpt_path, model, opt, sched, done, collapse)

        _save_ckpt(ckpt_path, model, opt, sched, end, collapse)
    if end < total:
        print(f"stopped at step {end}/{total}; re-run the same command to resume")
        writer.close()
        return run_dir

    weight = model.emb.weight.detach().float().cpu()
    onnx_path = export_onnx(weight, run_dir / "model.onnx")
    diff = verify_onnx(onnx_path, weight)
    vocab.save(run_dir / "vocab.json")
    words, matrix = load_onnx_embeddings(run_dir)
    final = {"onnx_max_abs_diff": diff, "collapse_warning": collapse, "steps": total,
             **evaluate_all(benchmarks, {w: i for i, w in enumerate(words)}, torch.from_numpy(matrix))}
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    if args.write_vec:
        write_vec(run_dir / "vectors.vec", words, matrix)
    writer.close()
    print(json.dumps(final, indent=2))
    return run_dir


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: 9 PASS (4 end-to-end variants + resume + config guard + pad guard + 2 unit). If the resume test fails on weight equality, debug with superpowers:systematic-debugging — the data batch, mask, negatives and VICReg sample are all seeded per step, so any difference is a real bug in checkpoint state.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: all PASS (UDPipe test passes if the model was downloaded in Task 2).

- [ ] **Step 7: Commit**

```bash
git add w2v/train.py tests/conftest.py tests/test_train.py
git commit -m "feat(w2v): training loop with resume, ONNX export and evaluation"
```

---

### Task 11: Results comparison

**Files:**
- Create: `scripts/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: run dirs with `config.json` (`lang, mode, jepa, dim, seed`) and `final_metrics.json` (Task 10).
- Produces: `collect(runs_dir) -> list[dict]` (one row per finished run, keys `run, lang, variant, dim, seed` + metrics); `main(argv)` writing `<out>/summary.csv` and `<out>/summary.md`.

- [ ] **Step 1: Write the failing tests** — `tests/test_compare.py`

```python
import csv
import json

from scripts.compare import collect, main


def _run(root, name, cfg, metrics):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    if metrics is not None:
        (d / "final_metrics.json").write_text(json.dumps(metrics))


def test_collect_and_write_tables(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "cs-bidir-jepa-d300-s0", {"lang": "cs", "mode": "bidir", "jepa": True, "dim": 300, "seed": 0},
         {"analogy_cz_semantic/acc": 0.41234, "collapse_warning": False})
    _run(runs, "en-causal-d100-s0", {"lang": "en", "mode": "causal", "jepa": False, "dim": 100, "seed": 0},
         {"sim_simlex999/spearman": 0.3})
    _run(runs, "unfinished", {"lang": "en", "mode": "bidir", "jepa": False, "dim": 100, "seed": 0}, None)
    rows = collect(runs)
    assert [r["variant"] for r in rows] == ["bidir-jepa", "causal"]
    main(["--runs-dir", str(runs), "--out", str(tmp_path / "results")])
    with open(tmp_path / "results/summary.csv") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 2 and table[0]["lang"] == "cs" and table[0]["dim"] == "300"
    md = (tmp_path / "results/summary.md").read_text()
    assert "| lang |" in md and "0.412" in md and "–" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.compare'`.

- [ ] **Step 3: Implement** — `scripts/compare.py`

```python
"""Aggregate final_metrics.json of all finished runs into summary.csv / summary.md.

    python -m scripts.compare --runs-dir runs --out results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ID_COLS = ["run", "lang", "variant", "dim", "seed"]


def collect(runs_dir) -> list[dict]:
    rows = []
    for metrics_path in sorted(Path(runs_dir).glob("*/final_metrics.json")):
        cfg = json.loads((metrics_path.parent / "config.json").read_text())
        metrics = json.loads(metrics_path.read_text())
        variant = cfg["mode"] + ("-jepa" if cfg["jepa"] else "")
        rows.append({"run": metrics_path.parent.name, "lang": cfg["lang"], "variant": variant,
                     "dim": cfg["dim"], "seed": cfg["seed"], **metrics})
    rows.sort(key=lambda r: (r["lang"], r["variant"], r["dim"], r["seed"]))
    return rows


def _fmt(v) -> str:
    if v is None or v == "":
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    rows = collect(a.runs_dir)
    if not rows:
        raise SystemExit(f"no finished runs (final_metrics.json) under {a.runs_dir}")
    metric_cols = sorted({k for r in rows for k in r} - set(ID_COLS))
    cols = ID_COLS + metric_cols
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md_cols = cols[1:]
    lines = ["| " + " | ".join(md_cols) + " |", "|" + "---|" * len(md_cols)]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in md_cols) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} runs -> {out}/summary.csv, {out}/summary.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compare.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/compare.py tests/test_compare.py
git commit -m "feat(w2v): results comparison table"
```

---

### Task 12: MetaCentrum jobs, README, local real-data smoke run

**Files:**
- Create: `jobs/config.sh`, `jobs/setup_env.sh`, `jobs/download.pbs`, `jobs/prep.pbs`, `jobs/vocab.pbs`, `jobs/train.pbs`, `jobs/submit_all.sh`, `README.md`

**Interfaces:**
- Consumes: CLIs `scripts.download`, `scripts.preprocess`, `scripts.build_vocab`, `scripts.encode`, `scripts.fetch_benchmarks`, `w2v.train`, `scripts.compare`.
- Job env vars: `W2V_LANG` (never `LANG`, which is the locale), `NUM_SHARDS`, `MODE`, `JEPA` (0/1), `DIM`, `SEED`, `EPOCHS`. Jobs are submitted from the project dir, so `$PBS_O_WORKDIR` is the code checkout.

- [ ] **Step 1: Write** `jobs/config.sh`

```bash
# Sourced by every job script. Edit W2V_STORAGE / W2V_CODE for your account.
export W2V_STORAGE="${W2V_STORAGE:-/storage/brno2/home/$USER/w2v}"
export W2V_CODE="${W2V_CODE:-$W2V_STORAGE/datalab/word2vec}"
export W2V_DATA="$W2V_STORAGE/data"
export W2V_RUNS="$W2V_STORAGE/runs"
export UV_CACHE_DIR="$W2V_STORAGE/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$W2V_STORAGE/.uv-python"
export UV_PROJECT_ENVIRONMENT="$W2V_STORAGE/.venv"
export UV_LINK_MODE=copy
export HF_HOME="$W2V_STORAGE/.hf"
export PATH="$W2V_STORAGE/.local/bin:$PATH"

# Preprocessing array size per language (one PBS subjob per shard).
declare -A W2V_SHARDS=([en]=64 [cs]=16 [uk]=32)

# Run project Python without re-syncing the shared environment.
w2v_py() { uv run --frozen --no-sync python "$@"; }
```

- [ ] **Step 2: Write** `jobs/setup_env.sh`

```bash
#!/bin/bash
# One-time setup on a MetaCentrum frontend: installs uv and the locked environment on storage.
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/config.sh
mkdir -p "$W2V_STORAGE/.local/bin" "$W2V_DATA" "$W2V_RUNS"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$W2V_STORAGE/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv sync --frozen --extra prep
w2v_py -c "import torch, onnxruntime, ufal.udpipe; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
```

- [ ] **Step 3: Write** `jobs/download.pbs`

```bash
#!/bin/bash
#PBS -N w2v-download
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=6:00:00
set -euo pipefail
cd "$PBS_O_WORKDIR"
source jobs/config.sh
: "${W2V_LANG:?set W2V_LANG}"
w2v_py -m scripts.download --lang "$W2V_LANG" --data-dir "$W2V_DATA"
w2v_py -m scripts.fetch_benchmarks --lang "$W2V_LANG" --data-dir "$W2V_DATA"
```

- [ ] **Step 4: Write** `jobs/prep.pbs`

```bash
#!/bin/bash
#PBS -N w2v-prep
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=8:00:00
set -euo pipefail
cd "$PBS_O_WORKDIR"
source jobs/config.sh
: "${W2V_LANG:?set W2V_LANG}" "${NUM_SHARDS:?set NUM_SHARDS}" "${PBS_ARRAY_INDEX:?submit as an array job (-J)}"
w2v_py -m scripts.preprocess --lang "$W2V_LANG" --shard "$PBS_ARRAY_INDEX" --num-shards "$NUM_SHARDS" \
    --workers "${PBS_NCPUS:-8}" --data-dir "$W2V_DATA"
```

- [ ] **Step 5: Write** `jobs/vocab.pbs`

```bash
#!/bin/bash
#PBS -N w2v-vocab
#PBS -l select=1:ncpus=2:mem=48gb
#PBS -l walltime=8:00:00
set -euo pipefail
cd "$PBS_O_WORKDIR"
source jobs/config.sh
: "${W2V_LANG:?set W2V_LANG}" "${NUM_SHARDS:?set NUM_SHARDS}"
CAP=()
if [[ "$W2V_LANG" == en ]]; then
    CAP=(--cap-from "$W2V_DATA/cs/stats.json" "$W2V_DATA/uk/stats.json")
fi
w2v_py -m scripts.build_vocab --lang "$W2V_LANG" --num-shards "$NUM_SHARDS" --data-dir "$W2V_DATA" "${CAP[@]}"
w2v_py -m scripts.encode --lang "$W2V_LANG" --pad 5 --data-dir "$W2V_DATA"
```

- [ ] **Step 6: Write** `jobs/train.pbs`

```bash
#!/bin/bash
#PBS -N w2v-train
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:gpu_mem=20gb:mem=32gb:scratch_local=40gb
#PBS -l walltime=4:00:00
set -euo pipefail
cd "$PBS_O_WORKDIR"
source jobs/config.sh
: "${W2V_LANG:?}" "${MODE:?}" "${DIM:?}" "${SEED:?}"
JEPA_FLAG=()
SUFFIX=""
if [[ "${JEPA:-0}" == 1 ]]; then JEPA_FLAG=(--jepa); SUFFIX="-jepa"; fi
NAME="${W2V_LANG}-${MODE}${SUFFIX}-d${DIM}-s${SEED}"  # mirrors w2v.train.run_name

D="$SCRATCHDIR/data"
mkdir -p "$D/$W2V_LANG" "$D/benchmarks" "$SCRATCHDIR/runs"
cp "$W2V_DATA/$W2V_LANG"/{tokens.bin,vocab.json,stats.json} "$D/$W2V_LANG/"
if [[ -d "$W2V_DATA/benchmarks/$W2V_LANG" ]]; then cp -r "$W2V_DATA/benchmarks/$W2V_LANG" "$D/benchmarks/"; fi
if [[ -d "$W2V_RUNS/$NAME" ]]; then cp -r "$W2V_RUNS/$NAME" "$SCRATCHDIR/runs/"; fi

copy_back() {
    mkdir -p "$W2V_RUNS"
    if cp -r "$SCRATCHDIR/runs/$NAME" "$W2V_RUNS/"; then
        clean_scratch || true
    else
        echo "copy back failed; results remain in $SCRATCHDIR on $(hostname)" >&2
    fi
}
trap copy_back EXIT
trap 'exit 143' TERM

nvidia-smi || true
w2v_py -m w2v.train --lang "$W2V_LANG" --mode "$MODE" "${JEPA_FLAG[@]}" --dim "$DIM" --seed "$SEED" \
    --epochs "${EPOCHS:-1}" --data-dir "$D" --run-dir "$SCRATCHDIR/runs/$NAME"
```

- [ ] **Step 7: Write** `jobs/submit_all.sh`

```bash
#!/bin/bash
# Submit data preparation and the training grid. Run from anywhere on a MetaCentrum frontend.
#   LANGS="en cs uk" DIMS="100 300" SEEDS="0 1" EPOCHS=1 SKIP_PREP=0 jobs/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/config.sh
LANGS=${LANGS:-"en cs uk"}
DIMS=${DIMS:-300}
SEEDS=${SEEDS:-0}
EPOCHS=${EPOCHS:-1}
VARIANTS=${VARIANTS:-"bidir causal bidir-jepa causal-jepa"}
SKIP_PREP=${SKIP_PREP:-0}

declare -A VOCAB_JOB=()
prep_lang() {  # $1 = lang, $2 = extra dependency string ("" or ":id:id")
    local L=$1 N=${W2V_SHARDS[$1]} dl prep
    dl=$(qsub -v W2V_LANG="$L" jobs/download.pbs)
    prep=$(qsub -W depend=afterok:"$dl" -J 0-$((N - 1)) -v W2V_LANG="$L",NUM_SHARDS="$N" jobs/prep.pbs)
    VOCAB_JOB[$L]=$(qsub -W depend=afterok:"$prep$2" -v W2V_LANG="$L",NUM_SHARDS="$N" jobs/vocab.pbs)
    echo "$L: download=$dl prep=$prep vocab=${VOCAB_JOB[$L]}"
}

if [[ "$SKIP_PREP" != 1 ]]; then
    for L in cs uk; do [[ " $LANGS " == *" $L "* ]] && prep_lang "$L" ""; done
    if [[ " $LANGS " == *" en "* ]]; then
        EN_DEPS=""
        for L in cs uk; do
            if [[ -n "${VOCAB_JOB[$L]:-}" ]]; then
                EN_DEPS+=":${VOCAB_JOB[$L]}"
            elif [[ ! -f "$W2V_DATA/$L/stats.json" ]]; then
                echo "en needs $L stats for its token cap: include $L in LANGS or prepare it first" >&2
                exit 1
            fi
        done
        prep_lang en "$EN_DEPS"
    fi
fi

for L in $LANGS; do
    DEP=()
    if [[ -n "${VOCAB_JOB[$L]:-}" ]]; then DEP=(-W depend=afterok:"${VOCAB_JOB[$L]}"); fi
    for V in $VARIANTS; do
        MODE=${V%-jepa}
        JEPA=0
        [[ "$V" == *-jepa ]] && JEPA=1
        for D in $DIMS; do
            for S in $SEEDS; do
                id=$(qsub "${DEP[@]}" -N "w2v-$L-$V-d$D-s$S" \
                    -v W2V_LANG="$L",MODE="$MODE",JEPA="$JEPA",DIM="$D",SEED="$S",EPOCHS="$EPOCHS" jobs/train.pbs)
                echo "train $L $V d$D s$S: $id"
            done
        done
    done
done
```

- [ ] **Step 8: Syntax-check the shell scripts**

Run: `chmod +x jobs/*.sh && for f in jobs/*.sh jobs/*.pbs; do bash -n "$f" && echo "ok $f"; done; command -v shellcheck && shellcheck -x jobs/*.sh jobs/*.pbs || true`
Expected: `ok` for all 7 files; shellcheck (if installed) reports no errors (warnings about `PBS_*` variables being unassigned are fine).

- [ ] **Step 9: Write** `README.md`

````markdown
# word2vec variants: bidirectional / causal × negative sampling / JEPA

Word2vec on Wikipedia (en, cs, uk — cs/uk lemmatized with UDPipe) with a fixed
convolutional context average, trained four ways:

| variant | context | objective |
|---|---|---|
| `bidir` | k left + k right | negative sampling |
| `causal` | k left | negative sampling |
| `bidir-jepa` | k left + k right | predict EMA target embedding (cosine + VICReg) |
| `causal-jepa` | k left | predict EMA target embedding (cosine + VICReg) |

Each run writes `runs/<lang>-<variant>-d<dim>-s<seed>/model.onnx`
(`ids (batch, seq) int64 -> embeddings (batch, seq, dim)`), `vocab.json`, `config.json`,
`metrics.jsonl`, TensorBoard logs and `final_metrics.json`.
Design: `docs/superpowers/specs/2026-10-13-word2vec-variants-design.md`.

## Local (CPU)

```bash
uv sync --extra prep
uv run pytest
# tiny real-data pipeline (first 1000 Czech articles)
uv run python -m scripts.download --lang cs --tiny --data-dir data-tiny
uv run python -m scripts.preprocess --lang cs --shard 0 --num-shards 1 --workers 4 --data-dir data-tiny
uv run python -m scripts.build_vocab --lang cs --num-shards 1 --min-count 2 --data-dir data-tiny
uv run python -m scripts.encode --lang cs --pad 5 --data-dir data-tiny
uv run python -m scripts.fetch_benchmarks --lang cs --data-dir data-tiny
uv run python -m w2v.train --lang cs --mode causal --jepa --dim 64 --batch 8 --seq-len 256 \
    --max-steps 50 --data-dir data-tiny --runs-dir runs-tiny
```

## MetaCentrum

```bash
# on a frontend, inside your storage home
git clone <this repo> "$W2V_STORAGE/datalab" && cd "$W2V_STORAGE/datalab/word2vec"
$EDITOR jobs/config.sh             # set W2V_STORAGE (and W2V_CODE if cloned elsewhere)
jobs/setup_env.sh                  # once: uv + locked environment on storage
jobs/submit_all.sh                 # download -> preprocess (array) -> vocab/encode -> 12 training jobs
DIMS="100 300 500" SKIP_PREP=1 jobs/submit_all.sh   # more sizes once data exists
qstat -u "$USER"
uv run --frozen --no-sync python -m scripts.compare --runs-dir "$W2V_RUNS" --out "$W2V_STORAGE/results"
```

Training jobs resume from `ckpt.pt` when resubmitted (same variables). English is capped
to the larger of the Czech/Ukrainian token counts, so its vocab job waits for both.
Czech/Ukrainian analogy benchmarks keep only semantic categories (lemmatization removes
the morphological ones); there is no Ukrainian similarity set.
````

- [ ] **Step 10: Local tiny real-data smoke run (needs network)**

Run the "Local (CPU)" block from the README exactly (with `uv run pytest` first).
Expected:
- `data-tiny/cs/text/shard_0000.txt` contains lowercase Czech lemmas, one sentence per line (`head -3` to inspect).
- `data-tiny/cs/stats.json` shows `tokens` > 100 000 and `pad: 5`.
- `data-tiny/benchmarks/cs/manifest.json` shows `analogy_cz_semantic` and `sim_ws353cs` with `kept > 0`.
- `runs-tiny/cs-causal-jepa-d64-s0/final_metrics.json` exists with `onnx_max_abs_diff` ≤ 1e-5.
Then delete `data-tiny/` and `runs-tiny/` (both ignored by git; add `runs-tiny/` to `.gitignore`).

- [ ] **Step 11: Commit**

```bash
printf 'runs-tiny/\n' >> .gitignore
git add jobs/ README.md .gitignore
git commit -m "feat(w2v): MetaCentrum PBS jobs and README"
```
