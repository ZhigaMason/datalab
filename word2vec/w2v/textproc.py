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
