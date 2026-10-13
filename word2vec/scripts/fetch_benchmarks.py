"""Download word-similarity / analogy benchmarks and normalize them like the corpus.

en: lowercased word forms. cs/uk: lemmatized with the same lemmatizer as the corpus (w2v.textproc.make_lemmatizer);
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

from w2v.textproc import UDPIPE_MODELS, make_lemmatizer, normalize_token

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
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["word1"] and r["word2"] and r["similarity"]]
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
        lemmatizer = make_lemmatizer(a.lang, data_dir / "models" / UDPIPE_MODELS[a.lang][0])
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
