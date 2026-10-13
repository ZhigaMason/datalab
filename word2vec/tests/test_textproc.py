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
