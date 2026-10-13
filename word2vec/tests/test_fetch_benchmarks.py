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


def test_parse_vecto_csv_skips_blank_trailing_rows():
    text = ",word1,word2,similarity\n0,tiger,cat,7.35\n203,,,\n"
    assert parse_vecto_csv(text) == [("tiger", "cat", 7.35)]
