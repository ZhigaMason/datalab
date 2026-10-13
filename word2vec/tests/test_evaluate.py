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


def test_eval_similarity_accepts_matrix_requiring_grad():
    index = {"a": 0, "b": 1, "c": 2}
    m = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]]).requires_grad_(True)
    r = eval_similarity([("a", "b", 10.0), ("a", "c", 1.0), ("b", "c", 2.0)], index, m)
    assert r["spearman"] == pytest.approx(1.0)
