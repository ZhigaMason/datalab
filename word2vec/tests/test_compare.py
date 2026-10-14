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
    _run(runs, "cs-bidir-jepa-d300-s0", {"lang": "cs", "mode": "bidir", "jepa": True, "dim": 300, "window": 5, "seed": 0},
         {"analogy_cz_semantic/acc": 0.41234, "collapse_warning": False})
    _run(runs, "en-causal-d100-s0", {"lang": "en", "mode": "causal", "jepa": False, "dim": 100, "window": 10, "seed": 0},
         {"sim_simlex999/spearman": 0.3})
    _run(runs, "unfinished", {"lang": "en", "mode": "bidir", "jepa": False, "dim": 100, "window": 10, "seed": 0}, None)
    rows = collect(runs)
    assert [r["variant"] for r in rows] == ["bidir-jepa", "causal"]
    main(["--runs-dir", str(runs), "--out", str(tmp_path / "results")])
    with open(tmp_path / "results/summary.csv") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 2 and table[0]["lang"] == "cs" and table[0]["dim"] == "300" and table[1]["window"] == "10"
    md = (tmp_path / "results/summary.md").read_text()
    assert "| lang |" in md and "0.412" in md and "–" in md
