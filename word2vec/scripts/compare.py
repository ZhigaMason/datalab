"""Aggregate final_metrics.json of all finished runs into summary.csv / summary.md.

    python -m scripts.compare --runs-dir runs --out results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ID_COLS = ["run", "lang", "variant", "dim", "window", "seed"]


def collect(runs_dir) -> list[dict]:
    rows = []
    for metrics_path in sorted(Path(runs_dir).glob("*/final_metrics.json")):
        cfg = json.loads((metrics_path.parent / "config.json").read_text())
        metrics = json.loads(metrics_path.read_text())
        variant = cfg["mode"] + ("-jepa" if cfg["jepa"] else "")
        rows.append({"run": metrics_path.parent.name, "lang": cfg["lang"], "variant": variant,
                     "dim": cfg["dim"], "window": cfg["window"], "seed": cfg["seed"], **metrics})
    rows.sort(key=lambda r: (r["lang"], r["variant"], r["dim"], r["window"], r["seed"]))
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
