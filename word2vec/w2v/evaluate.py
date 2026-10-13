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


@torch.no_grad()
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
