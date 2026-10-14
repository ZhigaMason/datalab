"""Convert finished runs' float16 ONNX artifacts to the int8 format of w2v.export, in place.

The original is kept as model_fp16.onnx. With --eval, the int8 table is re-evaluated on the run's
benchmarks and written to final_metrics_int8.json next to final_metrics.json.
    python -m scripts.quantize_onnx --runs-dir runs --glob '*-w5-*' --eval --data-dir data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from w2v.evaluate import evaluate_all, load_benchmarks
from w2v.export import export_onnx, load_onnx_embeddings, verify_onnx


def is_int8(path: Path) -> bool:
    return any(t.name == "embeddings_int8" for t in onnx.load(str(path), load_external_data=False).graph.initializer)


def convert(run_dir: Path) -> float | None:
    """Quantize run_dir/model.onnx; returns the verify diff, or None when it is already int8."""
    model, backup, tmp = run_dir / "model.onnx", run_dir / "model_fp16.onnx", run_dir / ".model_int8.onnx"
    if is_int8(model):
        return None
    _, matrix = load_onnx_embeddings(run_dir)
    weight = torch.from_numpy(matrix)
    export_onnx(weight, tmp)
    diff = verify_onnx(tmp, weight)
    model.replace(backup)
    tmp.replace(model)
    return diff


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--glob", default="*", help="run directory name pattern")
    p.add_argument("--eval", action="store_true", help="re-evaluate the int8 table on the run's benchmarks")
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)
    runs = sorted(d for d in Path(a.runs_dir).glob(a.glob) if (d / "model.onnx").is_file())
    for d in runs:
        diff = convert(d)
        line = f"{d.name}: " + ("already int8" if diff is None else f"converted (max abs diff {diff:.2g})")
        if a.eval and not (d / "final_metrics_int8.json").exists():
            lang = json.loads((d / "config.json").read_text())["lang"]
            words, matrix = load_onnx_embeddings(d)
            res = evaluate_all(load_benchmarks(Path(a.data_dir) / "benchmarks" / lang),
                               {w: i for i, w in enumerate(words)}, torch.from_numpy(matrix))
            (d / "final_metrics_int8.json").write_text(json.dumps(res, indent=2))
            ref = json.loads((d / "final_metrics.json").read_text())
            worst = max((abs(v - ref[k]), k) for k, v in res.items() if isinstance(ref.get(k), (int, float)))
            line += f"; largest metric change {worst[0]:.4f} ({worst[1]})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
