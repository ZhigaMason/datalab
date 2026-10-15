"""Write web/test/fixtures/tiny.onnx and tiny.json (the expected int8 rows and scales) for the web tests."""
import json
from pathlib import Path

import torch

from w2v.export import export_onnx, quantize_int8

OUT = Path(__file__).resolve().parents[1] / "web" / "test" / "fixtures"


def main():
    weight = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [1.0, 0.5, -0.25], [-2.0, 1.0, 0.0], [0.3, 0.3, 0.3]])
    OUT.mkdir(parents=True, exist_ok=True)
    export_onnx(weight, OUT / "tiny.onnx")
    q, scale = quantize_int8(weight)
    (OUT / "tiny.json").write_text(json.dumps({"dims": list(q.shape), "int8": q.flatten().tolist(),
                                               "scales": scale.flatten().tolist()}))


if __name__ == "__main__":
    main()
