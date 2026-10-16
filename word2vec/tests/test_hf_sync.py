from scripts.hf_sync import plan


def test_plan_replaces_root_runs_and_keeps_prefixes(tmp_path):
    for f in ("README.md", "vocab/en.json", "en-bidir-d16-w5-s0/model_int8.onnx", "en-bidir-d16-w5-s0/metrics.jsonl",
              ".cache/huggingface/upload/x.metadata"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("x")
    repo = [".gitattributes", "README.md", "vocab/en.json",
            "en-bidir-d16-w5-s0/model_int8.onnx", "en-bidir-d16-w5-s0/model_fp16.onnx",
            "e5/en-bidir-d16-w5-s0/model_int8.onnx", "e5-jepa-lemb0/en-bidir-jepa-d16-w5-s0/model_int8.onnx"]
    add, delete = plan(repo, tmp_path, ("e5-jepa-lemb0/",))
    assert add == ["README.md", "en-bidir-d16-w5-s0/metrics.jsonl", "en-bidir-d16-w5-s0/model_int8.onnx",
                   "vocab/en.json"]
    assert delete == ["e5/en-bidir-d16-w5-s0/model_int8.onnx", "en-bidir-d16-w5-s0/model_fp16.onnx"]
