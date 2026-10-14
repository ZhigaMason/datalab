import numpy as np
import onnxruntime as ort
import pytest
import torch

from w2v.data import Vocab
from w2v.export import export_onnx, load_onnx_embeddings, verify_onnx, write_vec


def test_export_and_verify_with_dynamic_shapes(tmp_path):
    torch.manual_seed(0)
    w = torch.randn(40, 6)
    path = export_onnx(w, tmp_path / "m.onnx")
    assert verify_onnx(path, w) <= 1e-5
    sess = ort.InferenceSession(str(path))
    assert [i.name for i in sess.get_inputs()] == ["ids"]
    assert [o.name for o in sess.get_outputs()] == ["embeddings"]
    out = sess.run(None, {"ids": np.array([[0, 5, 39]], dtype=np.int64)})[0]
    assert out.shape == (1, 3, 6)
    np.testing.assert_array_equal(out[0, 1], w[5].half().float().numpy())
    assert sess.run(None, {"ids": np.zeros((4, 13), dtype=np.int64)})[0].shape == (4, 13, 6)


def test_verify_onnx_detects_mismatch(tmp_path):
    w = torch.randn(10, 3)
    path = export_onnx(w, tmp_path / "m.onnx")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_onnx(path, w + 1.0)


def test_load_onnx_embeddings_and_write_vec(tmp_path):
    vocab = Vocab(["<pad>", "<unk>", "a", "b", "c"], [0, 0, 3, 2, 1])
    w = torch.randn(5, 4)
    export_onnx(w, tmp_path / "model.onnx")
    vocab.save(tmp_path / "vocab.json")
    words, mat = load_onnx_embeddings(tmp_path, chunk=2)
    assert words == vocab.words
    np.testing.assert_array_equal(mat, w.half().float().numpy())
    write_vec(tmp_path / "v.vec", words, mat)
    lines = (tmp_path / "v.vec").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "5 4" and lines[3].startswith("a ") and len(lines) == 6


def test_verify_onnx_fails_on_nan_weights(tmp_path):
    w = torch.randn(10, 3)
    w[0] = float("nan")  # row 0 is always probed by verify_onnx
    path = export_onnx(w, tmp_path / "m.onnx")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_onnx(path, w)


def test_write_vec_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        write_vec(tmp_path / "v.vec", ["a", "b", "c"], np.zeros((2, 4), dtype=np.float32))


def test_export_is_single_self_contained_file(tmp_path):
    import shutil

    w = torch.randn(30, 5)
    src = tmp_path / "src"
    src.mkdir()
    export_onnx(w, src / "model.onnx")
    assert list(src.glob("*.data")) == [] and [p.name for p in src.iterdir()] == ["model.onnx"]
    dst = tmp_path / "elsewhere"
    dst.mkdir()
    shutil.copy(src / "model.onnx", dst / "model.onnx")
    out = ort.InferenceSession(str(dst / "model.onnx")).run(None, {"ids": np.array([[3, 29]], dtype=np.int64)})[0]
    np.testing.assert_array_equal(out[0, 1], w[29].half().float().numpy())


def test_export_rejects_weights_over_protobuf_limit(tmp_path):
    class Huge:
        def numel(self):
            return 2**30

        def detach(self):
            return self

    with pytest.raises(ValueError, match="2 GB"):
        export_onnx(Huge(), tmp_path / "m.onnx")


def test_export_stores_float16_table_and_returns_float32(tmp_path):
    import onnx
    from onnx import TensorProto

    torch.manual_seed(0)
    w = torch.randn(30, 8)
    path = export_onnx(w, tmp_path / "m.onnx")
    model = onnx.load(str(path))
    assert [(t.name, t.data_type, list(t.dims)) for t in model.graph.initializer] == \
        [("embeddings_fp16", TensorProto.FLOAT16, [30, 8])]
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    out = ort.InferenceSession(str(path)).run(None, {"ids": np.array([[3, 0, 29]], dtype=np.int64)})[0]
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], w[[3, 0, 29]].half().float().numpy())
    assert verify_onnx(path, w) == 0.0
