import json

import pytest
import torch

from w2v.train import ema_tau, lr_lambda, parse_args, run_name, train


def test_run_name():
    assert run_name("cs", "causal", True, 300, 0) == "cs-causal-jepa-d300-s0"
    assert run_name("en", "bidir", False, 100, 2) == "en-bidir-d100-s2"


def test_schedules():
    assert lr_lambda(0, 100, 10) == pytest.approx(0.1)
    assert lr_lambda(9, 100, 10) == pytest.approx(1.0)
    assert lr_lambda(100, 100, 10) == pytest.approx(0.0)
    assert ema_tau(0, 100, 0.996) == pytest.approx(0.996)
    assert ema_tau(100, 100, 0.996) == pytest.approx(1.0)


def _args(root, *extra, mode="bidir", jepa=False, steps=6):
    argv = ["--lang", "en", "--mode", mode, "--dim", "16", "--batch", "4", "--seq-len", "64", "--neg", "32",
            "--max-steps", str(steps), "--log-every", "2", "--eval-every", "3", "--ckpt-every", "3",
            "--n-vicreg", "64", "--subsample-t", "0", "--device", "cpu", "--no-compile",
            "--data-dir", str(root / "data"), "--runs-dir", str(root / "runs"), *extra]
    if jepa:
        argv.append("--jepa")
    return parse_args(argv)


@pytest.mark.parametrize("mode", ["bidir", "causal"])
@pytest.mark.parametrize("jepa", [False, True])
def test_train_end_to_end(tiny_data, mode, jepa):
    run_dir = train(_args(tiny_data, "--write-vec", mode=mode, jepa=jepa))
    assert run_dir.name == run_name("en", mode, jepa, 16, 0)
    for f in ("config.json", "metrics.jsonl", "ckpt.pt", "model.onnx", "vocab.json", "final_metrics.json",
              "vectors.vec"):
        assert (run_dir / f).exists(), f
    final = json.loads((run_dir / "final_metrics.json").read_text())
    assert final["onnx_max_abs_diff"] <= 1e-5 and "sim_toy/spearman" in final and "collapse_warning" in final
    records = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_recs = [r for r in records if "loss" in r]
    assert [r["step"] for r in train_recs] == [2, 4, 6]
    assert all(r["loss"] == r["loss"] for r in train_recs)  # not NaN
    assert any("sim_toy/coverage" in r for r in records)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["dim"] == 16 and cfg["jepa"] == jepa and cfg["vocab_size"] == 52


def test_resume_matches_uninterrupted_run(tiny_data):
    straight = train(_args(tiny_data, "--run-dir", str(tiny_data / "runs/straight")))
    interrupted = tiny_data / "runs/interrupted"
    train(_args(tiny_data, "--run-dir", str(interrupted), "--stop-after", "3"))
    assert torch.load(interrupted / "ckpt.pt", weights_only=False)["step"] == 3
    assert not (interrupted / "model.onnx").exists()
    train(_args(tiny_data, "--run-dir", str(interrupted)))
    a = torch.load(straight / "ckpt.pt", weights_only=False)
    b = torch.load(interrupted / "ckpt.pt", weights_only=False)
    assert a["step"] == b["step"] == 6
    torch.testing.assert_close(a["model"]["emb.weight"], b["model"]["emb.weight"])


def test_resume_refuses_changed_config(tiny_data):
    run_dir = str(tiny_data / "runs/x")
    train(_args(tiny_data, "--run-dir", run_dir, "--stop-after", "1"))
    with pytest.raises(ValueError, match="different config"):
        train(_args(tiny_data, "--run-dir", run_dir, "--lr", "0.1"))


def test_window_wider_than_encoded_pad_is_rejected(tiny_data):
    with pytest.raises(ValueError, match="pad"):
        train(_args(tiny_data, "--window", "6"))


def test_jepa_resume_matches_uninterrupted_run(tiny_data):
    straight = train(_args(tiny_data, "--run-dir", str(tiny_data / "runs/straight"), jepa=True))
    interrupted = tiny_data / "runs/interrupted"
    train(_args(tiny_data, "--run-dir", str(interrupted), "--stop-after", "3", jepa=True))
    train(_args(tiny_data, "--run-dir", str(interrupted), jepa=True))
    a = torch.load(straight / "ckpt.pt", weights_only=False)["model"]
    b = torch.load(interrupted / "ckpt.pt", weights_only=False)["model"]
    for k in ("emb.weight", "emb_ema.weight", "predictor.weight", "predictor.bias"):
        torch.testing.assert_close(a[k], b[k])
